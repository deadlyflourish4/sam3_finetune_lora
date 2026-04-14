import yaml
import torch 
import json
from collections import defaultdict
import os
from tqdm import tqdm
from pathlib import Path

from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# SAM3 Imports
from sam3.model_builder import build_sam3_image_model
from sam3.model.model_misc import SAM3Output
from sam3.train.loss.loss_fns import IABCEMdetr, Boxes, Masks, CORE_LOSS_KEY
from sam3.train.loss.sam3_loss import Sam3LossWrapper
from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher
from sam3.train.data.collator import collate_fn_api
from sam3_finetune_lora.lora.lora_layers import LoRAConfig, apply_lora_to_model, save_lora_weights, count_parameters

from sam3_finetune_lora.utils.utils import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_world_size,
    get_rank,
    print_rank0,
    # launch_distributed_training
)

from sam3_finetune_lora.data.dataset import COCOSegmentDataset

class SAM3TrainerNative:
    def __init__(self, config_path, multi_gpu=False):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Multi-GPU setup
        self.multi_gpu = multi_gpu
        self.local_rank = 0
        self.world_size = 1

        if self.multi_gpu:
            self.local_rank = setup_distributed()
            self.world_size = get_world_size()
            self.device = torch.device(f"cuda:{self.local_rank}")
            print_rank0(f"Multi-GPU training enabled with {self.world_size} GPUs")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build Model
        print_rank0("Building SAM3 model...")
        # Load ckpt from local
        checkpoint_path = self.config["model"].get("checkpoint_path")
        load_from_hf = checkpoint_path is None
        training_cfg = self.config.get("training", {})
        self.enable_segmentation = self.config["model"].get(
            "enable_segmentation",
            training_cfg.get("enable_segmentation", True)
        )
        self.detection_only = training_cfg.get("detection_only", False)
        self.box_weights = {
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
        }
        self.cls_weights = {
            "loss_ce": 20.0,
            "presence_loss": 20.0,
        }
        self.mask_weights = {
            "loss_mask": 200.0,
            "loss_dice": 10.0,
        }
        if self.detection_only:
            self.mask_weights["loss_mask"] = 0.0
            self.mask_weights["loss_dice"] = 0.0
        self.mask_loss_enabled = any(
            self.mask_weights.get(key, 0.0) != 0.0
            for key in ("loss_mask", "loss_dice")
        )
        self.loss_weights = {
            **self.cls_weights,
            **self.box_weights,
            **self.mask_weights,
        }

        self.model = build_sam3_image_model(
            device=self.device.type,
            compile=False,
            load_from_HF=load_from_hf,  # Tries to download from HF if checkpoint_path is None
            checkpoint_path=checkpoint_path,
            enable_segmentation=self.enable_segmentation,
            bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            eval_mode=False
        )

        # Apply LoRA
        print_rank0("Applying LoRA...")
        lora_cfg = self.config["lora"]
        lora_config = LoRAConfig(
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
            apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
            apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
            apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
            apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
            apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
        )
        self.model = apply_lora_to_model(self.model, lora_config)

        stats = count_parameters(self.model)
        print_rank0(f"Trainable params: {stats['trainable_parameters']:,} ({stats['trainable_percentage']:.2f}%)")

        self.model.to(self.device)

        # Wrap model with DDP if multi-GPU
        if self.multi_gpu:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False  # Frozen params (requires_grad=False) don't need this flag
            )
            print_rank0(f"Model wrapped with DistributedDataParallel")

        # Store reference to unwrapped model for accessing custom methods
        self._unwrapped_model = self.model.module if self.multi_gpu else self.model

        # Optimizer
        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=float(self.config["training"]["learning_rate"]),
            weight_decay=self.config["training"]["weight_decay"]
        )
        
        # Matcher & Loss
        self.matcher = BinaryHungarianMatcherV2(
            cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, focal=True
        )

        # Create loss functions with correct weights (from original SAM3 training config)
        # Note: These weights are for mask-based training
        loss_fns = [
            Boxes(weight_dict=self.box_weights),
            IABCEMdetr(
                pos_weight=10.0,
                weight_dict=self.cls_weights,
                pos_focal=False,
                alpha=0.25,
                gamma=2,
                use_presence=True,
                pad_n_queries=200,
            ),
            Masks(
                weight_dict=self.mask_weights,
                focal_alpha=0.25,
                focal_gamma=2.0,
                compute_aux=False
            )
        ]

        # Create one-to-many matcher for auxiliary outputs
        o2m_matcher = BinaryOneToManyMatcher(
            alpha=0.3,
            threshold=0.4,
            topk=4
        )

        # Use Sam3LossWrapper for proper loss computation
        self.loss_wrapper = Sam3LossWrapper(
            loss_fns_find=loss_fns,
            matcher=self.matcher,
            o2m_matcher=o2m_matcher,
            o2m_weight=2.0,
            use_o2m_matcher_on_o2m_aux=False,
            normalization="local",  # Use local normalization (no distributed training)
            normalize_by_valid_object_num=False,
        )
        print_rank0(
            "Segmentation head: "
            f"{'enabled' if self.enable_segmentation else 'disabled'} | "
            f"Training mode: {'detection-only' if self.detection_only else 'detection+segmentation'}"
        )
        print_rank0(
            "Loss configuration: "
            f"enable_segmentation={self.enable_segmentation} | "
            f"detection_only={self.detection_only} | "
            f"mask_loss_enabled={self.mask_loss_enabled}"
        )
        print_rank0(f"Loss weights: {self.loss_weights}")

    @staticmethod
    def _extract_scalar_losses(loss_dict):
        scalar_losses = {}
        for key, value in loss_dict.items():
            if key != CORE_LOSS_KEY and "loss" not in key:
                continue
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                scalar_losses[key] = float(value.detach().item())
            elif isinstance(value, (int, float)):
                scalar_losses[key] = float(value)
        return scalar_losses

    @staticmethod
    def _update_loss_meter(loss_meter, loss_dict):
        for key, value in SAM3TrainerNative._extract_scalar_losses(loss_dict).items():
            loss_meter[key].append(value)

    @staticmethod
    def _average_loss_meter(loss_meter):
        return {
            key: sum(values) / len(values)
            for key, values in loss_meter.items()
            if values
        }

    @staticmethod
    def _format_loss_summary(losses):
        ordered_keys = [
            CORE_LOSS_KEY,
            "loss_ce",
            "presence_loss",
            "loss_bbox",
            "loss_giou",
            "loss_mask",
            "loss_dice",
        ]
        shown = []
        for key in ordered_keys:
            if key in losses:
                shown.append(f"{key}={losses[key]:.6f}")
        for key in sorted(losses):
            if key not in ordered_keys:
                shown.append(f"{key}={losses[key]:.6f}")
        return ", ".join(shown)
        
    def train(self):
        # Extract explicit paths if provided, or fallback to data_dir
        train_img_dir = self.config["training"].get("img_folder_train")
        train_ann_file = self.config["training"].get("ann_file_train")
        val_img_dir = self.config["training"].get("img_folder_val")
        val_ann_file = self.config["training"].get("ann_file_val")
        
        # If explicit paths aren't given, build backward-compatible paths
        if not train_img_dir:
            data_dir = self.config["training"].get("data_dir", "/workspace/data")
            train_img_dir = os.path.join(data_dir, "train")
            train_ann_file = os.path.join(data_dir, "train", "_annotations.coco.json")
            val_img_dir = os.path.join(data_dir, "valid")
            val_ann_file = os.path.join(data_dir, "valid", "_annotations.coco.json")

        # Load datasets using COCO format
        print_rank0(f"\nLoading training data from {train_img_dir}...")
        print_rank0(f"Annotation file: {train_ann_file}")
        train_ds = COCOSegmentDataset(img_dir=train_img_dir, ann_file=train_ann_file, split="train")

        # Check if validation data exists
        has_validation = False
        val_ds = None

        try:
            print_rank0(f"\nLoading validation data from {val_img_dir}...")
            print_rank0(f"Annotation file: {val_ann_file}")
            val_ds = COCOSegmentDataset(img_dir=val_img_dir, ann_file=val_ann_file, split="valid")
            if len(val_ds) > 0:
                has_validation = True
                print_rank0(f"Found validation data: {len(val_ds)} images")
            else:
                print_rank0(f"Validation dataset is empty.")
                val_ds = None
        except Exception as e:
            print_rank0(f"Could not load validation data: {e}")
            val_ds = None

        if not has_validation:
            val_ds = None

        def collate_fn(batch):
            return collate_fn_api(batch, dict_key="input", with_seg_masks=True)

        # Create samplers for distributed training
        train_sampler = None
        val_sampler = None

        if self.multi_gpu:
            train_sampler = DistributedSampler(
                train_ds,
                num_replicas=self.world_size,
                rank=get_rank(),
                shuffle=True
            )
            if has_validation:
                val_sampler = DistributedSampler(
                    val_ds,
                    num_replicas=self.world_size,
                    rank=get_rank(),
                    shuffle=False
                )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config["training"]["batch_size"],
            shuffle=(train_sampler is None),  # Only shuffle if not using sampler
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=self.config["training"].get("num_workers", 0),
            pin_memory=True
        )

        if has_validation:
            val_loader = DataLoader(
                val_ds,
                batch_size=self.config["training"]["batch_size"],
                shuffle=False,
                sampler=val_sampler,
                collate_fn=collate_fn,
                num_workers=self.config["training"].get("num_workers", 0),
                pin_memory=True
            )
        else:
            val_loader = None

        self.model.train()

        # Weights from a standard SAM config roughly
        weight_dict = {
            "loss_ce": 2.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_mask": 5.0,
            "loss_dice": 5.0
        }

        epochs = self.config["training"]["num_epochs"]
        best_val_loss = float('inf')
        print_rank0(f"Starting training for {epochs} epochs...")

        if has_validation:
            print_rank0(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
        else:
            print_rank0(f"Training samples: {len(train_ds)}")
            print_rank0("⚠️  No validation data found - training without validation")

        if self.multi_gpu:
            print_rank0(f"Effective batch size: {self.config['training']['batch_size']} x {self.world_size} = {self.config['training']['batch_size'] * self.world_size}")

        # Helper to move BatchedDatapoint to device
        def move_to_device(obj, device):
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            elif isinstance(obj, list):
                return [move_to_device(x, device) for x in obj]
            elif isinstance(obj, tuple):
                return tuple(move_to_device(x, device) for x in obj)
            elif isinstance(obj, dict):
                return {k: move_to_device(v, device) for k, v in obj.items()}
            elif hasattr(obj, "__dataclass_fields__"):
                for field in obj.__dataclass_fields__:
                    val = getattr(obj, field)
                    setattr(obj, field, move_to_device(val, device))
                return obj
            return obj

        # Create output directory
        out_dir = Path(self.config["output"]["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            # Set epoch for distributed sampler (required for proper shuffling)
            if self.multi_gpu and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            # Track training losses for this epoch
            train_losses = []
            train_loss_meter = defaultdict(list)

            # Only show progress bar on rank 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not is_main_process())
            for batch_dict in pbar:
                input_batch = batch_dict["input"]

                # Move to device
                input_batch = move_to_device(input_batch, self.device)

                # Forward pass
                # outputs_list is SAM3Output, we need to pass the whole thing to loss_wrapper
                outputs_list = self.model(input_batch)

                # Prepare targets for loss
                # input_batch.find_targets is a list of BatchedFindTarget (one per stage)
                find_targets = [self._unwrapped_model.back_convert(target) for target in input_batch.find_targets]

                # Move targets to device
                for targets in find_targets:
                    for k, v in targets.items():
                        if isinstance(v, torch.Tensor):
                            targets[k] = v.to(self.device)

                # Add matcher indices to outputs (required by Sam3LossWrapper)
                # Use SAM3Output.iteration_mode to properly iterate over outputs
                with SAM3Output.iteration_mode(
                    outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
                ) as outputs_iter:
                    for stage_outputs, stage_targets in zip(outputs_iter, find_targets):
                        # stage_targets is a single target dict, replicate for all steps
                        stage_targets_list = [stage_targets] * len(stage_outputs)
                        for outputs, targets in zip(stage_outputs, stage_targets_list):
                            # Compute indices for main output
                            outputs["indices"] = self.matcher(outputs, targets)

                            # Also add indices to auxiliary outputs if they exist
                            if "aux_outputs" in outputs:
                                for aux_out in outputs["aux_outputs"]:
                                    aux_out["indices"] = self.matcher(aux_out, targets)

                # Compute loss using Sam3LossWrapper
                # This handles num_boxes calculation and proper weighting
                loss_dict = self.loss_wrapper(outputs_list, find_targets)
                
                # Extract total loss
                total_loss = loss_dict[CORE_LOSS_KEY]

                # Backward
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # Track training loss
                train_losses.append(total_loss.item())
                self._update_loss_meter(train_loss_meter, loss_dict)
                postfix = {"loss": f"{total_loss.item():.4f}"}
                if "loss_mask" in loss_dict:
                    postfix["mask"] = f"{loss_dict['loss_mask'].item():.4f}"
                if "loss_ce" in loss_dict:
                    postfix["ce"] = f"{loss_dict['loss_ce'].item():.4f}"
                pbar.set_postfix(postfix)

            # Calculate average training loss for this epoch
            avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0.0
            avg_train_components = self._average_loss_meter(train_loss_meter)

            # Validation (only compute loss - no metrics, like SAM3)
            if has_validation and val_loader is not None:
                self.model.eval()
                val_losses = []
                val_loss_meter = defaultdict(list)

                with torch.no_grad():
                    val_pbar = tqdm(val_loader, desc=f"Validation", disable=not is_main_process())

                    for batch_dict in val_pbar:
                        input_batch = batch_dict["input"]
                        input_batch = move_to_device(input_batch, self.device)

                        # Forward pass
                        outputs_list = self.model(input_batch)

                        # Prepare targets
                        find_targets = [self._unwrapped_model.back_convert(target) for target in input_batch.find_targets]

                        # Move targets to device
                        for targets in find_targets:
                            for k, v in targets.items():
                                if isinstance(v, torch.Tensor):
                                    targets[k] = v.to(self.device)

                        # Add matcher indices to outputs (required by Sam3LossWrapper)
                        with SAM3Output.iteration_mode(
                            outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
                        ) as outputs_iter:
                            for stage_outputs, stage_targets in zip(outputs_iter, find_targets):
                                stage_targets_list = [stage_targets] * len(stage_outputs)
                                for outputs, targets in zip(stage_outputs, stage_targets_list):
                                    outputs["indices"] = self.matcher(outputs, targets)
                                    if "aux_outputs" in outputs:
                                        for aux_out in outputs["aux_outputs"]:
                                            aux_out["indices"] = self.matcher(aux_out, targets)

                        # Compute loss using Sam3LossWrapper
                        loss_dict = self.loss_wrapper(outputs_list, find_targets)
                        total_loss = loss_dict[CORE_LOSS_KEY]

                        val_losses.append(total_loss.item())
                        self._update_loss_meter(val_loss_meter, loss_dict)
                        postfix = {"val_loss": f"{total_loss.item():.4f}"}
                        if "loss_mask" in loss_dict:
                            postfix["val_mask"] = f"{loss_dict['loss_mask'].item():.4f}"
                        if "loss_ce" in loss_dict:
                            postfix["val_ce"] = f"{loss_dict['loss_ce'].item():.4f}"
                        val_pbar.set_postfix(postfix)

                avg_val_loss = sum(val_losses) / len(val_losses)
                avg_val_components = self._average_loss_meter(val_loss_meter)

                # Synchronize val_loss across all processes for consistent best model selection
                if self.multi_gpu:
                    val_loss_tensor = torch.tensor([avg_val_loss], device=self.device)
                    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                    avg_val_loss = val_loss_tensor.item()

                print_rank0(f"\nEpoch {epoch+1}/{epochs}")
                print_rank0(f"  Train: {self._format_loss_summary(avg_train_components)}")
                print_rank0(f"  Val:   {self._format_loss_summary(avg_val_components)}")

                # Save models based on validation loss (only on rank 0)
                if is_main_process():
                    # Get underlying model from DDP wrapper
                    model_to_save = self.model.module if self.multi_gpu else self.model
                    save_lora_weights(model_to_save, str(out_dir / "last_lora_weights.pt"))

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        save_lora_weights(model_to_save, str(out_dir / "best_lora_weights.pt"))
                        print(f"✓ New best model saved (val_loss: {avg_val_loss:.6f})")

                    # Log to file
                    stats_row = {
                        "epoch": epoch + 1,
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss
                    }
                    stats_row.update({f"train_{k}": v for k, v in avg_train_components.items()})
                    stats_row.update({f"val_{k}": v for k, v in avg_val_components.items()})
                    with open(out_dir / "val_stats.json", "a") as f:
                        f.write(json.dumps(stats_row) + "\n")

                torch.cuda.empty_cache()

                # Back to training mode
                self.model.train()
            else:
                print_rank0(f"\nEpoch {epoch+1}/{epochs}")
                print_rank0(f"  Train: {self._format_loss_summary(avg_train_components)}")

                # No validation - just save model each epoch (only on rank 0)
                if is_main_process():
                    model_to_save = self.model.module if self.multi_gpu else self.model
                    save_lora_weights(model_to_save, str(out_dir / "last_lora_weights.pt"))

        # Synchronize before final save
        if self.multi_gpu:
            dist.barrier()

        # Final save (only on rank 0)
        if is_main_process():
            if has_validation:
                print(f"\n{'='*80}")
                print(f"✅ Training complete!")
                print(f"{'='*80}")
                print(f"Best validation loss: {best_val_loss:.6f}")
                print(f"\nModels saved to {out_dir}:")
                print(f"  - best_lora_weights.pt (best validation loss)")
                print(f"  - last_lora_weights.pt (last epoch)")
                print(f"\n📊 To compute full metrics (mAP, cgF1) with NMS:")
                print(f"   python validate_sam3_lora.py \\")
                print(f"     --config <config_path> \\")
                print(f"     --weights {out_dir}/best_lora_weights.pt \\")
                print(f"     --val_data_dir <data_dir>/valid")
                print(f"{'='*80}")
            else:
                # If no validation, copy last to best
                import shutil
                last_path = out_dir / "last_lora_weights.pt"
                best_path = out_dir / "best_lora_weights.pt"
                if last_path.exists():
                    shutil.copy(last_path, best_path)

                print(f"\n{'='*80}")
                print(f"✅ Training complete!")
                print(f"{'='*80}")
                print(f"\nModels saved to {out_dir}:")
                print(f"  - best_lora_weights.pt (copy of last epoch)")
                print(f"  - last_lora_weights.pt (last epoch)")
                print(f"\nℹ️  No validation data - consider adding data/valid/ for better model selection")
                print(f"{'='*80}")

        # Cleanup distributed training
        if self.multi_gpu:
            cleanup_distributed()