from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU for two sets of xyxy boxes."""
    if box1.numel() == 0 or box2.numel() == 0:
        return torch.zeros(
            (box1.shape[0], box2.shape[0]), device=box1.device, dtype=box1.dtype
        )

    area1 = ((box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0))
    area2 = ((box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0))

    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-9)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute AP from recall and precision curves using COCO-style interpolation."""
    if len(recall) == 0 or len(precision) == 0:
        return 0.0, np.array([1.0, 0.0]), np.array([0.0, 1.0])

    mrec = np.concatenate(([0.0], recall, [recall[-1]], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    ap = np.trapz(np.interp(x, mrec, mpre), x)
    return float(ap), mpre, mrec


def ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    eps: float = 1e-16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ultralytics-style precision/recall/AP computation."""
    if target_cls.size == 0:
        zeros = np.zeros(1, dtype=float)
        return zeros, zeros, zeros, zeros

    if conf.size:
        order = np.argsort(-conf)
        tp, conf, pred_cls = tp[order], conf[order], pred_cls[order]

    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]
    x = np.linspace(0, 1, 1000)
    ap = np.zeros((nc, tp.shape[1] if tp.ndim == 2 else 10), dtype=float)
    p_curve = np.zeros((nc, 1000), dtype=float)
    r_curve = np.zeros((nc, 1000), dtype=float)

    for ci, cls in enumerate(unique_classes):
        cls_mask = pred_cls == cls
        n_l = nt[ci]
        n_p = int(cls_mask.sum())
        if n_l == 0 or n_p == 0:
            continue

        fpc = (1 - tp[cls_mask]).cumsum(0)
        tpc = tp[cls_mask].cumsum(0)

        recall = tpc / (n_l + eps)
        precision = tpc / (tpc + fpc + eps)

        r_curve[ci] = np.interp(-x, -conf[cls_mask], recall[:, 0], left=0.0)
        p_curve[ci] = np.interp(-x, -conf[cls_mask], precision[:, 0], left=1.0)

        for j in range(tp.shape[1]):
            ap[ci, j], _, _ = compute_ap(recall[:, j], precision[:, j])

    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    if f1_curve.size == 0:
        zeros = np.zeros(1, dtype=float)
        return zeros, zeros, zeros, zeros

    best = smooth(f1_curve.mean(0), 0.1).argmax()
    p = p_curve[:, best]
    r = r_curve[:, best]
    ap50 = ap[:, 0] if ap.size else np.zeros(nc, dtype=float)
    ap5095 = ap.mean(1) if ap.size else np.zeros(nc, dtype=float)
    return p, r, ap50, ap5095


def match_predictions(
    pred_classes: torch.Tensor, true_classes: torch.Tensor, iou: torch.Tensor, iouv: torch.Tensor
) -> torch.Tensor:
    """Match predictions to targets across multiple IoU thresholds."""
    correct = np.zeros((pred_classes.shape[0], iouv.shape[0]), dtype=bool)
    if pred_classes.numel() == 0 or true_classes.numel() == 0:
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

    correct_class = true_classes[:, None] == pred_classes
    iou = (iou * correct_class).cpu().numpy()
    for i, threshold in enumerate(iouv.cpu().tolist()):
        matches = np.array(np.nonzero(iou >= threshold)).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)


def smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    """Box-filter smoothing used by Ultralytics when selecting max-F1 point."""
    if y.size == 0:
        return y
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


@dataclass
class DetectionMetrics:
    precision: float = 0.0
    recall: float = 0.0
    map50: float = 0.0
    map5095: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "metrics/precision(B)": self.precision,
            "metrics/recall(B)": self.recall,
            "metrics/mAP50(B)": self.map50,
            "metrics/mAP50-95(B)": self.map5095,
        }


class DetectionMetricsEvaluator:
    """Accumulate detection statistics and compute Ultralytics-style box metrics."""

    def __init__(self, iouv: torch.Tensor | None = None) -> None:
        self.iouv = iouv if iouv is not None else torch.linspace(0.5, 0.95, 10)
        self.stats = {"tp": [], "conf": [], "pred_cls": [], "target_cls": []}

    def update(
        self,
        pred_boxes: torch.Tensor,
        pred_scores: torch.Tensor,
        pred_labels: torch.Tensor,
        target_boxes: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> None:
        pred_boxes = pred_boxes.detach().to(device="cpu", dtype=torch.float32)
        pred_scores = pred_scores.detach().to(device="cpu", dtype=torch.float32)
        pred_labels = pred_labels.detach().to(device="cpu", dtype=torch.int64)
        target_boxes = target_boxes.detach().to(device="cpu", dtype=torch.float32)
        target_labels = target_labels.detach().to(device="cpu", dtype=torch.int64)

        if pred_boxes.numel() == 0:
            tp = np.zeros((0, self.iouv.numel()), dtype=bool)
        elif target_boxes.numel() == 0:
            tp = np.zeros((pred_boxes.shape[0], self.iouv.numel()), dtype=bool)
        else:
            iou = box_iou(target_boxes, pred_boxes)
            tp = match_predictions(pred_labels, target_labels, iou, self.iouv).cpu().numpy()

        self.stats["tp"].append(tp)
        self.stats["conf"].append(pred_scores.cpu().numpy())
        self.stats["pred_cls"].append(pred_labels.cpu().numpy())
        self.stats["target_cls"].append(target_labels.cpu().numpy())

    def state_dict(self) -> dict[str, list[np.ndarray]]:
        return self.stats

    def load_state_dict(self, state: dict[str, list[np.ndarray]]) -> None:
        self.stats = state

    def merge(self, states: list[dict[str, list[np.ndarray]]]) -> None:
        merged = {k: [] for k in self.stats}
        for state in states:
            for key, values in state.items():
                merged[key].extend(values)
        self.stats = merged

    def compute(self) -> DetectionMetrics:
        target_cls_chunks = [x for x in self.stats["target_cls"] if x.size]
        if not target_cls_chunks:
            return DetectionMetrics()

        tp = np.concatenate(
            self.stats["tp"], 0
        ) if self.stats["tp"] else np.zeros((0, self.iouv.numel()), dtype=bool)
        conf = np.concatenate(
            self.stats["conf"], 0
        ) if self.stats["conf"] else np.zeros((0,), dtype=float)
        pred_cls = np.concatenate(
            self.stats["pred_cls"], 0
        ) if self.stats["pred_cls"] else np.zeros((0,), dtype=float)
        target_cls = np.concatenate(target_cls_chunks, 0)

        p, r, ap50, ap5095 = ap_per_class(tp, conf, pred_cls, target_cls)
        return DetectionMetrics(
            precision=float(p.mean()) if p.size else 0.0,
            recall=float(r.mean()) if r.size else 0.0,
            map50=float(ap50.mean()) if ap50.size else 0.0,
            map5095=float(ap5095.mean()) if ap5095.size else 0.0,
        )
