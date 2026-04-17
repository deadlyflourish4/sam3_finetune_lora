import math
from collections.abc import Callable
from pathlib import Path

from . import LOGGER

LOSS_COLUMNS = [
    "train/core_loss",
    "train/loss_ce",
    "train/presence_loss",
    "train/loss_bbox",
    "train/loss_giou",
    "train/loss_mask",
    "train/loss_dice",
    "val/core_loss",
    "val/loss_ce",
    "val/presence_loss",
    "val/loss_bbox",
    "val/loss_giou",
    "val/loss_mask",
    "val/loss_dice",
]

METRIC_COLUMNS = [
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
]


def _make_figure(plt, columns, title):
    """Create a subplot grid sized for the number of columns to plot."""
    n = len(columns)
    if n == 0:
        return None, None

    if columns == LOSS_COLUMNS:
        fig, axes = plt.subplots(2, 7, figsize=(28, 8), tight_layout=True)
        axes = axes.ravel()
        fig.suptitle(title, fontsize=14)
        return fig, axes

    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(12, ncols * 4.5), max(4, nrows * 3.5)),
        tight_layout=True,
    )
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=14)
    return fig, axes


def plot_results(
    file: str = "path/to/results.csv", dir: str = "", on_plot: Callable | None = None
):
    """Plot training results from a results CSV file. The function supports various types of data including
    segmentation, pose estimation, and classification. Plots are saved as 'results.png' in the directory where the
    CSV is located.

    Args:
        file (str, optional): Path to the CSV file containing the training results.
        dir (str, optional): Directory where the CSV file is located if 'file' is not provided.
        on_plot (Callable, optional): Callback function to be executed after plotting. Takes filename as an argument.

    Examples:
        >>> from ultralytics.utils.plotting import plot_results
        >>> plot_results("path/to/results.csv")
    """
    import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'
    import polars as pl
    from scipy.ndimage import gaussian_filter1d

    save_dir = Path(file).parent if file else Path(dir)
    files = list(save_dir.glob("results*.csv"))
    assert len(
        files
    ), f"No results.csv files found in {save_dir.resolve()}, nothing to plot."

    loss_keys, metric_keys = [], []
    loss_fig, loss_axes = None, None
    metric_fig, metric_axes = None, None
    for i, f in enumerate(files):
        try:
            data = pl.read_csv(f, infer_schema_length=None)
            if i == 0:
                loss_keys = [c for c in LOSS_COLUMNS if c in data.columns]
                metric_keys = [c for c in METRIC_COLUMNS if c in data.columns]
                loss_fig, loss_axes = _make_figure(plt, loss_keys, "Training and Validation Losses")
                metric_fig, metric_axes = _make_figure(plt, metric_keys, "Validation Metrics")
            x = data.select(data.columns[0]).to_numpy().flatten()

            for idx, column in enumerate(loss_keys):
                y = data.select(column).to_numpy().flatten().astype("float")
                loss_axes[idx].plot(
                    x, y, marker=".", label=f.stem, linewidth=2, markersize=8
                )
                loss_axes[idx].plot(
                    x, gaussian_filter1d(y, sigma=3), ":", label="smooth", linewidth=2
                )
                loss_axes[idx].set_title(column, fontsize=12)

            for idx, column in enumerate(metric_keys):
                y = data.select(column).to_numpy().flatten().astype("float")
                metric_axes[idx].plot(
                    x, y, marker=".", label=f.stem, linewidth=2, markersize=8
                )
                metric_axes[idx].plot(
                    x, gaussian_filter1d(y, sigma=3), ":", label="smooth", linewidth=2
                )
                metric_axes[idx].set_title(column, fontsize=12)
        except Exception as e:
            LOGGER.error(f"Plotting error for {f}: {e}")

    if loss_axes is not None:
        loss_axes[0].legend()
        loss_fname = save_dir / "results_losses.png"
        loss_fig.savefig(loss_fname, dpi=200)
        plt.close(loss_fig)
        if on_plot:
            on_plot(loss_fname)

    if metric_axes is not None:
        metric_axes[0].legend()
        metric_fname = save_dir / "results_metrics.png"
        metric_fig.savefig(metric_fname, dpi=200)
        plt.close(metric_fig)
        if on_plot:
            on_plot(metric_fname)
