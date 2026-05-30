from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from src.utils import ensure_dir


def _finalize_figure(fig: plt.Figure, save_path: str | Path | None = None) -> None:
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _project_for_plot(*arrays: np.ndarray) -> list[np.ndarray]:
    dims = [arr.shape[1] for arr in arrays if len(arr) > 0]
    if not dims:
        return [arr for arr in arrays]
    if max(dims) <= 2:
        return [arr[:, :2] if arr.shape[1] > 1 else np.pad(arr, ((0, 0), (0, 1))) for arr in arrays]
    stacked = np.concatenate(arrays, axis=0)
    pca = PCA(n_components=2)
    projected = pca.fit_transform(stacked)
    outputs: list[np.ndarray] = []
    offset = 0
    for arr in arrays:
        outputs.append(projected[offset : offset + len(arr)])
        offset += len(arr)
    return outputs


def plot_dataset_splits(x_train: np.ndarray, x_test: np.ndarray, title: str, save_path: str | Path) -> None:
    x_train_plot, x_test_plot = _project_for_plot(x_train, x_test)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(x_train_plot[:, 0], x_train_plot[:, 1], s=4, alpha=0.65, c="#1f77b4")
    axes[0].set_title(f"{title} - train")
    axes[1].scatter(x_test_plot[:, 0], x_test_plot[:, 1], s=4, alpha=0.65, c="#ff7f0e")
    axes[1].set_title(f"{title} - test")
    for ax in axes:
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
    _finalize_figure(fig, save_path)


def plot_real_vs_generated(
    x_real: np.ndarray,
    x_generated: np.ndarray,
    title: str,
    save_path: str | Path,
) -> None:
    x_real_plot, x_generated_plot = _project_for_plot(x_real, x_generated)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].scatter(x_real_plot[:, 0], x_real_plot[:, 1], s=4, alpha=0.65, c="#1f77b4")
    axes[0].set_title("Real")
    axes[1].scatter(x_generated_plot[:, 0], x_generated_plot[:, 1], s=4, alpha=0.65, c="#2ca02c")
    axes[1].set_title("Generated")
    axes[2].scatter(x_real_plot[:, 0], x_real_plot[:, 1], s=4, alpha=0.35, c="#1f77b4", label="Real")
    axes[2].scatter(x_generated_plot[:, 0], x_generated_plot[:, 1], s=4, alpha=0.35, c="#d62728", label="Generated")
    axes[2].set_title("Overlay")
    axes[2].legend()
    fig.suptitle(title)
    for ax in axes:
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
    _finalize_figure(fig, save_path)


def plot_generated_only(x_generated: np.ndarray, title: str, save_path: str | Path) -> None:
    (x_generated_plot,) = _project_for_plot(x_generated)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(x_generated_plot[:, 0], x_generated_plot[:, 1], s=4, alpha=0.65, c="#2ca02c")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    _finalize_figure(fig, save_path)


def plot_training_curves(history: Mapping[str, list[float]], title: str, save_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, values in history.items():
        ax.plot(values, label=key)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.25)
    ax.legend()
    _finalize_figure(fig, save_path)


def plot_diffusion_snapshots(
    snapshots: Mapping[int, np.ndarray],
    title: str,
    save_path: str | Path,
) -> None:
    steps = sorted(snapshots.keys(), reverse=True)
    if not steps:
        return
    n_cols = min(4, len(steps))
    selected_steps = steps[:n_cols]
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]
    for ax, step in zip(axes, selected_steps):
        (pts,) = _project_for_plot(snapshots[step])
        ax.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.65, c="#9467bd")
        ax.set_title(f"t = {step}")
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
    fig.suptitle(title)
    _finalize_figure(fig, save_path)


def plot_conditional_real_vs_generated(
    real_by_name: Mapping[str, np.ndarray],
    generated_by_name: Mapping[str, np.ndarray],
    title: str,
    save_path: str | Path,
) -> None:
    names = list(real_by_name.keys())
    fig, axes = plt.subplots(len(names), 3, figsize=(11, 3.2 * len(names)))
    if len(names) == 1:
        axes = np.array([axes])
    for row, name in enumerate(names):
        x_real = real_by_name[name]
        x_gen = generated_by_name[name]
        x_real_plot, x_gen_plot = _project_for_plot(x_real, x_gen)
        panels = [
            ("Real", x_real_plot, "#1f77b4", 0.65),
            ("Generated", x_gen_plot, "#2ca02c", 0.65),
            ("Overlay", None, None, None),
        ]
        for col, (panel_title, pts, color, alpha) in enumerate(panels):
            ax = axes[row, col]
            if panel_title == "Overlay":
                ax.scatter(x_real_plot[:, 0], x_real_plot[:, 1], s=4, alpha=0.35, c="#1f77b4", label="Real")
                ax.scatter(x_gen_plot[:, 0], x_gen_plot[:, 1], s=4, alpha=0.35, c="#d62728", label="Generated")
                ax.legend(fontsize=8, loc="upper right")
            else:
                ax.scatter(pts[:, 0], pts[:, 1], s=4, alpha=alpha, c=color)
            ax.set_title(f"{name} - {panel_title}")
            ax.set_aspect("equal")
            ax.grid(alpha=0.2)
    fig.suptitle(title)
    _finalize_figure(fig, save_path)


def plot_robustness_curves(
    x_values: list[float],
    series: Mapping[str, list[float]],
    metric_name: str,
    title: str,
    save_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, values in series.items():
        ax.plot(x_values, values, marker="o", linewidth=1.8, label=label)
    ax.set_title(title)
    ax.set_xlabel("Contamination ratio")
    ax.set_ylabel(metric_name)
    ax.grid(alpha=0.25)
    ax.legend()
    _finalize_figure(fig, save_path)
