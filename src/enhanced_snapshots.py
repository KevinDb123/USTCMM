"""Enhanced multi-step diffusion snapshot visualization.

Generates fine-grained denoising process snapshots (8-10 frames)
to better illustrate the "noise → structure" evolution.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.datasets import load_or_generate_dataset, make_tensor_dataset
from src.diffusion import DiffusionModel
from src.utils import ensure_dir, get_device, set_seed


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_diffusion_checkpoint(
    checkpoint_path: str, device: torch.device
) -> tuple[DiffusionModel, dict]:
    """Load a trained Diffusion model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = DiffusionModel(
        data_dim=config.get("input_dim", 2),
        timesteps=config.get("timesteps", 80),
        beta_start=config.get("beta_start", 1e-4),
        beta_end=config.get("beta_end", 2e-2),
        hidden_dim=config.get("hidden_dim", 128),
        depth=config.get("depth", 3),
        dropout=config.get("dropout", 0.0),
        schedule_type=config.get("schedule_type", "linear"),
        time_hidden_dim=config.get("time_hidden_dim", 0),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config


@torch.no_grad()
def generate_enhanced_snapshots(
    model: DiffusionModel,
    device: torch.device,
    n_samples: int,
    num_snapshots: int = 10,
) -> dict[int, np.ndarray]:
    """Generate enhanced snapshots with more frames."""
    model.eval()
    _sync_if_needed(device)

    # Create evenly-spaced snapshot steps from T-1 down to 0
    if num_snapshots >= model.timesteps:
        snapshot_steps = list(range(model.timesteps - 1, -1, -1))
    else:
        step_indices = np.linspace(0, model.timesteps - 1, num_snapshots, dtype=int)
        snapshot_steps = sorted(set(int(s) for s in step_indices), reverse=True)

    print(f"Snapshot steps: {snapshot_steps}")

    x = torch.randn(n_samples, model.data_dim, device=device)
    snapshots: dict[int, np.ndarray] = {}
    # Record initial noise
    snapshots[model.timesteps] = x.detach().cpu().numpy()

    schedule = list(reversed(range(model.timesteps)))
    for step in schedule:
        t = torch.full((x.size(0),), step, device=x.device, dtype=torch.long)
        eps_pred = model.noise_predictor(x, t)
        alpha_t = model.alphas[step]
        alpha_bar_t = model.alpha_bars[step]
        beta_t = model.betas[step]
        coef = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        mean = (x - coef * eps_pred) / torch.sqrt(alpha_t)
        if step > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(model.posterior_var[step]) * noise
        else:
            x = mean
        if step in snapshot_steps:
            snapshots[step] = x.detach().cpu().numpy()

    _sync_if_needed(device)
    return snapshots


def plot_enhanced_snapshots(
    snapshots: dict[int, np.ndarray],
    title: str,
    save_path: str,
    max_cols: int = 5,
) -> None:
    """Plot enhanced multi-step snapshots in a grid layout."""
    steps = sorted(snapshots.keys(), reverse=True)
    n_total = len(steps)
    n_cols = min(max_cols, n_total)
    n_rows = (n_total + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    for idx, step in enumerate(steps):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        pts = snapshots[step]
        if pts.shape[1] > 2:
            # PCA for higher dimensions
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            pts = pca.fit_transform(pts)
        ax.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.6, c="#9467bd", linewidths=0)
        label = f"t={step}" if step < 1000 else "t=T (noise)"
        ax.set_title(label, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused subplots
    for idx in range(n_total, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved enhanced snapshots to {save_path}")


def run_enhanced_snapshots(
    checkpoint_path: str,
    dataset: str,
    data_root: str = "data",
    figure_dir: str = "outputs/figures",
    num_snapshots: int = 10,
    n_samples: int = 2000,
    seed: int = 42,
) -> None:
    """Generate and plot enhanced diffusion snapshots."""
    set_seed(seed)
    device = get_device()
    ensure_dir(figure_dir)

    print(f"Loading checkpoint: {checkpoint_path}")
    model, config = load_diffusion_checkpoint(checkpoint_path, device)
    print(f"Model config: T={config.get('timesteps', 80)}, dim={config.get('input_dim', 2)}")

    snapshots = generate_enhanced_snapshots(
        model, device, n_samples=n_samples, num_snapshots=num_snapshots
    )

    plot_enhanced_snapshots(
        snapshots,
        title=f"Diffusion Denoising Process - {dataset}",
        save_path=f"{figure_dir}/diffusion_{dataset}_enhanced_snapshots.png",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate enhanced multi-step diffusion snapshots."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to Diffusion checkpoint (.pt file)",
    )
    parser.add_argument("--dataset", type=str, default="spiral")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--figure-dir", type=str, default="outputs/figures")
    parser.add_argument("--num-snapshots", type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_enhanced_snapshots(
        checkpoint_path=args.checkpoint,
        dataset=args.dataset,
        data_root=args.data_root,
        figure_dir=args.figure_dir,
        num_snapshots=args.num_snapshots,
        n_samples=args.n_samples,
        seed=args.seed,
    )
