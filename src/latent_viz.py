"""Latent space visualization for VAE models.

Generates latent space traversals, interpolations, and grid visualizations
to understand the structure learned by the VAE's latent representation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from src.datasets import AVAILABLE_DATASETS, load_or_generate_dataset, make_tensor_dataset
from src.utils import ensure_dir, get_device, set_seed
from src.vae import VAE


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_vae_checkpoint(
    checkpoint_path: str, device: torch.device
) -> tuple[VAE, dict]:
    """Load a trained VAE model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = VAE(
        input_dim=config.get("input_dim", 2),
        latent_dim=config.get("latent_dim", 4),
        hidden_dim=config.get("hidden_dim", 128),
        depth=config.get("depth", 3),
        dropout=config.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config


@torch.no_grad()
def plot_latent_grid(
    model: VAE,
    device: torch.device,
    save_path: str,
    grid_size: int = 15,
    bounds: tuple[float, float] = (-3.0, 3.0),
    pca: PCA | None = None,
    dim_indices: tuple[int, int] = (0, 1),
) -> None:
    """Generate a grid in latent space and decode each point.

    For latent_dim > 2, we set all other dimensions to 0 and vary
    the two specified dimensions (default: first two PCA components or dims 0,1).
    """
    model.eval()
    latent_dim = model.latent_dim

    xs = np.linspace(bounds[0], bounds[1], grid_size)
    ys = np.linspace(bounds[0], bounds[1], grid_size)

    grid_points = []
    for y in ys:
        for x in xs:
            z = np.zeros(latent_dim, dtype=np.float32)
            z[dim_indices[0]] = x
            z[dim_indices[1]] = y
            grid_points.append(z)

    z_grid = torch.from_numpy(np.stack(grid_points, axis=0)).to(device)
    decoded = model.decode(z_grid).cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    for i, (gx, gy) in enumerate(
        [(x, y) for y in ys for x in xs]
    ):
        # Normalize color to [0, 1] range
        r = (gx - bounds[0]) / (bounds[1] - bounds[0])
        b = (gy - bounds[0]) / (bounds[1] - bounds[0])
        ax.scatter(
            decoded[i : i + 1, 0],
            decoded[i : i + 1, 1],
            c=[[r, 0.5, b]],
            s=8,
            alpha=0.7,
        )

    ax.set_title(f"VAE Latent Grid (dims {dim_indices[0]}, {dim_indices[1]})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def plot_latent_interpolation(
    model: VAE,
    x_real: np.ndarray,
    device: torch.device,
    save_path: str,
    n_interp: int = 8,
    n_frames: int = 6,
) -> None:
    """Interpolate between pairs of real test samples in latent space.

    Takes n_interp pairs of real samples, encodes them, linearly interpolates
    their latent codes, and decodes the interpolated points.
    """
    model.eval()
    x_tensor = torch.from_numpy(x_real.astype(np.float32)).to(device)

    # Encode all real samples
    mu, logvar = model.encode(x_tensor)

    # Select random pairs for interpolation
    n_samples = len(x_real)
    rng = np.random.default_rng(42)
    indices_a = rng.choice(n_samples, size=n_interp, replace=False)
    indices_b = rng.choice(n_samples, size=n_interp, replace=False)

    fig, axes = plt.subplots(n_interp, n_frames + 2, figsize=(2.5 * (n_frames + 2), 2.5 * n_interp))
    if n_interp == 1:
        axes = np.array([axes])

    for row, (ia, ib) in enumerate(zip(indices_a, indices_b)):
        za = mu[ia].cpu().numpy()
        zb = mu[ib].cpu().numpy()

        # Plot start point
        axes[row, 0].scatter(x_real[ia : ia + 1, 0], x_real[ia : ia + 1, 1],
                             c="#1f77b4", s=30, edgecolors="k", linewidths=0.5)
        axes[row, 0].set_title("Sample A (real)")
        axes[row, 0].set_aspect("equal")

        # Plot interpolations
        for t_idx, alpha in enumerate(np.linspace(0, 1, n_frames)):
            z_interp = (1 - alpha) * za + alpha * zb
            z_tensor = torch.from_numpy(z_interp.astype(np.float32)).unsqueeze(0).to(device)
            decoded = model.decode(z_tensor).cpu().numpy()[0]
            color = plt.cm.viridis(alpha)
            axes[row, t_idx + 1].scatter(decoded[0], decoded[1], c=[color], s=30)
            axes[row, t_idx + 1].set_title(f"α={alpha:.1f}")
            axes[row, t_idx + 1].set_aspect("equal")

        # Plot end point
        axes[row, n_frames + 1].scatter(x_real[ib : ib + 1, 0], x_real[ib : ib + 1, 1],
                                        c="#d62728", s=30, edgecolors="k", linewidths=0.5)
        axes[row, n_frames + 1].set_title("Sample B (real)")
        axes[row, n_frames + 1].set_aspect("equal")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("VAE Latent Space Interpolation", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def plot_latent_encoding_scatter(
    model: VAE,
    dataset_name: str,
    data_root: str,
    device: torch.device,
    save_path: str,
    n_samples: int = 1000,
    seed: int = 42,
) -> None:
    """Encode real test samples and visualize their latent codes (PCA to 2D)."""
    set_seed(seed)
    bundle = load_or_generate_dataset(dataset_name, root=data_root)
    x_test = bundle.x_test[:n_samples]
    x_tensor = torch.from_numpy(x_test.astype(np.float32)).to(device)
    mu, _ = model.encode(x_tensor)
    mu_np = mu.cpu().numpy()

    # PCA to 2D for visualization
    if model.latent_dim > 2:
        pca = PCA(n_components=2)
        mu_2d = pca.fit_transform(mu_np)
        explained_var = pca.explained_variance_ratio_.sum()
    else:
        mu_2d = mu_np
        explained_var = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Data space
    axes[0].scatter(x_test[:, 0], x_test[:, 1], s=4, alpha=0.6, c="#1f77b4")
    axes[0].set_title(f"Data Space: {dataset_name}")
    axes[0].set_aspect("equal")
    axes[0].grid(alpha=0.2)

    # Latent space
    axes[1].scatter(mu_2d[:, 0], mu_2d[:, 1], s=4, alpha=0.6, c="#d62728")
    axes[1].set_title(
        f"Latent Space (PCA 2D, explained var: {explained_var:.2%})"
    )
    axes[1].set_aspect("equal")
    axes[1].grid(alpha=0.2)

    fig.suptitle("VAE: Data Space → Latent Space Mapping")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_latent_visualizations(
    checkpoint_path: str,
    dataset: str,
    data_root: str = "data",
    figure_dir: str = "outputs/figures/latent",
    seed: int = 42,
) -> None:
    """Run all latent space visualizations for a trained VAE model."""
    device = get_device()
    ensure_dir(figure_dir)
    model, config = load_vae_checkpoint(checkpoint_path, device)

    # Load test data
    set_seed(seed)
    bundle = load_or_generate_dataset(dataset, root=data_root)
    x_test = bundle.x_test

    prefix = f"{figure_dir}/vae_{dataset}"

    # 1. Latent grid traversal
    print(f"Generating latent grid for {dataset}...")
    plot_latent_grid(
        model, device,
        save_path=f"{prefix}_latent_grid.png",
        grid_size=15,
    )

    # 2. Latent interpolation
    print(f"Generating latent interpolation for {dataset}...")
    plot_latent_interpolation(
        model, x_test, device,
        save_path=f"{prefix}_latent_interpolation.png",
        n_interp=6,
        n_frames=5,
    )

    # 3. Encoding scatter
    print(f"Generating encoding scatter for {dataset}...")
    plot_latent_encoding_scatter(
        model, dataset, data_root, device,
        save_path=f"{prefix}_encoding_scatter.png",
    )

    print(f"Latent visualizations saved to {figure_dir}/")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate VAE latent space visualizations."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to VAE checkpoint (.pt file)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="spiral",
        choices=AVAILABLE_DATASETS,
        help="Dataset name",
    )
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument(
        "--figure-dir", type=str, default="outputs/figures/latent"
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_latent_visualizations(
        checkpoint_path=args.checkpoint,
        dataset=args.dataset,
        data_root=args.data_root,
        figure_dir=args.figure_dir,
        seed=args.seed,
    )
