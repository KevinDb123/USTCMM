"""Regenerate all figures from scratch by re-training models.

Covers all 4 datasets × 2 models (VAE + DDPM):
  - Sample comparison figures (real vs generated)
  - Training curves
  - VAE latent grids
  - DDPM enhanced denoising snapshots (10 steps)

Does NOT rely on any existing checkpoints.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.datasets import AVAILABLE_DATASETS, load_or_generate_dataset
from src.train_vae import train as train_vae
from src.train_diffusion import train as train_diffusion
from src.latent_viz import run_latent_visualizations
from src.enhanced_snapshots import run_enhanced_snapshots
from src.utils import ensure_dir


DATASETS = ["gaussian_mixture", "ring", "two_moons", "spiral"]
SEED = 42
FIGURE_DIR = "outputs/figures"
CHECKPOINT_DIR = "outputs/checkpoints/full"
LATENT_DIR = "outputs/figures/latent"


def main() -> None:
    ensure_dir(FIGURE_DIR)
    ensure_dir(CHECKPOINT_DIR)
    ensure_dir(LATENT_DIR)

    # ---- shared args ----
    n_train = 2000
    n_test = 2000

    for dataset in DATASETS:
        print(f"\n{'=' * 70}")
        print(f"=== [{dataset.upper()}] Training VAE + DDPM from scratch ===")
        print(f"{'=' * 70}")

        # ---------- VAE ----------
        print(f"\n--- VAE on {dataset} ---")
        vae_args = argparse.Namespace(
            dataset=dataset,
            data_root="data",
            figure_dir=FIGURE_DIR,
            checkpoint_dir=CHECKPOINT_DIR,
            table_dir="outputs/tables",
            n_train=n_train,
            n_test=n_test,
            epochs=150,
            batch_size=256,
            latent_dim=4,
            hidden_dim=128,
            depth=3,
            dropout=0.0,
            lr=1e-3,
            beta=0.5,
            seed=SEED,
            device=None,
            log_every=20,
            contamination_ratio=0.0,
            contamination_kind="uniform",
            output_tag="",
        )
        train_vae(vae_args)

        # VAE latent visualizations
        vae_ckpt = f"{CHECKPOINT_DIR}/vae_{dataset}.pt"
        print(f"  Generating VAE latent visualizations from {vae_ckpt}")
        run_latent_visualizations(
            checkpoint_path=vae_ckpt,
            dataset=dataset,
            data_root="data",
            figure_dir=LATENT_DIR,
            seed=SEED,
        )

        # ---------- DDPM ----------
        print(f"\n--- DDPM on {dataset} ---")
        diff_args = argparse.Namespace(
            dataset=dataset,
            data_root="data",
            figure_dir=FIGURE_DIR,
            checkpoint_dir=CHECKPOINT_DIR,
            table_dir="outputs/tables",
            n_train=n_train,
            n_test=n_test,
            epochs=250,
            batch_size=256,
            hidden_dim=128,
            depth=3,
            dropout=0.0,
            timesteps=80,
            beta_start=1e-4,
            beta_end=2e-2,
            schedule_type="linear",
            time_hidden_dim=0,
            lr=1e-3,
            seed=SEED,
            device=None,
            log_every=25,
            snapshot_steps=[79, 59, 39, 19],
            contamination_ratio=0.0,
            contamination_kind="uniform",
            sampler="ddpm",
            sampling_steps=None,
            output_tag="",
        )
        train_diffusion(diff_args)

        # DDPM enhanced denoising snapshots (10-step)
        diff_ckpt = f"{CHECKPOINT_DIR}/diffusion_{dataset}.pt"
        print(f"  Generating enhanced denoising snapshots from {diff_ckpt}")
        run_enhanced_snapshots(
            checkpoint_path=diff_ckpt,
            dataset=dataset,
            data_root="data",
            figure_dir=FIGURE_DIR,
            num_snapshots=10,
            n_samples=2000,
            seed=SEED,
        )

    print(f"\n{'=' * 70}")
    print("All figures regenerated successfully!")
    print(f"  Sample comparisons:   {FIGURE_DIR}/{{vae,diffusion}}_<dataset>_samples.png")
    print(f"  Training curves:      {FIGURE_DIR}/{{vae,diffusion}}_<dataset>_training.png")
    print(f"  Latent grids:         {LATENT_DIR}/vae_<dataset>_latent_grid.png")
    print(f"  Denoising snapshots:  {FIGURE_DIR}/diffusion_<dataset>_enhanced_snapshots.png")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
