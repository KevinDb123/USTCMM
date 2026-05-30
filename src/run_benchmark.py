from __future__ import annotations

import argparse

from src.datasets import AVAILABLE_DATASETS
from src.train_diffusion import train as train_diffusion
from src.train_vae import train as train_vae


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all VAE and diffusion experiments on all datasets.")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--figure-dir", type=str, default="outputs/figures")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--table-dir", type=str, default="outputs/tables")
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--vae-epochs", type=int, default=150)
    parser.add_argument("--diffusion-epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--vae-beta", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--timesteps", type=int, default=80)
    parser.add_argument("--schedule-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--time-hidden-dim", type=int, default=0)
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--output-tag", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    for dataset in AVAILABLE_DATASETS:
        print(f"\n=== Running VAE on {dataset} ===")
        vae_args = argparse.Namespace(
            dataset=dataset,
            data_root=args.data_root,
            figure_dir=args.figure_dir,
            checkpoint_dir=args.checkpoint_dir,
            table_dir=args.table_dir,
            n_train=args.n_train,
            n_test=args.n_test,
            epochs=args.vae_epochs,
            batch_size=args.batch_size,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            dropout=args.dropout,
            lr=1e-3,
            beta=args.vae_beta,
            seed=args.seed,
            device=args.device,
            log_every=max(1, args.vae_epochs // 10),
            contamination_ratio=0.0,
            contamination_kind="uniform",
            output_tag=args.output_tag,
        )
        train_vae(vae_args)

        print(f"\n=== Running Diffusion on {dataset} ===")
        diffusion_args = argparse.Namespace(
            dataset=dataset,
            data_root=args.data_root,
            figure_dir=args.figure_dir,
            checkpoint_dir=args.checkpoint_dir,
            table_dir=args.table_dir,
            n_train=args.n_train,
            n_test=args.n_test,
            epochs=args.diffusion_epochs,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            dropout=args.dropout,
            timesteps=args.timesteps,
            beta_start=1e-4,
            beta_end=2e-2,
            schedule_type=args.schedule_type,
            time_hidden_dim=args.time_hidden_dim,
            lr=1e-3,
            seed=args.seed,
            device=args.device,
            log_every=max(1, args.diffusion_epochs // 10),
            snapshot_steps=[args.timesteps - 1, max(args.timesteps * 3 // 4 - 1, 0), max(args.timesteps // 2 - 1, 0), max(args.timesteps // 4 - 1, 0)],
            contamination_ratio=0.0,
            contamination_kind="uniform",
            sampler=args.sampler,
            sampling_steps=args.sampling_steps,
            output_tag=args.output_tag,
        )
        train_diffusion(diffusion_args)


if __name__ == "__main__":
    main()
