from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.train_diffusion import train as train_diffusion
from src.train_vae import train as train_vae
from src.utils import ensure_dir
from src.visualize import plot_robustness_curves


def _write_rows(rows: list[dict[str, float | str]], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> list[dict[str, float | str]]:
    datasets = args.datasets
    ratios = args.ratios
    kinds = args.contamination_kinds
    rows: list[dict[str, float | str]] = []

    for dataset in datasets:
        for kind in kinds:
            for ratio in ratios:
                kind_tag = kind.replace("_", "-")
                tag = f"robust_{kind_tag}_r{str(ratio).replace('.', 'p')}"
                print(f"\n=== Robustness: VAE on {dataset}, kind={kind}, ratio={ratio:.2f} ===")
                vae_metrics = train_vae(
                    argparse.Namespace(
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
                        lr=args.lr,
                        beta=args.beta,
                        seed=args.seed,
                        device=args.device,
                        log_every=args.log_every,
                        contamination_ratio=ratio,
                        contamination_kind=kind,
                        output_tag=tag,
                    )
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "model": "vae",
                        "contamination_ratio": ratio,
                        "contamination_kind": kind,
                        **vae_metrics,
                    }
                )

                print(f"\n=== Robustness: Diffusion on {dataset}, kind={kind}, ratio={ratio:.2f} ===")
                snapshot_steps = [max(args.timesteps - 1, 0), max(args.timesteps // 2, 0), max(args.timesteps // 4, 0), 0]
                diffusion_metrics = train_diffusion(
                    argparse.Namespace(
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
                        beta_start=args.beta_start,
                        beta_end=args.beta_end,
                        schedule_type=args.schedule_type,
                        time_hidden_dim=args.time_hidden_dim,
                        lr=args.lr,
                        seed=args.seed,
                        device=args.device,
                        log_every=args.log_every,
                        snapshot_steps=snapshot_steps,
                        contamination_ratio=ratio,
                        contamination_kind=kind,
                        sampler=args.sampler,
                        sampling_steps=args.sampling_steps,
                        output_tag=tag,
                    )
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "model": "diffusion",
                        "contamination_ratio": ratio,
                        "contamination_kind": kind,
                        **diffusion_metrics,
                    }
                )

    _write_rows(rows, Path(args.table_dir) / "robustness_summary.csv")

    for dataset in datasets:
        for kind in kinds:
            for metric_name in ("mmd", "sliced_wasserstein", "chamfer", "kde_nll"):
                series = {
                    "VAE": [
                        float(row[metric_name])
                        for row in rows
                        if row["dataset"] == dataset and row["model"] == "vae" and row["contamination_kind"] == kind
                    ],
                    "Diffusion": [
                        float(row[metric_name])
                        for row in rows
                        if row["dataset"] == dataset and row["model"] == "diffusion" and row["contamination_kind"] == kind
                    ],
                }
                plot_robustness_curves(
                    x_values=list(ratios),
                    series=series,
                    metric_name=metric_name,
                    title=f"Robustness on {dataset} ({kind}, {metric_name})",
                    save_path=Path(args.figure_dir) / f"robustness_{dataset}_{kind}_{metric_name}.png",
                )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run robustness experiments with outlier contamination.")
    parser.add_argument("--datasets", nargs="*", default=["ring", "spiral"])
    parser.add_argument("--ratios", nargs="*", type=float, default=[0.0, 0.05, 0.10])
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--figure-dir", type=str, default="outputs/figures/optional")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints/optional")
    parser.add_argument("--table-dir", type=str, default="outputs/tables/optional")
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--vae-epochs", type=int, default=120)
    parser.add_argument("--diffusion-epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--timesteps", type=int, default=80)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--schedule-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--time-hidden-dim", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--contamination-kinds", nargs="*", default=["uniform", "cluster_shift", "heteroscedastic"])
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddpm", "ddim"])
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=20)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
