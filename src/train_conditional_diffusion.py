from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.conditional_diffusion import ConditionalDiffusionModel
from src.datasets import AVAILABLE_DATASETS, LABEL_TO_DATASET, load_all_labeled_datasets, make_labeled_tensor_dataset
from src.metrics import summarize_metrics
from src.utils import ensure_dir, get_device, save_json, set_seed, to_numpy
from src.visualize import plot_conditional_real_vs_generated, plot_training_curves


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _write_rows(rows: list[dict[str, float | str]], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train(args: argparse.Namespace) -> dict[str, float]:
    set_seed(args.seed)
    device = get_device(args.device)
    bundle = load_all_labeled_datasets(
        root=args.data_root,
        n_train=args.n_train,
        n_test=args.n_test,
        seed=args.seed,
    )
    loader = DataLoader(
        make_labeled_tensor_dataset(bundle.x_train, bundle.y_train),
        batch_size=args.batch_size,
        shuffle=True,
    )
    model = ConditionalDiffusionModel(
        data_dim=bundle.x_train.shape[1],
        condition_dim=len(AVAILABLE_DATASETS),
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
        schedule_type=args.schedule_type,
        time_hidden_dim=args.time_hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: dict[str, list[float]] = {"loss": []}
    _sync_if_needed(device)
    start = time.perf_counter()
    for epoch in range(args.epochs):
        running_loss = 0.0
        model.train()
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            losses = model.compute_loss(x, y)
            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()
            running_loss += float(losses["loss"].item()) * len(x)
        history["loss"].append(running_loss / len(bundle.x_train))
        if (epoch + 1) % args.log_every == 0 or epoch == 0 or epoch + 1 == args.epochs:
            print(
                f"[CDiffusion] epoch {epoch + 1:03d}/{args.epochs} "
                f"loss={history['loss'][-1]:.4f}"
            )
    _sync_if_needed(device)
    train_time = time.perf_counter() - start

    real_by_name: dict[str, np.ndarray] = {}
    generated_by_name: dict[str, np.ndarray] = {}
    rows: list[dict[str, float | str]] = []
    total_sample_time = 0.0

    model.eval()
    for label, name in LABEL_TO_DATASET.items():
        real = bundle.x_test[bundle.y_test == label]
        labels = torch.full((len(real),), label, dtype=torch.long, device=device)
        _sync_if_needed(device)
        sample_start = time.perf_counter()
        generated, _ = model.sample(
            labels,
            device=device,
            sampler=args.sampler,
            sampling_steps=args.sampling_steps,
        )
        generated_np = to_numpy(generated)
        _sync_if_needed(device)
        sample_time = time.perf_counter() - sample_start
        total_sample_time += sample_time
        metrics = summarize_metrics(real, generated_np, seed=args.seed + label)
        metrics["dataset"] = name
        metrics["model"] = "conditional_diffusion"
        metrics["train_time_sec"] = train_time
        metrics["sample_time_sec"] = sample_time
        metrics["sampler"] = args.sampler
        metrics["sampling_steps"] = args.sampling_steps or args.timesteps
        rows.append(metrics)
        real_by_name[name] = real
        generated_by_name[name] = generated_np

    avg_metrics = {
        "mmd": float(np.mean([float(row["mmd"]) for row in rows])),
        "sliced_wasserstein": float(np.mean([float(row["sliced_wasserstein"]) for row in rows])),
        "chamfer": float(np.mean([float(row["chamfer"]) for row in rows])),
        "kde_nll": float(np.mean([float(row["kde_nll"]) for row in rows])),
        "mode_coverage": float(np.mean([float(row["mode_coverage"]) for row in rows])),
        "train_time_sec": train_time,
        "sample_time_sec": total_sample_time,
        "sampler": args.sampler,
        "sampling_steps": args.sampling_steps or args.timesteps,
    }

    ensure_dir(args.figure_dir)
    ensure_dir(args.checkpoint_dir)
    ensure_dir(args.table_dir)

    plot_training_curves(
        history,
        title="Conditional Diffusion training",
        save_path=f"{args.figure_dir}/conditional_diffusion_training.png",
    )
    plot_conditional_real_vs_generated(
        real_by_name,
        generated_by_name,
        title="Conditional Diffusion across four distributions",
        save_path=f"{args.figure_dir}/conditional_diffusion_samples.png",
    )

    checkpoint_path = f"{args.checkpoint_dir}/conditional_diffusion_all.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "data_dim": bundle.x_train.shape[1],
                "condition_dim": len(AVAILABLE_DATASETS),
                "timesteps": args.timesteps,
                "hidden_dim": args.hidden_dim,
                "depth": args.depth,
                "dropout": args.dropout,
                "beta_start": args.beta_start,
                "beta_end": args.beta_end,
                "schedule_type": args.schedule_type,
                "time_hidden_dim": args.time_hidden_dim,
                "sampler": args.sampler,
                "sampling_steps": args.sampling_steps or args.timesteps,
            },
            "metrics": avg_metrics,
        },
        checkpoint_path,
    )
    save_json(avg_metrics, f"{args.table_dir}/conditional_diffusion_summary.json")
    _write_rows(rows, f"{args.table_dir}/conditional_diffusion_per_dataset.csv")
    print(f"Saved checkpoint to {checkpoint_path}")
    print(avg_metrics)
    return avg_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a conditional diffusion model on all four datasets jointly.")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--figure-dir", type=str, default="outputs/figures/optional")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints/optional")
    parser.add_argument("--table-dir", type=str, default="outputs/tables/optional")
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--timesteps", type=int, default=80)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--schedule-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--time-hidden-dim", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddpm", "ddim"])
    parser.add_argument("--sampling-steps", type=int, default=20)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
