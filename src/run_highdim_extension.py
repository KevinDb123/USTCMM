from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets import AVAILABLE_DATASETS, DatasetBundle, load_or_generate_dataset, make_tensor_dataset
from src.diffusion import DiffusionModel
from src.metrics import summarize_metrics
from src.utils import ensure_dir, get_device, save_json, set_seed, to_numpy
from src.vae import VAE
from src.visualize import plot_dataset_splits, plot_real_vs_generated, plot_training_curves


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def lift_to_highdim(x: np.ndarray) -> np.ndarray:
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    radius = np.sqrt(np.sum(x**2, axis=1, keepdims=True))
    lifted = np.concatenate(
        [
            x1,
            x2,
            x1 * x2,
            x1**2,
            x2**2,
            np.sin(x1),
            np.sin(x2),
            radius,
        ],
        axis=1,
    )
    return lifted.astype(np.float32)


def make_highdim_bundle(dataset: str, data_root: str, n_train: int, n_test: int, seed: int) -> DatasetBundle:
    bundle = load_or_generate_dataset(dataset, root=data_root, n_train=n_train, n_test=n_test, seed=seed)
    meta = dict(bundle.meta)
    meta["lifted_dim"] = 8
    meta["source"] = f"{meta.get('source', 'data')}+lifted8d"
    return DatasetBundle(
        name=f"{dataset}_8d",
        x_train=lift_to_highdim(bundle.x_train),
        x_test=lift_to_highdim(bundle.x_test),
        meta=meta,
    )


def train_highdim_vae(bundle: DatasetBundle, args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    loader = DataLoader(make_tensor_dataset(bundle.x_train), batch_size=args.batch_size, shuffle=True)
    model = VAE(
        input_dim=bundle.x_train.shape[1],
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history: dict[str, list[float]] = {"loss": [], "recon_loss": [], "kl_loss": []}

    _sync_if_needed(device)
    start = time.perf_counter()
    for epoch in range(args.vae_epochs):
        epoch_stats = {key: 0.0 for key in history}
        model.train()
        for (x,) in loader:
            x = x.to(device)
            losses = model.compute_loss(x, beta=args.beta)
            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()
            for key in history:
                epoch_stats[key] += float(losses[key].item()) * len(x)
        for key in history:
            history[key].append(epoch_stats[key] / len(bundle.x_train))
    _sync_if_needed(device)
    train_time = time.perf_counter() - start

    model.eval()
    _sync_if_needed(device)
    sample_start = time.perf_counter()
    generated = to_numpy(model.sample(len(bundle.x_test), device))
    _sync_if_needed(device)
    sample_time = time.perf_counter() - sample_start
    metrics = summarize_metrics(bundle.x_test, generated, seed=args.seed)
    metrics["train_time_sec"] = train_time
    metrics["sample_time_sec"] = sample_time

    plot_training_curves(history, title=f"High-dim VAE - {bundle.name}", save_path=Path(args.figure_dir) / f"vae_{bundle.name}_training.png")
    plot_real_vs_generated(bundle.x_test, generated, title=f"High-dim VAE - {bundle.name}", save_path=Path(args.figure_dir) / f"vae_{bundle.name}_samples.png")
    return metrics


def train_highdim_diffusion(bundle: DatasetBundle, args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    loader = DataLoader(make_tensor_dataset(bundle.x_train), batch_size=args.batch_size, shuffle=True)
    model = DiffusionModel(
        data_dim=bundle.x_train.shape[1],
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
    for epoch in range(args.diffusion_epochs):
        running_loss = 0.0
        model.train()
        for (x,) in loader:
            x = x.to(device)
            losses = model.compute_loss(x)
            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()
            running_loss += float(losses["loss"].item()) * len(x)
        history["loss"].append(running_loss / len(bundle.x_train))
    _sync_if_needed(device)
    train_time = time.perf_counter() - start

    model.eval()
    _sync_if_needed(device)
    sample_start = time.perf_counter()
    generated, _ = model.sample(
        len(bundle.x_test),
        device=device,
        sampler=args.sampler,
        sampling_steps=args.sampling_steps,
    )
    generated_np = to_numpy(generated)
    _sync_if_needed(device)
    sample_time = time.perf_counter() - sample_start
    metrics = summarize_metrics(bundle.x_test, generated_np, seed=args.seed)
    metrics["train_time_sec"] = train_time
    metrics["sample_time_sec"] = sample_time
    metrics["sampler"] = args.sampler
    metrics["sampling_steps"] = args.sampling_steps or args.timesteps
    metrics["schedule_type"] = args.schedule_type
    metrics["time_hidden_dim"] = args.time_hidden_dim

    plot_training_curves(history, title=f"High-dim Diffusion - {bundle.name}", save_path=Path(args.figure_dir) / f"diffusion_{bundle.name}_training.png")
    plot_real_vs_generated(bundle.x_test, generated_np, title=f"High-dim Diffusion - {bundle.name}", save_path=Path(args.figure_dir) / f"diffusion_{bundle.name}_samples.png")
    return metrics


def run(args: argparse.Namespace) -> list[dict[str, float | str]]:
    set_seed(args.seed)
    device = get_device(args.device)
    ensure_dir(args.figure_dir)
    ensure_dir(args.table_dir)
    rows: list[dict[str, float | str]] = []

    for dataset in args.datasets:
        bundle = make_highdim_bundle(dataset, data_root=args.data_root, n_train=args.n_train, n_test=args.n_test, seed=args.seed)
        plot_dataset_splits(bundle.x_train, bundle.x_test, title=bundle.name, save_path=Path(args.figure_dir) / f"{bundle.name}_dataset.png")

        vae_metrics = train_highdim_vae(bundle, args, device)
        rows.append({"dataset": bundle.name, "model": "vae", **vae_metrics})

        diffusion_metrics = train_highdim_diffusion(bundle, args, device)
        rows.append({"dataset": bundle.name, "model": "diffusion", **diffusion_metrics})

    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    output_path = Path(args.table_dir) / f"highdim_summary{suffix}.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    save_json({"rows": rows}, Path(args.table_dir) / f"highdim_summary{suffix}.json")
    print(f"Saved high-dimensional summary to {output_path}")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run lifted high-dimensional experiments.")
    parser.add_argument("--datasets", nargs="*", default=["gaussian_mixture", "spiral"], choices=AVAILABLE_DATASETS)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--figure-dir", type=str, default="outputs/figures/optional")
    parser.add_argument("--table-dir", type=str, default="outputs/tables/optional")
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--vae-epochs", type=int, default=120)
    parser.add_argument("--diffusion-epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--timesteps", type=int, default=80)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--schedule-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--time-hidden-dim", type=int, default=0)
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddpm", "ddim"])
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-suffix", type=str, default="")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
