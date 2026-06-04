"""Training script for RealNVP Normalizing Flow on 2D distributions."""

from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader

from src.datasets import (
    AVAILABLE_CONTAMINATION_KINDS,
    AVAILABLE_DATASETS,
    load_or_generate_dataset,
    make_contaminated_bundle,
    make_tensor_dataset,
)
from src.flow import RealNVP
from src.metrics import summarize_metrics
from src.utils import ensure_dir, get_device, save_json, set_seed, to_numpy
from src.visualize import plot_dataset_splits, plot_real_vs_generated, plot_training_curves


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train(args: argparse.Namespace) -> dict[str, float]:
    set_seed(args.seed)
    device = get_device(args.device)
    bundle = load_or_generate_dataset(
        args.dataset,
        root=args.data_root,
        n_train=args.n_train,
        n_test=args.n_test,
        seed=args.seed,
    )
    if args.contamination_ratio > 0:
        bundle = make_contaminated_bundle(
            bundle,
            ratio=args.contamination_ratio,
            seed=args.seed,
            kind=args.contamination_kind,
        )
    loader = DataLoader(
        make_tensor_dataset(bundle.x_train), batch_size=args.batch_size, shuffle=True
    )
    input_dim = bundle.x_train.shape[1]

    model = RealNVP(
        data_dim=input_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history: dict[str, list[float]] = {"loss": []}
    _sync_if_needed(device)
    start = time.perf_counter()
    for epoch in range(args.epochs):
        running_loss = 0.0
        model.train()
        for (x,) in loader:
            x = x.to(device)
            losses = model.compute_loss(x)
            optimizer.zero_grad()
            losses["loss"].backward()
            # Gradient clipping for stable training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            running_loss += float(losses["loss"].item()) * len(x)
        scheduler.step()
        history["loss"].append(running_loss / len(bundle.x_train))
        if (epoch + 1) % args.log_every == 0 or epoch == 0 or epoch + 1 == args.epochs:
            print(
                f"[Flow][{args.dataset}] epoch {epoch + 1:03d}/{args.epochs} "
                f"loss={history['loss'][-1]:.4f} lr={scheduler.get_last_lr()[0]:.2e}"
            )
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
    metrics["contamination_ratio"] = args.contamination_ratio
    metrics["contamination_kind"] = args.contamination_kind

    ensure_dir(args.figure_dir)
    ensure_dir(args.checkpoint_dir)
    ensure_dir(args.table_dir)

    suffix = f"_{args.output_tag}" if args.output_tag else ""

    plot_dataset_splits(
        bundle.x_train,
        bundle.x_test,
        title=args.dataset,
        save_path=f"{args.figure_dir}/{args.dataset}{suffix}_dataset.png",
    )
    plot_training_curves(
        history,
        title=f"Flow training - {args.dataset}",
        save_path=f"{args.figure_dir}/flow_{args.dataset}{suffix}_training.png",
    )
    plot_real_vs_generated(
        bundle.x_test,
        generated,
        title=f"RealNVP Flow - {args.dataset}",
        save_path=f"{args.figure_dir}/flow_{args.dataset}{suffix}_samples.png",
    )

    checkpoint_path = f"{args.checkpoint_dir}/flow_{args.dataset}{suffix}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "hidden_dim": args.hidden_dim,
                "depth": args.depth,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "input_dim": input_dim,
                "dataset": args.dataset,
            },
            "metrics": metrics,
        },
        checkpoint_path,
    )
    save_json(metrics, f"{args.table_dir}/flow_{args.dataset}{suffix}_metrics.json")
    print(f"Saved checkpoint to {checkpoint_path}")
    print(metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a RealNVP Normalizing Flow on a 2D dataset."
    )
    parser.add_argument(
        "--dataset", type=str, default="spiral", choices=AVAILABLE_DATASETS
    )
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--figure-dir", type=str, default="outputs/figures")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--table-dir", type=str, default="outputs/tables")
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=30)
    parser.add_argument("--contamination-ratio", type=float, default=0.0)
    parser.add_argument(
        "--contamination-kind",
        type=str,
        default="uniform",
        choices=AVAILABLE_CONTAMINATION_KINDS,
    )
    parser.add_argument("--output-tag", type=str, default="")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
