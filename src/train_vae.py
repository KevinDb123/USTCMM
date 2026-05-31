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
from src.metrics import summarize_metrics
from src.utils import ensure_dir, get_device, save_json, set_seed, to_numpy
from src.vae import VAE
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
    loader = DataLoader(make_tensor_dataset(bundle.x_train), batch_size=args.batch_size, shuffle=True)
    input_dim = bundle.x_train.shape[1]
    model = VAE(
        input_dim=input_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: dict[str, list[float]] = {"loss": [], "recon_loss": [], "kl_loss": []}
    _sync_if_needed(device)
    start = time.perf_counter()
    for epoch in range(args.epochs):
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
        if (epoch + 1) % args.log_every == 0 or epoch == 0 or epoch + 1 == args.epochs:
            print(
                f"[VAE][{args.dataset}] epoch {epoch + 1:03d}/{args.epochs} "
                f"loss={history['loss'][-1]:.4f} recon={history['recon_loss'][-1]:.4f} kl={history['kl_loss'][-1]:.4f}"
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
        title=f"VAE training - {args.dataset}",
        save_path=f"{args.figure_dir}/vae_{args.dataset}{suffix}_training.png",
    )
    plot_real_vs_generated(
        bundle.x_test,
        generated,
        title=f"VAE - {args.dataset}",
        save_path=f"{args.figure_dir}/vae_{args.dataset}{suffix}_samples.png",
    )

    checkpoint_path = f"{args.checkpoint_dir}/vae_{args.dataset}{suffix}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "latent_dim": args.latent_dim,
                "hidden_dim": args.hidden_dim,
                "depth": args.depth,
                "dropout": args.dropout,
                "input_dim": input_dim,
                "dataset": args.dataset,
            },
            "metrics": metrics,
        },
        checkpoint_path,
    )
    save_json(metrics, f"{args.table_dir}/vae_{args.dataset}{suffix}_metrics.json")
    print(f"Saved checkpoint to {checkpoint_path}")
    print(metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a VAE on a 2D dataset.")
    parser.add_argument("--dataset", type=str, default="spiral", choices=AVAILABLE_DATASETS)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--figure-dir", type=str, default="outputs/figures")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--table-dir", type=str, default="outputs/tables")
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--contamination-ratio", type=float, default=0.0)
    parser.add_argument("--contamination-kind", type=str, default="uniform", choices=AVAILABLE_CONTAMINATION_KINDS)
    parser.add_argument("--output-tag", type=str, default="")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
