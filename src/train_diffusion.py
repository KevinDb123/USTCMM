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
from src.diffusion import DiffusionModel
from src.metrics import summarize_metrics
from src.utils import ensure_dir, get_device, save_json, set_seed, to_numpy
from src.visualize import (
    plot_dataset_splits,
    plot_diffusion_snapshots,
    plot_real_vs_generated,
    plot_training_curves,
)


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
    model = DiffusionModel(
        data_dim=input_dim,
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
        for (x,) in loader:
            x = x.to(device)
            losses = model.compute_loss(x)
            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()
            running_loss += float(losses["loss"].item()) * len(x)
        history["loss"].append(running_loss / len(bundle.x_train))
        if (epoch + 1) % args.log_every == 0 or epoch == 0 or epoch + 1 == args.epochs:
            print(
                f"[Diffusion][{args.dataset}] epoch {epoch + 1:03d}/{args.epochs} "
                f"loss={history['loss'][-1]:.4f}"
            )
    _sync_if_needed(device)
    train_time = time.perf_counter() - start

    model.eval()
    _sync_if_needed(device)
    sample_start = time.perf_counter()
    generated, snapshots = model.sample(
        len(bundle.x_test),
        device=device,
        snapshot_steps=args.snapshot_steps,
        sampler=args.sampler,
        sampling_steps=args.sampling_steps,
    )
    _sync_if_needed(device)
    sample_time = time.perf_counter() - sample_start
    generated_np = to_numpy(generated)
    metrics = summarize_metrics(bundle.x_test, generated_np, seed=args.seed)
    metrics["train_time_sec"] = train_time
    metrics["sample_time_sec"] = sample_time
    metrics["contamination_ratio"] = args.contamination_ratio
    metrics["contamination_kind"] = args.contamination_kind
    metrics["sampler"] = args.sampler
    metrics["sampling_steps"] = args.sampling_steps or args.timesteps

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
        title=f"Diffusion training - {args.dataset}",
        save_path=f"{args.figure_dir}/diffusion_{args.dataset}{suffix}_training.png",
    )
    plot_real_vs_generated(
        bundle.x_test,
        generated_np,
        title=f"Diffusion - {args.dataset}",
        save_path=f"{args.figure_dir}/diffusion_{args.dataset}{suffix}_samples.png",
    )
    plot_diffusion_snapshots(
        snapshots,
        title=f"Diffusion snapshots - {args.dataset}",
        save_path=f"{args.figure_dir}/diffusion_{args.dataset}{suffix}_snapshots.png",
    )

    checkpoint_path = f"{args.checkpoint_dir}/diffusion_{args.dataset}{suffix}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
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
                "input_dim": input_dim,
                "dataset": args.dataset,
            },
            "metrics": metrics,
        },
        checkpoint_path,
    )
    save_json(metrics, f"{args.table_dir}/diffusion_{args.dataset}{suffix}_metrics.json")
    print(f"Saved checkpoint to {checkpoint_path}")
    print(metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a 2D diffusion model.")
    parser.add_argument("--dataset", type=str, default="spiral", choices=AVAILABLE_DATASETS)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--figure-dir", type=str, default="outputs/figures")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--table-dir", type=str, default="outputs/tables")
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
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
    parser.add_argument("--snapshot-steps", type=int, nargs="*", default=[79, 59, 39, 19])
    parser.add_argument("--contamination-ratio", type=float, default=0.0)
    parser.add_argument("--contamination-kind", type=str, default="uniform", choices=AVAILABLE_CONTAMINATION_KINDS)
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--output-tag", type=str, default="")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
