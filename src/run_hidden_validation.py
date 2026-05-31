from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from src.conditional_diffusion import ConditionalDiffusionModel
from src.conditional_vae import ConditionalVAE
from src.datasets import (
    AVAILABLE_DATASETS,
    DATASET_TO_LABEL,
    LABEL_TO_DATASET,
    load_all_hidden_labeled_datasets,
    load_hidden_dataset,
)
from src.diffusion import DiffusionModel
from src.metrics import summarize_metrics
from src.utils import ensure_dir, get_device, set_seed, to_numpy
from src.vae import VAE


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_vae(checkpoint_path: Path, device: torch.device) -> VAE:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = VAE(
        input_dim=config.get("input_dim", 2),
        latent_dim=config["latent_dim"],
        hidden_dim=config["hidden_dim"],
        depth=config.get("depth", 3),
        dropout=config.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _load_diffusion(checkpoint_path: Path, device: torch.device) -> tuple[DiffusionModel, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = DiffusionModel(
        data_dim=config.get("input_dim", 2),
        timesteps=config["timesteps"],
        beta_start=config["beta_start"],
        beta_end=config["beta_end"],
        hidden_dim=config["hidden_dim"],
        depth=config.get("depth", 3),
        dropout=config.get("dropout", 0.0),
        schedule_type=config.get("schedule_type", "linear"),
        time_hidden_dim=config.get("time_hidden_dim", 0),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def _load_cvae(checkpoint_path: Path, device: torch.device) -> ConditionalVAE:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = ConditionalVAE(
        input_dim=config.get("input_dim", 2),
        latent_dim=config["latent_dim"],
        condition_dim=config["condition_dim"],
        hidden_dim=config["hidden_dim"],
        depth=config.get("depth", 3),
        dropout=config.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _load_conditional_diffusion(checkpoint_path: Path, device: torch.device) -> tuple[ConditionalDiffusionModel, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = ConditionalDiffusionModel(
        data_dim=config.get("data_dim", 2),
        condition_dim=config["condition_dim"],
        timesteps=config["timesteps"],
        beta_start=config["beta_start"],
        beta_end=config["beta_end"],
        hidden_dim=config["hidden_dim"],
        depth=config.get("depth", 3),
        dropout=config.get("dropout", 0.0),
        schedule_type=config.get("schedule_type", "linear"),
        time_hidden_dim=config.get("time_hidden_dim", 0),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def _write_rows(rows: list[dict[str, float | str]], path: Path) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device(args.device)
    table_dir = Path(args.table_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    rows: list[dict[str, float | str]] = []

    for dataset in AVAILABLE_DATASETS:
        hidden_bundle = load_hidden_dataset(dataset, root=args.data_root)
        n_samples = len(hidden_bundle.x_test)

        vae_model = _load_vae(ckpt_dir / f"vae_{dataset}_enhanced.pt", device)
        _sync_if_needed(device)
        vae_generated = to_numpy(vae_model.sample(n_samples, device))
        vae_metrics = summarize_metrics(hidden_bundle.x_test, vae_generated, seed=args.seed)
        rows.append({"dataset": dataset, "model": "vae", "split": "hidden_test", **vae_metrics})

        diffusion_model, diffusion_config = _load_diffusion(ckpt_dir / f"diffusion_{dataset}_enhanced.pt", device)
        _sync_if_needed(device)
        diffusion_generated, _ = diffusion_model.sample(
            n_samples,
            device=device,
            sampler=str(diffusion_config.get("sampler", "ddpm")),
            sampling_steps=diffusion_config.get("sampling_steps"),
        )
        diffusion_metrics = summarize_metrics(hidden_bundle.x_test, to_numpy(diffusion_generated), seed=args.seed)
        rows.append(
            {
                "dataset": dataset,
                "model": "diffusion",
                "split": "hidden_test",
                "sampler": diffusion_config.get("sampler", "ddpm"),
                "sampling_steps": diffusion_config.get("sampling_steps"),
                **diffusion_metrics,
            }
        )

    hidden_labeled = load_all_hidden_labeled_datasets(root=args.data_root)
    cvae = _load_cvae(ckpt_dir / "cvae_all.pt", device)
    cdiff, cdiff_config = _load_conditional_diffusion(ckpt_dir / "conditional_diffusion_all.pt", device)
    for dataset in AVAILABLE_DATASETS:
        label = DATASET_TO_LABEL[dataset]
        mask = hidden_labeled.y_test == label
        real = hidden_labeled.x_test[mask]
        labels = torch.full((len(real),), label, dtype=torch.long, device=device)

        _sync_if_needed(device)
        cvae_generated = to_numpy(cvae.sample(labels, device))
        cvae_metrics = summarize_metrics(real, cvae_generated, seed=args.seed + label)
        rows.append({"dataset": dataset, "model": "cvae", "split": "hidden_test", **cvae_metrics})

        _sync_if_needed(device)
        cdiff_generated, _ = cdiff.sample(
            labels,
            device=device,
            sampler=str(cdiff_config.get("sampler", "ddpm")),
            sampling_steps=cdiff_config.get("sampling_steps"),
        )
        cdiff_metrics = summarize_metrics(real, to_numpy(cdiff_generated), seed=args.seed + label)
        rows.append(
            {
                "dataset": dataset,
                "model": "conditional_diffusion",
                "split": "hidden_test",
                "sampler": cdiff_config.get("sampler", "ddpm"),
                "sampling_steps": cdiff_config.get("sampling_steps"),
                **cdiff_metrics,
            }
        )

    _write_rows(rows, table_dir / "hidden_validation.csv")
    print(f"Saved hidden validation results to {table_dir / 'hidden_validation.csv'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate trained models on the hidden official split.")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints/optional")
    parser.add_argument("--table-dir", type=str, default="outputs/tables/optional")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
