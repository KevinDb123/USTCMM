from __future__ import annotations

import argparse

import torch

from src.diffusion import DiffusionModel
from src.utils import get_device, to_numpy
from src.visualize import plot_diffusion_snapshots, plot_generated_only


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a trained diffusion checkpoint and sample points.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save-prefix", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    model = DiffusionModel(
        data_dim=config.get("input_dim", 2),
        timesteps=config["timesteps"],
        hidden_dim=config["hidden_dim"],
        depth=config.get("depth", 3),
        dropout=config.get("dropout", 0.0),
        beta_start=config["beta_start"],
        beta_end=config["beta_end"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    generated, snapshots = model.sample(
        args.n_samples,
        device,
        snapshot_steps=[config["timesteps"] - 1, 50, 25, 0],
        sampler=config.get("sampler", "ddpm"),
        sampling_steps=config.get("sampling_steps"),
    )
    generated_np = to_numpy(generated)
    plot_generated_only(generated_np, title=f"Diffusion samples - {config['dataset']}", save_path=f"{args.save_prefix}_samples.png")
    plot_diffusion_snapshots(
        snapshots,
        title=f"Diffusion snapshots - {config['dataset']}",
        save_path=f"{args.save_prefix}_snapshots.png",
    )
    print(f"Saved samples to {args.save_prefix}_*.png")


if __name__ == "__main__":
    main()
