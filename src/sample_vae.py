from __future__ import annotations

import argparse

import torch

from src.utils import get_device, to_numpy
from src.vae import VAE
from src.visualize import plot_generated_only


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a trained VAE checkpoint and sample points.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    model = VAE(latent_dim=config["latent_dim"], hidden_dim=config["hidden_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    generated = to_numpy(model.sample(args.n_samples, device))
    plot_generated_only(generated, title=f"VAE samples - {config['dataset']}", save_path=args.save_path)
    print(f"Saved samples to {args.save_path}")


if __name__ == "__main__":
    main()
