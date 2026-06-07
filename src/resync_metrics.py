from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.datasets import load_dataset
from src.diffusion import DiffusionModel, sinusoidal_embedding
from src.metrics import summarize_metrics
from src.utils import get_device, save_json, set_seed, to_numpy
from src.vae import VAE

REQUIRED_METRICS = ("mmd", "sliced_wasserstein", "chamfer", "kde_nll", "gmm_nll", "mode_coverage")
SUPPORTED_MODELS = ("vae", "diffusion")


class LegacyVAE(nn.Module):
    def __init__(self, input_dim: int = 2, latent_dim: int = 4, hidden_dim: int = 128) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    @torch.no_grad()
    def sample(self, n_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(n_samples, self.latent_dim, device=device)
        return self.decode(z)


class LegacyNoisePredictor(nn.Module):
    def __init__(self, data_dim: int = 2, hidden_dim: int = 128, time_dim: int = 32) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.net = nn.Sequential(
            nn.Linear(data_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_embedding(timesteps, self.time_dim)
        return self.net(torch.cat([x, t_emb], dim=1))


class LegacyDiffusionModel(nn.Module):
    def __init__(
        self,
        data_dim: int = 2,
        timesteps: int = 80,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.timesteps = timesteps
        self.noise_predictor = LegacyNoisePredictor(data_dim=data_dim, hidden_dim=hidden_dim)

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]], dim=0)
        posterior_var = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)
        self.register_buffer("posterior_var", posterior_var.clamp(min=1e-8))

    def predict_x0(self, x: torch.Tensor, timesteps: torch.Tensor, eps_pred: torch.Tensor) -> torch.Tensor:
        alpha_bar_t = self.alpha_bars[timesteps].unsqueeze(1)
        return (x - torch.sqrt(1.0 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)

    def _ddpm_step(self, x: torch.Tensor, step: int) -> torch.Tensor:
        t = torch.full((x.size(0),), step, device=x.device, dtype=torch.long)
        eps_pred = self.noise_predictor(x, t)
        alpha_t = self.alphas[step]
        alpha_bar_t = self.alpha_bars[step]
        beta_t = self.betas[step]
        coef = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        mean = (x - coef * eps_pred) / torch.sqrt(alpha_t)
        if step > 0:
            noise = torch.randn_like(x)
            return mean + torch.sqrt(self.posterior_var[step]) * noise
        return mean

    def _ddim_schedule(self, sampling_steps: int) -> list[int]:
        if sampling_steps >= self.timesteps:
            return list(reversed(range(self.timesteps)))
        schedule = np.linspace(0, self.timesteps - 1, sampling_steps, dtype=int)
        return list(reversed(sorted(set(int(step) for step in schedule))))

    def _ddim_step(self, x: torch.Tensor, step: int, next_step: int) -> torch.Tensor:
        t = torch.full((x.size(0),), step, device=x.device, dtype=torch.long)
        eps_pred = self.noise_predictor(x, t)
        x0_pred = self.predict_x0(x, t, eps_pred)
        if next_step < 0:
            return x0_pred
        alpha_bar_next = self.alpha_bars[next_step]
        return torch.sqrt(alpha_bar_next) * x0_pred + torch.sqrt(1.0 - alpha_bar_next) * eps_pred

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        device: torch.device,
        snapshot_steps: list[int] | None = None,
        sampler: str = "ddpm",
        sampling_steps: int | None = None,
    ) -> tuple[torch.Tensor, dict[int, np.ndarray]]:
        x = torch.randn(n_samples, self.data_dim, device=device)
        snapshots: dict[int, np.ndarray] = {}
        snapshot_steps = set(snapshot_steps or [])

        if sampler == "ddpm":
            for step in reversed(range(self.timesteps)):
                x = self._ddpm_step(x, step)
                if step in snapshot_steps:
                    snapshots[step] = x.detach().cpu().numpy()
            return x, snapshots

        if sampler != "ddim":
            raise ValueError(f"Unknown sampler: {sampler}")

        schedule = self._ddim_schedule(sampling_steps or self.timesteps)
        next_schedule = schedule[1:] + [-1]
        for step, next_step in zip(schedule, next_schedule):
            x = self._ddim_step(x, step, next_step)
            if step in snapshot_steps:
                snapshots[step] = x.detach().cpu().numpy()
        return x, snapshots


def _build_vae(config: dict[str, object]) -> VAE:
    return VAE(
        input_dim=int(config.get("input_dim", 2)),
        latent_dim=int(config.get("latent_dim", 4)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        depth=int(config.get("depth", 3)),
        dropout=float(config.get("dropout", 0.0)),
    )


def _build_diffusion(config: dict[str, object]) -> DiffusionModel:
    return DiffusionModel(
        data_dim=int(config.get("input_dim", config.get("data_dim", 2))),
        timesteps=int(config.get("timesteps", 80)),
        beta_start=float(config.get("beta_start", 1e-4)),
        beta_end=float(config.get("beta_end", 2e-2)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        depth=int(config.get("depth", 3)),
        dropout=float(config.get("dropout", 0.0)),
        schedule_type=str(config.get("schedule_type", "linear")),
        time_hidden_dim=int(config.get("time_hidden_dim", 0)),
    )


def _build_legacy_vae(config: dict[str, object]) -> LegacyVAE:
    return LegacyVAE(
        input_dim=int(config.get("input_dim", 2)),
        latent_dim=int(config.get("latent_dim", 4)),
        hidden_dim=int(config.get("hidden_dim", 128)),
    )


def _build_legacy_diffusion(config: dict[str, object]) -> LegacyDiffusionModel:
    return LegacyDiffusionModel(
        data_dim=int(config.get("input_dim", config.get("data_dim", 2))),
        timesteps=int(config.get("timesteps", 80)),
        beta_start=float(config.get("beta_start", 1e-4)),
        beta_end=float(config.get("beta_end", 2e-2)),
        hidden_dim=int(config.get("hidden_dim", 128)),
    )


def _load_model(
    model_name: str,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[dict[str, object], torch.nn.Module]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = dict(checkpoint.get("config", {}))
    state_dict = checkpoint["model_state_dict"]
    if model_name == "vae":
        if any(key.startswith("encoder.net.") for key in state_dict):
            model = _build_vae(config)
        else:
            model = _build_legacy_vae(config)
    elif model_name == "diffusion":
        if any(".net.2.net." in key or key.startswith("noise_predictor.time_mlp.") for key in state_dict):
            model = _build_diffusion(config)
        else:
            model = _build_legacy_diffusion(config)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return checkpoint, model


def _sample_generated(
    model_name: str,
    model: torch.nn.Module,
    n_samples: int,
    device: torch.device,
    config: dict[str, object],
) -> torch.Tensor:
    with torch.no_grad():
        if model_name == "vae":
            return model.sample(n_samples, device)
        generated, _ = model.sample(
            n_samples,
            device=device,
            sampler=str(config.get("sampler", "ddpm")),
            sampling_steps=config.get("sampling_steps"),
        )
        return generated


def run(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    table_dir = Path(args.table_dir)
    checkpoint_dir = Path(args.checkpoint_dir)

    updated = 0
    skipped = 0

    for metrics_path in sorted(table_dir.glob("*_metrics.json")):
        stem = metrics_path.stem.replace("_metrics", "")
        model_name, dataset = stem.split("_", 1)
        if model_name not in SUPPORTED_MODELS:
            skipped += 1
            print(f"Skipping unsupported metrics file: {metrics_path.name}")
            continue

        with metrics_path.open("r", encoding="utf-8") as f:
            current_metrics = json.load(f)
        missing = [key for key in REQUIRED_METRICS if key not in current_metrics]
        if not missing and not args.force:
            skipped += 1
            print(f"Already synced: {metrics_path.name}")
            continue

        checkpoint_path = checkpoint_dir / f"{model_name}_{dataset}.pt"
        if not checkpoint_path.exists():
            skipped += 1
            print(f"Checkpoint missing, skipping: {checkpoint_path.name}")
            continue

        set_seed(args.seed)
        dataset_bundle = load_dataset(dataset, root=args.data_root)
        checkpoint, model = _load_model(model_name, checkpoint_path, device)
        config = dict(checkpoint.get("config", {}))
        generated = _sample_generated(model_name, model, len(dataset_bundle.x_test), device, config)
        recomputed_metrics = summarize_metrics(dataset_bundle.x_test, to_numpy(generated), seed=args.seed)

        synced_metrics = dict(current_metrics)
        for key, value in recomputed_metrics.items():
            if args.force or key not in synced_metrics:
                synced_metrics[key] = value
        save_json(synced_metrics, metrics_path)

        checkpoint_metrics = dict(checkpoint.get("metrics", {}))
        for key, value in recomputed_metrics.items():
            if args.force or key not in checkpoint_metrics:
                checkpoint_metrics[key] = value
        checkpoint["metrics"] = checkpoint_metrics
        torch.save(checkpoint, checkpoint_path)

        updated += 1
        changed_keys = [key for key in recomputed_metrics if args.force or key not in current_metrics]
        print(f"Synced {metrics_path.name}; updated: {', '.join(sorted(changed_keys))}")

    print(f"Done. Updated {updated} file(s), skipped {skipped} file(s).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resync legacy metrics JSON files with the current metric set.")
    parser.add_argument("--table-dir", type=str, default="outputs/tables")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--force", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
