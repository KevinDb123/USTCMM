from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.vae import ResidualBlock


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half, 1)
    )
    angles = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class NoisePredictor(nn.Module):
    def __init__(
        self,
        data_dim: int = 2,
        time_dim: int = 32,
        hidden_dim: int = 128,
        depth: int = 3,
        condition_dim: int = 0,
        dropout: float = 0.0,
        time_hidden_dim: int = 0,
    ) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.condition_dim = condition_dim
        self.time_mlp = (
            nn.Sequential(
                nn.Linear(time_dim, time_hidden_dim),
                nn.SiLU(),
                nn.Linear(time_hidden_dim, time_dim),
                nn.SiLU(),
            )
            if time_hidden_dim and time_hidden_dim > 0
            else nn.Identity()
        )
        input_dim = data_dim + time_dim + condition_dim
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(max(depth - 1, 1)):
            layers.append(ResidualBlock(hidden_dim, dropout=dropout))
        layers.append(nn.Linear(hidden_dim, data_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t_emb = self.time_mlp(sinusoidal_embedding(timesteps, self.time_dim))
        pieces = [x, t_emb]
        if condition is not None:
            pieces.append(condition)
        return self.net(torch.cat(pieces, dim=1))


class DiffusionModel(nn.Module):
    def __init__(
        self,
        data_dim: int = 2,
        timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        hidden_dim: int = 128,
        depth: int = 3,
        dropout: float = 0.0,
        schedule_type: str = "linear",
        time_hidden_dim: int = 0,
    ) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.timesteps = timesteps
        self.schedule_type = schedule_type
        self.noise_predictor = NoisePredictor(
            data_dim=data_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            time_hidden_dim=time_hidden_dim,
        )

        if schedule_type == "linear":
            betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        elif schedule_type == "cosine":
            steps = torch.arange(timesteps + 1, dtype=torch.float32)
            s = 0.008
            alphas_cumprod = torch.cos(((steps / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = betas.clamp(min=1e-5, max=0.999)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]], dim=0)
        posterior_var = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)
        self.register_buffer("posterior_var", posterior_var.clamp(min=1e-8))

    def q_sample(self, x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = torch.sqrt(self.alpha_bars[timesteps]).unsqueeze(1)
        sqrt_one_minus_ab = torch.sqrt(1.0 - self.alpha_bars[timesteps]).unsqueeze(1)
        return sqrt_ab * x0 + sqrt_one_minus_ab * noise

    def predict_x0(self, x: torch.Tensor, timesteps: torch.Tensor, eps_pred: torch.Tensor) -> torch.Tensor:
        alpha_bar_t = self.alpha_bars[timesteps].unsqueeze(1)
        return (x - torch.sqrt(1.0 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)

    def compute_loss(self, x0: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = x0.size(0)
        timesteps = torch.randint(0, self.timesteps, (batch_size,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, timesteps, noise)
        pred_noise = self.noise_predictor(xt, timesteps)
        loss = F.mse_loss(pred_noise, noise, reduction="mean")
        return {"loss": loss}

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
    ) -> tuple[torch.Tensor, Mapping[int, np.ndarray]]:
        x = torch.randn(n_samples, self.data_dim, device=device)
        snapshots: dict[int, np.ndarray] = {}
        snapshot_steps = set(snapshot_steps or [])

        if sampler == "ddpm":
            schedule = list(reversed(range(self.timesteps)))
            for step in schedule:
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
