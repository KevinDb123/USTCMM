from __future__ import annotations

import torch
from torch.nn import functional as F

from src.diffusion import DiffusionModel, NoisePredictor


class ConditionalDiffusionModel(DiffusionModel):
    def __init__(
        self,
        data_dim: int = 2,
        condition_dim: int = 4,
        timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        hidden_dim: int = 128,
        depth: int = 3,
        dropout: float = 0.0,
        schedule_type: str = "linear",
        time_hidden_dim: int = 0,
    ) -> None:
        super().__init__(
            data_dim=data_dim,
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            schedule_type=schedule_type,
            time_hidden_dim=time_hidden_dim,
        )
        self.condition_dim = condition_dim
        self.noise_predictor = NoisePredictor(
            data_dim=data_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            condition_dim=condition_dim,
            dropout=dropout,
            time_hidden_dim=time_hidden_dim,
        )

    def _one_hot(self, labels: torch.Tensor) -> torch.Tensor:
        return F.one_hot(labels, num_classes=self.condition_dim).float()

    def compute_loss(self, x0: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = x0.size(0)
        timesteps = torch.randint(0, self.timesteps, (batch_size,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, timesteps, noise)
        condition = self._one_hot(labels)
        pred_noise = self.noise_predictor(xt, timesteps, condition=condition)
        loss = F.mse_loss(pred_noise, noise, reduction="mean")
        return {"loss": loss}

    def _ddpm_step(self, x: torch.Tensor, step: int, labels: torch.Tensor) -> torch.Tensor:
        t = torch.full((x.size(0),), step, device=x.device, dtype=torch.long)
        condition = self._one_hot(labels)
        eps_pred = self.noise_predictor(x, t, condition=condition)
        alpha_t = self.alphas[step]
        alpha_bar_t = self.alpha_bars[step]
        beta_t = self.betas[step]
        coef = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        mean = (x - coef * eps_pred) / torch.sqrt(alpha_t)
        if step > 0:
            noise = torch.randn_like(x)
            return mean + torch.sqrt(self.posterior_var[step]) * noise
        return mean

    def _ddim_step(self, x: torch.Tensor, step: int, next_step: int, labels: torch.Tensor) -> torch.Tensor:
        t = torch.full((x.size(0),), step, device=x.device, dtype=torch.long)
        condition = self._one_hot(labels)
        eps_pred = self.noise_predictor(x, t, condition=condition)
        x0_pred = self.predict_x0(x, t, eps_pred)
        if next_step < 0:
            return x0_pred
        alpha_bar_next = self.alpha_bars[next_step]
        return torch.sqrt(alpha_bar_next) * x0_pred + torch.sqrt(1.0 - alpha_bar_next) * eps_pred

    @torch.no_grad()
    def sample(
        self,
        labels: torch.Tensor,
        device: torch.device,
        snapshot_steps: list[int] | None = None,
        sampler: str = "ddpm",
        sampling_steps: int | None = None,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        labels = labels.to(device)
        x = torch.randn(labels.size(0), self.data_dim, device=device)
        snapshots: dict[int, torch.Tensor] = {}
        snapshot_steps = set(snapshot_steps or [])

        if sampler == "ddpm":
            schedule = list(reversed(range(self.timesteps)))
            for step in schedule:
                x = self._ddpm_step(x, step, labels=labels)
                if step in snapshot_steps:
                    snapshots[step] = x.detach().cpu().numpy()
            return x, snapshots

        schedule = self._ddim_schedule(sampling_steps or self.timesteps)
        next_schedule = schedule[1:] + [-1]
        for step, next_step in zip(schedule, next_schedule):
            x = self._ddim_step(x, step, next_step, labels=labels)
            if step in snapshot_steps:
                snapshots[step] = x.detach().cpu().numpy()
        return x, snapshots
