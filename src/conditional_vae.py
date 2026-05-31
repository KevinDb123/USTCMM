from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.vae import DecoderBackbone, MLPBackbone


class ConditionalVAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        latent_dim: int = 4,
        condition_dim: int = 4,
        hidden_dim: int = 128,
        depth: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim

        self.encoder = MLPBackbone(
            input_dim=input_dim + condition_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.decoder = DecoderBackbone(
            latent_dim=latent_dim + condition_dim,
            output_dim=input_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
        )

    def _one_hot(self, labels: torch.Tensor) -> torch.Tensor:
        return F.one_hot(labels, num_classes=self.condition_dim).float()

    def encode(self, x: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cond = self._one_hot(labels)
        h = self.encoder(torch.cat([x, cond], dim=1))
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cond = self._one_hot(labels)
        return self.decoder(torch.cat([z, cond], dim=1))

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, labels)
        return recon, mu, logvar

    def compute_loss(self, x: torch.Tensor, labels: torch.Tensor, beta: float = 1.0) -> dict[str, torch.Tensor]:
        recon, mu, logvar = self(x, labels)
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + beta * kl_loss
        return {"loss": loss, "recon_loss": recon_loss, "kl_loss": kl_loss}

    @torch.no_grad()
    def sample(self, labels: torch.Tensor, device: torch.device) -> torch.Tensor:
        z = torch.randn(labels.size(0), self.latent_dim, device=device)
        return self.decode(z, labels.to(device))
