from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class MLPBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        blocks = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(max(depth - 1, 1)):
            blocks.append(ResidualBlock(hidden_dim, dropout=dropout))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBackbone(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int, depth: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(latent_dim, hidden_dim), nn.SiLU()]
        for _ in range(max(depth - 1, 1)):
            layers.append(ResidualBlock(hidden_dim, dropout=dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        latent_dim: int = 2,
        hidden_dim: int = 128,
        depth: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = MLPBackbone(input_dim=input_dim, hidden_dim=hidden_dim, depth=depth, dropout=dropout)
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.decoder = DecoderBackbone(
            latent_dim=latent_dim,
            output_dim=input_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def compute_loss(self, x: torch.Tensor, beta: float = 1.0) -> dict[str, torch.Tensor]:
        recon, mu, logvar = self(x)
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + beta * kl_loss
        return {"loss": loss, "recon_loss": recon_loss, "kl_loss": kl_loss}

    @torch.no_grad()
    def sample(self, n_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(n_samples, self.latent_dim, device=device)
        return self.decode(z)
