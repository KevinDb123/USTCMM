"""RealNVP Normalizing Flow for 2D distribution modeling.

RealNVP (Real-valued Non-Volume Preserving) uses affine coupling layers
to construct an invertible transformation from data space to latent space.
Each coupling layer splits the input and transforms one part conditioned
on the other, yielding a triangular Jacobian for efficient determinant computation.
"""

from __future__ import annotations

import torch
from torch import nn


class AffineCouplingLayer(nn.Module):
    """Single affine coupling layer that transforms half of the input.

    Splits the input into two parts [x_a, x_b]. x_a passes through unchanged;
    x_b is transformed as: z_b = x_b * exp(s(x_a)) + t(x_a).
    The log-determinant of the Jacobian is simply sum(s(x_a)).
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        dropout: float = 0.0,
        mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        if mask is None:
            # Default: first half unchanged, second half transformed
            mask = torch.zeros(dim)
            mask[dim // 2 :] = 1.0
        self.register_buffer("mask", mask)

        # Scale and translation network conditioned on the unchanged part
        unchanged_dim = int((1 - self.mask).sum().item())
        layers: list[nn.Module] = [nn.Linear(unchanged_dim, hidden_dim), nn.SiLU()]
        for _ in range(max(depth - 1, 1)):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, dim - unchanged_dim))
        # Initialize the last layer with small weights for stable training
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        self.s_net = nn.Sequential(*layers)

        # Separate t network for more expressiveness
        t_layers: list[nn.Module] = [nn.Linear(unchanged_dim, hidden_dim), nn.SiLU()]
        for _ in range(max(depth - 1, 1)):
            t_layers.append(nn.Linear(hidden_dim, hidden_dim))
            t_layers.append(nn.SiLU())
            if dropout > 0:
                t_layers.append(nn.Dropout(dropout))
        t_layers.append(nn.Linear(hidden_dim, dim - unchanged_dim))
        nn.init.zeros_(t_layers[-1].weight)
        nn.init.zeros_(t_layers[-1].bias)
        self.t_net = nn.Sequential(*t_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward transformation: data -> latent. Returns (z, log_det)."""
        mask = self.mask.to(x.device)
        x_masked = x * (1 - mask)  # unchanged part
        x_transformed = x * mask    # part to be transformed

        # Get the unchanged dimensions
        unchanged = x_masked[:, mask == 0]

        s = self.s_net(unchanged)
        # Clamp scale to prevent numerical overflow in exp(s)
        s = torch.tanh(s) * 3.0
        t = self.t_net(unchanged)

        # Transform: z_transformed = x_transformed * exp(s) + t
        z_transformed = x_transformed[:, mask == 1] * torch.exp(s) + t

        # Recombine
        z = torch.zeros_like(x)
        z[:, mask == 0] = unchanged
        z[:, mask == 1] = z_transformed

        log_det = s.sum(dim=1)
        return z, log_det

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse transformation: latent -> data."""
        mask = self.mask.to(z.device)
        z_unchanged = z[:, mask == 0]
        z_transformed = z[:, mask == 1]

        s = self.s_net(z_unchanged)
        s = torch.tanh(s) * 3.0  # same clamping as forward
        t = self.t_net(z_unchanged)

        # Inverse: x_transformed = (z_transformed - t) * exp(-s)
        x_transformed = (z_transformed - t) * torch.exp(-s)

        x = torch.zeros_like(z)
        x[:, mask == 0] = z_unchanged
        x[:, mask == 1] = x_transformed
        return x


class RealNVP(nn.Module):
    """RealNVP normalizing flow for 2D distribution modeling.

    Stacks multiple affine coupling layers with alternating masks
    to construct a flexible invertible transformation.
    The prior is a standard Gaussian.
    """

    def __init__(
        self,
        data_dim: int = 2,
        hidden_dim: int = 128,
        depth: int = 3,
        num_layers: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.data_dim = data_dim

        # Build coupling layers with alternating masks
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # Alternate which half is transformed
            mask = torch.zeros(data_dim)
            if i % 2 == 0:
                mask[data_dim // 2 :] = 1.0
            else:
                mask[: data_dim // 2] = 1.0
            self.layers.append(
                AffineCouplingLayer(
                    dim=data_dim,
                    hidden_dim=hidden_dim,
                    depth=depth,
                    dropout=dropout,
                    mask=mask,
                )
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: data -> latent. Returns (z, total_log_det)."""
        total_log_det = torch.zeros(x.size(0), device=x.device)
        z = x
        for layer in self.layers:
            z, log_det = layer(z)
            total_log_det = total_log_det + log_det
        return z, total_log_det

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse pass: latent -> data (generation)."""
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x

    def compute_loss(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute negative log-likelihood loss.

        Loss = -log p(x) = -[log p(z) + log |det(dz/dx)|]
        where p(z) is standard Gaussian and log_det comes from coupling layers.
        """
        z, log_det = self(x)
        # log p(z) = -0.5 * ||z||^2 - (D/2) * log(2*pi)
        prior_ll = -0.5 * torch.sum(z**2, dim=1) - 0.5 * self.data_dim * torch.log(
            torch.tensor(2.0 * torch.pi, device=z.device)
        )
        log_prob = prior_ll + log_det
        loss = -log_prob.mean()
        return {"loss": loss, "nll": loss.detach()}

    @torch.no_grad()
    def sample(self, n_samples: int, device: torch.device) -> torch.Tensor:
        """Generate samples by sampling from prior and applying inverse."""
        z = torch.randn(n_samples, self.data_dim, device=device)
        return self.inverse(z)
