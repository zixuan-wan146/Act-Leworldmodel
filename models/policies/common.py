"""Shared latent goal-conditioning modules for learned controllers."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalHorizonEncoding(nn.Module):
    """Encode a normalized remaining horizon with fixed Fourier features."""

    def __init__(self, dim: int, max_horizon: int) -> None:
        super().__init__()
        if dim < 2 or dim % 2:
            raise ValueError("horizon encoding dimension must be a positive even number")
        if max_horizon < 1:
            raise ValueError("max_horizon must be positive")
        self.dim = dim
        self.max_horizon = max_horizon
        frequencies = torch.exp(
            torch.arange(dim // 2, dtype=torch.float32)
            * (-math.log(10_000.0) / max(dim // 2 - 1, 1))
        )
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, steps_remaining: torch.Tensor, batch_size: int) -> torch.Tensor:
        steps = torch.as_tensor(
            steps_remaining,
            device=self.frequencies.device,
            dtype=self.frequencies.dtype,
        )
        if steps.ndim == 0:
            steps = steps.expand(batch_size)
        elif steps.ndim == 2 and steps.size(-1) == 1:
            steps = steps[:, 0]
        if steps.ndim != 1 or steps.size(0) != batch_size:
            raise ValueError("steps_remaining must be scalar, [batch], or [batch, 1]")
        normalized = steps.clamp(0, self.max_horizon) / self.max_horizon
        angles = 2.0 * math.pi * normalized.unsqueeze(-1) * self.frequencies
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class LatentGoalTrunk(nn.Module):
    """Goal-conditioned MLP with zero-initialized horizon AdaLN."""

    def __init__(
        self,
        *,
        latent_dim: int,
        hidden_dim: int,
        depth: int,
        dropout: float,
        horizon_dim: int,
        max_horizon: int,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be positive")
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.horizon_encoding = SinusoidalHorizonEncoding(horizon_dim, max_horizon)
        layers = []
        input_dim = 2 * latent_dim
        for _ in range(depth):
            linear = nn.Linear(input_dim, hidden_dim)
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            nn.init.zeros_(linear.bias)
            layers.append(
                nn.Sequential(
                    linear,
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            input_dim = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.horizon_mlp = nn.Sequential(
            nn.Linear(horizon_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.horizon_scale = nn.Linear(hidden_dim, hidden_dim)
        self.horizon_shift = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.horizon_scale.weight)
        nn.init.zeros_(self.horizon_scale.bias)
        nn.init.zeros_(self.horizon_shift.weight)
        nn.init.zeros_(self.horizon_shift.bias)

    def forward(
        self,
        current_latent: torch.Tensor,
        goal_latent: torch.Tensor,
        steps_remaining: torch.Tensor,
    ) -> torch.Tensor:
        if current_latent.ndim != 2 or goal_latent.ndim != 2:
            raise ValueError("current_latent and goal_latent must be [batch, latent_dim]")
        if current_latent.shape != goal_latent.shape:
            raise ValueError("current_latent and goal_latent must have identical shapes")
        if current_latent.size(-1) != self.latent_dim:
            raise ValueError(f"expected latent dimension {self.latent_dim}")
        hidden = torch.cat((current_latent, goal_latent), dim=-1)
        for layer in self.layers:
            hidden = layer(hidden)
        horizon = self.horizon_encoding(steps_remaining, current_latent.size(0))
        horizon = horizon.to(device=hidden.device, dtype=hidden.dtype)
        condition = self.horizon_mlp(horizon)
        scale = self.horizon_scale(condition)
        shift = self.horizon_shift(condition)
        return hidden * (1.0 + scale) + shift
