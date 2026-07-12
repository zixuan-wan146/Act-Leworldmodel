"""Goal-conditioned inverse dynamics policy over frozen latents."""

from __future__ import annotations

import torch
from torch import nn

from models.policies.common import LatentGoalTrunk


class GCIDMPolicy(nn.Module):
    """Predict one closed-loop action from current/goal latents and horizon."""

    def __init__(
        self,
        *,
        latent_dim: int = 192,
        action_dim: int,
        hidden_dim: int = 512,
        depth: int = 3,
        dropout: float = 0.1,
        horizon_dim: int = 64,
        max_horizon: int = 50,
        output_init_std: float = 0.01,
    ) -> None:
        super().__init__()
        if action_dim < 1:
            raise ValueError("action_dim must be positive")
        if output_init_std <= 0:
            raise ValueError("output_init_std must be positive")
        self.action_dim = action_dim
        self.trunk = LatentGoalTrunk(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            horizon_dim=horizon_dim,
            max_horizon=max_horizon,
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)
        nn.init.normal_(self.action_head.weight, std=output_init_std)
        nn.init.zeros_(self.action_head.bias)

    def forward(
        self,
        current_latent: torch.Tensor,
        goal_latent: torch.Tensor,
        steps_remaining: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.trunk(current_latent, goal_latent, steps_remaining)
        return self.action_head(hidden)
