"""Adapter for the original LeWM one-step dynamics baseline."""

from __future__ import annotations

import torch
from torch import nn


class OneStepDynamics(nn.Module):
    """Autoregressively apply a LeWM-style local transition model."""

    def __init__(self, *, action_encoder: nn.Module, predictor: nn.Module) -> None:
        super().__init__()
        self.action_encoder = action_encoder
        self.predictor = predictor

    def forward(self, anchor_latent: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if anchor_latent.ndim != 2:
            raise ValueError("anchor_latent must have shape [batch, latent_dim]")
        if actions.ndim != 3:
            raise ValueError("actions must have shape [batch, horizon, action_dim]")
        if anchor_latent.size(0) != actions.size(0):
            raise ValueError("anchor_latent and actions must share their batch dimension")
        if actions.size(1) < 1:
            raise ValueError("action horizon must be positive")
        current = anchor_latent
        predictions = []
        for step in range(actions.size(1)):
            action_embedding = self.action_encoder(actions[:, step : step + 1])
            current = self.predictor(current.unsqueeze(1), action_embedding)[:, -1]
            predictions.append(current)
        return torch.stack(predictions, dim=1)
