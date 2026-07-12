"""Projection modules shared with the LeWM visual-latent interface."""

from __future__ import annotations

import torch
from torch import nn


class ProjectionMLP(nn.Module):
    """Two-layer projection head used after the visual CLS token."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)
