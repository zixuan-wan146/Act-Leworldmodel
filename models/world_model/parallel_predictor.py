"""Parallel, action-modulated latent predictor for Fast-LeWM."""

from __future__ import annotations

import torch
from torch import nn


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift


class ActionModulatedResidualBlock(nn.Module):
    """Residual MLP with zero-initialized AdaLN modulation."""

    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        hidden_dim: int,
        fusion_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.Dropout(dropout),
        )
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, fusion_dim),
            nn.SiLU(),
            nn.Linear(fusion_dim, 3 * latent_dim),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, latent: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.modulation(condition).chunk(3, dim=-1)
        residual = self.mlp(_modulate(self.norm(latent), shift, scale))
        return latent + gate * residual


class ParallelLatentPredictor(nn.Module):
    """Predict all prefix endpoints without recurrent latent-state feedback."""

    def __init__(
        self,
        *,
        latent_dim: int = 192,
        prefix_dim: int = 192,
        hidden_dim: int = 2048,
        fusion_dim: int = 768,
        depth: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.prefix_dim = prefix_dim
        self.blocks = nn.ModuleList(
            ActionModulatedResidualBlock(
                latent_dim=latent_dim,
                condition_dim=prefix_dim,
                hidden_dim=hidden_dim,
                fusion_dim=fusion_dim,
                dropout=dropout,
            )
            for _ in range(depth)
        )

    def forward(
        self, anchor_latent: torch.Tensor, prefix_tokens: torch.Tensor
    ) -> torch.Tensor:
        if anchor_latent.ndim != 2:
            raise ValueError("anchor_latent must have shape [batch, latent_dim]")
        if prefix_tokens.ndim != 3:
            raise ValueError("prefix_tokens must have shape [batch, horizon, prefix_dim]")
        if anchor_latent.size(0) != prefix_tokens.size(0):
            raise ValueError("anchor_latent and prefix_tokens must share their batch dimension")
        if anchor_latent.size(-1) != self.latent_dim:
            raise ValueError(f"expected latent dimension {self.latent_dim}")
        if prefix_tokens.size(-1) != self.prefix_dim:
            raise ValueError(f"expected prefix dimension {self.prefix_dim}")
        predicted = anchor_latent.unsqueeze(1).expand(-1, prefix_tokens.size(1), -1)
        for block in self.blocks:
            predicted = block(predicted, prefix_tokens)
        return predicted
