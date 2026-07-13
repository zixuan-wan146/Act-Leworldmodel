"""Dense prefix prediction and anti-collapse objectives."""

from __future__ import annotations

import torch
from torch import nn


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer from LeWorldModel.

    Adapted from the MIT-licensed implementation in
    `third_party/le-wm/module.py`. Inputs follow `[batch, time, latent_dim]`.
    """

    def __init__(self, knots: int = 17, num_projections: int = 1024) -> None:
        super().__init__()
        if knots < 2:
            raise ValueError("knots must be at least 2")
        if num_projections < 1:
            raise ValueError("num_projections must be positive")
        self.num_projections = num_projections
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        characteristic = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", characteristic)
        self.register_buffer("weights", weights * characteristic)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 3:
            raise ValueError("SIGReg expects [batch, time, latent_dim]")
        latents = latents.float()
        projections = torch.randn(
            latents.size(-1),
            self.num_projections,
            device=latents.device,
            dtype=latents.dtype,
        )
        projections = projections / projections.norm(p=2, dim=0).clamp_min(1e-12)
        projected = latents.transpose(0, 1) @ projections
        t = self.t.float()
        phi = self.phi.float()
        weights = self.weights.float()
        samples = projected.unsqueeze(-1) * t
        error = (samples.cos().mean(dim=1) - phi).square()
        error = error + samples.sin().mean(dim=1).square()
        statistic = (error @ weights) * latents.size(0)
        return statistic.mean()


class DensePrefixObjective(nn.Module):
    """Average prefix-level latent MSE plus SIGReg."""

    def __init__(
        self,
        sigreg_weight: float = 0.09,
        sigreg_knots: int = 17,
        sigreg_num_projections: int = 1024,
    ) -> None:
        super().__init__()
        if sigreg_weight < 0:
            raise ValueError("sigreg_weight cannot be negative")
        self.sigreg_weight = sigreg_weight
        self.sigreg = SIGReg(knots=sigreg_knots, num_projections=sigreg_num_projections)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        encoded_sequence: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if predictions.shape != targets.shape:
            raise ValueError(
                f"prediction shape {tuple(predictions.shape)} does not match "
                f"target shape {tuple(targets.shape)}"
            )
        prefix_loss = (predictions - targets).square().mean()
        if self.sigreg_weight:
            if encoded_sequence is None:
                raise ValueError("encoded_sequence is required when SIGReg is enabled")
            sigreg_loss = self.sigreg(encoded_sequence)
        else:
            sigreg_loss = prefix_loss.new_zeros(())
        total = prefix_loss + self.sigreg_weight * sigreg_loss
        return {
            "loss": total,
            "prefix_loss": prefix_loss,
            "sigreg_loss": sigreg_loss,
        }
