"""Dense prefix prediction objective for frozen-latent Fast-LeWM training."""

from __future__ import annotations

import torch
from torch import nn


class DensePrefixObjective(nn.Module):
    """Average latent MSE with equal weight for every predicted prefix."""

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if predictions.shape != targets.shape:
            raise ValueError(
                f"prediction shape {tuple(predictions.shape)} does not match "
                f"target shape {tuple(targets.shape)}"
            )
        prefix_loss = (predictions - targets).square().mean()
        return {"loss": prefix_loss, "prefix_loss": prefix_loss}
