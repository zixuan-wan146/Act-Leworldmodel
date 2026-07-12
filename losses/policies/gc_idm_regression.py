"""Supervised inverse-dynamics regression objective."""

from __future__ import annotations

import torch
from torch import nn


class GCIDMRegressionObjective(nn.Module):
    """Regress the first action toward a future goal latent."""

    def forward(
        self,
        predicted_action: torch.Tensor,
        target_action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if predicted_action.shape != target_action.shape:
            raise ValueError("predicted and target actions must have identical shapes")
        regression = (predicted_action - target_action).square().mean()
        return {"loss": regression, "action_mse": regression}
