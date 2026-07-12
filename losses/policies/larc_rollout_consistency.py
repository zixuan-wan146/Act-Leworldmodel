"""Behavior cloning plus frozen-world-model rollout consistency for LARC."""

from __future__ import annotations

import torch
from torch import nn

from models.world_model.interfaces import LatentDynamics


class LARCRolloutConsistencyObjective(nn.Module):
    """Constrain a predicted action chunk to terminate near the goal latent."""

    def __init__(self, behavior_weight: float = 1.0, consistency_weight: float = 1.0) -> None:
        super().__init__()
        if behavior_weight < 0 or consistency_weight < 0:
            raise ValueError("loss weights cannot be negative")
        if behavior_weight == 0 and consistency_weight == 0:
            raise ValueError("at least one loss weight must be positive")
        self.behavior_weight = behavior_weight
        self.consistency_weight = consistency_weight

    def forward(
        self,
        *,
        predicted_actions: torch.Tensor,
        expert_actions: torch.Tensor | None,
        current_latent: torch.Tensor,
        goal_latent: torch.Tensor,
        world_model: LatentDynamics,
    ) -> dict[str, torch.Tensor]:
        zero = predicted_actions.new_zeros(())
        behavior_loss = zero
        if self.behavior_weight:
            if expert_actions is None:
                raise ValueError("expert_actions are required when behavior_weight is non-zero")
            if predicted_actions.shape != expert_actions.shape:
                raise ValueError("predicted and expert action chunks must have identical shapes")
            behavior_loss = (predicted_actions - expert_actions).square().mean()
        consistency_loss = zero
        terminal_latent = None
        if self.consistency_weight:
            predicted_latents = world_model.predict_latents(
                current_latent.detach(), predicted_actions
            )
            terminal_latent = predicted_latents[:, -1]
            consistency_loss = (terminal_latent - goal_latent.detach()).square().mean()
        total = (
            self.behavior_weight * behavior_loss
            + self.consistency_weight * consistency_loss
        )
        output = {
            "loss": total,
            "behavior_loss": behavior_loss,
            "rollout_consistency_loss": consistency_loss,
        }
        if terminal_latent is not None:
            output["terminal_latent"] = terminal_latent
        return output
