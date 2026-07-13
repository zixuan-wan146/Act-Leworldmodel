"""Behavior cloning plus frozen-world-model rollout consistency for LARC."""

from __future__ import annotations

import torch
from torch import nn

from models.world_model.interfaces import LatentDynamics


class LARCRolloutConsistencyObjective(nn.Module):
    """Constrain a predicted action chunk to terminate near the goal latent."""

    def __init__(
        self,
        behavior_weight: float = 1.0,
        consistency_weight: float = 1.0,
        frameskip: int = 1,
    ) -> None:
        super().__init__()
        if behavior_weight < 0 or consistency_weight < 0:
            raise ValueError("loss weights cannot be negative")
        if behavior_weight == 0 and consistency_weight == 0:
            raise ValueError("at least one loss weight must be positive")
        self.behavior_weight = behavior_weight
        self.consistency_weight = consistency_weight
        if frameskip < 1:
            raise ValueError("frameskip must be positive")
        self.frameskip = int(frameskip)

    def forward(
        self,
        *,
        predicted_actions: torch.Tensor,
        expert_actions: torch.Tensor | None,
        current_latent: torch.Tensor,
        goal_latent: torch.Tensor,
        world_model: LatentDynamics,
        steps_remaining: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        zero = predicted_actions.new_zeros(())
        behavior_loss = zero
        if self.behavior_weight:
            if expert_actions is None:
                raise ValueError("expert_actions are required when behavior_weight is non-zero")
            if predicted_actions.shape != expert_actions.shape:
                raise ValueError("predicted and expert action chunks must have identical shapes")
            squared_error = (predicted_actions - expert_actions).square()
            if action_mask is None:
                behavior_loss = squared_error.mean()
            else:
                if action_mask.shape != predicted_actions.shape[:-1]:
                    raise ValueError("action_mask must have shape [batch, chunk_size]")
                weights = action_mask.to(squared_error).unsqueeze(-1)
                denominator = weights.sum() * predicted_actions.size(-1)
                behavior_loss = (squared_error * weights).sum() / denominator.clamp_min(1)
        consistency_loss = zero
        terminal_latent = None
        if self.consistency_weight:
            predicted_latents = world_model.predict_latents(
                current_latent.detach(), predicted_actions
            )
            if steps_remaining is None:
                terminal_latent = predicted_latents[:, -1]
            else:
                remaining = torch.as_tensor(
                    steps_remaining, device=predicted_latents.device
                ).reshape(-1)
                if remaining.size(0) != predicted_latents.size(0):
                    raise ValueError("steps_remaining must contain one value per batch item")
                rollout_steps = torch.div(
                    remaining + self.frameskip - 1,
                    self.frameskip,
                    rounding_mode="floor",
                ).clamp(1, predicted_latents.size(1))
                batch_indices = torch.arange(
                    predicted_latents.size(0), device=predicted_latents.device
                )
                terminal_latent = predicted_latents[batch_indices, rollout_steps - 1]
            consistency_loss = (terminal_latent - goal_latent.detach()).square().mean()
        total = self.behavior_weight * behavior_loss + self.consistency_weight * consistency_loss
        output = {
            "loss": total,
            "behavior_loss": behavior_loss,
            "rollout_consistency_loss": consistency_loss,
        }
        if terminal_latent is not None:
            output["terminal_latent"] = terminal_latent
        return output
