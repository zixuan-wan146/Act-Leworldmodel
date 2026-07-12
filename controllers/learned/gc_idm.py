"""Closed-loop GC-IDM controller."""

from __future__ import annotations

import torch

from controllers.base import ActionCommand, Controller
from data.action_transform import ActionTransform, IdentityActionTransform
from models.policies.gc_idm import GCIDMPolicy
from models.world_model.interfaces import LatentEncoder


class GCIDMController(Controller):
    """Re-encode the observation and predict one action at every step."""

    def __init__(
        self,
        encoder: LatentEncoder,
        policy: GCIDMPolicy,
        action_transform: ActionTransform | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.policy = policy
        self.action_transform = action_transform or IdentityActionTransform()
        self.register_buffer("_goal_latent", None, persistent=False)

    def reset(self, goal_observation: torch.Tensor) -> None:
        self._goal_latent = self.encoder.encode_observations(goal_observation).detach()

    def act(
        self,
        observation: torch.Tensor,
        steps_remaining: int | torch.Tensor,
    ) -> ActionCommand:
        if self._goal_latent is None:
            raise RuntimeError("reset(goal_observation) must be called before act")
        with torch.no_grad():
            current = self.encoder.encode_observations(observation)
            model_action = self.policy(current, self._goal_latent, steps_remaining)
            action = self.action_transform.decode(model_action)
        return ActionCommand(
            actions=action.unsqueeze(1),
            replan_after=1,
            diagnostics={"current_latent": current, "model_actions": model_action.unsqueeze(1)},
        )
