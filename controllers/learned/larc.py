"""Closed-loop LARC action-chunk controller."""

from __future__ import annotations

import torch

from controllers.base import ActionCommand, Controller
from data.action_transform import ActionTransform, IdentityActionTransform
from models.policies.larc_chunk import LARCChunkPolicy
from models.world_model.interfaces import LatentEncoder


class LARCController(Controller):
    """Predict a chunk, execute a configured prefix, then replan from pixels."""

    def __init__(
        self,
        encoder: LatentEncoder,
        policy: LARCChunkPolicy,
        commit_steps: int | None = None,
        action_transform: ActionTransform | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.policy = policy
        self.action_transform = action_transform or IdentityActionTransform()
        self.commit_steps = policy.chunk_size if commit_steps is None else commit_steps
        if not 1 <= self.commit_steps <= policy.chunk_size:
            raise ValueError("commit_steps must fall inside the policy chunk")
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
            model_actions = self.policy(current, self._goal_latent, steps_remaining)
            actions = self.action_transform.decode(model_actions)
        return ActionCommand(
            actions=actions,
            replan_after=self.commit_steps,
            diagnostics={"current_latent": current, "model_actions": model_actions},
        )
