"""Thin adapter retaining reproduced CEM as an evaluation-only baseline."""

from __future__ import annotations

from collections.abc import Callable

import torch

from controllers.base import ActionCommand, Controller


class CEMController(Controller):
    """Wrap an existing LeWM/stable-worldmodel CEM planning callable."""

    def __init__(
        self,
        planner: Callable[[torch.Tensor, torch.Tensor, int | torch.Tensor], torch.Tensor],
        commit_steps: int,
    ) -> None:
        super().__init__()
        if commit_steps < 1:
            raise ValueError("commit_steps must be positive")
        self.planner = planner
        self.commit_steps = commit_steps
        self.register_buffer("_goal_observation", None, persistent=False)

    def reset(self, goal_observation: torch.Tensor) -> None:
        self._goal_observation = goal_observation

    def act(
        self,
        observation: torch.Tensor,
        steps_remaining: int | torch.Tensor,
    ) -> ActionCommand:
        if self._goal_observation is None:
            raise RuntimeError("reset(goal_observation) must be called before act")
        actions = self.planner(observation, self._goal_observation, steps_remaining)
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        return ActionCommand(
            actions=actions,
            replan_after=min(self.commit_steps, actions.size(1)),
        )
