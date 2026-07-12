"""Common closed-loop controller contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


@dataclass
class ActionCommand:
    """A controller output with an explicit open-loop commitment length."""

    actions: torch.Tensor
    replan_after: int
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.actions.ndim != 3:
            raise ValueError("actions must have shape [batch, chunk, action_dim]")
        if not 1 <= self.replan_after <= self.actions.size(1):
            raise ValueError("replan_after must fall inside the returned action chunk")


class Controller(nn.Module, ABC):
    """Stateful controller used by the shared closed-loop evaluation protocol."""

    @abstractmethod
    def reset(self, goal_observation: torch.Tensor) -> None:
        """Start a new episode and cache goal-dependent state."""

    @abstractmethod
    def act(
        self,
        observation: torch.Tensor,
        steps_remaining: int | torch.Tensor,
    ) -> ActionCommand:
        """Return one action chunk and its commitment length."""
