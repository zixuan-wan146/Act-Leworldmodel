"""Project-owned cross-entropy planning baseline."""

from __future__ import annotations

from collections.abc import Callable

import torch

from controllers.base import ActionCommand, Controller, select_batch_rows
from data.action_transform import ActionBlockTransform


class CEMPlanner:
    """Optimize normalized action blocks against a latent world model."""

    def __init__(
        self,
        *,
        model,
        action_transform: ActionBlockTransform,
        horizon: int,
        receding_horizon: int,
        batch_size: int,
        num_samples: int,
        var_scale: float,
        n_steps: int,
        topk: int,
        device: str,
        seed: int,
        warm_start: bool,
    ) -> None:
        if not 1 <= receding_horizon <= horizon:
            raise ValueError("receding horizon must fall inside the planning horizon")
        if batch_size < 1 or num_samples < 2 or n_steps < 1:
            raise ValueError("CEM batch, sample, and iteration counts must be positive")
        if not 2 <= topk <= num_samples:
            raise ValueError("CEM topk must contain at least two candidates")
        self.model = model.to(device).eval().requires_grad_(False)
        self.action_transform = action_transform.to(device)
        self.horizon = int(horizon)
        self.receding_horizon = int(receding_horizon)
        self.batch_size = int(batch_size)
        self.num_samples = int(num_samples)
        self.var_scale = float(var_scale)
        self.n_steps = int(n_steps)
        self.topk = int(topk)
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.warm_start = bool(warm_start)
        self._next_init: torch.Tensor | None = None
        self._pool_size: int | None = None

    @property
    def action_dim(self) -> int:
        return self.action_transform.block_action_dim

    @property
    def commit_steps(self) -> int:
        return self.receding_horizon * self.action_transform.frameskip

    def reset(self, pool_size: int) -> None:
        if pool_size < 1:
            raise ValueError("CEM pool size must be positive")
        self._next_init = None
        self._pool_size = int(pool_size)

    def _initial_mean(
        self,
        batch: int,
        batch_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.warm_start and self._next_init is not None:
            if batch_indices is None:
                if self._next_init.size(0) != batch:
                    raise ValueError("warm-start batch differs from the controller pool")
                prefix = self._next_init
            else:
                prefix = self._next_init.index_select(0, batch_indices)
        else:
            prefix = torch.empty(batch, 0, self.action_dim, device=self.device)
        remaining = self.horizon - prefix.size(1)
        if remaining < 0:
            raise ValueError("warm-start prefix exceeds the planning horizon")
        tail = torch.zeros(batch, remaining, self.action_dim, device=self.device)
        return torch.cat((prefix, tail), dim=1)

    def _store_warm_start(
        self,
        rest: torch.Tensor,
        batch_indices: torch.Tensor | None,
    ) -> None:
        if not self.warm_start or not rest.size(1):
            self._next_init = None
            return
        if batch_indices is None:
            self._next_init = rest.detach()
            return
        if self._pool_size is None:
            raise RuntimeError("reset(pool_size) must be called before indexed CEM planning")
        if self._next_init is None:
            self._next_init = torch.zeros(
                (self._pool_size, *rest.shape[1:]),
                device=self.device,
                dtype=rest.dtype,
            )
        self._next_init.index_copy_(0, batch_indices, rest.detach())

    @torch.inference_mode()
    def __call__(
        self,
        observation: torch.Tensor,
        goal_observation: torch.Tensor,
        steps_remaining: int | torch.Tensor,
        *,
        batch_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del steps_remaining
        if observation.shape != goal_observation.shape or observation.ndim != 4:
            raise ValueError("current and goal images must have matching [B,C,H,W] shapes")
        observation = observation.to(self.device)
        goal_observation = goal_observation.to(self.device)
        total = observation.size(0)
        indices = None
        if batch_indices is not None:
            indices = torch.as_tensor(batch_indices, dtype=torch.long, device=self.device)
            if indices.shape != (total,):
                raise ValueError("batch_indices must contain one index per observation")
            if (
                self._pool_size is None
                or torch.any(indices < 0)
                or torch.any(indices >= self._pool_size)
                or indices.unique().numel() != total
            ):
                raise ValueError("batch_indices fall outside the configured CEM pool")
        mean = self._initial_mean(total, indices)
        variance = torch.full_like(mean, self.var_scale)

        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            batch_mean = mean[start:end]
            batch_variance = variance[start:end]
            for _ in range(self.n_steps):
                candidates = torch.randn(
                    end - start,
                    self.num_samples,
                    self.horizon,
                    self.action_dim,
                    generator=self.generator,
                    device=self.device,
                    dtype=batch_mean.dtype,
                )
                candidates = candidates * batch_variance[:, None] + batch_mean[:, None]
                candidates[:, 0] = batch_mean
                costs = self.model.candidate_cost(
                    observation[start:end],
                    goal_observation[start:end],
                    candidates,
                )
                elite_indices = torch.topk(costs, self.topk, dim=1, largest=False).indices
                elite_batch_indices = torch.arange(end - start, device=self.device)[:, None]
                elite = candidates[elite_batch_indices, elite_indices]
                batch_mean = elite.mean(dim=1)
                batch_variance = elite.std(dim=1)
            mean[start:end] = batch_mean
            variance[start:end] = batch_variance

        rest = mean[:, self.receding_horizon :]
        self._store_warm_start(rest, indices)
        return self.action_transform.decode(mean).float()


class CEMController(Controller):
    """Expose project-owned CEM planning through the controller contract."""

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
        if hasattr(self.planner, "reset"):
            self.planner.reset(goal_observation.size(0))

    def act(
        self,
        observation: torch.Tensor,
        steps_remaining: int | torch.Tensor,
        *,
        batch_indices: torch.Tensor | None = None,
    ) -> ActionCommand:
        if self._goal_observation is None:
            raise RuntimeError("reset(goal_observation) must be called before act")
        goal_observation = select_batch_rows(self._goal_observation, batch_indices)
        if goal_observation.size(0) != observation.size(0):
            raise ValueError("goal and observation batches differ")
        actions = self.planner(
            observation,
            goal_observation,
            steps_remaining,
            batch_indices=batch_indices,
        )
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        return ActionCommand(
            actions=actions,
            replan_after=min(self.commit_steps, actions.size(1)),
        )
