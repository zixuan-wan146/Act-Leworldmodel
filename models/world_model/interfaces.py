"""Capability-level interfaces exposed by latent world models."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class LatentEncoder(Protocol):
    """Minimum capability required by reactive latent policies."""

    @property
    def latent_dim(self) -> int: ...

    def encode_observations(self, pixels: torch.Tensor) -> torch.Tensor: ...


@runtime_checkable
class LatentDynamics(LatentEncoder, Protocol):
    """Additional transition capability required by rollout-based methods."""

    @property
    def max_horizon(self) -> int: ...

    def predict_latents(
        self,
        anchor_latent: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor: ...
