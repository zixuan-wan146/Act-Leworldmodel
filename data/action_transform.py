"""Explicit conversion between environment and world-model action spaces."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class ActionTransform(Protocol):
    def encode(self, environment_actions: torch.Tensor) -> torch.Tensor: ...

    def decode(self, model_actions: torch.Tensor) -> torch.Tensor: ...


class IdentityActionTransform(nn.Module):
    def encode(self, environment_actions: torch.Tensor) -> torch.Tensor:
        return environment_actions

    def decode(self, model_actions: torch.Tensor) -> torch.Tensor:
        return model_actions


class ZScoreActionTransform(nn.Module):
    """Persist the normalization statistics used by the world model."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32)
        std = torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6)
        if mean.shape != std.shape:
            raise ValueError("action mean and standard deviation must have identical shapes")
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def encode(self, environment_actions: torch.Tensor) -> torch.Tensor:
        return (environment_actions - self.mean) / self.std

    def decode(self, model_actions: torch.Tensor) -> torch.Tensor:
        return model_actions * self.std + self.mean
