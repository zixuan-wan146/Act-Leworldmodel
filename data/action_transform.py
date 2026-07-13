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


class ActionBlockTransform(ZScoreActionTransform):
    """Normalize raw actions and pack/unpack fixed-size world-model blocks.

    Push-T exposes two-dimensional actions at every environment step.  The
    world model advances by ``frameskip`` environment steps, so one model
    action is the concatenation of ``frameskip`` normalized raw actions.
    Keeping this conversion in one object prevents policies from confusing a
    raw action chunk with the block-valued action sequence used by dynamics.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, frameskip: int) -> None:
        super().__init__(mean=mean, std=std)
        if frameskip < 1:
            raise ValueError("frameskip must be positive")
        if self.mean.ndim != 1:
            raise ValueError("action statistics must be one-dimensional")
        self.frameskip = int(frameskip)

    @property
    def raw_action_dim(self) -> int:
        return self.mean.numel()

    @property
    def block_action_dim(self) -> int:
        return self.frameskip * self.raw_action_dim

    def encode(self, environment_actions: torch.Tensor) -> torch.Tensor:
        actions = torch.as_tensor(environment_actions)
        if actions.ndim < 2 or actions.size(-1) != self.raw_action_dim:
            raise ValueError(f"environment actions must end in raw dimension {self.raw_action_dim}")
        if actions.size(-2) % self.frameskip:
            raise ValueError("environment action length must be divisible by frameskip")
        normalized = super().encode(actions)
        leading = normalized.shape[:-2]
        blocks = normalized.size(-2) // self.frameskip
        return normalized.reshape(*leading, blocks, self.block_action_dim)

    def decode(self, model_actions: torch.Tensor) -> torch.Tensor:
        blocks = torch.as_tensor(model_actions)
        if blocks.ndim < 2 or blocks.size(-1) != self.block_action_dim:
            raise ValueError(f"model actions must end in block dimension {self.block_action_dim}")
        leading = blocks.shape[:-2]
        raw = blocks.reshape(
            *leading,
            blocks.size(-2) * self.frameskip,
            self.raw_action_dim,
        )
        return super().decode(raw)
