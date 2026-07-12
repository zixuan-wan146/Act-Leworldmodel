"""Pixel encoder and latent dynamics composition."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from models.world_model.dynamics import PrefixDynamics


class FastLeWMBackbone(nn.Module):
    """Planner-independent visual Fast-LeWM backbone."""

    def __init__(
        self,
        *,
        encoder: nn.Module,
        dynamics: PrefixDynamics,
        projector: nn.Module | None = None,
        interpolate_pos_encoding: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.dynamics = dynamics
        self.projector = projector or nn.Identity()
        self.interpolate_pos_encoding = interpolate_pos_encoding

    @property
    def latent_dim(self) -> int:
        return self.dynamics.latent_dim

    @property
    def max_horizon(self) -> int:
        return self.dynamics.max_horizon

    @staticmethod
    def _extract_cls_token(encoder_output: Any) -> torch.Tensor:
        if hasattr(encoder_output, "last_hidden_state"):
            hidden = encoder_output.last_hidden_state
        elif isinstance(encoder_output, (tuple, list)):
            hidden = encoder_output[0]
        elif torch.is_tensor(encoder_output):
            hidden = encoder_output
        else:
            raise TypeError("visual encoder output does not expose hidden states")
        if hidden.ndim == 3:
            return hidden[:, 0]
        if hidden.ndim == 2:
            return hidden
        raise ValueError("visual encoder hidden states must have rank 2 or 3")

    def encode_observations(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode `[B,C,H,W]` or `[B,T,C,H,W]` pixels into latent tokens."""

        if pixels.ndim not in (4, 5):
            raise ValueError("pixels must have shape [B,C,H,W] or [B,T,C,H,W]")
        has_time = pixels.ndim == 5
        if has_time:
            batch, time = pixels.shape[:2]
            flat_pixels = pixels.reshape(batch * time, *pixels.shape[2:])
        else:
            batch, time = pixels.size(0), 1
            flat_pixels = pixels
        encoder_output = self.encoder(
            flat_pixels.float(),
            interpolate_pos_encoding=self.interpolate_pos_encoding,
        )
        latents = self.projector(self._extract_cls_token(encoder_output))
        if latents.size(-1) != self.latent_dim:
            raise ValueError(
                f"projected visual latent has dimension {latents.size(-1)}; "
                f"expected {self.latent_dim}"
            )
        latents = latents.reshape(batch, time, self.latent_dim)
        return latents if has_time else latents[:, 0]

    def predict_latents(
        self,
        anchor_latent: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        return self.dynamics(anchor_latent, actions)

    def forward(self, current_pixels: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.predict_latents(self.encode_observations(current_pixels), actions)
