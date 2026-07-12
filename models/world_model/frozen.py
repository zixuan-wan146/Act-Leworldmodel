"""Frozen world-model view used by learned controllers and policy losses."""

from __future__ import annotations

import torch
from torch import nn

from models.world_model.interfaces import LatentDynamics


class FrozenWorldModel(nn.Module):
    """Freeze parameters while retaining action gradients through dynamics.

    Observation encoding is detached because policies consume latents as fixed
    features. Dynamics prediction deliberately remains under normal autograd:
    LARC needs gradients from terminal latent loss back to its proposed actions.
    """

    def __init__(self, backbone: LatentDynamics) -> None:
        super().__init__()
        if not isinstance(backbone, nn.Module):
            raise TypeError("backbone must be a torch module")
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

    @property
    def latent_dim(self) -> int:
        return self.backbone.latent_dim

    @property
    def max_horizon(self) -> int:
        return self.backbone.max_horizon

    def train(self, mode: bool = True):
        super().train(False)
        self.backbone.eval()
        return self

    def encode_observations(self, pixels: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.backbone.encode_observations(pixels).detach()

    def predict_latents(
        self,
        anchor_latent: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        return self.backbone.predict_latents(anchor_latent.detach(), actions)

    def forward(self, current_pixels: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        anchor = self.encode_observations(current_pixels)
        return self.predict_latents(anchor, actions)
