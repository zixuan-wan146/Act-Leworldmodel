"""Construction helpers for portable world-model checkpoints."""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

from models.world_model.frozen import FrozenWorldModel


def initialize_representation_from_lewm(
    backbone,
    checkpoint_path: str | Path,
    *,
    freeze: bool = True,
) -> None:
    """Copy the released LeWM encoder/projector into a Fast-LeWM backbone."""

    source = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if not hasattr(source, "encoder") or not hasattr(source, "projector"):
        raise TypeError("LeWM checkpoint does not expose encoder and projector modules")
    backbone.encoder.load_state_dict(source.encoder.state_dict(), strict=True)
    backbone.projector.load_state_dict(source.projector.state_dict(), strict=True)
    if freeze:
        backbone.encoder.requires_grad_(False).eval()
        backbone.projector.requires_grad_(False).eval()


def load_frozen_world_model(
    config_path: str | Path,
    weights_path: str | Path,
) -> FrozenWorldModel:
    config = OmegaConf.load(Path(config_path))
    backbone = hydra.utils.instantiate(config)
    state_dict = torch.load(Path(weights_path), map_location="cpu", weights_only=True)
    backbone.load_state_dict(state_dict, strict=True)
    return FrozenWorldModel(backbone)
