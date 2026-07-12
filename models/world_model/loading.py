"""Construction helpers for portable world-model checkpoints."""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

from models.world_model.frozen import FrozenWorldModel


def load_frozen_world_model(
    config_path: str | Path,
    weights_path: str | Path,
) -> FrozenWorldModel:
    config = OmegaConf.load(Path(config_path))
    backbone = hydra.utils.instantiate(config)
    state_dict = torch.load(Path(weights_path), map_location="cpu", weights_only=True)
    backbone.load_state_dict(state_dict, strict=True)
    return FrozenWorldModel(backbone)
