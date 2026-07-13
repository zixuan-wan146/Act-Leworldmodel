"""Construction helpers for portable world-model checkpoints."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import OmegaConf

from models.world_model.artifacts import (
    RELEASED_LEWM_KIND,
    load_portable_artifact,
    load_tensor_state_dict,
)
from models.world_model.frozen import FrozenWorldModel


def initialize_representation_from_lewm(
    backbone,
    weights_path: str | Path,
    *,
    freeze: bool = True,
) -> None:
    """Copy the released LeWM encoder/projector into a Fast-LeWM backbone."""

    state_dict, _ = load_portable_artifact(weights_path, expected_kind=RELEASED_LEWM_KIND)
    encoder = {
        key.removeprefix("encoder."): value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
    }
    projector = {
        key.removeprefix("projector."): value
        for key, value in state_dict.items()
        if key.startswith("projector.")
    }
    backbone.encoder.load_state_dict(encoder, strict=True)
    backbone.projector.load_state_dict(projector, strict=True)
    if freeze:
        backbone.encoder.requires_grad_(False).eval()
        backbone.projector.requires_grad_(False).eval()


def load_frozen_world_model(
    config_path: str | Path,
    weights_path: str | Path,
) -> FrozenWorldModel:
    config = OmegaConf.load(Path(config_path))
    backbone = hydra.utils.instantiate(config)
    state_dict = load_tensor_state_dict(weights_path)
    backbone.load_state_dict(state_dict, strict=True)
    return FrozenWorldModel(backbone)


def load_released_lewm(config_path: str | Path, weights_path: str | Path):
    """Reconstruct the released architecture and load a tensor-only artifact."""

    config = OmegaConf.load(Path(config_path))
    model = hydra.utils.instantiate(config)
    state_dict, metadata = load_portable_artifact(
        weights_path,
        expected_kind=RELEASED_LEWM_KIND,
    )
    model.load_state_dict(state_dict, strict=True)
    return model, metadata
