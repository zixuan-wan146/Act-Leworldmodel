"""Publish validated released-LeWM tensor artifacts."""

from __future__ import annotations

import os
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import torch

from models.world_model.artifacts import (
    PORTABLE_ARTIFACT_VERSION,
    RELEASED_LEWM_KIND,
    load_portable_artifact,
    load_tensor_state_dict,
)
from utils import file_sha256


def publish_released_lewm_artifact(
    *,
    source_weights: str | Path,
    model_config: str | Path,
    output_path: str | Path,
    overwrite: bool = False,
) -> dict:
    """Validate a bare tensor state dict and publish the portable schema."""

    source_weights = Path(source_weights).resolve()
    model_config = Path(model_config).resolve()
    output_path = Path(output_path).resolve()
    if source_weights == output_path:
        raise ValueError("source and published artifact paths must differ")
    source_hash = file_sha256(source_weights)
    config = OmegaConf.load(model_config)

    if output_path.exists() and not overwrite:
        state_dict, metadata = load_portable_artifact(
            output_path,
            expected_kind=RELEASED_LEWM_KIND,
        )
        if metadata["source_checkpoint_sha256"] != source_hash:
            raise ValueError("existing artifact was published from different source bytes")
        model = hydra.utils.instantiate(config)
        model.load_state_dict(state_dict, strict=True)
        return metadata

    state_dict = load_tensor_state_dict(source_weights)
    model = hydra.utils.instantiate(config)
    model.load_state_dict(state_dict, strict=True)
    metadata = {
        "source_checkpoint_sha256": source_hash,
        "source_model_config_sha256": file_sha256(model_config),
    }
    payload = {
        "format_version": PORTABLE_ARTIFACT_VERSION,
        "model_kind": RELEASED_LEWM_KIND,
        "metadata": metadata,
        "state_dict": state_dict,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output_path)

    published_state, published_metadata = load_portable_artifact(
        output_path,
        expected_kind=RELEASED_LEWM_KIND,
    )
    verified_model = hydra.utils.instantiate(config)
    verified_model.load_state_dict(published_state, strict=True)
    return published_metadata
