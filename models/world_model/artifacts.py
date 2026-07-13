"""Validated tensor-only checkpoint artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from utils import is_sha256


PORTABLE_ARTIFACT_VERSION = 1
RELEASED_LEWM_KIND = "released_lewm"


def _validate_state_dict(value: Any) -> dict[str, torch.Tensor]:
    if not isinstance(value, dict) or not value:
        raise ValueError("checkpoint state_dict must be a non-empty mapping")
    if not all(isinstance(key, str) and torch.is_tensor(tensor) for key, tensor in value.items()):
        raise ValueError("checkpoint state_dict may contain only string tensor entries")
    return value


def load_portable_artifact(
    path: str | Path,
    *,
    expected_kind: str,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Load a project artifact without invoking Python pickle globals."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("portable checkpoint must contain a mapping")
    if payload.get("format_version") != PORTABLE_ARTIFACT_VERSION:
        raise ValueError("unsupported portable checkpoint format version")
    if payload.get("model_kind") != expected_kind:
        raise ValueError(f"portable checkpoint is not a {expected_kind!r} artifact")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("portable checkpoint metadata must be a mapping")
    source_hash = metadata.get("source_checkpoint_sha256")
    if not is_sha256(source_hash):
        raise ValueError("portable checkpoint lacks its source checkpoint SHA-256")
    return _validate_state_dict(payload.get("state_dict")), metadata


def load_tensor_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a bare state dict produced by this project's training callbacks."""

    return _validate_state_dict(torch.load(Path(path), map_location="cpu", weights_only=True))
