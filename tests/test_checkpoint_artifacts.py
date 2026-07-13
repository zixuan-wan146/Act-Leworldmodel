import pickle

import pytest
import torch

from models.world_model.artifacts import (
    RELEASED_LEWM_KIND,
    load_portable_artifact,
    load_tensor_state_dict,
)


class LegacyObjectCheckpoint(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))


def _artifact(state_dict, source_hash="a" * 64):
    return {
        "format_version": 1,
        "model_kind": RELEASED_LEWM_KIND,
        "metadata": {"source_checkpoint_sha256": source_hash},
        "state_dict": state_dict,
    }


def test_portable_artifact_loads_only_tensor_state(tmp_path):
    path = tmp_path / "released.pt"
    torch.save(_artifact({"weight": torch.arange(3)}), path)

    state, metadata = load_portable_artifact(path, expected_kind=RELEASED_LEWM_KIND)

    torch.testing.assert_close(state["weight"], torch.arange(3))
    assert metadata["source_checkpoint_sha256"] == "a" * 64


def test_portable_artifact_rejects_non_tensor_entries_and_bad_hashes(tmp_path):
    non_tensor = tmp_path / "non_tensor.pt"
    torch.save(_artifact({"weight": torch.ones(1), "epoch": 3}), non_tensor)
    with pytest.raises(ValueError, match="only string tensor entries"):
        load_portable_artifact(non_tensor, expected_kind=RELEASED_LEWM_KIND)

    bad_hash = tmp_path / "bad_hash.pt"
    torch.save(_artifact({"weight": torch.ones(1)}, source_hash="g" * 64), bad_hash)
    with pytest.raises(ValueError, match="source checkpoint SHA-256"):
        load_portable_artifact(bad_hash, expected_kind=RELEASED_LEWM_KIND)


def test_tensor_loader_rejects_legacy_python_objects(tmp_path):
    path = tmp_path / "legacy.ckpt"
    torch.save(LegacyObjectCheckpoint(), path)

    with pytest.raises(pickle.UnpicklingError):
        load_tensor_state_dict(path)
