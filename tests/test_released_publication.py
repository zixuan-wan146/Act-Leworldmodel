from pathlib import Path

import pytest
import torch

from models.world_model import publish_released_lewm_artifact
from models.world_model.artifacts import RELEASED_LEWM_KIND, load_portable_artifact
from utils import file_sha256


def _linear_source(tmp_path: Path, value: float) -> tuple[Path, Path]:
    config = tmp_path / "model.yaml"
    config.write_text("_target_: torch.nn.Linear\nin_features: 2\nout_features: 1\n")
    source = tmp_path / f"source_{value}.pt"
    torch.save(
        {
            "weight": torch.full((1, 2), value),
            "bias": torch.full((1,), value),
        },
        source,
    )
    return config, source


def test_released_publication_is_tensor_only_strict_and_idempotent(tmp_path):
    config, source = _linear_source(tmp_path, 3.0)
    destination = tmp_path / "released.pt"

    metadata = publish_released_lewm_artifact(
        source_weights=source,
        model_config=config,
        output_path=destination,
    )
    state, loaded_metadata = load_portable_artifact(
        destination,
        expected_kind=RELEASED_LEWM_KIND,
    )
    assert metadata["source_checkpoint_sha256"] == file_sha256(source)
    assert loaded_metadata == metadata
    torch.testing.assert_close(state["weight"], torch.full((1, 2), 3.0))

    second = publish_released_lewm_artifact(
        source_weights=source,
        model_config=config,
        output_path=destination,
    )
    assert second == metadata


def test_released_publication_rejects_source_mismatch_and_bad_shapes(tmp_path):
    config, first_source = _linear_source(tmp_path, 1.0)
    destination = tmp_path / "released.pt"
    publish_released_lewm_artifact(
        source_weights=first_source,
        model_config=config,
        output_path=destination,
    )
    _, second_source = _linear_source(tmp_path, 2.0)
    with pytest.raises(ValueError, match="different source bytes"):
        publish_released_lewm_artifact(
            source_weights=second_source,
            model_config=config,
            output_path=destination,
        )

    malformed = tmp_path / "malformed.pt"
    torch.save({"weight": torch.ones(2, 2), "bias": torch.ones(1)}, malformed)
    with pytest.raises(RuntimeError):
        publish_released_lewm_artifact(
            source_weights=malformed,
            model_config=config,
            output_path=tmp_path / "bad.pt",
        )
