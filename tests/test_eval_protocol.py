import json

import h5py
import numpy as np
import pytest
from omegaconf import OmegaConf

from eval.closed_loop import (
    FRAME_LINEAGE_FIELDS,
    HORIZON_VIEW_FIELDS,
    _validate_evaluation_protocol,
    _validate_learned_artifacts,
)
from eval.protocol import create_or_load_manifest


def test_manifest_uses_only_validation_episodes_and_is_reused(tmp_path):
    dataset_path = tmp_path / "data.h5"
    lengths = np.array([60, 61, 62, 63, 64, 65], dtype=np.int32)
    with h5py.File(dataset_path, "w") as dataset:
        dataset.create_dataset("ep_len", data=lengths)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    np.save(cache_dir / "frame_latents.npy", np.zeros((1, 2), dtype=np.float16))
    metadata = {
        "version": 2,
        "dataset_path": str(dataset_path),
        "dataset_size": dataset_path.stat().st_size,
        "dataset_mtime_ns": dataset_path.stat().st_mtime_ns,
        "source_checkpoint_sha256": "checkpoint",
        "seed": 3072,
        "train_fraction": 0.9,
        "validation_episode_ids": [2, 3, 4, 5],
    }
    (cache_dir / "metadata.json").write_text(json.dumps(metadata))
    manifest_path = tmp_path / "manifest.json"
    first = create_or_load_manifest(
        dataset_path=dataset_path,
        latent_cache_dir=cache_dir,
        output_path=manifest_path,
        seed=42,
        num_eval=3,
        goal_offsets=[25, 35, 50],
    )
    second = create_or_load_manifest(
        dataset_path=dataset_path,
        latent_cache_dir=cache_dir,
        output_path=manifest_path,
        seed=42,
        num_eval=3,
        goal_offsets=[25, 35, 50],
    )
    assert first == second
    assert first["goal_offsets"] == [25, 35, 50]
    assert first["max_goal_offset"] == 50
    assert set(first["episode_indices"]).issubset({2, 3, 4, 5})
    assert len(set(first["episode_indices"])) == 3
    for episode, start in zip(first["episode_indices"], first["start_steps"]):
        assert 0 <= start < lengths[episode] - 50

    with pytest.raises(ValueError, match="incompatible num_eval"):
        create_or_load_manifest(
            dataset_path=dataset_path,
            latent_cache_dir=cache_dir,
            output_path=manifest_path,
            seed=42,
            num_eval=2,
            goal_offsets=[25, 35, 50],
        )


def test_resolved_protocol_enforces_budget_and_cem_horizon():
    cfg = OmegaConf.create(
        {
            "protocol": {
                "goal_offsets": [25, 35, 50],
                "goal_offset": 35,
                "budget_multiplier": 2,
                "eval_budget": 70,
            },
            "cem": {
                "action_block": 5,
                "horizon": 7,
                "receding_horizon": 1,
            },
        }
    )
    assert _validate_evaluation_protocol(cfg, "cem") == (25, 35, 50)

    cfg.protocol.eval_budget = 71
    with pytest.raises(ValueError, match="eval_budget"):
        _validate_evaluation_protocol(cfg, "cem")
    cfg.protocol.eval_budget = 70
    cfg.cem.horizon = 10
    with pytest.raises(ValueError, match="CEM horizon"):
        _validate_evaluation_protocol(cfg, "cem")


def test_learned_artifact_validation_rejects_mixed_action_protocols():
    common = {field: 0 for field in FRAME_LINEAGE_FIELDS + HORIZON_VIEW_FIELDS}
    common.update(
        {
            "source_checkpoint_sha256": "encoder",
            "seed": 3072,
            "train_fraction": 0.9,
            "frameskip": 5,
            "max_horizon": 10,
            "max_goal_offset": 50,
            "latent_dim": 192,
            "action_statistics": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
            "training_seed": 3072,
            "training_code_revision": "a" * 40,
        }
    )
    policy = {**common, "method": "larc"}
    _validate_learned_artifacts(
        policy,
        common,
        common,
        method="larc",
        training_seed=3072,
        goal_offset=25,
        code_revision="a" * 40,
    )
    incompatible_world = {
        **common,
        "action_statistics": {"mean": [1.0, 0.0], "std": [1.0, 1.0]},
    }
    with pytest.raises(ValueError, match="action_statistics"):
        _validate_learned_artifacts(
            policy,
            incompatible_world,
            common,
            method="larc",
            training_seed=3072,
            goal_offset=25,
            code_revision="a" * 40,
        )
