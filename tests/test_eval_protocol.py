import json

import h5py
import numpy as np
import pytest

from eval.closed_loop import LINEAGE_FIELDS, _validate_learned_artifacts
from eval.protocol import create_or_load_manifest


def test_manifest_uses_only_validation_episodes_and_is_reused(tmp_path):
    dataset_path = tmp_path / "data.h5"
    lengths = np.array([40, 41, 42, 43, 44, 45], dtype=np.int32)
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
        goal_offset=25,
    )
    second = create_or_load_manifest(
        dataset_path=dataset_path,
        latent_cache_dir=cache_dir,
        output_path=manifest_path,
        seed=42,
        num_eval=3,
        goal_offset=25,
    )
    assert first == second
    assert set(first["episode_indices"]).issubset({2, 3, 4, 5})
    assert len(set(first["episode_indices"])) == 3
    for episode, start in zip(first["episode_indices"], first["start_steps"]):
        assert 0 <= start < lengths[episode] - 25

    with pytest.raises(ValueError, match="incompatible num_eval"):
        create_or_load_manifest(
            dataset_path=dataset_path,
            latent_cache_dir=cache_dir,
            output_path=manifest_path,
            seed=42,
            num_eval=2,
            goal_offset=25,
        )


def test_learned_artifact_validation_rejects_mixed_action_protocols():
    common = {field: 0 for field in LINEAGE_FIELDS}
    common.update(
        {
            "source_checkpoint_sha256": "encoder",
            "seed": 3072,
            "train_fraction": 0.9,
            "frameskip": 5,
            "max_horizon": 5,
            "max_goal_offset": 25,
            "latent_dim": 192,
            "action_statistics": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
            "training_seed": 3072,
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
        )
