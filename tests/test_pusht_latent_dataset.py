import hashlib
import json

import h5py
import numpy as np
import pytest
import torch

from data.pusht_latent import (
    PushTLatentDynamicsDataset,
    PushTLatentPolicyDataset,
    build_frame_latent_cache,
    split_episode_ids,
)


def _make_cache(tmp_path):
    dataset_path = tmp_path / "pusht.h5"
    lengths = np.array([31, 31], dtype=np.int32)
    offsets = np.array([0, 31], dtype=np.int64)
    actions = np.arange(62 * 2, dtype=np.float32).reshape(62, 2)
    with h5py.File(dataset_path, "w") as dataset:
        dataset.create_dataset("ep_len", data=lengths)
        dataset.create_dataset("ep_offset", data=offsets)
        dataset.create_dataset("action", data=actions)
    latents = np.arange(62 * 4, dtype=np.float16).reshape(62, 4)
    np.save(tmp_path / "frame_latents.npy", latents)
    metadata = {
        "version": 1,
        "dataset_path": str(dataset_path),
        "dataset_size": dataset_path.stat().st_size,
        "dataset_mtime_ns": dataset_path.stat().st_mtime_ns,
        "source_checkpoint": "unused",
        "source_checkpoint_sha256": "unused",
        "seed": 3,
        "train_fraction": 0.5,
        "train_episode_ids": [0],
        "validation_episode_ids": [1],
        "episode_lengths": lengths.tolist(),
        "episode_offsets": offsets.tolist(),
        "frame_count": 62,
        "latent_dim": 4,
        "latent_dtype": "float16",
        "raw_action_dim": 2,
        "frameskip": 5,
        "max_horizon": 5,
        "max_goal_offset": 25,
        "action_statistics": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    return latents, actions


def test_episode_split_is_deterministic_and_disjoint():
    train_a, validation_a = split_episode_ids(100, 0.8, 42)
    train_b, validation_b = split_episode_ids(100, 0.8, 42)
    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(validation_a, validation_b)
    assert not set(train_a).intersection(validation_a)
    assert len(train_a) == 80


def test_dense_dynamics_indices_and_action_blocks(tmp_path):
    latents, actions = _make_cache(tmp_path)
    dataset = PushTLatentDynamicsDataset(tmp_path, "train", frameskip=5, max_horizon=5)
    assert len(dataset) == 6
    sample = dataset[0]
    torch.testing.assert_close(sample["anchor_latent"], torch.from_numpy(latents[0]).float())
    expected_targets = torch.from_numpy(latents[[5, 10, 15, 20, 25]]).float()
    torch.testing.assert_close(sample["target_latents"], expected_targets)
    torch.testing.assert_close(
        sample["action_blocks"], torch.from_numpy(actions[:25]).reshape(5, 10)
    )
    batched = dataset.__getitems__([0, 1])
    for key in sample:
        torch.testing.assert_close(batched[0][key], sample[key])


def test_horizon_view_is_not_fixed_by_legacy_cache_metadata(tmp_path):
    _, actions = _make_cache(tmp_path)
    dataset = PushTLatentDynamicsDataset(tmp_path, "train", frameskip=5, max_horizon=6)
    assert len(dataset) == 1
    torch.testing.assert_close(
        dataset[0]["action_blocks"], torch.from_numpy(actions[:30]).reshape(6, 10)
    )


def test_policy_pairs_stay_inside_split_and_obey_block_protocol(tmp_path):
    latents, _ = _make_cache(tmp_path)
    larc = PushTLatentPolicyDataset(
        tmp_path,
        "validation",
        method="larc",
        frameskip=5,
        max_horizon=5,
        sample_seed=11,
    )
    sample = larc[0]
    assert sample["action_chunk"].shape == (5, 10)
    assert sample["action_mask"].shape == (5,)
    assert sample["steps_remaining"].item() in (5, 10, 15, 20, 25)
    assert sample["action_mask"].sum().item() == sample["steps_remaining"].item() // 5
    torch.testing.assert_close(sample["current_latent"], torch.from_numpy(latents[31]).float())
    larc_batched = larc.__getitems__([0, 1])
    for key in sample:
        torch.testing.assert_close(larc_batched[0][key], sample[key])

    gc_idm = PushTLatentPolicyDataset(
        tmp_path,
        "train",
        method="gc_idm",
        frameskip=5,
        max_horizon=5,
        sample_seed=11,
    )
    gc_sample = gc_idm[0]
    assert gc_sample["action"].shape == (2,)
    assert 1 <= gc_sample["steps_remaining"].item() <= 25
    gc_batched = gc_idm.__getitems__([0, 1])
    for key in gc_sample:
        torch.testing.assert_close(gc_batched[0][key], gc_sample[key])


def _make_source_artifact(tmp_path):
    _make_cache(tmp_path)
    source_hash = hashlib.sha256(b"checkpoint").hexdigest()
    config_path = tmp_path / "source.yaml"
    config_path.write_text("_target_: unused\n")
    weights_path = tmp_path / "source.pt"
    torch.save(
        {
            "format_version": 1,
            "model_kind": "released_lewm",
            "metadata": {"source_checkpoint_sha256": source_hash},
            "state_dict": {"weight": torch.ones(1)},
        },
        weights_path,
    )
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["source_checkpoint_sha256"] = source_hash
    metadata_path.write_text(json.dumps(metadata))
    return config_path, weights_path


def test_existing_cache_rejects_incompatible_protocol(tmp_path):
    config_path, weights_path = _make_source_artifact(tmp_path)

    with pytest.raises(ValueError, match="incompatible seed"):
        build_frame_latent_cache(
            dataset_path=tmp_path / "pusht.h5",
            source_config=config_path,
            source_weights=weights_path,
            output_dir=tmp_path,
            seed=4,
            train_fraction=0.5,
            batch_size=1,
            device="cpu",
        )


def test_existing_cache_migrates_metadata_without_reencoding(tmp_path, monkeypatch):
    config_path, weights_path = _make_source_artifact(tmp_path)
    latent_path = tmp_path / "frame_latents.npy"
    latent_sha256 = hashlib.sha256(latent_path.read_bytes()).hexdigest()

    def reject_encoder_load(*args, **kwargs):
        raise AssertionError("cache reuse must not construct the encoder")

    monkeypatch.setattr("data.pusht_latent.load_released_lewm", reject_encoder_load)
    metadata = build_frame_latent_cache(
        dataset_path=tmp_path / "pusht.h5",
        source_config=config_path,
        source_weights=weights_path,
        output_dir=tmp_path,
        seed=3,
        train_fraction=0.5,
        batch_size=1,
        device="cpu",
    )

    assert metadata["version"] == 2
    assert metadata["legacy_default_view"]["max_goal_offset"] == 25
    assert "max_horizon" not in metadata
    assert json.loads((tmp_path / "metadata.json").read_text()) == metadata
    assert hashlib.sha256(latent_path.read_bytes()).hexdigest() == latent_sha256
