import hashlib
import json
from types import SimpleNamespace

import h5py
import hdf5plugin
import numpy as np
import pytest
import torch

from controllers.base import ActionCommand
from data.evaluation import TrajectoryEvaluationDataset
from eval.closed_loop import (
    FRAME_LINEAGE_FIELDS,
    HORIZON_VIEW_FIELDS,
    LearnedPushTPolicy,
    _validate_cem_source_artifacts,
    _validate_learned_artifacts,
)
from eval.protocol import create_or_load_manifest
from eval.pusht_env import PushTEnv
from utils import file_sha256


def test_compressed_evaluation_dataset_reads_arbitrary_episode_order(tmp_path):
    path = tmp_path / "pusht.h5"
    lengths = np.array([3, 4], dtype=np.int32)
    offsets = np.array([0, 3], dtype=np.int64)
    pixels = np.arange(7 * 4 * 4 * 3, dtype=np.uint8).reshape(7, 4, 4, 3)
    states = np.arange(7 * 7, dtype=np.float32).reshape(7, 7)
    actions = np.arange(7 * 2, dtype=np.float32).reshape(7, 2)
    with h5py.File(path, "w") as dataset:
        dataset.create_dataset("ep_len", data=lengths)
        dataset.create_dataset("ep_offset", data=offsets)
        dataset.create_dataset(
            "pixels",
            data=pixels,
            **hdf5plugin.Blosc(cname="lz4", clevel=1, shuffle=hdf5plugin.Blosc.SHUFFLE),
        )
        dataset.create_dataset("state", data=states)
        dataset.create_dataset("action", data=actions)

    with TrajectoryEvaluationDataset(
        path,
        state_key="state",
        state_dim=7,
        action_dim=2,
    ) as dataset:
        rows = dataset.evaluation_rows([1, 0], [0, 0], goal_offset=2)
        np.testing.assert_array_equal(rows["pixels"], pixels[[3, 0]])
        np.testing.assert_array_equal(rows["goal_pixels"], pixels[[5, 2]])
        np.testing.assert_array_equal(rows["state"], states[[3, 0]])
        np.testing.assert_array_equal(rows["goal_state"], states[[5, 2]])
        np.testing.assert_array_equal(dataset.action_array(), actions)


def test_environment_goal_does_not_change_initial_physics_state():
    initial = np.array([256, 400, 400, 100, 0, 0, 0], dtype=np.float64)
    first_goal = np.array([250, 390, 390, 110, 0.2, 0, 0], dtype=np.float64)
    colliding_goal = np.array([400, 100, 400, 100, 2.5, 0, 0], dtype=np.float64)
    first = PushTEnv(resolution=64)
    second = PushTEnv(resolution=64)
    try:
        first_state, first_info = first.reset(options={"state": initial, "goal_state": first_goal})
        second_state, second_info = second.reset(
            options={"state": initial, "goal_state": colliding_goal}
        )
        np.testing.assert_array_equal(first_state, second_state)
        np.testing.assert_array_equal(first_info["pixels"], second_info["pixels"])

        action = np.array([0.25, -0.5], dtype=np.float32)
        first_next = first.step(action)
        second_next = second.step(action)
        np.testing.assert_array_equal(first_next[0], second_next[0])
        np.testing.assert_array_equal(first_next[4]["pixels"], second_next[4]["pixels"])
    finally:
        first.close()
        second.close()


def test_environment_reset_keeps_reference_physics_flush():
    initial = np.array([100.0, 100.0, 400.0, 400.0, 0.7, 30.0, -40.0])
    goal = initial.copy()
    expected = initial.copy()
    expected[:2] += initial[-2:] * 0.01
    environment = PushTEnv(resolution=64)
    try:
        restored, _ = environment.reset(options={"state": initial, "goal_state": goal})
        np.testing.assert_allclose(restored, expected, rtol=0.0, atol=1e-12)
    finally:
        environment.close()


def test_environment_nonzero_angle_rollout_matches_golden_reference():
    initial = np.array([225.0, 365.0, 335.0, 175.0, 0.73, 0.0, 0.0], dtype=np.float64)
    goal = np.array([255.0, 330.0, 300.0, 220.0, 1.17, 0.0, 0.0], dtype=np.float64)
    actions = (
        np.array([0.31, -0.47], dtype=np.float32),
        np.array([-0.22, 0.18], dtype=np.float32),
    )
    expected = (
        (
            initial,
            "57ffd482374b4458d24ec548e19c43755df6690b44b128908131245e585ac002",
        ),
        (
            np.array(
                [
                    234.1658255386353,
                    351.10342697143545,
                    335.0,
                    175.0,
                    0.73,
                    115.98577880859375,
                    -175.84938049316406,
                ],
                dtype=np.float64,
            ),
            "877c4504f87fa0e56e1394be3cca51d584b856c86d052d60ffc4501805204b38",
        ),
        (
            np.array(
                [
                    231.13271194458014,
                    351.1620288085937,
                    335.0,
                    175.0,
                    0.73,
                    -83.07244873046875,
                    68.49878692626953,
                ],
                dtype=np.float64,
            ),
            "9b17dad6e06c899b3bf504ab4b776ff164cb67afac5a5aafdc1c0801688c4389",
        ),
    )
    environment = PushTEnv(resolution=64)
    try:
        state, info = environment.reset(options={"state": initial, "goal_state": goal})
        np.testing.assert_array_equal(state, expected[0][0])
        assert hashlib.sha256(info["pixels"].tobytes()).hexdigest() == expected[0][1]

        for action, (expected_state, expected_pixels) in zip(actions, expected[1:]):
            state, _, _, _, info = environment.step(action)
            np.testing.assert_array_equal(state, expected_state)
            assert hashlib.sha256(info["pixels"].tobytes()).hexdigest() == expected_pixels
    finally:
        environment.close()


class _RecordingController(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_indices = []

    def reset(self, goal_observation: torch.Tensor) -> None:
        self.goal_batch = goal_observation.size(0)

    def act(
        self,
        observation: torch.Tensor,
        steps_remaining: torch.Tensor,
        *,
        batch_indices: torch.Tensor | None = None,
    ) -> ActionCommand:
        del steps_remaining
        self.batch_indices.append(batch_indices.detach().cpu().tolist())
        return ActionCommand(
            actions=torch.zeros(observation.size(0), 1, 2, device=observation.device),
            replan_after=1,
        )


def test_policy_never_plans_terminated_environment_rows():
    controller = _RecordingController()
    policy = LearnedPushTPolicy(
        controller=controller,
        goal_offset=25,
        minimum_horizon=1,
        device="cpu",
    )
    space = SimpleNamespace(
        low=np.full(2, -1.0, dtype=np.float32),
        high=np.full(2, 1.0, dtype=np.float32),
        shape=(2,),
    )
    policy.set_env(SimpleNamespace(num_envs=3, single_action_space=space))
    pixels = np.zeros((3, 4, 4, 3), dtype=np.uint8)
    actions = policy.get_action(
        {
            "pixels": pixels[:, None],
            "goal": pixels[:, None],
            "terminated": np.array([False, True, False]),
            "_needs_flush": np.ones(3, dtype=bool),
        }
    )

    assert controller.goal_batch == 3
    assert controller.batch_indices == [[0, 2]]
    np.testing.assert_array_equal(actions[[0, 2]], np.zeros((2, 2), dtype=np.float32))
    assert np.isnan(actions[1]).all()
    timing = policy.timing_metrics()
    assert timing["episode_planner_calls"] == [1, 0, 1]
    assert timing["planning_batch_calls"] == 1
    assert timing["planning_row_calls"] == 2
    assert timing["planning_elapsed_seconds"] >= 0.0
    assert timing["goal_encoding_seconds"] >= 0.0


def test_learned_policy_lineage_is_checked_against_evaluation_cache():
    common = {field: 0 for field in FRAME_LINEAGE_FIELDS + HORIZON_VIEW_FIELDS}
    common.update(
        {
            "max_goal_offset": 25,
            "training_seed": 3072,
            "training_code_revision": "a" * 40,
        }
    )
    policy = {**common, "method": "larc"}
    foreign_cache = {**common, "dataset_path": "different-dataset"}

    with pytest.raises(ValueError, match="latent cache.*dataset_path"):
        _validate_learned_artifacts(
            policy,
            common,
            foreign_cache,
            method="larc",
            training_seed=3072,
            goal_offset=25,
            code_revision="a" * 40,
        )


def test_cem_files_must_match_latent_cache_source_hashes(tmp_path):
    config_path = tmp_path / "released.yaml"
    weights_path = tmp_path / "released.pt"
    config_path.write_text("model config")
    weights_path.write_bytes(b"weights")
    cfg = SimpleNamespace(
        cem=SimpleNamespace(
            config_path=str(config_path),
            weights_path=str(weights_path),
        )
    )
    metadata = {
        "source_model_config_sha256": file_sha256(config_path),
        "source_weights_sha256": file_sha256(weights_path),
    }
    _validate_cem_source_artifacts(metadata, cfg)

    weights_path.write_bytes(b"different weights")
    with pytest.raises(ValueError, match="CEM model weights differ"):
        _validate_cem_source_artifacts(metadata, cfg)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("episode_indices", [2, 2], "must be unique"),
        ("episode_indices", [0, 3], "non-validation"),
        ("start_steps", [15, 0], "out-of-range start"),
        ("start_steps", [True, 0], "only integers"),
    ),
)
def test_corrupt_manifest_entries_are_rejected(tmp_path, field, value, message):
    dataset_path = tmp_path / "data.h5"
    lengths = np.array([40, 40, 40, 40], dtype=np.int32)
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
        "train_fraction": 0.5,
        "validation_episode_ids": [2, 3],
    }
    (cache_dir / "metadata.json").write_text(json.dumps(metadata))
    manifest_path = tmp_path / "manifest.json"
    manifest = create_or_load_manifest(
        dataset_path=dataset_path,
        latent_cache_dir=cache_dir,
        output_path=manifest_path,
        seed=42,
        num_eval=2,
        goal_offsets=[25],
    )
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=message):
        create_or_load_manifest(
            dataset_path=dataset_path,
            latent_cache_dir=cache_dir,
            output_path=manifest_path,
            seed=42,
            num_eval=2,
            goal_offsets=[25],
        )
