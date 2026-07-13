import hashlib

import numpy as np

from eval.tworoom_env import TwoRoomEnv


def _pixel_sha256(pixels: np.ndarray) -> str:
    return hashlib.sha256(pixels.tobytes()).hexdigest()


def test_tworoom_reset_and_rollout_match_dataset_golden():
    environment = TwoRoomEnv()
    try:
        initial = np.array([169.432373046875, 65.93359375], dtype=np.float32)
        goal = np.array([40.0, 150.0], dtype=np.float32)
        state, info = environment.reset(options={"state": initial, "goal_state": goal})
        np.testing.assert_array_equal(state, initial)
        assert (
            _pixel_sha256(info["pixels"])
            == "40b0703b33ff20b32cea6bdb267ac3785fbddb361893f8077ca86b4615340e44"
        )

        action = np.array([-0.4363047480583191, -1.0], dtype=np.float32)
        state, _, terminated, truncated, info = environment.step(action)
        expected = np.array([167.2508544921875, 60.93359375], dtype=np.float32)
        np.testing.assert_array_equal(state, expected)
        assert not terminated
        assert not truncated
        assert (
            _pixel_sha256(info["pixels"])
            == "5257da7a00e6d29f0e5423ad7af3e0a3eff48cc71033987c221aa0998e095980"
        )
    finally:
        environment.close()


def test_tworoom_wall_collision_matches_dataset_golden():
    environment = TwoRoomEnv()
    try:
        initial = np.array([107.9933090209961, 63.259727478027344], dtype=np.float32)
        goal = np.array([180.0, 180.0], dtype=np.float32)
        environment.reset(options={"state": initial, "goal_state": goal})
        state, _, _, _, info = environment.step(np.array([1.0, 1.0], dtype=np.float32))
        expected = np.array([99.5, 68.25972747802734], dtype=np.float32)
        np.testing.assert_array_equal(state, expected)
        assert (
            _pixel_sha256(info["pixels"])
            == "0e7ec92d7706bc5236ba8ce6e68cefff5024ef24cb789ee046b16800dd2564e7"
        )
    finally:
        environment.close()


def test_tworoom_goal_changes_termination_but_not_rendered_observation():
    state = np.array([80.0, 80.0], dtype=np.float32)
    first = TwoRoomEnv()
    second = TwoRoomEnv()
    try:
        _, first_info = first.reset(options={"state": state, "goal_state": [80.0, 80.0]})
        _, second_info = second.reset(options={"state": state, "goal_state": [180.0, 180.0]})
        np.testing.assert_array_equal(first_info["pixels"], second_info["pixels"])
        assert first.step([0.0, 0.0])[2]
        assert not second.step([0.0, 0.0])[2]
    finally:
        first.close()
        second.close()
