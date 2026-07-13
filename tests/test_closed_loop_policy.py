from types import SimpleNamespace

import numpy as np
import torch

from controllers.learned import LARCController
from data import ActionBlockTransform
from eval.closed_loop import LearnedPushTPolicy


class _Encoder(torch.nn.Module):
    latent_dim = 4

    def encode_observations(self, pixels):
        return torch.zeros(pixels.size(0), self.latent_dim, device=pixels.device)


class _ChunkPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, current, goal, steps_remaining):
        self.calls += 1
        values = torch.arange(50, device=current.device, dtype=current.dtype)
        return values.reshape(1, 5, 10).expand(current.size(0), -1, -1)


def test_larc_adapter_commits_decoded_raw_actions_before_replanning():
    policy = _ChunkPolicy()
    controller = LARCController(
        _Encoder(),
        policy,
        commit_steps=5,
        action_transform=ActionBlockTransform(torch.zeros(2), torch.ones(2), frameskip=5),
    )
    adapter = LearnedPushTPolicy(
        controller=controller,
        goal_offset=25,
        minimum_horizon=5,
        device="cpu",
    )
    space = SimpleNamespace(
        low=np.full(2, -100.0, dtype=np.float32),
        high=np.full(2, 100.0, dtype=np.float32),
        shape=(2,),
    )
    adapter.set_env(
        SimpleNamespace(
            num_envs=1,
            single_action_space=space,
            action_space=SimpleNamespace(shape=(1, 2)),
        )
    )
    image = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    info = {
        "goal": image,
        "pixels": image,
        "terminated": np.array([False]),
        "_needs_flush": np.array([True]),
    }
    actions = []
    for step in range(6):
        info["_needs_flush"] = np.array([step == 0])
        actions.append(adapter.get_action(info).copy())

    expected_first_commit = np.arange(10, dtype=np.float32).reshape(5, 1, 2)
    np.testing.assert_array_equal(np.stack(actions[:5]), expected_first_commit)
    assert policy.calls == 2
