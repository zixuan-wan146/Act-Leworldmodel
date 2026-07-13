import torch
import numpy as np

from data import calculate_action_statistics
from data.action_transform import ActionBlockTransform, ZScoreActionTransform


def test_action_block_round_trip_and_shape():
    transform = ActionBlockTransform(
        mean=torch.tensor([1.0, -2.0]),
        std=torch.tensor([2.0, 4.0]),
        frameskip=5,
    )
    raw = torch.randn(3, 25, 2)
    blocks = transform.encode(raw)
    assert blocks.shape == (3, 5, 10)
    torch.testing.assert_close(transform.decode(blocks), raw)


def test_zscore_raw_action_round_trip():
    transform = ZScoreActionTransform(mean=torch.tensor([0.5, -0.5]), std=torch.tensor([0.25, 2.0]))
    actions = torch.randn(7, 2)
    torch.testing.assert_close(transform.decode(transform.encode(actions)), actions)


def test_action_block_rejects_non_divisible_sequence():
    transform = ActionBlockTransform(torch.zeros(2), torch.ones(2), frameskip=5)
    try:
        transform.encode(torch.zeros(2, 6, 2))
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("expected a non-divisible action sequence to fail")


def test_action_statistics_respect_scope_and_ignore_nonfinite_rows():
    actions = np.array([[1.0, 2.0], [3.0, 6.0], [np.nan, 9.0]], dtype=np.float32)
    full = calculate_action_statistics(actions)
    selected = calculate_action_statistics(actions, np.array([True, False, True]))
    assert full.mean == (2.0, 4.0)
    assert full.std == (1.0, 2.0)
    assert selected.mean == (1.0, 2.0)
    assert selected.std == (1e-6, 1e-6)
