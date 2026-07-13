import torch

from controllers.baselines import CEMPlanner
from data import ActionBlockTransform


class _QuadraticCost(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    def candidate_cost(self, observation, goal_observation, candidates):
        targets = observation.mean(dim=(1, 2, 3)) - goal_observation.mean(dim=(1, 2, 3))
        return (candidates - targets[:, None, None, None]).square().sum(dim=(2, 3))


def _make_planner(seed: int) -> CEMPlanner:
    return CEMPlanner(
        model=_QuadraticCost(),
        action_transform=ActionBlockTransform(torch.zeros(2), torch.ones(2), frameskip=2),
        horizon=3,
        receding_horizon=1,
        batch_size=2,
        num_samples=8,
        var_scale=1.0,
        n_steps=2,
        topk=3,
        device="cpu",
        seed=seed,
        warm_start=True,
    )


def test_cem_warm_start_updates_only_selected_environment_rows():
    planner = _make_planner(seed=7)
    planner.reset(pool_size=3)
    observation = torch.zeros(3, 3, 4, 4)
    goal = torch.ones_like(observation)

    first = planner(
        observation,
        goal,
        steps_remaining=25,
        batch_indices=torch.arange(3),
    )
    assert first.shape == (3, 6, 2)
    assert planner._next_init.shape == (3, 2, 4)
    untouched = planner._next_init[1].clone()

    selected = torch.tensor([0, 2])
    second = planner(
        observation[selected],
        goal[selected],
        steps_remaining=torch.tensor([20, 20]),
        batch_indices=selected,
    )

    assert second.shape == (2, 6, 2)
    torch.testing.assert_close(planner._next_init[1], untouched)


def test_cem_excluding_terminated_pool_row_preserves_live_random_stream():
    indexed = _make_planner(seed=19)
    compact = _make_planner(seed=19)
    indexed.reset(pool_size=3)
    compact.reset(pool_size=2)
    observation = torch.linspace(0.0, 1.0, 2 * 3 * 4 * 4).reshape(2, 3, 4, 4)
    goal = torch.flip(observation, dims=(0,))

    first_indexed = indexed(
        observation,
        goal,
        steps_remaining=torch.tensor([25, 25]),
        batch_indices=torch.tensor([1, 2]),
    )
    first_compact = compact(
        observation,
        goal,
        steps_remaining=torch.tensor([25, 25]),
        batch_indices=torch.tensor([0, 1]),
    )
    second_indexed = indexed(
        observation,
        goal,
        steps_remaining=torch.tensor([20, 20]),
        batch_indices=torch.tensor([1, 2]),
    )
    second_compact = compact(
        observation,
        goal,
        steps_remaining=torch.tensor([20, 20]),
        batch_indices=torch.tensor([0, 1]),
    )

    torch.testing.assert_close(first_indexed, first_compact, rtol=0.0, atol=0.0)
    torch.testing.assert_close(second_indexed, second_compact, rtol=0.0, atol=0.0)
    torch.testing.assert_close(indexed._next_init[1:], compact._next_init, rtol=0.0, atol=0.0)
