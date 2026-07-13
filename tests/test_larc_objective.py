import torch

from losses import LARCRolloutConsistencyObjective


class CumulativeWorldModel:
    def predict_latents(self, anchor_latent, actions):
        return anchor_latent[:, None] + actions.cumsum(dim=1)


def test_larc_uses_remaining_horizon_and_masks_bc():
    predicted = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            [[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]],
        ],
        requires_grad=True,
    )
    expert = predicted.detach().clone()
    expert[0, 1:] += 100.0
    mask = torch.tensor([[True, False, False], [True, True, True]])
    steps = torch.tensor([5, 15])
    predicted_latents = CumulativeWorldModel().predict_latents(torch.zeros(2, 2), predicted)
    goals = torch.stack((predicted_latents[0, 0], predicted_latents[1, 2])).detach()
    objective = LARCRolloutConsistencyObjective(frameskip=5)
    losses = objective(
        predicted_actions=predicted,
        expert_actions=expert,
        current_latent=torch.zeros(2, 2),
        goal_latent=goals,
        world_model=CumulativeWorldModel(),
        steps_remaining=steps,
        action_mask=mask,
    )
    torch.testing.assert_close(losses["behavior_loss"], torch.zeros(()))
    torch.testing.assert_close(losses["rollout_consistency_loss"], torch.zeros(()))
    losses["loss"].backward()
    assert predicted.grad is not None
