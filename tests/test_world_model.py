import torch

from losses.world_model.prefix_prediction import DensePrefixObjective
from models.world_model import (
    ActionPrefixEncoder,
    FrozenWorldModel,
    ParallelLatentPredictor,
    PrefixDynamics,
)


def _small_dynamics():
    return PrefixDynamics(
        prefix_encoder=ActionPrefixEncoder(
            action_dim=6,
            latent_dim=4,
            token_dim=8,
            state_hidden_dim=8,
            action_hidden_dim=8,
            depth=1,
            heads=2,
            head_dim=4,
            mlp_dim=16,
            dropout=0.0,
            max_horizon=3,
        ),
        predictor=ParallelLatentPredictor(
            latent_dim=4,
            prefix_dim=8,
            hidden_dim=16,
            fusion_dim=8,
            depth=2,
            dropout=0.0,
        ),
    )


def test_causal_prefix_does_not_see_future_actions():
    dynamics = _small_dynamics().eval()
    anchor = torch.randn(2, 4)
    actions = torch.randn(2, 3, 6)
    changed = actions.clone()
    changed[:, 2] += 100.0
    with torch.no_grad():
        prefix = dynamics.prefix_encoder(anchor, actions)
        changed_prefix = dynamics.prefix_encoder(anchor, changed)
    torch.testing.assert_close(prefix[:, :2], changed_prefix[:, :2])


def test_adaln_zero_predictor_starts_at_anchor():
    dynamics = _small_dynamics().eval()
    anchor = torch.randn(2, 4)
    actions = torch.randn(2, 3, 6)
    with torch.no_grad():
        predicted = dynamics(anchor, actions)
    torch.testing.assert_close(predicted, anchor[:, None].expand_as(predicted))


def test_dense_prefix_backward_without_sigreg():
    predictions = torch.randn(2, 3, 4, requires_grad=True)
    targets = torch.randn_like(predictions)
    objective = DensePrefixObjective(sigreg_weight=0.0)
    loss = objective(predictions, targets, encoded_sequence=None)["loss"]
    loss.backward()
    assert predictions.grad is not None
    assert torch.isfinite(predictions.grad).all()


def test_frozen_world_model_keeps_action_gradients():
    class Backbone(torch.nn.Module):
        latent_dim = 4
        max_horizon = 3

        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(6, 4)

        def predict_latents(self, anchor, actions):
            return anchor[:, None] + self.projection(actions)

        def encode_observations(self, pixels):
            return pixels

    frozen = FrozenWorldModel(Backbone())
    actions = torch.randn(2, 3, 6, requires_grad=True)
    loss = frozen.predict_latents(torch.randn(2, 4), actions).square().mean()
    loss.backward()
    assert actions.grad is not None and actions.grad.abs().sum() > 0
    assert all(not parameter.requires_grad for parameter in frozen.parameters())
