from models.policies import GCIDMPolicy, LARCChunkPolicy
from models.world_model import (
    ActionPrefixEncoder,
    ParallelLatentPredictor,
    PrefixDynamics,
)


def _parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def test_configured_model_parameter_budgets():
    dynamics = PrefixDynamics(
        prefix_encoder=ActionPrefixEncoder(
            action_dim=10,
            latent_dim=192,
            token_dim=192,
            state_hidden_dim=768,
            action_hidden_dim=768,
            depth=3,
            heads=6,
            head_dim=32,
            mlp_dim=768,
            dropout=0.1,
            max_horizon=5,
        ),
        predictor=ParallelLatentPredictor(
            latent_dim=192,
            prefix_dim=192,
            hidden_dim=2048,
            fusion_dim=768,
            depth=6,
            dropout=0.1,
        ),
    )
    gc_idm = GCIDMPolicy(action_dim=2)
    larc = LARCChunkPolicy(action_dim=10, chunk_size=5)

    assert _parameter_count(dynamics) == 10_065_984
    assert _parameter_count(gc_idm) == 1_547_778
    assert _parameter_count(larc) == 1_572_402
