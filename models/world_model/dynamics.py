"""Pure latent action-prefix dynamics."""

from __future__ import annotations

import torch
from torch import nn

from models.world_model.parallel_predictor import ParallelLatentPredictor
from models.world_model.prefix_encoder import ActionPrefixEncoder


class PrefixDynamics(nn.Module):
    """Map `(z_t, a_t:t+H)` to one predicted latent per action prefix."""

    def __init__(
        self,
        *,
        prefix_encoder: ActionPrefixEncoder,
        predictor: ParallelLatentPredictor,
    ) -> None:
        super().__init__()
        if prefix_encoder.latent_dim != predictor.latent_dim:
            raise ValueError("prefix encoder and predictor latent dimensions differ")
        if prefix_encoder.token_dim != predictor.prefix_dim:
            raise ValueError("prefix encoder output and predictor condition dimensions differ")
        self.prefix_encoder = prefix_encoder
        self.predictor = predictor

    @property
    def latent_dim(self) -> int:
        return self.predictor.latent_dim

    @property
    def max_horizon(self) -> int:
        return self.prefix_encoder.max_horizon

    @property
    def action_dim(self) -> int:
        return self.prefix_encoder.action_dim

    def predict_prefix(self, anchor_latent: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Predict every action-prefix endpoint while retaining action gradients."""

        return self.forward(anchor_latent, actions)

    def forward(self, anchor_latent: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if anchor_latent.ndim < 2 or actions.ndim < 3:
            raise ValueError("anchor_latent and actions need at least one batch dimension")
        if anchor_latent.shape[:-1] != actions.shape[:-2]:
            raise ValueError("anchor_latent and actions have incompatible leading dimensions")
        batch_shape = anchor_latent.shape[:-1]
        flat_anchor = anchor_latent.reshape(-1, anchor_latent.size(-1))
        flat_actions = actions.reshape(-1, actions.size(-2), actions.size(-1))
        prefix_tokens = self.prefix_encoder(flat_anchor, flat_actions)
        predictions = self.predictor(flat_anchor, prefix_tokens)
        return predictions.reshape(*batch_shape, actions.size(-2), self.latent_dim)
