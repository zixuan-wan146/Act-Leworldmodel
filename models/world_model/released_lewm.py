"""Project-owned implementation of the released autoregressive LeWM model."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn


def _modulate(inputs: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return inputs * (1.0 + scale) + shift


class _FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, dropout: float) -> None:
        super().__init__()
        inner_dim = heads * head_dim
        self.heads = heads
        self.head_dim = head_dim
        self.dropout = float(dropout)
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(inputs)
        batch, time, _ = normalized.shape
        query, key, value = self.to_qkv(normalized).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(batch, time, self.heads, self.head_dim).transpose(1, 2)

        output = functional.scaled_dot_product_attention(
            split_heads(query),
            split_heads(key),
            split_heads(value),
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        output = output.transpose(1, 2).reshape(batch, time, self.heads * self.head_dim)
        return self.to_out(output)


class _ConditionalBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        head_dim: int,
        mlp_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attn = _Attention(dim, heads, head_dim, dropout)
        self.mlp = _FeedForward(dim, mlp_dim, dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, inputs: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift_attention, scale_attention, gate_attention, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(condition).chunk(6, dim=-1)
        )
        inputs = inputs + gate_attention * self.attn(
            _modulate(self.norm1(inputs), shift_attention, scale_attention)
        )
        return inputs + gate_mlp * self.mlp(_modulate(self.norm2(inputs), shift_mlp, scale_mlp))


class _ConditionalTransformer(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
        heads: int,
        head_dim: int,
        mlp_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList(
            [_ConditionalBlock(hidden_dim, heads, head_dim, mlp_dim, dropout) for _ in range(depth)]
        )
        self.input_proj = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        )
        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        )
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        inputs = self.input_proj(inputs)
        condition = self.cond_proj(condition)
        for block in self.layers:
            inputs = block(inputs, condition)
        return self.output_proj(self.norm(inputs))


class ReleasedLeWMPredictor(nn.Module):
    """Autoregressive latent predictor reconstructed from explicit config."""

    def __init__(
        self,
        *,
        history_size: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
        heads: int,
        head_dim: int,
        mlp_dim: int,
        dropout: float,
        embedding_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.history_size = int(history_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, history_size, input_dim))
        self.dropout = nn.Dropout(embedding_dropout)
        self.transformer = _ConditionalTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            depth=depth,
            heads=heads,
            head_dim=head_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

    def forward(self, inputs: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        time = inputs.size(1)
        if time > self.pos_embedding.size(1):
            raise ValueError("predictor input exceeds configured history size")
        return self.transformer(self.dropout(inputs + self.pos_embedding[:, :time]), condition)


class ReleasedLeWMActionEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        smoothed_dim: int,
        embedding_dim: int,
        mlp_scale: int = 4,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * embedding_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * embedding_dim, embedding_dim),
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.size(-1) != self.input_dim:
            raise ValueError(f"actions must end in dimension {self.input_dim}")
        embedded = self.patch_embed(actions.float().transpose(1, 2)).transpose(1, 2)
        return self.embed(embedded)


class ReleasedLeWM(nn.Module):
    """Released LeWM architecture with project-owned inference behavior."""

    def __init__(
        self,
        *,
        encoder: nn.Module,
        predictor: ReleasedLeWMPredictor,
        action_encoder: ReleasedLeWMActionEncoder,
        projector: nn.Module,
        pred_proj: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector
        self.pred_proj = pred_proj

    def encode_observations(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.ndim not in (4, 5):
            raise ValueError("pixels must have shape [B,C,H,W] or [B,T,C,H,W]")
        has_time = pixels.ndim == 5
        if has_time:
            batch, time = pixels.shape[:2]
            flat = pixels.reshape(batch * time, *pixels.shape[2:])
        else:
            batch, time = pixels.size(0), 1
            flat = pixels
        output = self.encoder(flat.float(), interpolate_pos_encoding=True)
        latent = self.projector(output.last_hidden_state[:, 0])
        latent = latent.reshape(batch, time, -1)
        return latent if has_time else latent[:, 0]

    def predict_next(
        self, latent_history: torch.Tensor, action_history: torch.Tensor
    ) -> torch.Tensor:
        action_latent = self.action_encoder(action_history)
        predicted = self.predictor(latent_history, action_latent)
        batch, time, width = predicted.shape
        return self.pred_proj(predicted.reshape(batch * time, width)).reshape(batch, time, width)

    def candidate_cost(
        self,
        current_pixels: torch.Tensor,
        goal_pixels: torch.Tensor,
        action_candidates: torch.Tensor,
    ) -> torch.Tensor:
        """Return terminal latent MSE sums for candidates shaped [B,S,H,A]."""

        if action_candidates.ndim != 4:
            raise ValueError("action candidates must have shape [B,S,H,A]")
        batch, samples, horizon, action_dim = action_candidates.shape
        if horizon < 1 or action_dim != self.action_encoder.input_dim:
            raise ValueError("action candidates have an incompatible horizon or action dimension")
        current = self.encode_observations(current_pixels)
        goal = self.encode_observations(goal_pixels)
        if current.ndim == 2:
            current = current[:, None]
        if goal.ndim == 3:
            goal = goal[:, -1]
        if current.size(0) != batch or goal.size(0) != batch:
            raise ValueError("pixel and candidate batches differ")

        latent_history = (
            current[:, None]
            .expand(batch, samples, *current.shape[1:])
            .reshape(batch * samples, current.size(1), current.size(2))
        )
        actions = action_candidates.reshape(batch * samples, horizon, action_dim)
        for step in range(horizon):
            history_start = max(0, step + 1 - self.predictor.history_size)
            latent_window = latent_history[:, -self.predictor.history_size :]
            action_window = actions[:, history_start : step + 1]
            next_latent = self.predict_next(latent_window, action_window)[:, -1:]
            latent_history = torch.cat((latent_history, next_latent), dim=1)
        terminal = latent_history[:, -1].reshape(batch, samples, -1)
        return (terminal - goal[:, None]).square().sum(dim=-1)
