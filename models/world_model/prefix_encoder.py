"""Causal action-prefix encoder for Fast-LeWM dynamics."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionEncoding(nn.Module):
    """Fixed sine/cosine features at exponentially spaced frequencies."""

    def __init__(self, dim: int, max_length: int) -> None:
        super().__init__()
        if dim < 2:
            raise ValueError("position encoding dimension must be at least 2")
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10_000.0) / dim)
        )
        encoding = torch.zeros(max_length, dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        odd_width = encoding[:, 1::2].size(1)
        encoding[:, 1::2] = torch.cos(positions * frequencies[:odd_width])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.size(1) > self.encoding.size(1):
            raise ValueError(
                f"sequence length {tokens.size(1)} exceeds configured maximum "
                f"{self.encoding.size(1)}"
            )
        positions = self.encoding[:, : tokens.size(1)].to(
            device=tokens.device, dtype=tokens.dtype
        )
        return tokens + positions


class CausalTransformerBlock(nn.Module):
    """Pre-norm causal self-attention followed by a residual MLP."""

    def __init__(
        self,
        dim: int,
        heads: int,
        head_dim: int,
        mlp_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if heads * head_dim != dim:
            raise ValueError("heads * head_dim must equal token_dim")
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.size(1)
        causal_mask = torch.triu(
            torch.ones(length, length, device=tokens.device, dtype=torch.bool), diagonal=1
        )
        normalized = self.attention_norm(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            need_weights=False,
        )
        tokens = tokens + attended
        return tokens + self.mlp(self.mlp_norm(tokens))


class ActionPrefixEncoder(nn.Module):
    """Map each action prefix to a horizon-specific, state-conditioned token."""

    def __init__(
        self,
        *,
        action_dim: int,
        latent_dim: int = 192,
        token_dim: int = 192,
        state_hidden_dim: int = 768,
        action_hidden_dim: int = 768,
        depth: int = 3,
        heads: int = 6,
        head_dim: int = 32,
        mlp_dim: int = 768,
        dropout: float = 0.1,
        max_horizon: int = 5,
    ) -> None:
        super().__init__()
        if action_dim < 1:
            raise ValueError("action_dim must be positive")
        if max_horizon < 1:
            raise ValueError("max_horizon must be positive")
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.token_dim = token_dim
        self.max_horizon = max_horizon
        self.state_tokenizer = nn.Sequential(
            nn.Linear(latent_dim, state_hidden_dim),
            nn.SiLU(),
            nn.Linear(state_hidden_dim, token_dim),
        )
        self.action_tokenizer = nn.Sequential(
            nn.Linear(action_dim, action_hidden_dim),
            nn.SiLU(),
            nn.Linear(action_hidden_dim, token_dim),
        )
        self.position_encoding = SinusoidalPositionEncoding(
            token_dim, max_length=max_horizon + 1
        )
        self.blocks = nn.ModuleList(
            CausalTransformerBlock(token_dim, heads, head_dim, mlp_dim, dropout)
            for _ in range(depth)
        )
        self.output_norm = nn.LayerNorm(token_dim)

    def forward(self, anchor_latent: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if anchor_latent.ndim != 2:
            raise ValueError("anchor_latent must have shape [batch, latent_dim]")
        if actions.ndim != 3:
            raise ValueError("actions must have shape [batch, horizon, action_dim]")
        if anchor_latent.size(0) != actions.size(0):
            raise ValueError("anchor_latent and actions must share their batch dimension")
        if anchor_latent.size(-1) != self.latent_dim:
            raise ValueError(f"expected latent dimension {self.latent_dim}")
        if actions.size(-1) != self.action_dim:
            raise ValueError(f"expected action dimension {self.action_dim}")
        if not 1 <= actions.size(1) <= self.max_horizon:
            raise ValueError(f"action horizon must be in [1, {self.max_horizon}]")
        state_token = self.state_tokenizer(anchor_latent).unsqueeze(1)
        actions = actions.to(device=state_token.device, dtype=state_token.dtype)
        action_tokens = self.action_tokenizer(actions)
        tokens = self.position_encoding(torch.cat((state_token, action_tokens), dim=1))
        for block in self.blocks:
            tokens = block(tokens)
        return self.output_norm(tokens)[:, 1:]
