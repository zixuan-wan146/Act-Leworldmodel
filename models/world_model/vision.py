"""Project-owned visual encoder construction."""

from __future__ import annotations

from transformers import ViTConfig, ViTModel


def build_vit_encoder(
    *,
    image_size: int = 224,
    patch_size: int = 14,
    hidden_size: int = 192,
    depth: int = 12,
    heads: int = 3,
    mlp_dim: int = 768,
) -> ViTModel:
    """Build the ViT-Tiny architecture used by the released representation.

    The architecture is declared by this project and reconstructed from config.
    No upstream training package is imported at runtime.
    """

    config = ViTConfig(
        image_size=image_size,
        patch_size=patch_size,
        num_channels=3,
        hidden_size=hidden_size,
        num_hidden_layers=depth,
        num_attention_heads=heads,
        intermediate_size=mlp_dim,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        qkv_bias=True,
        layer_norm_eps=1e-12,
        encoder_stride=patch_size,
    )
    return ViTModel(config, add_pooling_layer=False, use_mask_token=False)
