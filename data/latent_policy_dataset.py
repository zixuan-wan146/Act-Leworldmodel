"""Portable tensor-cache datasets for amortized policy training."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch
from torch.utils.data import Dataset


class LatentPolicyDataset(Dataset):
    """Read aligned frozen latents and actions from a tensor dictionary."""

    def __init__(self, path: str | Path, required_keys: Iterable[str]) -> None:
        super().__init__()
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise TypeError("latent cache must contain a dictionary of tensors")
        keys = tuple(required_keys)
        missing = [key for key in keys if key not in payload]
        if missing:
            raise KeyError(f"latent cache is missing keys: {missing}")
        tensors = {key: payload[key] for key in keys}
        if any(not torch.is_tensor(value) for value in tensors.values()):
            raise TypeError("all latent-cache values must be tensors")
        lengths = {value.size(0) for value in tensors.values()}
        if len(lengths) != 1:
            raise ValueError("all latent-cache tensors must have the same first dimension")
        self.tensors = tensors
        self.length = lengths.pop()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.tensors.items()}
