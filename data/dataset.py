"""LeWM-compatible offline trajectory loading and preprocessing."""

from __future__ import annotations

import os

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from stable_pretraining import data as data_api


def get_image_preprocessor(source: str, target: str, image_size: int = 224):
    imagenet_stats = data_api.dataset_stats.ImageNet
    to_image = data_api.transforms.ToImage(
        **imagenet_stats, source=source, target=target
    )
    resize = data_api.transforms.Resize(image_size, source=source, target=target)
    return data_api.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    """Picklable normalizer that is safe with worker-spawned data loaders."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean = mean
        self.std = std.clamp_min(1e-6)

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        return ((values - self.mean) / self.std).float()


def get_column_normalizer(dataset, source: str, target: str):
    values = torch.from_numpy(np.asarray(dataset.get_col_data(source))).float()
    if values.size(0) == 0:
        raise ValueError(f"column {source!r} is empty")
    finite_rows = torch.isfinite(values.reshape(values.size(0), -1)).all(dim=1)
    values = values[finite_rows]
    if values.numel() == 0:
        raise ValueError(f"column {source!r} has no finite rows")
    normalizer = ZScoreNormalizer(
        mean=values.mean(dim=0, keepdim=True).clone(),
        std=values.std(dim=0, keepdim=True, unbiased=False).clone(),
    )
    return data_api.transforms.WrapTorchTransform(
        normalizer, source=source, target=target
    )


def load_training_dataset(cfg: DictConfig):
    """Load a configured trajectory dataset and resolve the action-block width."""

    dataset_config = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_config.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR")
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_config
    )
    transforms = [
        get_image_preprocessor("pixels", "pixels", image_size=cfg.image_size)
    ]
    with open_dict(cfg):
        for column in cfg.data.dataset.keys_to_load:
            if not column.startswith("pixels"):
                transforms.append(get_column_normalizer(dataset, column, column))
        cfg.model.dynamics.prefix_encoder.action_dim = (
            cfg.data.dataset.frameskip * dataset.get_dim("action")
        )
    dataset.transform = spt.data.transforms.Compose(*transforms)
    return dataset
