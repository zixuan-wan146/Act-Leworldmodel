"""Shared deterministic execution helpers for training and evaluation."""

from __future__ import annotations

import os
import random

import lightning as pl
import numpy as np
import torch


def configure_reproducibility(seed: int) -> None:
    """Seed every stochastic source used by this project."""

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True, verbose=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        import stable_pretraining as spt

        spt.set(
            progress_bar="none",
            default_callbacks={
                "progress_bar": False,
                "registry": False,
                "logging": False,
                "env_dump": False,
                "trainer_info": False,
                "sklearn_checkpoint": False,
                "wandb_checkpoint": False,
                "trackio_checkpoint": False,
                "swanlab_checkpoint": False,
                "module_summary": False,
                "slurm_info": False,
                "unused_params": False,
                "hf_checkpoint": False,
            },
        )
    except ImportError:
        pass


def make_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)
