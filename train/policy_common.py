"""Shared orchestration utilities for latent-policy training."""

from __future__ import annotations

from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig, OmegaConf


class PolicyWeightsCheckpoint(Callback):
    """Save only the learned policy, never the frozen world model."""

    def __init__(
        self,
        output_dir: str | Path,
        policy_config: DictConfig,
        filename_prefix: str,
        interval: int = 1,
    ) -> None:
        super().__init__()
        if interval < 1:
            raise ValueError("checkpoint interval must be positive")
        self.output_dir = Path(output_dir)
        self.policy_config = OmegaConf.create(
            OmegaConf.to_container(policy_config, resolve=True)
        )
        self.filename_prefix = filename_prefix
        self.interval = interval

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        epoch = trainer.current_epoch + 1
        if not trainer.is_global_zero:
            return
        if epoch % self.interval and epoch != trainer.max_epochs:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            pl_module.policy.state_dict(),
            self.output_dir / f"{self.filename_prefix}_epoch_{epoch}.pt",
        )
        OmegaConf.save(
            config=self.policy_config,
            f=self.output_dir / "policy_config.yaml",
        )


def configure_adamw(
    parameters,
    optimizer_config: DictConfig,
    max_epochs: int,
):
    optimizer = torch.optim.AdamW(
        parameters,
        lr=optimizer_config.lr,
        weight_decay=optimizer_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epochs,
        eta_min=optimizer_config.lr / 100.0,
    )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
    }


def make_loaders(dataset, cfg: DictConfig):
    generator = torch.Generator().manual_seed(cfg.seed)
    train_size = int(len(dataset) * cfg.train_split)
    validation_size = len(dataset) - train_size
    if train_size < 1 or validation_size < 1:
        raise ValueError("train_split must leave at least one train and validation sample")
    train_set, validation_set = torch.utils.data.random_split(
        dataset,
        [train_size, validation_size],
        generator=generator,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        shuffle=True,
        drop_last=True,
        generator=generator,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_set,
        **cfg.loader,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, validation_loader
