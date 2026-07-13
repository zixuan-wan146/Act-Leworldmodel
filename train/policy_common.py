"""Shared orchestration utilities for latent-policy training."""

from __future__ import annotations

import json
import os
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
        metadata: dict,
        interval: int = 1,
    ) -> None:
        super().__init__()
        if interval < 1:
            raise ValueError("checkpoint interval must be positive")
        self.output_dir = Path(output_dir)
        self.policy_config = OmegaConf.create(OmegaConf.to_container(policy_config, resolve=True))
        self.filename_prefix = filename_prefix
        self.metadata = metadata
        self.interval = interval
        self.best_loss = float("inf")

    def _save(self, trainer, pl_module, suffix: str) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        destination = self.output_dir / f"{self.filename_prefix}_{suffix}.pt"
        temporary = destination.with_suffix(".pt.tmp")
        torch.save(pl_module.policy.state_dict(), temporary)
        os.replace(temporary, destination)
        OmegaConf.save(
            config=self.policy_config,
            f=self.output_dir / "policy_config.yaml",
        )
        metadata_path = self.output_dir / "policy_metadata.json"
        temporary_metadata = metadata_path.with_suffix(".json.tmp")
        temporary_metadata.write_text(json.dumps(self.metadata, indent=2, sort_keys=True) + "\n")
        os.replace(temporary_metadata, metadata_path)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        metric = trainer.callback_metrics.get("validation/loss")
        if metric is None or not trainer.is_global_zero:
            return
        value = float(metric.detach().cpu())
        improved = value < self.best_loss
        print(
            f"policy epoch={trainer.current_epoch + 1} "
            f"validation/loss={value:.8f} best={min(value, self.best_loss):.8f}",
            flush=True,
        )
        if improved:
            self.best_loss = value
            self._save(trainer, pl_module, "best")

    def state_dict(self) -> dict:
        return {"best_loss": self.best_loss}

    def load_state_dict(self, state_dict: dict) -> None:
        self.best_loss = float(state_dict.get("best_loss", float("inf")))

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        epoch = trainer.current_epoch + 1
        if not trainer.is_global_zero:
            return
        if epoch % self.interval == 0 or epoch == trainer.max_epochs:
            self._save(trainer, pl_module, f"epoch_{epoch}")
        self._save(trainer, pl_module, "last")


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


def make_loaders(train_set, validation_set, cfg: DictConfig):
    generator = torch.Generator().manual_seed(cfg.seed)
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
