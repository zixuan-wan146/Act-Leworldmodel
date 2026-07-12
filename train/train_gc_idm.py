"""Train GC-IDM from a cache of frozen world-model latents."""

from __future__ import annotations

from pathlib import Path

import hydra
import lightning as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf, open_dict

from data import LatentPolicyDataset
from train.policy_common import PolicyWeightsCheckpoint, configure_adamw, make_loaders


class GCIDMTrainingModule(pl.LightningModule):
    def __init__(self, policy, objective, optimizer_config: DictConfig, max_epochs: int) -> None:
        super().__init__()
        self.policy = policy
        self.objective = objective
        self.optimizer_config = optimizer_config
        self.max_training_epochs = max_epochs

    def _step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        predicted = self.policy(
            batch["current_latent"],
            batch["goal_latent"],
            batch["steps_remaining"],
        )
        losses = self.objective(predicted, batch["action"])
        self.log_dict(
            {f"{stage}/{name}": value for name, value in losses.items()},
            on_step=stage == "train",
            on_epoch=True,
            sync_dist=True,
        )
        return losses["loss"]

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "validation")

    def configure_optimizers(self):
        return configure_adamw(
            self.policy.parameters(),
            self.optimizer_config,
            self.max_training_epochs,
        )


def run(cfg: DictConfig) -> None:
    dataset = LatentPolicyDataset(
        cfg.latent_cache,
        required_keys=("current_latent", "goal_latent", "steps_remaining", "action"),
    )
    action_dim = dataset.tensors["action"].size(-1)
    with open_dict(cfg):
        cfg.policy.action_dim = action_dim
    policy = hydra.utils.instantiate(cfg.policy)
    objective = hydra.utils.instantiate(cfg.objective)
    train_loader, validation_loader = make_loaders(dataset, cfg)
    module = GCIDMTrainingModule(
        policy,
        objective,
        optimizer_config=cfg.optimizer,
        max_epochs=cfg.trainer.max_epochs,
    )
    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    callback = PolicyWeightsCheckpoint(
        output_dir=Path(cfg.output_dir),
        policy_config=cfg.policy,
        filename_prefix=cfg.output_model_name,
        interval=cfg.checkpoint_interval,
    )
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[callback],
        logger=logger,
        enable_checkpointing=False,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=validation_loader)


@hydra.main(version_base=None, config_path="../configs", config_name="train_gc_idm")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
