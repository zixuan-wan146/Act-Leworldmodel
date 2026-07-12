"""Train LARC-Chunk with frozen-world-model rollout consistency."""

from __future__ import annotations

from pathlib import Path

import hydra
import lightning as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf, open_dict

from data import LatentPolicyDataset
from models.world_model import load_frozen_world_model
from train.policy_common import PolicyWeightsCheckpoint, configure_adamw, make_loaders


class LARCTrainingModule(pl.LightningModule):
    def __init__(
        self,
        policy,
        objective,
        world_model,
        optimizer_config: DictConfig,
        max_epochs: int,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.objective = objective
        self.world_model = world_model
        self.optimizer_config = optimizer_config
        self.max_training_epochs = max_epochs

    def _step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        predicted_actions = self.policy(
            batch["current_latent"],
            batch["goal_latent"],
            batch["steps_remaining"],
        )
        losses = self.objective(
            predicted_actions=predicted_actions,
            expert_actions=batch["action_chunk"],
            current_latent=batch["current_latent"],
            goal_latent=batch["goal_latent"],
            world_model=self.world_model,
        )
        scalar_losses = {
            f"{stage}/{name}": value
            for name, value in losses.items()
            if name == "loss" or name.endswith("_loss")
        }
        self.log_dict(
            scalar_losses,
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
        required_keys=(
            "current_latent",
            "goal_latent",
            "steps_remaining",
            "action_chunk",
        ),
    )
    action_chunks = dataset.tensors["action_chunk"]
    action_dim = action_chunks.size(-1)
    chunk_size = action_chunks.size(-2)
    with open_dict(cfg):
        cfg.policy.action_dim = action_dim
        cfg.policy.chunk_size = chunk_size
    policy = hydra.utils.instantiate(cfg.policy)
    objective = hydra.utils.instantiate(cfg.objective)
    world_model = load_frozen_world_model(
        cfg.world_model.config_path,
        cfg.world_model.weights_path,
    )
    if chunk_size > world_model.max_horizon:
        raise ValueError(
            f"LARC chunk size {chunk_size} exceeds world-model horizon "
            f"{world_model.max_horizon}"
        )
    train_loader, validation_loader = make_loaders(dataset, cfg)
    module = LARCTrainingModule(
        policy,
        objective,
        world_model,
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


@hydra.main(version_base=None, config_path="../configs", config_name="train_larc")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
