"""Train GC-IDM on episode-separated Push-T frozen latents."""

from __future__ import annotations

from pathlib import Path

import hydra
import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf, open_dict

from data import (
    LatentPolicyDataset,
    load_latent_metadata,
    with_horizon_view,
)
from train.artifacts import finalize_training_artifacts
from train.policy_common import PolicyWeightsCheckpoint, configure_adamw, make_loaders
from train.reproducibility import configure_reproducibility, reject_external_lightning_callbacks
from utils import validate_code_revision


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
            batch_size=predicted.size(0),
        )
        return losses["loss"]

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "validation")

    def configure_optimizers(self):
        return configure_adamw(
            self.policy.parameters(), self.optimizer_config, self.max_training_epochs
        )


def run(cfg: DictConfig) -> None:
    reject_external_lightning_callbacks()
    configure_reproducibility(cfg.seed)
    code_revision = validate_code_revision(str(cfg.code_revision))
    metadata = load_latent_metadata(cfg.latent_cache_dir)
    view_metadata = with_horizon_view(
        metadata,
        frameskip=int(cfg.data.frameskip),
        max_horizon=int(cfg.data.max_horizon),
    )
    train_dataset = LatentPolicyDataset(
        cfg.latent_cache_dir,
        "train",
        method="gc_idm",
        frameskip=view_metadata["frameskip"],
        max_horizon=view_metadata["max_horizon"],
        max_samples=cfg.data.max_train_samples,
        sample_seed=cfg.seed,
    )
    validation_dataset = LatentPolicyDataset(
        cfg.latent_cache_dir,
        "validation",
        method="gc_idm",
        frameskip=view_metadata["frameskip"],
        max_horizon=view_metadata["max_horizon"],
        max_samples=cfg.data.max_validation_samples,
        sample_seed=cfg.seed,
    )
    with open_dict(cfg):
        cfg.policy.latent_dim = train_dataset.latent_dim
        cfg.policy.action_dim = train_dataset.action_dim
        cfg.policy.max_horizon = train_dataset.max_goal_offset
    policy = hydra.utils.instantiate(cfg.policy)
    objective = hydra.utils.instantiate(cfg.objective)
    train_loader, validation_loader = make_loaders(train_dataset, validation_dataset, cfg)
    module = GCIDMTrainingModule(
        policy,
        objective,
        optimizer_config=cfg.optimizer,
        max_epochs=cfg.trainer.max_epochs,
    )
    logger = False
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    training_metadata = {
        **view_metadata,
        "method": "gc_idm",
        "training_seed": int(cfg.seed),
        "training_code_revision": code_revision,
        "training_batch_size": int(cfg.loader.batch_size),
    }
    callback = PolicyWeightsCheckpoint(
        output_dir=Path(cfg.output_dir),
        policy_config=cfg.policy,
        filename_prefix=cfg.output_model_name,
        metadata=training_metadata,
    )
    lightning_checkpoint = ModelCheckpoint(
        dirpath=Path(cfg.output_dir) / "lightning",
        filename="epoch-{epoch:02d}",
        monitor="validation/loss",
        mode="min",
        save_last=True,
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_kwargs.setdefault("default_root_dir", str(Path(cfg.output_dir)))
    trainer_kwargs.setdefault("deterministic", True)
    trainer = pl.Trainer(
        **trainer_kwargs,
        callbacks=[callback, lightning_checkpoint],
        logger=logger,
    )
    resume_path = Path(cfg.output_dir) / "lightning" / "last.ckpt"
    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
        ckpt_path=str(resume_path) if resume_path.exists() else None,
    )
    if trainer.is_global_zero:
        finalize_training_artifacts(cfg.output_dir, str(cfg.output_model_name))


@hydra.main(version_base=None, config_path="../configs", config_name="train_gc_idm")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
