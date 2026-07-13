"""Train Fast-LeWM dynamics on cached latents from the released LeWM encoder."""

from __future__ import annotations

import json
import os
from pathlib import Path

import hydra
import lightning as pl
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf, open_dict

from data import (
    PushTLatentDynamicsDataset,
    load_latent_metadata,
    with_horizon_view,
)
from models.world_model import initialize_representation_from_lewm
from models.world_model.artifacts import RELEASED_LEWM_KIND, load_portable_artifact
from train.reproducibility import configure_reproducibility, make_generator
from utils import file_sha256


class LatentDynamicsTrainingModule(pl.LightningModule):
    def __init__(self, model, objective, optimizer_config: DictConfig, max_epochs: int):
        super().__init__()
        self.model = model
        self.objective = objective
        self.optimizer_config = optimizer_config
        self.max_training_epochs = max_epochs

    def _step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        predictions = self.model.predict_latents(batch["anchor_latent"], batch["action_blocks"])
        losses = self.objective(
            predictions=predictions,
            targets=batch["target_latents"],
            encoded_sequence=None,
        )
        self.log_dict(
            {f"{stage}/{name}": value for name, value in losses.items()},
            on_step=stage == "train",
            on_epoch=True,
            sync_dist=True,
            batch_size=batch["anchor_latent"].size(0),
        )
        return losses["loss"]

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "validation")

    def configure_optimizers(self):
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.optimizer_config.lr,
            weight_decay=self.optimizer_config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_training_epochs,
            eta_min=self.optimizer_config.lr / 100.0,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


class PortableWorldModelCheckpoint(Callback):
    """Save model-only weights, construction config, and data protocol metadata."""

    def __init__(
        self,
        output_dir: str | Path,
        model_config: DictConfig,
        metadata: dict,
        filename_prefix: str,
        interval: int = 1,
    ) -> None:
        super().__init__()
        if interval < 1:
            raise ValueError("checkpoint interval must be positive")
        self.output_dir = Path(output_dir)
        self.model_config = OmegaConf.create(OmegaConf.to_container(model_config, resolve=True))
        self.metadata = metadata
        self.filename_prefix = filename_prefix
        self.interval = interval
        self.best_loss = float("inf")

    def _save(self, module, suffix: str) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        destination = self.output_dir / f"{self.filename_prefix}_{suffix}.pt"
        temporary = destination.with_suffix(".pt.tmp")
        torch.save(module.model.state_dict(), temporary)
        os.replace(temporary, destination)
        OmegaConf.save(self.model_config, self.output_dir / "model_config.yaml")
        metadata_path = self.output_dir / "model_metadata.json"
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
            f"world-model epoch={trainer.current_epoch + 1} "
            f"validation/loss={value:.8f} best={min(value, self.best_loss):.8f}",
            flush=True,
        )
        if improved:
            self.best_loss = value
            self._save(pl_module, "best")

    def state_dict(self) -> dict:
        return {"best_loss": self.best_loss}

    def load_state_dict(self, state_dict: dict) -> None:
        self.best_loss = float(state_dict.get("best_loss", float("inf")))

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        epoch = trainer.current_epoch + 1
        if not trainer.is_global_zero:
            return
        if epoch % self.interval == 0 or epoch == trainer.max_epochs:
            self._save(pl_module, f"epoch_{epoch}")
        self._save(pl_module, "last")


def _make_loader(dataset, cfg: DictConfig, *, shuffle: bool, seed: int):
    return torch.utils.data.DataLoader(
        dataset,
        **cfg.loader,
        shuffle=shuffle,
        drop_last=shuffle,
        generator=make_generator(seed),
    )


def run(cfg: DictConfig) -> None:
    configure_reproducibility(cfg.seed)
    metadata = load_latent_metadata(cfg.latent_cache_dir)
    _, source_metadata = load_portable_artifact(
        cfg.source_model.weights_path,
        expected_kind=RELEASED_LEWM_KIND,
    )
    if source_metadata["source_checkpoint_sha256"] != metadata["source_checkpoint_sha256"]:
        raise ValueError("configured source representation differs from the latent cache")
    if metadata.get("source_model_config_sha256") not in (
        None,
        file_sha256(cfg.source_model.config_path),
    ):
        raise ValueError("source model config differs from the latent cache")
    if metadata.get("source_weights_sha256") not in (
        None,
        file_sha256(cfg.source_model.weights_path),
    ):
        raise ValueError("source model weights differ from the latent cache")
    if int(cfg.latent_dim) != int(metadata["latent_dim"]):
        raise ValueError("configured latent width differs from the latent cache")
    if not cfg.freeze_representation:
        raise ValueError("cached-latent training requires a frozen representation")
    view_metadata = with_horizon_view(
        metadata,
        frameskip=int(cfg.data.frameskip),
        max_horizon=int(cfg.max_horizon),
    )
    with open_dict(cfg):
        cfg.model.dynamics.prefix_encoder.action_dim = (
            view_metadata["frameskip"] * metadata["raw_action_dim"]
        )
        cfg.model.dynamics.prefix_encoder.latent_dim = metadata["latent_dim"]
        cfg.model.dynamics.predictor.latent_dim = metadata["latent_dim"]

    train_dataset = PushTLatentDynamicsDataset(
        cfg.latent_cache_dir,
        "train",
        frameskip=view_metadata["frameskip"],
        max_horizon=view_metadata["max_horizon"],
        max_samples=cfg.data.max_train_samples,
        sample_seed=cfg.seed,
    )
    validation_dataset = PushTLatentDynamicsDataset(
        cfg.latent_cache_dir,
        "validation",
        frameskip=view_metadata["frameskip"],
        max_horizon=view_metadata["max_horizon"],
        max_samples=cfg.data.max_validation_samples,
        sample_seed=cfg.seed,
    )
    train_loader = _make_loader(train_dataset, cfg, shuffle=True, seed=cfg.seed)
    validation_loader = _make_loader(validation_dataset, cfg, shuffle=False, seed=cfg.seed + 1)

    model = hydra.utils.instantiate(cfg.model)
    initialize_representation_from_lewm(
        model,
        cfg.source_model.weights_path,
        freeze=cfg.freeze_representation,
    )
    objective = hydra.utils.instantiate(cfg.objective)
    module = LatentDynamicsTrainingModule(
        model,
        objective,
        optimizer_config=cfg.optimizer,
        max_epochs=cfg.trainer.max_epochs,
    )
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = False
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    portable = PortableWorldModelCheckpoint(
        output_dir=output_dir,
        model_config=cfg.model,
        metadata={
            **view_metadata,
            "training_seed": int(cfg.seed),
            "freeze_representation": bool(cfg.freeze_representation),
            "source_model_config": str(Path(cfg.source_model.config_path).resolve()),
            "source_weights": str(Path(cfg.source_model.weights_path).resolve()),
        },
        filename_prefix=cfg.output_model_name,
        interval=cfg.checkpoint_interval,
    )
    lightning_checkpoint = ModelCheckpoint(
        dirpath=output_dir / "lightning",
        filename="epoch-{epoch:02d}",
        monitor="validation/loss",
        mode="min",
        save_last=True,
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_kwargs.setdefault("default_root_dir", str(output_dir))
    trainer_kwargs.setdefault("deterministic", True)
    trainer = pl.Trainer(
        **trainer_kwargs,
        callbacks=[portable, lightning_checkpoint],
        logger=logger,
    )
    resume_path = output_dir / "lightning" / "last.ckpt"
    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
        ckpt_path=str(resume_path) if resume_path.exists() else None,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="train_world_model")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
