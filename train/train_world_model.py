"""Train the CEM-free Fast-LeWM dynamics backbone."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from data import load_training_dataset


def dynamics_forward(self, batch: dict, stage: str, max_horizon: int):
    """Encode one segment and supervise every available action prefix."""

    pixels = batch["pixels"]
    actions = torch.nan_to_num(batch["action"], nan=0.0)
    horizon = min(max_horizon, pixels.size(1) - 1, actions.size(1))
    if horizon < 1:
        raise ValueError("a sample must contain an anchor and at least one future step")
    encoded = self.model.encode_observations(pixels[:, : horizon + 1])
    predictions = self.model.predict_latents(encoded[:, 0], actions[:, :horizon])
    targets = encoded[:, 1 : horizon + 1]
    losses = self.objective(
        predictions=predictions,
        targets=targets,
        encoded_sequence=encoded,
    )
    self.log_dict(
        {f"{stage}/{name}": value.detach() for name, value in losses.items()},
        on_step=True,
        sync_dist=True,
    )
    return {
        **losses,
        "predicted_latents": predictions,
        "target_latents": targets,
    }


class PortableWeightsCheckpoint(Callback):
    """Persist model-only weights and the resolved construction config."""

    def __init__(
        self,
        output_dir: Path,
        model_config: DictConfig,
        filename_prefix: str,
        interval: int = 1,
    ) -> None:
        super().__init__()
        if interval < 1:
            raise ValueError("checkpoint interval must be positive")
        self.output_dir = output_dir
        self.model_config = OmegaConf.create(
            OmegaConf.to_container(model_config, resolve=True)
        )
        self.filename_prefix = filename_prefix
        self.interval = interval

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        epoch = trainer.current_epoch + 1
        should_save = epoch % self.interval == 0 or epoch == trainer.max_epochs
        if trainer.is_global_zero and should_save:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                pl_module.model.state_dict(),
                self.output_dir / f"{self.filename_prefix}_epoch_{epoch}.pt",
            )
            OmegaConf.save(
                config=self.model_config,
                f=self.output_dir / "model_config.yaml",
            )


def run(cfg: DictConfig) -> None:
    dataset = load_training_dataset(cfg)
    random_generator = torch.Generator().manual_seed(cfg.seed)
    train_set, validation_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=random_generator,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        shuffle=True,
        drop_last=True,
        generator=random_generator,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_set,
        **cfg.loader,
        shuffle=False,
        drop_last=False,
    )
    model = hydra.utils.instantiate(cfg.model)
    objective = hydra.utils.instantiate(cfg.objective)
    optimizer_config = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        }
    }
    module = spt.Module(
        model=model,
        objective=objective,
        forward=partial(dynamics_forward, max_horizon=cfg.max_horizon),
        optim=optimizer_config,
    )
    data_module = spt.data.DataModule(train=train_loader, val=validation_loader)
    run_id = cfg.get("subdir") or "default"
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), run_id)
    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    callbacks = [
        ModelCheckpoint(
            dirpath=run_dir,
            filename=f"{cfg.output_model_name}-{{epoch}}",
            save_last=True,
            every_n_epochs=1,
        ),
        PortableWeightsCheckpoint(
            output_dir=run_dir,
            model_config=cfg.model,
            filename_prefix=cfg.output_model_name,
            interval=cfg.checkpoint_interval,
        ),
    ]
    trainer = pl.Trainer(**cfg.trainer, callbacks=callbacks, logger=logger)
    resume_path = run_dir / "last.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=data_module,
        ckpt_path=resume_path if resume_path.exists() else None,
    )
    manager()


@hydra.main(version_base=None, config_path="../configs", config_name="train_world_model")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
