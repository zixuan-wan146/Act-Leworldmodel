"""CLI for the reusable Push-T frame-latent cache."""

from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig

from data import build_frame_latent_cache
from train.reproducibility import configure_reproducibility


def run(cfg: DictConfig) -> dict:
    configure_reproducibility(cfg.seed)
    metadata = build_frame_latent_cache(
        dataset_path=cfg.dataset_path,
        source_checkpoint=cfg.source_checkpoint,
        output_dir=cfg.output_dir,
        seed=cfg.seed,
        train_fraction=cfg.train_fraction,
        frameskip=cfg.frameskip,
        max_horizon=cfg.max_horizon,
        batch_size=cfg.batch_size,
        device=cfg.device,
        overwrite=cfg.overwrite,
    )
    summary = {
        "output_dir": str(cfg.output_dir),
        "frame_count": metadata["frame_count"],
        "latent_dim": metadata["latent_dim"],
        "train_episodes": len(metadata["train_episode_ids"]),
        "validation_episodes": len(metadata["validation_episode_ids"]),
        "action_statistics": metadata["action_statistics"],
        "source_checkpoint_sha256": metadata["source_checkpoint_sha256"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return metadata


@hydra.main(version_base=None, config_path="../configs", config_name="cache_pusht_latents")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
