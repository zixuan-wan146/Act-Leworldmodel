"""Measure and plot Fast-LeWM dense-prefix error on held-out Push-T episodes."""

from __future__ import annotations

import json
import os
from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig, OmegaConf

from data import LatentDynamicsDataset, collate_latent_batch
from eval.provenance import artifact_record, validate_artifact_records
from models.world_model import load_frozen_world_model
from train.reproducibility import configure_reproducibility, make_generator
from utils import validate_code_revision


def run(cfg: DictConfig) -> dict:
    configure_reproducibility(cfg.seed)
    code_revision = validate_code_revision(str(cfg.code_revision))
    world_config = Path(cfg.model.config_path)
    artifacts = {
        "latent_cache_metadata": artifact_record(Path(cfg.latent_cache_dir) / "metadata.json"),
        "world_model_config": artifact_record(world_config),
        "world_model_weights": artifact_record(cfg.model.weights_path),
        "world_model_metadata": artifact_record(world_config.with_name("model_metadata.json")),
    }
    dataset = LatentDynamicsDataset(
        cfg.latent_cache_dir,
        "validation",
        frameskip=cfg.data.frameskip,
        max_horizon=cfg.data.max_horizon,
        max_samples=cfg.max_samples,
        sample_seed=cfg.seed,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        **cfg.loader,
        shuffle=False,
        collate_fn=collate_latent_batch,
        drop_last=False,
        generator=make_generator(cfg.seed),
    )
    model = load_frozen_world_model(cfg.model.config_path, cfg.model.weights_path)
    model = model.to(cfg.device).eval()
    squared_error = torch.zeros(dataset.max_horizon, dtype=torch.float64)
    anchor_error = torch.zeros_like(squared_error)
    sample_count = 0
    device_type = torch.device(cfg.device).type
    with torch.inference_mode():
        for batch in loader:
            anchor = batch["anchor_latent"].to(cfg.device, non_blocking=True)
            actions = batch["action_blocks"].to(cfg.device, non_blocking=True)
            targets = batch["target_latents"].to(cfg.device, non_blocking=True)
            with torch.autocast(
                device_type=device_type,
                dtype=torch.bfloat16,
                enabled=device_type == "cuda",
            ):
                predictions = model.predict_latents(anchor, actions)
            squared_error += (predictions.float() - targets).square().mean(dim=-1).sum(dim=0).cpu()
            anchor_error += (anchor[:, None] - targets).square().mean(dim=-1).sum(dim=0).cpu()
            sample_count += anchor.size(0)
    if sample_count == 0:
        raise ValueError("open-loop evaluation dataset is empty")
    mse = (squared_error / sample_count).tolist()
    persistence_mse = (anchor_error / sample_count).tolist()
    environment_steps = [dataset.frameskip * (index + 1) for index in range(dataset.max_horizon)]
    validate_artifact_records(artifacts, verify_files=True)
    result = {
        "seed": int(cfg.seed),
        "num_samples": sample_count,
        "environment_steps": environment_steps,
        "fast_lewm_mse": mse,
        "persistence_mse": persistence_mse,
        "code_revision": code_revision,
        "artifacts": artifacts,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "open_loop_metrics.json"
    temporary = metrics_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, metrics_path)

    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.plot(environment_steps, mse, marker="o", label="Fast-LeWM")
    axis.plot(
        environment_steps,
        persistence_mse,
        marker="s",
        linestyle="--",
        label="Persistence",
    )
    axis.set_xlabel("Prediction horizon (environment steps)")
    axis.set_ylabel("Held-out latent MSE")
    axis.set_xticks(environment_steps)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "open_loop_curve.png", dpi=180)
    plt.close(figure)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


@hydra.main(version_base=None, config_path="../configs", config_name="eval_open_loop")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
