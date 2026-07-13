"""Fixed, reusable episode/start manifests for fair Push-T comparison."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

from data import load_latent_metadata


def create_or_load_manifest(
    *,
    dataset_path: str | Path,
    latent_cache_dir: str | Path,
    output_path: str | Path,
    seed: int,
    num_eval: int,
    goal_offset: int,
) -> dict:
    """Choose unique validation episodes and fixed starts once for all methods."""

    if num_eval < 1 or goal_offset < 1:
        raise ValueError("num_eval and goal_offset must be positive")
    dataset_path = Path(dataset_path).resolve()
    output_path = Path(output_path)
    cache_metadata = load_latent_metadata(latent_cache_dir)
    if Path(cache_metadata["dataset_path"]) != dataset_path:
        raise ValueError("latent cache and evaluation dataset do not match")
    dataset_stat = dataset_path.stat()
    if dataset_stat.st_size != cache_metadata["dataset_size"]:
        raise ValueError("evaluation dataset size changed after latent caching")
    if dataset_stat.st_mtime_ns != cache_metadata["dataset_mtime_ns"]:
        raise ValueError("evaluation dataset timestamp changed after latent caching")
    validation = np.asarray(cache_metadata["validation_episode_ids"], dtype=np.int64)
    validation_digest = hashlib.sha256(validation.astype("<i8", copy=False).tobytes()).hexdigest()
    expected = {
        "version": 1,
        "dataset_path": str(dataset_path),
        "dataset_size": dataset_stat.st_size,
        "dataset_mtime_ns": dataset_stat.st_mtime_ns,
        "source_checkpoint_sha256": cache_metadata["source_checkpoint_sha256"],
        "split_seed": int(cache_metadata["seed"]),
        "train_fraction": float(cache_metadata["train_fraction"]),
        "validation_episode_ids_sha256": validation_digest,
        "evaluation_seed": int(seed),
        "num_eval": int(num_eval),
        "goal_offset": int(goal_offset),
    }
    if output_path.exists():
        manifest = json.loads(output_path.read_text())
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"existing evaluation manifest has incompatible {key}")
        return manifest

    with h5py.File(dataset_path, "r", swmr=True) as dataset:
        lengths = dataset["ep_len"][:].astype(np.int64)
    eligible = validation[lengths[validation] > goal_offset]
    if len(eligible) < num_eval:
        raise ValueError("not enough validation episodes for the evaluation protocol")
    rng = np.random.default_rng(seed)
    episodes = np.sort(rng.choice(eligible, num_eval, replace=False))
    starts = np.asarray(
        [rng.integers(0, int(lengths[episode]) - goal_offset) for episode in episodes],
        dtype=np.int64,
    )
    manifest = {
        **expected,
        "episode_indices": episodes.tolist(),
        "start_steps": starts.tolist(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_path)
    return manifest
