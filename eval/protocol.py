"""Fixed, reusable episode/start manifests for fair Push-T comparison."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
import json
import os
from pathlib import Path

import h5py
import numpy as np

from data import load_latent_metadata


def _integer_list(manifest: dict, key: str, expected_length: int) -> np.ndarray:
    value = manifest.get(key)
    if not isinstance(value, list) or len(value) != expected_length:
        raise ValueError(f"evaluation manifest {key} must contain {expected_length} integers")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"evaluation manifest {key} must contain only integers")
    return np.asarray(value, dtype=np.int64)


def _normalize_goal_offsets(goal_offsets: Sequence[int]) -> np.ndarray:
    if isinstance(goal_offsets, (str, bytes)) or not isinstance(goal_offsets, Sequence):
        raise ValueError("goal_offsets must be a non-empty integer sequence")
    values = list(goal_offsets)
    if not values or any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("goal_offsets must be a non-empty integer sequence")
    offsets = np.asarray(values, dtype=np.int64)
    if np.any(offsets < 1):
        raise ValueError("goal_offsets must be positive")
    if not np.array_equal(offsets, np.unique(offsets)):
        raise ValueError("goal_offsets must be strictly increasing and unique")
    return offsets


def _validate_manifest_entries(
    manifest: dict,
    *,
    validation_episodes: np.ndarray,
    episode_lengths: np.ndarray,
    num_eval: int,
    max_goal_offset: int,
) -> None:
    episodes = _integer_list(manifest, "episode_indices", num_eval)
    starts = _integer_list(manifest, "start_steps", num_eval)
    if len(np.unique(episodes)) != num_eval:
        raise ValueError("evaluation manifest episode_indices must be unique")
    if np.any(episodes < 0) or np.any(episodes >= len(episode_lengths)):
        raise ValueError("evaluation manifest contains an out-of-range episode")
    if not np.isin(episodes, validation_episodes).all():
        raise ValueError("evaluation manifest contains a non-validation episode")
    upper_bounds = episode_lengths[episodes] - max_goal_offset
    if np.any(upper_bounds <= 0):
        raise ValueError("evaluation manifest contains an episode shorter than the goal offset")
    if np.any(starts < 0) or np.any(starts >= upper_bounds):
        raise ValueError("evaluation manifest contains an out-of-range start step")


def create_or_load_manifest(
    *,
    dataset_path: str | Path,
    latent_cache_dir: str | Path,
    output_path: str | Path,
    seed: int,
    num_eval: int,
    goal_offsets: Sequence[int],
) -> dict:
    """Choose paired validation starts once for every method and goal offset."""

    if num_eval < 1:
        raise ValueError("num_eval must be positive")
    offsets = _normalize_goal_offsets(goal_offsets)
    max_goal_offset = int(offsets[-1])
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
    if validation.ndim != 1 or len(np.unique(validation)) != len(validation):
        raise ValueError("latent cache validation episode IDs must be a unique vector")
    validation_digest = hashlib.sha256(validation.astype("<i8", copy=False).tobytes()).hexdigest()
    expected = {
        "version": 2,
        "dataset_path": str(dataset_path),
        "dataset_size": dataset_stat.st_size,
        "dataset_mtime_ns": dataset_stat.st_mtime_ns,
        "source_checkpoint_sha256": cache_metadata["source_checkpoint_sha256"],
        "split_seed": int(cache_metadata["seed"]),
        "train_fraction": float(cache_metadata["train_fraction"]),
        "validation_episode_ids_sha256": validation_digest,
        "evaluation_seed": int(seed),
        "num_eval": int(num_eval),
        "goal_offsets": offsets.tolist(),
        "max_goal_offset": max_goal_offset,
    }
    with h5py.File(dataset_path, "r", swmr=True) as dataset:
        lengths = dataset["ep_len"][:].astype(np.int64)
    if np.any(validation < 0) or np.any(validation >= len(lengths)):
        raise ValueError("latent cache contains an out-of-range validation episode")
    if output_path.exists():
        manifest = json.loads(output_path.read_text())
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"existing evaluation manifest has incompatible {key}")
        _validate_manifest_entries(
            manifest,
            validation_episodes=validation,
            episode_lengths=lengths,
            num_eval=num_eval,
            max_goal_offset=max_goal_offset,
        )
        return manifest
    eligible = validation[lengths[validation] > max_goal_offset]
    if len(eligible) < num_eval:
        raise ValueError("not enough validation episodes for the evaluation protocol")
    rng = np.random.default_rng(seed)
    episodes = np.sort(rng.choice(eligible, num_eval, replace=False))
    starts = np.asarray(
        [rng.integers(0, int(lengths[episode]) - max_goal_offset) for episode in episodes],
        dtype=np.int64,
    )
    manifest = {
        **expected,
        "episode_indices": episodes.tolist(),
        "start_steps": starts.tolist(),
    }
    _validate_manifest_entries(
        manifest,
        validation_episodes=validation,
        episode_lengths=lengths,
        num_eval=num_eval,
        max_goal_offset=max_goal_offset,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_path)
    return manifest
