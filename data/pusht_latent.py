"""Episode-safe Push-T latent caches and training datasets."""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import hdf5plugin  # noqa: F401  # Register compressed HDF5 filters before reads.
import numpy as np
import torch
from torch.utils.data import Dataset

from models.world_model.artifacts import RELEASED_LEWM_KIND, load_portable_artifact
from models.world_model.loading import load_released_lewm
from utils import file_sha256


CACHE_VERSION = 2
_LATENT_FILENAME = "frame_latents.npy"
_METADATA_FILENAME = "metadata.json"
_VALIDATION_SAMPLE_SEED_OFFSET = 1_000_003
_POLICY_GOAL_SEED_OFFSET = 2_000_003


@dataclass(frozen=True)
class ActionStatistics:
    mean: tuple[float, ...]
    std: tuple[float, ...]

    @classmethod
    def from_mapping(cls, value: dict) -> "ActionStatistics":
        return cls(
            mean=tuple(float(x) for x in value["mean"]),
            std=tuple(float(x) for x in value["std"]),
        )

    def as_dict(self) -> dict[str, list[float]]:
        return {"mean": list(self.mean), "std": list(self.std)}


def calculate_action_statistics(
    actions: np.ndarray,
    row_mask: np.ndarray | None = None,
) -> ActionStatistics:
    """Calculate finite-row action statistics for a declared data scope."""

    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError("actions must have shape [frames, action_dim]")
    finite = np.isfinite(actions).all(axis=1)
    if row_mask is not None:
        row_mask = np.asarray(row_mask, dtype=bool)
        if row_mask.shape != finite.shape:
            raise ValueError("row_mask must contain one value per action row")
        finite &= row_mask
    if not finite.any():
        raise ValueError("selected data scope contains no finite actions")
    mean = actions[finite].mean(axis=0, dtype=np.float64)
    std = actions[finite].std(axis=0, dtype=np.float64)
    return ActionStatistics(
        mean=tuple(mean.tolist()),
        std=tuple(np.maximum(std, 1e-6).tolist()),
    )


def _atomic_json_dump(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_latent_metadata(cache_dir: str | Path) -> dict:
    path = Path(cache_dir) / _METADATA_FILENAME
    metadata = json.loads(path.read_text())
    if metadata.get("version") != CACHE_VERSION:
        raise ValueError(
            f"unsupported latent cache version {metadata.get('version')!r}; "
            f"expected {CACHE_VERSION}"
        )
    latent_path = Path(cache_dir) / _LATENT_FILENAME
    if not latent_path.is_file():
        raise FileNotFoundError(latent_path)
    return metadata


def with_horizon_view(metadata: dict, *, frameskip: int, max_horizon: int) -> dict:
    """Attach one action-horizon view to immutable frame-cache lineage."""

    if isinstance(frameskip, bool) or not isinstance(frameskip, int) or frameskip < 1:
        raise ValueError("frameskip must be a positive integer")
    if isinstance(max_horizon, bool) or not isinstance(max_horizon, int) or max_horizon < 1:
        raise ValueError("max_horizon must be a positive integer")
    return {
        **metadata,
        "frameskip": frameskip,
        "max_horizon": max_horizon,
        "max_goal_offset": frameskip * max_horizon,
    }


def _validate_cache_request(
    metadata: dict,
    *,
    dataset_path: Path,
    source_config: Path,
    source_weights: Path,
    seed: int,
    train_fraction: float,
) -> dict[str, str]:
    dataset_stat = dataset_path.stat()
    _, artifact_metadata = load_portable_artifact(
        source_weights,
        expected_kind=RELEASED_LEWM_KIND,
    )
    expected = {
        "dataset_path": str(dataset_path),
        "dataset_size": dataset_stat.st_size,
        "dataset_mtime_ns": dataset_stat.st_mtime_ns,
        "source_checkpoint_sha256": artifact_metadata["source_checkpoint_sha256"],
        "seed": int(seed),
        "train_fraction": float(train_fraction),
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(
                f"existing latent cache has incompatible {field}: "
                f"{metadata.get(field)!r} != {value!r}"
            )
    artifact_fields = {
        "source_model_config": str(source_config),
        "source_model_config_sha256": file_sha256(source_config),
        "source_weights": str(source_weights),
        "source_weights_sha256": file_sha256(source_weights),
    }
    return artifact_fields


def split_episode_ids(
    episode_count: int,
    train_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split whole episodes deterministically, never overlapping clips."""

    if episode_count < 2:
        raise ValueError("at least two episodes are required")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    permutation = np.random.default_rng(seed).permutation(episode_count)
    train_count = int(episode_count * train_fraction)
    train_count = min(max(train_count, 1), episode_count - 1)
    return np.sort(permutation[:train_count]), np.sort(permutation[train_count:])


def _validate_episode_layout(
    lengths: np.ndarray,
    offsets: np.ndarray,
    frame_count: int,
) -> None:
    if lengths.ndim != 1 or offsets.ndim != 1 or lengths.shape != offsets.shape:
        raise ValueError("episode lengths and offsets must be same-length vectors")
    if len(lengths) < 2 or np.any(lengths <= 0):
        raise ValueError("dataset must contain at least two non-empty episodes")
    expected_offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(lengths[:-1], dtype=np.int64))
    )
    if not np.array_equal(offsets, expected_offsets):
        raise ValueError("episode offsets are inconsistent with episode lengths")
    if int(lengths.sum()) != frame_count:
        raise ValueError("episode lengths do not cover the declared frame count")


def _rows_for_episodes(
    offsets: np.ndarray,
    lengths: np.ndarray,
    episodes: np.ndarray,
) -> np.ndarray:
    mask = np.zeros(int(lengths.sum()), dtype=bool)
    for episode in episodes:
        start = int(offsets[episode])
        mask[start : start + int(lengths[episode])] = True
    return mask


def preprocess_pusht_pixels(
    pixels: np.ndarray | torch.Tensor, device: torch.device | str
) -> torch.Tensor:
    """Apply the exact ImageNet preprocessing used by released LeWM weights."""

    device = torch.device(device)
    if torch.is_tensor(pixels):
        tensor = pixels.to(device=device, non_blocking=True)
    else:
        tensor = torch.from_numpy(np.asarray(pixels)).to(device=device, non_blocking=True)
    if tensor.ndim != 4:
        raise ValueError("pixels must have shape [batch,H,W,C] or [batch,C,H,W]")
    if tensor.size(-1) == 3:
        tensor = tensor.permute(0, 3, 1, 2)
    elif tensor.size(1) != 3:
        raise ValueError("pixels must expose an RGB channel dimension of size 3")
    tensor = tensor.float().div_(255.0)
    mean = tensor.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = tensor.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return tensor.sub_(mean).div_(std)


def build_frame_latent_cache(
    *,
    dataset_path: str | Path,
    source_config: str | Path,
    source_weights: str | Path,
    output_dir: str | Path,
    seed: int,
    train_fraction: float,
    batch_size: int,
    device: str = "cuda",
    overwrite: bool = False,
) -> dict:
    """Encode each dataset frame once using the released LeWM encoder."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    dataset_path = Path(dataset_path).resolve()
    source_config = Path(source_config).resolve()
    source_weights = Path(source_weights).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    latent_path = output_dir / _LATENT_FILENAME
    metadata_path = output_dir / _METADATA_FILENAME
    if (latent_path.exists() or metadata_path.exists()) and not overwrite:
        metadata = load_latent_metadata(output_dir)
        artifact_fields = _validate_cache_request(
            metadata,
            dataset_path=dataset_path,
            source_config=source_config,
            source_weights=source_weights,
            seed=seed,
            train_fraction=train_fraction,
        )
        if any(metadata.get(key) != value for key, value in artifact_fields.items()):
            metadata.update(artifact_fields)
            _atomic_json_dump(metadata, metadata_path)
        return metadata

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    target_device = torch.device(device)
    source_model, source_metadata = load_released_lewm(source_config, source_weights)
    encoder = source_model.encoder.to(target_device).eval().requires_grad_(False)
    projector = source_model.projector.to(target_device).eval().requires_grad_(False)

    temporary_latents = latent_path.with_suffix(".npy.tmp")
    if temporary_latents.exists():
        temporary_latents.unlink()

    with h5py.File(dataset_path, "r", swmr=True) as dataset:
        lengths = dataset["ep_len"][:].astype(np.int64)
        offsets = dataset["ep_offset"][:].astype(np.int64)
        frame_count = int(dataset["pixels"].shape[0])
        _validate_episode_layout(lengths, offsets, frame_count)
        if dataset["action"].shape[0] != frame_count:
            raise ValueError("action row count does not match episode metadata")
        train_episodes, validation_episodes = split_episode_ids(len(lengths), train_fraction, seed)
        train_rows = _rows_for_episodes(offsets, lengths, train_episodes)
        actions = dataset["action"][:].astype(np.float32)
        action_statistics = calculate_action_statistics(actions, train_rows)

        sample_pixels = preprocess_pusht_pixels(dataset["pixels"][:1], target_device)
        with torch.inference_mode():
            encoded = encoder(sample_pixels, interpolate_pos_encoding=True)
            latent_dim = int(projector(encoded.last_hidden_state[:, 0]).size(-1))

        latents = np.lib.format.open_memmap(
            temporary_latents,
            mode="w+",
            dtype=np.float16,
            shape=(frame_count, latent_dim),
        )
        started = time.time()
        report_interval = max(batch_size, (frame_count // 100 // batch_size) * batch_size)
        with torch.inference_mode():
            for start in range(0, frame_count, batch_size):
                end = min(start + batch_size, frame_count)
                pixels = preprocess_pusht_pixels(dataset["pixels"][start:end], target_device)
                with torch.autocast(
                    device_type=target_device.type,
                    dtype=torch.bfloat16,
                    enabled=target_device.type == "cuda",
                ):
                    output = encoder(pixels, interpolate_pos_encoding=True)
                    batch_latents = projector(output.last_hidden_state[:, 0])
                latents[start:end] = batch_latents.float().cpu().numpy().astype(np.float16)
                if end == frame_count or end % report_interval < batch_size:
                    elapsed = max(time.time() - started, 1e-6)
                    frames_per_second = end / elapsed
                    eta = (frame_count - end) / max(frames_per_second, 1e-6)
                    print(
                        f"latent-cache {end}/{frame_count} frames "
                        f"({100.0 * end / frame_count:.1f}%), "
                        f"{frames_per_second:.1f} frames/s, ETA {eta / 60.0:.1f} min",
                        flush=True,
                    )
        latents.flush()
        del latents

    os.replace(temporary_latents, latent_path)
    metadata = {
        "version": CACHE_VERSION,
        "dataset_path": str(dataset_path),
        "dataset_size": dataset_path.stat().st_size,
        "dataset_mtime_ns": dataset_path.stat().st_mtime_ns,
        "source_model_config": str(source_config),
        "source_model_config_sha256": file_sha256(source_config),
        "source_weights": str(source_weights),
        "source_weights_sha256": file_sha256(source_weights),
        "source_checkpoint_sha256": source_metadata["source_checkpoint_sha256"],
        "seed": int(seed),
        "train_fraction": float(train_fraction),
        "train_episode_ids": train_episodes.tolist(),
        "validation_episode_ids": validation_episodes.tolist(),
        "episode_lengths": lengths.tolist(),
        "episode_offsets": offsets.tolist(),
        "frame_count": frame_count,
        "latent_dim": latent_dim,
        "latent_dtype": "float16",
        "raw_action_dim": int(actions.shape[-1]),
        "action_statistics": action_statistics.as_dict(),
    }
    _atomic_json_dump(metadata, metadata_path)
    return metadata


def _deterministic_subsample(
    values: np.ndarray,
    max_samples: int | None,
    seed: int,
) -> np.ndarray:
    if max_samples is None or max_samples >= len(values):
        return values
    if max_samples < 1:
        raise ValueError("max_samples must be positive when provided")
    selected = np.random.default_rng(seed).choice(len(values), max_samples, replace=False)
    return values[np.sort(selected)]


def _balanced_goal_bins(count: int, num_bins: int, seed: int) -> np.ndarray:
    """Assign every goal bin deterministically with counts differing by at most one."""

    if count < 1 or num_bins < 1:
        raise ValueError("balanced goal assignment requires positive sizes")
    bins = np.resize(np.arange(1, num_bins + 1, dtype=np.int64), count)
    np.random.default_rng(seed).shuffle(bins)
    return bins


def collate_latent_batch(
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return one batch already vectorized by a latent dataset.

    PyTorch calls Dataset.__getitems__ with all indices selected by a batch
    sampler. The latent datasets gather those indices in NumPy once and return
    stacked tensors directly, so applying default_collate again would be both
    incorrect and needlessly expensive for production-sized batches.
    """

    if not isinstance(batch, dict) or not batch:
        raise TypeError("latent batch must be a non-empty tensor mapping")
    if not all(torch.is_tensor(value) for value in batch.values()):
        raise TypeError("every latent batch value must be a tensor")
    return batch


class _PushTLatentDataset(Dataset):
    def __init__(
        self,
        cache_dir: str | Path,
        split: Literal["train", "validation"],
        *,
        frameskip: int,
        max_horizon: int,
        max_samples: int | None = None,
        sample_seed: int = 0,
    ) -> None:
        super().__init__()
        if split not in ("train", "validation"):
            raise ValueError("split must be 'train' or 'validation'")
        self.cache_dir = Path(cache_dir)
        self.metadata = load_latent_metadata(self.cache_dir)
        view = with_horizon_view({}, frameskip=frameskip, max_horizon=max_horizon)
        self.frameskip = view["frameskip"]
        self.max_horizon = view["max_horizon"]
        self.max_goal_offset = view["max_goal_offset"]
        self.split = split
        self.latents = np.load(self.cache_dir / _LATENT_FILENAME, mmap_mode="r")
        dataset_path = Path(self.metadata["dataset_path"])
        stat = dataset_path.stat()
        if stat.st_size != self.metadata["dataset_size"]:
            raise ValueError("source Push-T dataset size changed after cache creation")
        if stat.st_mtime_ns != self.metadata["dataset_mtime_ns"]:
            raise ValueError("source Push-T dataset timestamp changed after cache creation")
        with h5py.File(dataset_path, "r", swmr=True) as dataset:
            self.actions = dataset["action"][:].astype(np.float32)
        self.lengths = np.asarray(self.metadata["episode_lengths"], dtype=np.int64)
        self.offsets = np.asarray(self.metadata["episode_offsets"], dtype=np.int64)
        self.episode_ids = np.asarray(self.metadata[f"{split}_episode_ids"], dtype=np.int64)
        statistics = ActionStatistics.from_mapping(self.metadata["action_statistics"])
        self.action_mean = np.asarray(statistics.mean, dtype=np.float32)
        self.action_std = np.asarray(statistics.std, dtype=np.float32)
        self.raw_action_dim = int(self.metadata["raw_action_dim"])
        self.latent_dim = int(self.metadata["latent_dim"])
        frame_count = int(self.metadata["frame_count"])
        _validate_episode_layout(self.lengths, self.offsets, frame_count)
        if self.latents.shape != (frame_count, self.latent_dim):
            raise ValueError("latent cache shape does not match its metadata")
        if self.actions.shape != (frame_count, self.raw_action_dim):
            raise ValueError("action shape does not match latent-cache metadata")
        self.max_samples = max_samples
        self.sample_seed = sample_seed

    def _eligible_starts(self, span: int) -> np.ndarray:
        pieces = []
        for episode in self.episode_ids:
            count = int(self.lengths[episode]) - span
            if count > 0:
                pieces.append(self.offsets[episode] + np.arange(count, dtype=np.int64))
        if not pieces:
            raise ValueError(f"split {self.split!r} contains no clips with span {span}")
        starts = np.concatenate(pieces)
        split_offset = 0 if self.split == "train" else _VALIDATION_SAMPLE_SEED_OFFSET
        return _deterministic_subsample(starts, self.max_samples, self.sample_seed + split_offset)

    def _normalize_actions(self, actions: np.ndarray) -> torch.Tensor:
        normalized = (actions - self.action_mean) / self.action_std
        return torch.from_numpy(np.asarray(normalized, dtype=np.float32))


class PushTLatentDynamicsDataset(_PushTLatentDataset):
    """Dense prefix targets backed by one latent per raw dataset frame."""

    def __init__(
        self,
        cache_dir,
        split,
        *,
        frameskip: int,
        max_horizon: int,
        max_samples=None,
        sample_seed=0,
    ) -> None:
        super().__init__(
            cache_dir,
            split,
            frameskip=frameskip,
            max_horizon=max_horizon,
            max_samples=max_samples,
            sample_seed=sample_seed,
        )
        self.starts = self._eligible_starts(self.max_goal_offset)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = int(self.starts[index])
        target_indices = start + self.frameskip * np.arange(1, self.max_horizon + 1, dtype=np.int64)
        raw_actions = self.actions[start : start + self.max_goal_offset]
        action_blocks = self._normalize_actions(raw_actions).reshape(
            self.max_horizon, self.frameskip * self.raw_action_dim
        )
        return {
            "anchor_latent": torch.from_numpy(self.latents[start].astype(np.float32)),
            "target_latents": torch.from_numpy(self.latents[target_indices].astype(np.float32)),
            "action_blocks": action_blocks,
        }

    def __getitems__(self, indices: list[int]) -> dict[str, torch.Tensor]:
        """Vectorize memmap/action gathers for one DataLoader batch."""

        item_indices = np.asarray(indices, dtype=np.int64)
        starts = self.starts[item_indices]
        target_indices = (
            starts[:, None]
            + self.frameskip * np.arange(1, self.max_horizon + 1, dtype=np.int64)[None]
        )
        action_indices = starts[:, None] + np.arange(self.max_goal_offset, dtype=np.int64)[None]
        anchors = torch.from_numpy(self.latents[starts].astype(np.float32))
        targets = torch.from_numpy(self.latents[target_indices].astype(np.float32))
        normalized = (self.actions[action_indices] - self.action_mean) / self.action_std
        action_blocks = torch.from_numpy(normalized.astype(np.float32)).reshape(
            len(starts), self.max_horizon, self.frameskip * self.raw_action_dim
        )
        return {
            "anchor_latent": anchors,
            "target_latents": targets,
            "action_blocks": action_blocks,
        }


class PushTLatentPolicyDataset(_PushTLatentDataset):
    """Goal/action pairs with deterministic horizon coverage per episode split."""

    def __init__(
        self,
        cache_dir,
        split,
        *,
        method: Literal["gc_idm", "larc"],
        frameskip: int,
        max_horizon: int,
        max_samples=None,
        sample_seed=0,
    ) -> None:
        super().__init__(
            cache_dir,
            split,
            frameskip=frameskip,
            max_horizon=max_horizon,
            max_samples=max_samples,
            sample_seed=sample_seed,
        )
        if method not in ("gc_idm", "larc"):
            raise ValueError("method must be 'gc_idm' or 'larc'")
        self.method = method
        self.starts = self._eligible_starts(self.max_goal_offset)
        split_offset = 0 if split == "train" else _VALIDATION_SAMPLE_SEED_OFFSET
        goal_seed = sample_seed + split_offset + _POLICY_GOAL_SEED_OFFSET
        if method == "gc_idm":
            self.goal_offsets = _balanced_goal_bins(
                len(self.starts), self.max_goal_offset, goal_seed
            )
        else:
            blocks = _balanced_goal_bins(len(self.starts), self.max_horizon, goal_seed)
            self.goal_offsets = blocks * self.frameskip

    @property
    def action_dim(self) -> int:
        if self.method == "gc_idm":
            return self.raw_action_dim
        return self.frameskip * self.raw_action_dim

    @property
    def chunk_size(self) -> int:
        return self.max_horizon

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = int(self.starts[index])
        goal_offset = int(self.goal_offsets[index])
        output = {
            "current_latent": torch.from_numpy(self.latents[start].astype(np.float32)),
            "goal_latent": torch.from_numpy(self.latents[start + goal_offset].astype(np.float32)),
            "steps_remaining": torch.tensor(goal_offset, dtype=torch.long),
        }
        if self.method == "gc_idm":
            output["action"] = self._normalize_actions(self.actions[start])
            return output

        raw_actions = self.actions[start : start + self.max_goal_offset]
        output["action_chunk"] = self._normalize_actions(raw_actions).reshape(
            self.max_horizon, self.frameskip * self.raw_action_dim
        )
        valid_blocks = goal_offset // self.frameskip
        output["action_mask"] = torch.arange(self.max_horizon) < valid_blocks
        return output

    def __getitems__(self, indices: list[int]) -> dict[str, torch.Tensor]:
        """Vectorize latent and action gathers for one policy batch."""

        item_indices = np.asarray(indices, dtype=np.int64)
        starts = self.starts[item_indices]
        goal_offsets = self.goal_offsets[item_indices]
        current = torch.from_numpy(self.latents[starts].astype(np.float32))
        goals = torch.from_numpy(self.latents[starts + goal_offsets].astype(np.float32))
        steps = torch.from_numpy(goal_offsets.copy()).long()
        if self.method == "gc_idm":
            normalized = (self.actions[starts] - self.action_mean) / self.action_std
            actions = torch.from_numpy(normalized.astype(np.float32))
            return {
                "current_latent": current,
                "goal_latent": goals,
                "steps_remaining": steps,
                "action": actions,
            }

        action_indices = starts[:, None] + np.arange(self.max_goal_offset, dtype=np.int64)[None]
        normalized = (self.actions[action_indices] - self.action_mean) / self.action_std
        chunks = torch.from_numpy(normalized.astype(np.float32)).reshape(
            len(starts), self.max_horizon, self.frameskip * self.raw_action_dim
        )
        valid_blocks = torch.from_numpy(goal_offsets // self.frameskip)
        masks = torch.arange(self.max_horizon)[None] < valid_blocks[:, None]
        return {
            "current_latent": current,
            "goal_latent": goals,
            "steps_remaining": steps,
            "action_chunk": chunks,
            "action_mask": masks,
        }
