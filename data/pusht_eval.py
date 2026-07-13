"""Minimal HDF5 access for the fixed Push-T evaluation protocol."""

from __future__ import annotations

from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  # Register compressed HDF5 filters before reads.
import numpy as np


def _integer_vector(values: list[int], name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer vector")
    return array.astype(np.int64, copy=False)


def _take_rows(dataset: h5py.Dataset, rows: np.ndarray) -> np.ndarray:
    """Support arbitrary row order while satisfying h5py's sorted-index rule."""

    order = np.argsort(rows)
    sorted_rows = rows[order]
    if len(np.unique(sorted_rows)) != len(sorted_rows):
        raise ValueError("evaluation rows must be unique")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return dataset[sorted_rows][inverse]


class PushTEvaluationDataset:
    """Read only the initial and goal rows required by evaluation."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._file = h5py.File(self.path, "r", swmr=True)
        required = {"ep_len", "ep_offset", "pixels", "state", "action"}
        missing = required.difference(self._file.keys())
        if missing:
            self.close()
            raise ValueError(f"Push-T dataset is missing columns: {sorted(missing)}")
        self.lengths = self._file["ep_len"][:].astype(np.int64)
        self.offsets = self._file["ep_offset"][:].astype(np.int64)
        if self.lengths.ndim != 1 or self.offsets.shape != self.lengths.shape:
            self.close()
            raise ValueError("episode metadata must be same-length vectors")
        if not len(self.lengths) or np.any(self.lengths <= 0):
            self.close()
            raise ValueError("episode lengths must describe non-empty episodes")
        expected = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(self.lengths[:-1], dtype=np.int64))
        )
        if not np.array_equal(self.offsets, expected):
            self.close()
            raise ValueError("episode offsets are inconsistent with lengths")
        self.frame_count = int(self.lengths.sum())
        row_counts = {
            name: int(self._file[name].shape[0]) for name in ("pixels", "state", "action")
        }
        if any(count != self.frame_count for count in row_counts.values()):
            self.close()
            raise ValueError("episode metadata does not cover every evaluation column")
        if self._file["pixels"].ndim != 4 or self._file["pixels"].shape[-1] != 3:
            self.close()
            raise ValueError("Push-T pixels must have shape [frames,height,width,3]")
        if self._file["state"].shape[1:] != (7,) or self._file["action"].shape[1:] != (2,):
            self.close()
            raise ValueError("Push-T state and action columns have incompatible shapes")

    def evaluation_rows(
        self,
        episode_indices: list[int],
        start_steps: list[int],
        goal_offset: int,
    ) -> dict[str, np.ndarray]:
        episodes = _integer_vector(episode_indices, "episode_indices")
        starts = _integer_vector(start_steps, "start_steps")
        if not len(episodes) or starts.shape != episodes.shape:
            raise ValueError(
                "episode_indices and start_steps must be same-length non-empty vectors"
            )
        if isinstance(goal_offset, bool) or not isinstance(goal_offset, int) or goal_offset < 1:
            raise ValueError("goal_offset must be a positive integer")
        if np.any(episodes < 0) or np.any(episodes >= len(self.lengths)):
            raise ValueError("episode_indices contain an out-of-range episode")
        upper_bounds = self.lengths[episodes] - goal_offset
        if np.any(starts < 0) or np.any(starts >= upper_bounds):
            raise ValueError("start_steps contain an out-of-range step")
        initial_rows = self.offsets[episodes] + starts
        goal_rows = initial_rows + goal_offset
        return {
            "pixels": _take_rows(self._file["pixels"], initial_rows),
            "goal_pixels": _take_rows(self._file["pixels"], goal_rows),
            "state": _take_rows(self._file["state"], initial_rows),
            "goal_state": _take_rows(self._file["state"], goal_rows),
        }

    def action_array(self) -> np.ndarray:
        return self._file["action"][:]

    def close(self) -> None:
        if getattr(self, "_file", None) is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "PushTEvaluationDataset":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
