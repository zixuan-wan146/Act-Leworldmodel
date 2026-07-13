"""Training-checkpoint lifecycle helpers."""

from __future__ import annotations

import shutil
from pathlib import Path


def finalize_training_artifacts(output_dir: str | Path, filename_prefix: str) -> None:
    """Keep the best tensor export and discard completed-run recovery state."""

    if Path(filename_prefix).name != filename_prefix:
        raise ValueError("checkpoint filename prefix must not contain a path")
    output_dir = Path(output_dir)
    best = output_dir / f"{filename_prefix}_best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"best tensor checkpoint was not written: {best}")

    for checkpoint in output_dir.glob(f"{filename_prefix}_epoch_*.pt"):
        checkpoint.unlink()
    last = output_dir / f"{filename_prefix}_last.pt"
    if last.exists():
        if not last.is_file():
            raise RuntimeError(f"last-checkpoint path is not a file: {last}")
        last.unlink()

    recovery = output_dir / "lightning"
    if recovery.is_symlink():
        raise RuntimeError(f"refusing to remove symlinked recovery directory: {recovery}")
    if recovery.exists():
        if not recovery.is_dir():
            raise RuntimeError(f"recovery path is not a directory: {recovery}")
        shutil.rmtree(recovery)
