import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from train.reproducibility import reject_external_lightning_callbacks


def test_project_entrypoints_import_without_reference_packages():
    repository = Path(__file__).resolve().parents[1]
    script = r"""
import importlib
import importlib.abc
from pathlib import Path
import sys

blocked = {"stable_worldmodel", "stable_pretraining", "jepa", "module"}

class ReferencePackageBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked:
            raise ImportError(f"blocked reference-package import: {fullname}")
        return None

sys.meta_path.insert(0, ReferencePackageBlocker())
entrypoints = (
    "train.cache_latents",
    "train.train_world_model",
    "train.train_gc_idm",
    "train.train_larc",
    "eval.open_loop_curve",
    "eval.closed_loop",
    "eval.summarize",
)
for entrypoint in entrypoints:
    importlib.import_module(entrypoint)
reference_source = (Path.cwd() / "third_party" / "le-wm").resolve()
if any(Path(path).resolve() == reference_source for path in sys.path if path):
    raise RuntimeError("reference source unexpectedly appears on the runtime module path")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_external_lightning_callbacks_are_rejected(monkeypatch):
    candidate = SimpleNamespace(
        name="reference_callbacks",
        value="stable_pretraining.callbacks:factory",
    )

    def fake_entry_points(*, group):
        if group == "lightning.pytorch.callbacks_factory":
            return [candidate]
        return []

    monkeypatch.setattr("train.reproducibility.entry_points", fake_entry_points)
    with pytest.raises(RuntimeError, match="stable_pretraining"):
        reject_external_lightning_callbacks()
