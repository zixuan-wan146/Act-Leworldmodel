"""Evaluation code and file provenance validation."""

from __future__ import annotations

from pathlib import Path

from utils import file_sha256, is_sha256


def artifact_record(path: str | Path) -> dict[str, str]:
    """Record the absolute location and SHA-256 of one evaluated file."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"evaluation artifact does not exist: {resolved}")
    return {"path": str(resolved), "sha256": file_sha256(resolved)}


def validate_artifact_record(
    value: object,
    *,
    name: str,
    verify_file: bool,
) -> None:
    """Validate one result artifact record and optionally its current bytes."""

    if not isinstance(value, dict):
        raise ValueError(f"evaluation artifact {name} must be a mapping")
    path_value = value.get("path")
    digest = value.get("sha256")
    if not isinstance(path_value, str) or not Path(path_value).is_absolute():
        raise ValueError(f"evaluation artifact {name} must use an absolute path")
    if not is_sha256(digest):
        raise ValueError(f"evaluation artifact {name} must declare a valid SHA-256")
    if verify_file:
        path = Path(path_value)
        if not path.is_file():
            raise ValueError(f"evaluation artifact {name} is missing")
        if file_sha256(path) != digest:
            raise ValueError(f"evaluation artifact {name} content changed")


def validate_artifact_records(records: object, *, verify_files: bool) -> None:
    """Validate a non-empty named collection of evaluation file records."""

    if not isinstance(records, dict) or not records:
        raise ValueError("evaluation artifacts must be a non-empty mapping")
    for name, record in records.items():
        if not isinstance(name, str) or not name:
            raise ValueError("evaluation artifact names must be non-empty strings")
        validate_artifact_record(record, name=name, verify_file=verify_files)
