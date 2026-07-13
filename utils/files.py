"""File identity helpers shared by data, model, and evaluation layers."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def file_sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the lowercase SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: object) -> bool:
    """Return whether a value is a canonical lowercase SHA-256 digest."""

    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def validate_code_revision(value: object) -> str:
    """Return one full lowercase Git commit or reject ambiguous provenance."""

    if not isinstance(value, str) or _GIT_COMMIT_PATTERN.fullmatch(value) is None:
        raise ValueError("code_revision must be a full lowercase Git commit")
    return value
