"""Bind resumable training to installed runtime bytes and the reviewed lockfile."""

from __future__ import annotations

import platform
from importlib.metadata import version
from pathlib import Path

import numpy
import torch
from agari_kit.provenance import source_fingerprint
from riichienv import _riichienv

from .manifest import file_sha256


def runtime_provenance() -> dict:
    lockfile = Path(__file__).resolve().parents[2] / "uv.lock"
    if not lockfile.is_file():
        raise ValueError("Reproducible training requires a source checkout containing uv.lock")
    return {
        "schema_version": 1,
        "agari_kit": source_fingerprint(),
        "engine": {"version": version("riichienv"), "binary_sha256": file_sha256(Path(_riichienv.__file__))},
        "python": {"implementation": platform.python_implementation(), "version": platform.python_version()},
        "torch": str(torch.__version__), "numpy": str(numpy.__version__),
        "lockfile_sha256": file_sha256(lockfile),
    }


def require_matching_runtime(recorded: dict | None, current: dict) -> None:
    if not isinstance(recorded, dict):
        raise ValueError("Resume runtime provenance mismatch: missing or invalid metadata")
    if recorded != current:
        keys = sorted(key for key in set(current) | set(recorded)
                      if recorded.get(key) != current.get(key))
        raise ValueError("Resume runtime provenance mismatch: " + ", ".join(keys))
