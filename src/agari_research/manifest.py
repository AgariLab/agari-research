"""Portable model provenance independent of the model implementation."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_manifest(*, weights: Path, code_revision: str, architecture: str,
                   encoding_version: str, dataset_sha256: str, evaluation_report: Path,
                   upstream_revision: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", code_revision):
        raise ValueError("Record the full code commit, not a branch name")
    if not re.fullmatch(r"[0-9a-f]{40}", upstream_revision):
        raise ValueError("Record the verified upstream commit, not a mutable reference")
    if not re.fullmatch(r"[0-9a-f]{64}", dataset_sha256):
        raise ValueError("Record the frozen dataset SHA-256")
    if not architecture.strip() or not encoding_version.strip():
        raise ValueError("Architecture and encoding version are required")
    report = json.loads(evaluation_report.read_text())
    if not isinstance(report.get("games"), int) or report["games"] <= 0:
        raise ValueError("Evaluation report must contain completed games")
    return {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": code_revision, "upstream_revision": upstream_revision,
        "architecture": architecture, "encoding_version": encoding_version,
        "dataset_sha256": dataset_sha256,
        "rules": {"mode": "4p-red-half", "preset": "tenhou", "riichienv": "0.4.10"},
        "weights": {"filename": weights.name, "sha256": file_sha256(weights), "bytes": weights.stat().st_size},
        "evaluation": {"filename": evaluation_report.name, "sha256": file_sha256(evaluation_report),
                       "games": report["games"]},
    }
