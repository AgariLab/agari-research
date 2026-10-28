#!/usr/bin/env python3
"""Prepare an immutable PPO release after the independent 100-ranked-game audit.

Use the recorded kit/research runtime, before merging changes to either source
package. Default execution is read-only: an unfinished campaign or unpublished
final audit returns pending and never creates a release. Once 100 real games
pass the separate auditor, run audit_ranked.py --write first, then this script
with --write. --verify-release rechecks an existing release without changing it.

The v1 generator, v1 manifests, model artifacts and inference source are never
modified. No network connections, game execution or credential reads occur.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True

from agari_kit.storage import Storage

RUN = "full-coverage-trace-1000-v1"
V1_SHA256 = "8e5d2467cda1a4f0570f64eceb3a862e253d6bfcd81e235cd31133c1d15a4f1a"
V1_GENERATOR_SHA256 = "0f9e5c1b62b500af4b3482658915363d1bb25f550a9335f709d0fdef65fc4747"
EXPECTED_LAUNCHER_SHA256 = "cd570e8abc1b5b028661f6dfa7881ebdea7c650db11b7d5a0c7fd53adbe4aede"
EXPERIMENT = Path(__file__).resolve().parent
REPO = EXPERIMENT.parents[1]
RANKED_EXPERIMENT = REPO / "experiments/riichi-ranked-100-20261028-v1"
AUDITOR = RANKED_EXPERIMENT / "audit_ranked.py"
FINAL_AUDIT = RANKED_EXPERIMENT / "final-audit"
FINAL_FILES = {"report.json", "games.json", "raw-sha256.json", "SHA256SUMS"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def artifact(path: Path, base: Path) -> dict:
    return {"path": path.relative_to(base).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256(path)}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "Cannot load evidence verifier")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses in the independently maintained auditor need the registered
    # module while their annotations are inspected.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_v1(storage: Storage) -> dict:
    """Bind immutable published bytes, then independently re-read their inputs."""
    external = storage.path(f"models/{RUN}/release-v1.json")
    public = EXPERIMENT / "model-release-v1.json"
    generator = EXPERIMENT / "build_model_release.py"
    require(all(path.is_file() and not path.is_symlink() for path in (external, public, generator)),
            "Immutable v1 input is missing or a symlink")
    require(sha256(external) == sha256(public) == V1_SHA256, "Immutable v1 hash differs")
    require(external.read_bytes() == public.read_bytes(), "Immutable v1 copies differ")
    require(sha256(generator) == V1_GENERATOR_SHA256, "Immutable v1 verifier has changed")
    existing = read_json(public)
    require(existing["verification"]["generator"] == artifact(generator, REPO), "V1 generator binding differs")
    v1 = load_module("agari_release_v1_verifier", generator)
    actual = v1.validate_inputs(storage)
    actual["created_at"] = existing["created_at"]
    require(actual == existing, "V1 differs from independently checked model/run evidence")
    return existing


def verify_published_audit(fresh: dict) -> dict:
    """Require a separate final publication and reproduce every audit field."""
    require(FINAL_AUDIT.is_dir() and not FINAL_AUDIT.is_symlink(), "Final audit directory is invalid")
    require({path.name for path in FINAL_AUDIT.iterdir()} == FINAL_FILES,
            "Final audit must contain exactly its four declared files")
    require(all((FINAL_AUDIT/name).is_file() and not (FINAL_AUDIT/name).is_symlink() for name in FINAL_FILES),
            "Final audit contains a non-file or symlink")
    checksum_rows = (FINAL_AUDIT / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    checksums = {}
    for row in checksum_rows:
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)", row)
        require(match is not None, "Malformed final-audit SHA256SUMS")
        digest, name = match.groups()
        require(name not in checksums, "Duplicate final-audit checksum entry")
        checksums[name] = digest
    require(set(checksums) == FINAL_FILES - {"SHA256SUMS"}, "Incomplete final-audit checksums")
    for name, digest in checksums.items():
        require(sha256(FINAL_AUDIT / name) == digest, f"Final-audit hash differs: {name}")
    published = read_json(FINAL_AUDIT / "report.json")
    require(published.get("status") == "passed" and published.get("final_acceptance") is True,
            "Ranked audit was not separately accepted")
    timestamp = published.get("published_at")
    require(isinstance(timestamp, str), "Missing ranked audit publication time")
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, "Ranked audit publication time has no timezone")
    normalized = copy.deepcopy(published)
    normalized.pop("published_at")
    normalized["status"] = "ready"
    normalized["final_acceptance"] = False
    require(normalized == fresh, "Published audit differs from fresh independent raw-event audit")
    require(read_json(FINAL_AUDIT / "games.json") == fresh["games"], "Published ranked game rows differ")
    require(read_json(FINAL_AUDIT / "raw-sha256.json") == fresh["raw_files"], "Published raw-file inventory differs")
    return published


def prepare(storage: Storage) -> tuple[dict, dict | None]:
    storage.check(min_free_bytes=0)
    prior = verify_v1(storage)
    if not AUDITOR.is_file():
        return {"status": "pending", "reason": "independent-ranked-auditor-not-yet-available",
                "immutable_v1_verified": True}, None
    auditor_hash = sha256(AUDITOR)
    auditor = load_module("agari_ranked_release_auditor", AUDITOR)
    first = storage.path("runs/riichi-ranked-first-v1")
    campaign = storage.path("runs/riichi-ranked-campaign-remaining99-v1")
    # Production defaults inspect actual model/descriptor, package source,
    # native engine and launcher bytes. Never inject test identity overrides.
    fresh = auditor.audit_ranked(first, campaign)
    require(sha256(AUDITOR) == auditor_hash, "Independent auditor changed during verification")
    require(fresh.get("auditor_sha256") == auditor_hash, "Independent audit lacks its source binding")
    if fresh.get("status") == "pending":
        return {"status": "pending", "reason": "ranked-campaign-not-complete",
                "immutable_v1_verified": True, "ranked_totals": fresh.get("totals"),
                "pending_reasons": fresh.get("pending_reasons", []), "errors": fresh.get("errors", [])}, None
    require(fresh.get("status") == "ready" and fresh.get("final_acceptance") is False
            and not fresh.get("errors") and not fresh.get("pending_reasons"),
            "Independent ranked audit is not ready for acceptance")
    require(fresh.get("target_games") == len(fresh["games"]) == 100
            and fresh["totals"]["games_verified_complete"] == 100
            and fresh["totals"]["invalid_games"] == fresh["totals"]["pending_games"] == 0,
            "Release v2 requires exactly 100 audited real ranked games")
    expected_identity = {
        "weights": prior["weights"]["sha256"],
        "descriptor": prior["agent"]["descriptor_artifact"]["sha256"],
        "kit": prior["online_validation"]["recorded_runtime"]["agari_kit"]["source_sha256"],
        "research": prior["online_validation"]["recorded_source"]["source_sha256"],
        "engine": prior["online_validation"]["recorded_runtime"]["engine"]["binary_sha256"],
        "launcher": EXPECTED_LAUNCHER_SHA256, "kit_files": 19, "research_files": 17,
    }
    require(fresh["identity_checks"] == {"actual": expected_identity, "expected": expected_identity, "matches": True},
            "Ranked identities do not match the model and runtime inherited from v1")
    if not FINAL_AUDIT.exists():
        return {"status": "pending", "reason": "independent-final-audit-not-published",
                "immutable_v1_verified": True, "ranked_totals": fresh["totals"],
                "next_step": "Run audit_ranked.py --write before publishing release v2"}, None
    published = verify_published_audit(fresh)
    first_manifest = read_json(first / "manifest.json")
    require(first_manifest["rules_expected"] == prior["evaluation"]["rules"], "Ranked and local evaluation rules differ")
    launcher = REPO.parent / "agari-kit/scripts/run_online_campaign.py"
    require(sha256(launcher) == EXPECTED_LAUNCHER_SHA256, "Ranked campaign launcher differs")
    require(sha256(AUDITOR) == auditor_hash, "Independent auditor changed before release creation")
    release = copy.deepcopy(prior)
    release.update({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "release_id": RUN + "/release-v2",
        "scope": "Completed training, 1000 local hanchan, one server validation East game, and 100 independently audited ranked games",
        "previous_release": {"release_id": prior["release_id"], "sha256": V1_SHA256,
            "storage": artifact(storage.path(f"models/{RUN}/release-v1.json"), storage.root),
            "repository": artifact(EXPERIMENT/"model-release-v1.json", REPO),
            "verification": "Both immutable copies, original verifier and all v1 evidence reverified before ranked acceptance"},
        "ranked": {"games": 100, "status": "accepted", "model_run": RUN,
            "rules_expected": first_manifest["rules_expected"],
            "first_game_run": "riichi-ranked-first-v1",
            "campaign_run": "riichi-ranked-campaign-remaining99-v1",
            "totals": fresh["totals"], "identity_checks": fresh["identity_checks"],
            "statistics_method": fresh["statistics_method"], "audit_limitations": fresh["limitations"],
            "recorded_inference_identity": {
                "research_package_source_sha256": prior["online_validation"]["recorded_source"]["source_sha256"],
                "research_source_commit_matched_by_bytes": prior["online_validation"]["source_commit_matched_by_bytes"],
                "kit": prior["online_validation"]["recorded_runtime"]["agari_kit"],
                "engine": prior["online_validation"]["recorded_runtime"]["engine"],
                "weights_sha256": prior["weights"]["sha256"],
                "descriptor_sha256": prior["agent"]["descriptor_artifact"]["sha256"],
                "launcher": {"path": "agari-kit/scripts/run_online_campaign.py", "bytes": launcher.stat().st_size,
                             "sha256": EXPECTED_LAUNCHER_SHA256},
                "commit_note": "Source commits are byte matches, not fabricated execution Git commits; every game's manifest is independently audited"},
            "final_audit_artifacts": {name: artifact(FINAL_AUDIT/name, REPO) for name in sorted(FINAL_FILES)},
            "auditor": artifact(AUDITOR, REPO), "final_audit_published_at": published["published_at"],
            "acceptance_basis": "Recomputed all actual per-game events and acknowledgements, parent indices and identities; fresh core audit equals separately published acceptance",
            "limitation": "100 ranked games establish initial online engineering acceptance, not the 1000-game/seven-day leaderboard milestone or formal strength promotion"},
        "inherited_verification": prior["verification"],
        "verification": {"generator": artifact(Path(__file__).resolve(), REPO),
            "immutable_v1_sha256": V1_SHA256, "v1_verifier_sha256": V1_GENERATOR_SHA256,
            "fresh_ranked_auditor_sha256": auditor_hash,
            "fresh_audit_matches_published_core": True, "ranked_final_files_checksums_verified": True,
            "no_training_or_game_connections": True,
            "current_runtime_requirement": "Run before merging new inference/kit source, or restore the recorded runtime to verify later"},
        "release_policy": {"immutable": True, "overwrite_allowed": False,
            "ranked_campaign_included": True, "supersedes_without_modifying": prior["release_id"],
            "next_version": "Record later evidence only in a new immutable release"},
        "limitations": [line for line in prior["limitations"]
                        if "running 100-game ranked campaign" not in line] + [
            "Ranked results cover the complete first 100-game campaign, including any failures in the independent audit; no result-based selection is allowed.",
            "Formal 10000-game promotion and 1000-ranked-game/seven-day leaderboard milestones remain outside this release."]})
    require(release["training"] == prior["training"] and release["evaluation"] == prior["evaluation"]
            and release["online_validation"] == prior["online_validation"], "Historical stage evidence changed")
    return {"status": "ready", "immutable_v1_verified": True, "ranked_games": 100}, release


def publish_new(path: Path, content: bytes) -> None:
    """Refuse overwrites on both APFS and the external volume without hard links."""
    require(path.parent.is_dir(), "Release destination parent is missing")
    created = False
    try:
        with path.open("xb") as stream:
            created = True
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if created:
            path.unlink()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--verify-release", action="store_true")
    args = parser.parse_args()
    created_paths = []
    try:
        storage = Storage.from_env()
        destinations = (storage.path(f"models/{RUN}/release-v2.json"), EXPERIMENT/"model-release-v2.json")
        if args.write:
            require(not any(path.exists() for path in destinations), "Release v2 already exists and is immutable")
        status, release = prepare(storage)
        if release is None:
            print(json.dumps(status, ensure_ascii=False))
            return 2 if args.write or args.verify_release else 0
        if args.verify_release:
            require(destinations[0].read_bytes() == destinations[1].read_bytes(), "Published v2 copies differ")
            existing = read_json(destinations[0])
            release["created_at"] = existing["created_at"]
            require(existing == release, "Published v2 differs from independently reverified evidence")
            status["status"] = "verified-existing"
        elif args.write:
            content = (json.dumps(release, indent=2, ensure_ascii=False) + "\n").encode()
            for destination in destinations:
                publish_new(destination, content)
                created_paths.append(str(destination))
            status["status"] = "published"
        status.update({"release_id": release["release_id"],
            "release_written": args.write, "created_paths": created_paths,
            "paths": [str(path) for path in destinations],
            "release_sha256": sha256(destinations[0]) if args.write or args.verify_release else None})
        print(json.dumps(status, ensure_ascii=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "partial" if created_paths else "invalid", "error": str(exc),
            "release_written": bool(created_paths), "created_paths": created_paths}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
