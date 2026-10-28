#!/usr/bin/env python3
"""Publish release v2 only after the separately audited 22 + 78 continuation.

Read-only by default. A complete result means 100 completed games across 101
connection attempts, including the preserved connection_closed interruption.
It never means the original uninterrupted zero-failure criterion was met.
The original v1/v2 generators and manifests remain immutable. No credentials,
network connections, training, source edits or model edits are involved.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True

from agari_kit.storage import Storage

RUN = "full-coverage-trace-1000-v1"
V1_SHA256 = "8e5d2467cda1a4f0570f64eceb3a862e253d6bfcd81e235cd31133c1d15a4f1a"
BASE_HELPER_SHA256 = "5368fbf3b2f14288b52af52df0ccef67caa11f5b6a1fb8a2d8cc9f937bc16ee8"
LEGACY_AUDITOR_SHA256 = "191c396e01574782942059c902ff6ae7ad915c7a68e000cf663f9c43077bbc40"
LAUNCHER_SHA256 = "cd570e8abc1b5b028661f6dfa7881ebdea7c650db11b7d5a0c7fd53adbe4aede"
EXPERIMENT = Path(__file__).resolve().parent
REPO = EXPERIMENT.parents[1]
RANKED_EXPERIMENT = REPO / "experiments/riichi-ranked-100-20261028-v1"
BASE_HELPER = EXPERIMENT / "build_model_release_v2.py"
LEGACY_AUDITOR = RANKED_EXPERIMENT / "audit_ranked.py"
AUDITOR = RANKED_EXPERIMENT / "audit_ranked_continuation.py"
FINAL_AUDIT = RANKED_EXPERIMENT / "continued-audit"
FINAL_FILES = {"report.json", "attempts.json", "raw-sha256.json", "SHA256SUMS"}
FIRST = "riichi-ranked-first-v1"
STOPPED = "riichi-ranked-campaign-remaining99-v1"
CONTINUATION = "riichi-ranked-continuation78-v1"


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
    return {"path": path.relative_to(base).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "Cannot load evidence verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_base_helper():
    require(BASE_HELPER.is_file() and not BASE_HELPER.is_symlink()
            and sha256(BASE_HELPER) == BASE_HELPER_SHA256, "Original v2 helper changed")
    return load_module("agari_release_v2_immutable_helper", BASE_HELPER)


def require_complete_continuation(fresh: dict, prior: dict) -> None:
    """Completion, integrity and the original failure criterion stay separate."""
    require(fresh.get("status") == "ready_with_interruption" and fresh.get("final_acceptance") is False
            and not fresh.get("integrity_errors") and not fresh.get("pending_reasons"), "Continuation audit is not ready")
    require(fresh.get("original_zero_failure_criterion_met") is False,
            "Continuation cannot claim the original zero-failure criterion")
    totals = fresh["totals"]
    require(totals["attempts"] == len(fresh["attempts"]) == 101
            and totals["games_completed"] == totals["healthy_completed"] == 100
            and totals["failed_attempts"] == len(fresh["attempt_failures"]) == 1
            and totals["invalid_attempts"] == totals["pending_attempts"] == 0,
            "Expected 100 complete games and exactly one preserved interruption in 101 attempts")
    require(fresh.get("target_completed_games") == 100 and fresh.get("expected_attempts") == 101
            and fresh["stage_totals"] == {
                "first": {"attempts": 1, "games_completed": 1, "failed_attempts": 0},
                "stopped_batch": {"attempts": 22, "games_completed": 21, "failed_attempts": 1},
                "continuation": {"attempts": 78, "games_completed": 78, "failed_attempts": 0}},
            "Continuation must preserve the 1 + 21 + 78 completed-game split")
    require(len(fresh["raw_files"]) == 410
            and len({row["path"] for row in fresh["raw_files"]}) == 410,
            "Continuation needs all 410 distinct raw artifacts")
    failure = fresh["attempt_failures"][0]
    expected_failure = {"global_attempt": 23, "campaign_game": 22, "failure": "connection_closed",
        "start_game_observed": False, "events_count": 0, "requests": 0, "seat": None, "scores": None,
        "server_outcome": "unknown", "rating_impact": "unknown", "source_stage": "stopped_batch",
        "integrity_verified": True}
    require(all(key in failure and type(failure[key]) is type(value) and failure[key] == value
                for key, value in expected_failure.items()),
            "The interruption is missing, changed, or its server outcome was invented")
    expected_identity = {"weights": prior["weights"]["sha256"],
        "descriptor": prior["agent"]["descriptor_artifact"]["sha256"],
        "kit": prior["online_validation"]["recorded_runtime"]["agari_kit"]["source_sha256"],
        "research": prior["online_validation"]["recorded_source"]["source_sha256"],
        "engine": prior["online_validation"]["recorded_runtime"]["engine"]["binary_sha256"],
        "launcher": LAUNCHER_SHA256, "kit_files": 19, "research_files": 17}
    require(fresh["identity_checks"] == {"actual": expected_identity, "expected": expected_identity, "matches": True},
            "Continuation identities do not match immutable v1")


def verify_published_audit(fresh: dict) -> dict:
    require(FINAL_AUDIT.is_dir() and not FINAL_AUDIT.is_symlink(), "Missing independent continued-audit directory")
    require({path.name for path in FINAL_AUDIT.iterdir()} == FINAL_FILES
            and all((FINAL_AUDIT/name).is_file() and not (FINAL_AUDIT/name).is_symlink() for name in FINAL_FILES),
            "Unexpected continued-audit files or symlinks")
    checksums = {}
    for line in (FINAL_AUDIT/"SHA256SUMS").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)", line)
        require(match is not None, "Malformed continued-audit checksum")
        digest, name = match.groups()
        require(name not in checksums, "Duplicate continued-audit checksum")
        checksums[name] = digest
    require(set(checksums) == FINAL_FILES - {"SHA256SUMS"}, "Incomplete continued-audit checksums")
    for name, digest in checksums.items():
        require(sha256(FINAL_AUDIT/name) == digest, f"Continued-audit hash differs: {name}")
    published = read_json(FINAL_AUDIT/"report.json")
    require(published.get("status") == "completed_with_interruption" and published.get("final_acceptance") is True,
            "Continuation was not separately published with its interruption")
    timestamp = published.get("published_at")
    require(isinstance(timestamp, str)
            and datetime.fromisoformat(timestamp.replace("Z", "+00:00")).tzinfo is not None,
            "Invalid continuation publication time")
    normalized = copy.deepcopy(published)
    normalized.pop("published_at")
    normalized.update(status="ready_with_interruption", final_acceptance=False)
    require(normalized == fresh, "Published continuation differs from fresh independent audit")
    require(read_json(FINAL_AUDIT/"attempts.json") == fresh["attempts"], "Published attempts omit or alter evidence")
    require(read_json(FINAL_AUDIT/"raw-sha256.json") == fresh["raw_files"], "Published raw inventory differs")
    return published


def verify_source_split(storage: Storage) -> dict:
    """Directly verify the stopped batch and its unknown-outcome interruption."""
    stopped = storage.path("runs/" + STOPPED)
    continuation = storage.path("runs/" + CONTINUATION)
    old_manifest, old_summary = read_json(stopped/"manifest.json"), read_json(stopped/"summary.json")
    new_manifest, new_summary = read_json(continuation/"manifest.json"), read_json(continuation/"summary.json")
    require(old_manifest["status"] == old_summary["status"] == "stopped"
            and old_manifest["games_requested"] == old_summary["games_requested"] == 99
            and old_manifest["games_recorded"] == old_summary["games_attempted"] == 22
            and old_summary["games_completed"] == old_summary["healthy_games"] == 21
            and old_manifest["error"] == old_summary["error"] == "game_incomplete",
            "Original 99-game batch must remain recorded as stopped after 21 completed games")
    require(new_manifest["status"] == new_summary["status"] == "complete"
            and new_manifest["games_requested"] == new_summary["games_requested"] == 78
            and new_manifest["games_recorded"] == new_summary["games_attempted"]
            == new_summary["games_completed"] == new_summary["healthy_games"] == 78
            and new_manifest["error"] is None and new_summary["error"] is None,
            "The continuation must finish all 78 games without another interruption")
    failed = stopped/"game-000022"
    require((failed/"events.jsonl").read_bytes() == b"", "The preserved failure event stream changed")
    game_rows = [json.loads(line) for line in (failed/"games.jsonl").read_text().splitlines()]
    require(game_rows == [{"mode": "ranked", "status": "interrupted", "seat": None, "scores": None,
        "validation_passed": None, "degraded": False, "counts": {}, "failure": "connection_closed", "game_index": 0}],
        "The recorded failure result changed")
    report, manifest = read_json(failed/"report.json"), read_json(failed/"manifest.json")
    require(report == {"games_requested": 1, "games_completed": 0, "validation_passed": None,
                       "degraded_games": 0, "counts": {}}
            and manifest["status"] == "incomplete" and manifest["games_recorded"] == 1,
            "The failed attempt was relabeled or completed")
    return {"completed_before_continuation": 22, "completed_in_continuation": 78,
        "stages": [
            {"run": FIRST, "attempts": 1, "completed": 1, "failed": 0, "status": "complete"},
            {"run": STOPPED, "requested": 99, "attempts": 22, "completed": 21, "failed": 1,
             "status": "stopped", "error": "game_incomplete"},
            {"run": CONTINUATION, "requested": 78, "attempts": 78, "completed": 78, "failed": 0, "status": "complete"}],
        "failure_raw_files": {name: artifact(failed/name, storage.root)
            for name in ("manifest.json", "report.json", "games.jsonl", "events.jsonl")},
        "failure_outcome": {"locally_observed_start_game": False, "events_count": 0, "requests": 0,
            "seat": None, "scores": None, "server_outcome": "unknown", "rating_impact": "unknown"}}


def prepare(storage: Storage) -> tuple[dict, dict | None]:
    storage.check(min_free_bytes=0)
    base = load_base_helper()
    prior = base.verify_v1(storage)
    if not AUDITOR.is_file():
        return {"status": "pending", "reason": "continuation-auditor-not-yet-available",
                "immutable_v1_verified": True, "original_zero_failure_criterion_met": False}, None
    require(LEGACY_AUDITOR.is_file() and not LEGACY_AUDITOR.is_symlink()
            and sha256(LEGACY_AUDITOR) == LEGACY_AUDITOR_SHA256, "Original strict auditor changed")
    auditor_hash = sha256(AUDITOR)
    auditor = load_module("agari_continued_release_auditor", AUDITOR)
    fresh = auditor.audit_ranked_continuation(storage.path("runs/"+FIRST), storage.path("runs/"+STOPPED),
                                            storage.path("runs/"+CONTINUATION))
    require(sha256(AUDITOR) == auditor_hash == fresh.get("auditor_sha256")
            and fresh.get("legacy_auditor_sha256") == LEGACY_AUDITOR_SHA256, "Continuation auditor source binding differs")
    require(fresh.get("inputs") == {"first_game": str(storage.path("runs/"+FIRST).resolve()),
            "stopped_campaign": str(storage.path("runs/"+STOPPED).resolve()),
            "continuation": str(storage.path("runs/"+CONTINUATION).resolve())}, "Continuation audit inputs differ")
    if fresh.get("status") == "pending":
        return {"status": "pending", "reason": "continuation-not-complete", "immutable_v1_verified": True,
            "original_zero_failure_criterion_met": False, "totals": fresh.get("totals"),
            "attempt_failures": fresh.get("attempt_failures"), "integrity_errors": fresh.get("integrity_errors", [])}, None
    require_complete_continuation(fresh, prior)
    if not FINAL_AUDIT.exists():
        return {"status": "pending", "reason": "independent-continued-audit-not-published",
                "original_zero_failure_criterion_met": False,
                "next_step": "Run audit_ranked_continuation.py --write before publishing v2"}, None
    published = verify_published_audit(fresh)
    split = verify_source_split(storage)
    first_manifest = read_json(storage.path("runs/"+FIRST+"/manifest.json"))
    require(first_manifest["rules_expected"] == prior["evaluation"]["rules"], "Ranked rules differ from fixed half-game evaluation")
    launcher = REPO.parent/"agari-kit/scripts/run_online_campaign.py"
    require(sha256(launcher) == LAUNCHER_SHA256, "Continuation launcher changed")
    release = copy.deepcopy(prior)
    release.update({
        "created_at": datetime.now(timezone.utc).isoformat(), "release_id": RUN+"/release-v2",
        "status": "completed_with_interruption", "original_zero_failure_criterion_met": False,
        "scope": "Completed training, 1000 local hanchan, server validation, and 100 completed ranked games across 101 attempts with one preserved unknown-outcome connection interruption",
        "previous_release": {"release_id": prior["release_id"], "sha256": V1_SHA256,
            "storage": artifact(storage.path(f"models/{RUN}/release-v1.json"), storage.root),
            "repository": artifact(EXPERIMENT/"model-release-v1.json", REPO)},
        "ranked": {"status": "completed_with_interruption", "completion_target_met": True,
            "original_zero_failure_criterion_met": False, "original_99_game_batch_completed": False,
            "completed_games": 100, "attempts": 101, "failed_attempts": 1,
            "rules_expected": first_manifest["rules_expected"], "source_split": split,
            "attempt_failures": fresh["attempt_failures"], "stage_totals": fresh["stage_totals"],
            "totals": fresh["totals"], "identity_checks": fresh["identity_checks"],
            "statistics_method": fresh["statistics_method"], "audit_limitations": fresh["limitations"],
            "recorded_inference_identity": {"research_source": prior["online_validation"]["recorded_source"],
                "research_source_commit_matched_by_bytes": prior["online_validation"]["source_commit_matched_by_bytes"],
                "runtime": prior["online_validation"]["recorded_runtime"],
                "weights_sha256": prior["weights"]["sha256"], "descriptor_sha256": prior["agent"]["descriptor_artifact"]["sha256"],
                "launcher_sha256": LAUNCHER_SHA256,
                "commit_note": "Source commits are byte matches, not fabricated execution Git commits"},
            "continued_audit_artifacts": {name: artifact(FINAL_AUDIT/name, REPO) for name in sorted(FINAL_FILES)},
            "auditor": artifact(AUDITOR, REPO), "legacy_strict_auditor": artifact(LEGACY_AUDITOR, REPO),
            "audit_published_at": published["published_at"],
            "acceptance_basis": "Fresh independent continuation audit matches all published fields; stopped batch and failure are retained, not relabeled as uninterrupted success"},
        "inherited_verification": prior["verification"],
        "verification": {"generator": artifact(Path(__file__).resolve(), REPO),
            "immutable_v1_sha256": V1_SHA256, "immutable_base_helper": artifact(BASE_HELPER, REPO),
            "fresh_continuation_auditor_sha256": auditor_hash, "legacy_auditor_sha256": LEGACY_AUDITOR_SHA256,
            "fresh_audit_matches_published_core": True, "all_410_raw_files_in_independent_inventory": True,
            "direct_failure_and_stage_split_verified": True, "no_training_or_game_connections": True,
            "runtime_requirement": "Verify before merging kit/research source changes, or restore the recorded runtime later"},
        "release_policy": {"immutable": True, "overwrite_allowed": False, "ranked_campaign_included": True,
            "generator_plan": "continued-ranked-campaign-v1", "original_strict_v2_generator_preserved": BASE_HELPER_SHA256,
            "supersedes_without_modifying": prior["release_id"], "next_version": "Add later evidence only in a new immutable release"},
        "limitations": [line for line in prior["limitations"] if "running 100-game ranked campaign" not in line] + [
            "The original 99-game campaign stopped after 21 complete games and one connection_closed attempt; it was not completed or rewritten.",
            "The interrupted attempt has no locally observed start event, requests, seat or score; server game outcome and rating impact remain unknown.",
            "Completion of 22 prior plus 78 continued games meets the 100-completed-game target with an interruption; the original zero-failure criterion remains false.",
            "Results summarize the 100 known completed games while preserving all 101 attempts; unknown outcomes are not assigned a rank or score.",
            "Formal 10000-game promotion and 1000-ranked-game/seven-day leaderboard milestones remain outside this release."]})
    require(all(release[key] == prior[key] for key in ("training", "evaluation", "online_validation")), "Historical stage evidence changed")
    require(sha256(AUDITOR) == auditor_hash and sha256(BASE_HELPER) == BASE_HELPER_SHA256,
            "A release verifier changed during verification")
    return {"status": "ready_with_interruption", "completed_games": 100, "attempts": 101,
            "original_zero_failure_criterion_met": False}, release


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--verify-release", action="store_true")
    args = parser.parse_args()
    created_paths = []
    try:
        storage = Storage.from_env()
        targets = (storage.path(f"models/{RUN}/release-v2.json"), EXPERIMENT/"model-release-v2.json")
        if args.write:
            require(not any(path.exists() for path in targets), "Release v2 already exists and is immutable")
        status, release = prepare(storage)
        if release is None:
            print(json.dumps(dict(status, release_written=False), ensure_ascii=False))
            return 2 if args.write or args.verify_release else 0
        if args.verify_release:
            require(targets[0].read_bytes() == targets[1].read_bytes(), "Published v2 copies differ")
            existing = read_json(targets[0])
            release["created_at"] = existing["created_at"]
            require(existing == release, "Published continued v2 differs from fresh evidence")
            status["status"] = "verified-existing"
        elif args.write:
            publish = load_base_helper().publish_new
            content = (json.dumps(release, ensure_ascii=False, indent=2)+"\n").encode()
            for target in targets:
                publish(target, content)
                created_paths.append(str(target))
            status["status"] = "published"
        status.update(release_id=release["release_id"], release_status=release["status"],
            release_written=args.write, created_paths=created_paths, paths=[str(path) for path in targets],
            release_sha256=sha256(targets[0]) if args.write or args.verify_release else None)
        print(json.dumps(status, ensure_ascii=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "partial" if created_paths else "invalid", "error": str(exc),
            "release_written": bool(created_paths), "created_paths": created_paths,
            "original_zero_failure_criterion_met": False}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
