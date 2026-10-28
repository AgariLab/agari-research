#!/usr/bin/env python3
"""Prepare immutable v2 from the independently published complete ranked series.

Default execution is read-only. Completion of 100 games retains every interrupted
attempt and unknown server outcome; the original zero-failure criterion stays
false. No credentials, network, training, model edits, or inference edits occur.
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
WRAPPER_SHA256 = "9c8dd327662276771082a5dd543b7bd4b00d8fa419ce3de64962a808cd10e294"
REGISTRY_ROOT_SHA256 = "564d5f67baaa9dd0c901313ae3c5879b4e81ffb437c1d7e38c5f760a4fa5cc2f"
SCRIPT = Path(__file__).resolve()
EXPERIMENT = SCRIPT.parent
REPO = EXPERIMENT.parents[1]
RANKED_EXPERIMENT = REPO / "experiments/riichi-ranked-100-20261028-v1"
BASE_HELPER = EXPERIMENT / "build_model_release_v2.py"
LEGACY_AUDITOR = RANKED_EXPERIMENT / "audit_ranked.py"
AUDITOR = RANKED_EXPERIMENT / "audit_ranked_series.py"
REGISTRY = RANKED_EXPERIMENT / "series-registry-v1.json"
FINAL_AUDIT = RANKED_EXPERIMENT / "series-audit"
LAUNCHER = REPO.parent / "agari-kit/scripts/run_online_campaign.py"
WRAPPER = LAUNCHER.with_name("run_online_diagnostic.py")
FINAL_FILES = {"report.json", "attempts.json", "source-sha256.json", "SHA256SUMS"}


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


def same_json(left, right):
    # Python equality considers True == 1; audit fields must retain JSON types.
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def artifact(path: Path, base: Path) -> dict:
    return {"path": path.relative_to(base).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256(path)}


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
    return load_module("agari_series_immutable_v2_helper", BASE_HELPER)


def require_complete_series(fresh: dict, prior: dict) -> None:
    require(fresh.get("status") == "ready" and fresh.get("final_acceptance") is False
            and fresh.get("integrity_status") == "verified" and not fresh.get("integrity_errors")
            and not fresh.get("pending_reasons"), "Series audit is not ready with verified integrity")
    require(fresh.get("target_complete_games") == 100 and fresh.get("completion_target_met") is True
            and fresh.get("outcome") == "completed_with_interruptions"
            and fresh.get("original_zero_failure_criterion_met") is False,
            "Series completion cannot claim uninterrupted success")
    totals = fresh["totals"]
    for name in ("attempts", "games_completed", "healthy_completed", "interrupted_attempts",
                 "invalid_attempts", "pending_attempts"):
        require(type(totals.get(name)) is int and totals[name] >= 0, "Invalid series totals")
    require(totals["games_completed"] == totals["healthy_completed"] == 100
            and totals["interrupted_attempts"] >= 2
            and totals["attempts"] == 100 + totals["interrupted_attempts"] == len(fresh["attempts"])
            and totals["interrupted_attempts"] == len(fresh["failures"])
            and totals["invalid_attempts"] == totals["pending_attempts"] == 0,
            "Release requires exactly 100 healthy completed games and all interrupted attempts")
    attempts = fresh["attempts"]
    require(same_json([row.get("global_attempt") for row in attempts], list(range(1, totals["attempts"] + 1))),
            "Series attempts are missing, duplicated, or reordered")
    completed = [row for row in attempts if row.get("status") == "complete"]
    interrupted = [row for row in attempts if row.get("status") == "interrupted"]
    require(len(completed) == 100 and all(row.get("healthy") is True for row in completed)
            and len(interrupted) == totals["interrupted_attempts"]
            and all(row.get("healthy") is False for row in interrupted),
            "Attempt statuses do not match terminal completion totals")
    failure_ids = [row.get("global_attempt") for row in fresh["failures"]]
    require(failure_ids == [row["global_attempt"] for row in interrupted],
            "Failure inventory must retain every interrupted attempt")
    for failure in fresh["failures"]:
        require(failure.get("server_outcome") == failure.get("rating_impact") == "unknown",
                "Unknown server outcome or rating impact was invented")
    for number, requests, started in ((23, 0, False), (30, 115, True)):
        failure = next((row for row in fresh["failures"] if row["global_attempt"] == number), {})
        require(failure.get("failure") == "connection_closed" and type(failure.get("requests")) is int
                and failure["requests"] == requests
                and failure.get("start_game_observed") is started
                and failure.get("end_game_observed") is False and failure.get("scores") is None,
                "Historical interruption was omitted or rewritten")
    variants = fresh["runtime_variants"]
    require(set(variants) == {"uninstrumented", "parent_transport_observer"},
            "Transport runtime variants cannot be merged or invented")
    original, observed = variants["uninstrumented"], variants["parent_transport_observer"]
    require(original.get("attempts") == 30 and original.get("games_completed") == 28
            and original.get("interrupted_attempts") == 2 and original.get("wrapper_sha256") is None,
            "Original runtime history changed")
    require(observed.get("wrapper_sha256") == WRAPPER_SHA256
            and observed.get("games_completed") == 72
            and observed.get("attempts") == totals["attempts"] - 30
            and observed.get("interrupted_attempts") == totals["interrupted_attempts"] - 2,
            "Observed runtime completion or identity differs")
    for name, values in variants.items():
        selected = [row for row in attempts if row.get("runtime_variant") == name]
        require(len(selected) == values["attempts"]
                and sum(row["status"] == "complete" for row in selected) == values["games_completed"]
                and sum(row["status"] == "interrupted" for row in selected) == values["interrupted_attempts"],
                "Runtime grouping differs from actual attempts")
    expected = {"weights": prior["weights"]["sha256"],
                "descriptor": prior["agent"]["descriptor_artifact"]["sha256"],
                "kit": prior["online_validation"]["recorded_runtime"]["agari_kit"]["source_sha256"],
                "research": prior["online_validation"]["recorded_source"]["source_sha256"],
                "engine": prior["online_validation"]["recorded_runtime"]["engine"]["binary_sha256"],
                "launcher": LAUNCHER_SHA256, "kit_files": 19, "research_files": 17}
    require(same_json(fresh["identity_checks"], {"actual": expected, "expected": expected, "matches": True}),
            "Series identities differ from immutable v1")
    require(fresh["registry"].get("root_sha256") == REGISTRY_ROOT_SHA256,
            "Series registry does not retain the frozen historical root")


def verify_published_audit(fresh: dict) -> dict:
    require(FINAL_AUDIT.is_dir() and not FINAL_AUDIT.is_symlink(), "Missing independent series-audit directory")
    require({path.name for path in FINAL_AUDIT.iterdir()} == FINAL_FILES
            and all((FINAL_AUDIT / name).is_file() and not (FINAL_AUDIT / name).is_symlink() for name in FINAL_FILES),
            "Unexpected series-audit files or symlinks")
    checksums = {}
    for line in (FINAL_AUDIT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)", line)
        require(match is not None, "Malformed series-audit checksum")
        digest, name = match.groups()
        require(name not in checksums, "Duplicate series-audit checksum")
        checksums[name] = digest
    require(set(checksums) == FINAL_FILES - {"SHA256SUMS"}, "Incomplete series-audit checksums")
    for name, digest in checksums.items():
        require(sha256(FINAL_AUDIT / name) == digest, "Series-audit file hash differs")
    published = read_json(FINAL_AUDIT / "report.json")
    require(published.get("status") == "published" and published.get("final_acceptance") is True,
            "Series audit was not independently published")
    timestamp = published.get("published_at")
    require(isinstance(timestamp, str)
            and datetime.fromisoformat(timestamp.replace("Z", "+00:00")).tzinfo is not None,
            "Invalid series-audit publication time")
    normalized = copy.deepcopy(published)
    normalized.pop("published_at")
    normalized.update(status="ready", final_acceptance=False)
    require(same_json(normalized, fresh), "Published series audit differs from fresh independent audit")
    require(same_json(read_json(FINAL_AUDIT / "attempts.json"), fresh["attempts"]),
            "Published attempts omit or alter evidence")
    require(same_json(read_json(FINAL_AUDIT / "source-sha256.json"), fresh["source_files"]),
            "Published source inventory differs")
    return published


def verify_source_inventory(fresh: dict, registry_path: Path | None = None) -> None:
    """Re-read files declared by the independently validated fixed registry."""
    inventory = fresh["source_files"]
    require(bool(inventory) and len({row["path"] for row in inventory}) == len(inventory),
            "Empty or duplicate source inventory")
    sources = {}
    for row in inventory:
        path = Path(row["path"])
        require(path.is_absolute() and path.is_file() and not path.is_symlink()
                and type(row.get("bytes")) is int and row["bytes"] >= 0
                and re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", "")) is not None,
                "Invalid source inventory entry")
        require(path.stat().st_size == row["bytes"] and sha256(path) == row["sha256"],
                "Series source bytes changed after audit")
        sources[row["path"]] = row
    raw = fresh["raw_files"]
    require(bool(raw) and len({row["path"] for row in raw}) == len(raw), "Empty or duplicate raw inventory")
    require(all(row["path"] in sources and all(row[key] == sources[row["path"]][key]
                for key in ("path", "bytes", "sha256")) for row in raw),
            "Raw files missing from complete source inventory")
    for path in (AUDITOR, REGISTRY, registry_path or REGISTRY, LEGACY_AUDITOR, LAUNCHER, WRAPPER):
        require(str(path.resolve()) in sources, "Source inventory omitted a registry or runtime verifier")
    chain = fresh["registry_chain"]
    require(bool(chain) and same_json([row.get("revision") for row in chain],
                                     list(range(1, fresh["registry"]["revision"] + 1)))
            and chain[0].get("path") == str(REGISTRY.resolve())
            and chain[0].get("sha256") == REGISTRY_ROOT_SHA256
            and chain[-1].get("path") == fresh["registry"]["path"]
            and chain[-1].get("sha256") == fresh["registry"]["sha256"],
            "Registry ancestor chain differs from the frozen root and selected head")
    for entry in chain:
        require(entry["path"] in sources and sources[entry["path"]]["sha256"] == entry["sha256"],
                "Source inventory omitted a registry ancestor")
        ancestor = read_json(Path(entry["path"]))
        for plan in ancestor["plans"]:
            require(plan["path"] in sources and all(plan[key] == sources[plan["path"]][key]
                        for key in ("path", "bytes", "sha256")),
                    "Source inventory omitted or changed a historical execution plan")


def prepare(storage: Storage, *, registry_path: Path | None = None) -> tuple[dict, dict | None]:
    registry_path = REGISTRY if registry_path is None else Path(registry_path)
    storage.check(min_free_bytes=0)
    generator_sha = sha256(SCRIPT)
    base = load_base_helper()
    prior = base.verify_v1(storage)
    if not AUDITOR.exists() or not registry_path.exists():
        return {"status": "pending", "reason": "series-auditor-or-registry-not-available",
                "immutable_v1_verified": True, "original_zero_failure_criterion_met": False}, None
    for path, expected in ((LEGACY_AUDITOR, LEGACY_AUDITOR_SHA256), (LAUNCHER, LAUNCHER_SHA256),
                           (WRAPPER, WRAPPER_SHA256)):
        require(path.is_file() and not path.is_symlink() and sha256(path) == expected,
                "Frozen auditor or transport launcher changed")
    require(AUDITOR.is_file() and not AUDITOR.is_symlink()
            and registry_path.is_file() and not registry_path.is_symlink(), "Invalid series auditor or registry")
    registry_path = registry_path.resolve()
    auditor_sha, registry_sha = sha256(AUDITOR), sha256(registry_path)
    auditor = load_module("agari_series_release_auditor", AUDITOR)
    fresh = auditor.audit_ranked_series(registry_path=registry_path)
    require(sha256(AUDITOR) == auditor_sha == fresh.get("auditor_sha256")
            and fresh.get("legacy_auditor_sha256") == LEGACY_AUDITOR_SHA256,
            "Series auditor source binding differs")
    require(fresh["registry"].get("path") == str(registry_path)
            and fresh["registry"].get("sha256") == registry_sha == sha256(registry_path),
            "Series audit registry binding differs")
    if fresh.get("status") == "pending":
        require(not fresh.get("integrity_errors")
                and fresh.get("original_zero_failure_criterion_met") is False,
                "Pending series has integrity errors or rewritten failure criterion")
        return {"status": "pending", "reason": "ranked-series-not-complete",
                "immutable_v1_verified": True, "original_zero_failure_criterion_met": False,
                "totals": fresh["totals"], "failures": fresh["failures"],
                "pending_reasons": fresh["pending_reasons"]}, None
    require_complete_series(fresh, prior)
    verify_source_inventory(fresh, registry_path)
    if not FINAL_AUDIT.exists() and not FINAL_AUDIT.is_symlink():
        return {"status": "pending", "reason": "independent-series-audit-not-published",
                "original_zero_failure_criterion_met": False,
                "next_step": "Publish audit_ranked_series.py --write before publishing v2"}, None
    published = verify_published_audit(fresh)
    first_manifest = read_json(storage.path("runs/riichi-ranked-first-v1/manifest.json"))
    require(same_json(first_manifest["rules_expected"], prior["evaluation"]["rules"]),
            "Ranked rules differ from immutable evaluation rules")
    # A second independent pass catches a changed registry, source, or run while
    # the separate audit and inherited release evidence were being checked.
    repeated = auditor.audit_ranked_series(registry_path=registry_path)
    require(same_json(repeated, fresh), "Series evidence changed during release preparation")
    require(same_json(base.verify_v1(storage), prior), "Historical v1 evidence changed during preparation")
    verify_source_inventory(fresh, registry_path)
    require(same_json(verify_published_audit(fresh), published), "Published audit changed during preparation")
    require(sha256(AUDITOR) == auditor_sha and sha256(BASE_HELPER) == BASE_HELPER_SHA256
            and sha256(SCRIPT) == generator_sha, "Release verifier changed during preparation")
    release = copy.deepcopy(prior)
    updates = {
        "created_at": datetime.now(timezone.utc).isoformat(), "release_id": RUN + "/release-v2",
        "status": "completed_with_interruptions", "original_zero_failure_criterion_met": False,
        "scope": "Completed training, 1000 local hanchan, server validation, and exactly 100 healthy completed ranked games with all interrupted attempts and unknown outcomes retained",
        "previous_release": {"release_id": prior["release_id"], "sha256": V1_SHA256,
            "storage": artifact(storage.path(f"models/{RUN}/release-v1.json"), storage.root),
            "repository": artifact(EXPERIMENT / "model-release-v1.json", REPO)},
        "ranked": {"status": "completed_with_interruptions", "completion_target_met": True,
            "original_zero_failure_criterion_met": False, "completed_games": 100,
            "attempts": fresh["totals"]["attempts"], "interrupted_attempts": fresh["totals"]["interrupted_attempts"],
            "rules_expected": first_manifest["rules_expected"],
            # Keep the entire fresh report, including future audit fields. In
            # particular failures, unknown outcomes, all attempts, request-count
            # scopes, batches, and transport variants must never be projected out.
            "series_audit": copy.deepcopy(fresh),
            "series_audit_artifacts": {name: artifact(FINAL_AUDIT / name, REPO) for name in sorted(FINAL_FILES)},
            "auditor": artifact(AUDITOR, REPO), "registry": artifact(registry_path, REPO),
            "audit_published_at": published["published_at"],
            "recorded_inference_identity": {"research_source": prior["online_validation"]["recorded_source"],
                "research_source_commit_matched_by_bytes": prior["online_validation"]["source_commit_matched_by_bytes"],
                "runtime": prior["online_validation"]["recorded_runtime"],
                "weights_sha256": prior["weights"]["sha256"],
                "descriptor_sha256": prior["agent"]["descriptor_artifact"]["sha256"],
                "launcher_sha256": LAUNCHER_SHA256, "observer_sha256": WRAPPER_SHA256,
                "commit_note": "Historical source commits are byte matches; current Git HEAD is not substituted for execution identity"}},
        "verification": {"generator": artifact(SCRIPT, REPO),
            "immutable_v1_sha256": V1_SHA256, "immutable_base_helper": artifact(BASE_HELPER, REPO),
            "legacy_auditor_sha256": LEGACY_AUDITOR_SHA256, "fresh_series_auditor_sha256": auditor_sha,
            "fresh_audit_matches_published_all_fields": True, "fresh_audit_repeated_unchanged": True,
            "source_inventory_rehashed": True, "historical_v1_reverified": True,
            "no_training_or_game_connections": True},
        "release_policy": {"immutable": True, "overwrite_allowed": False, "ranked_campaign_included": True,
            "generator_plan": "ranked-series-v1", "original_strict_v2_generator_preserved": BASE_HELPER_SHA256,
            "supersedes_without_modifying": prior["release_id"],
            "next_version": "Add later evidence only in a new immutable release"},
        "limitations": [
            "Completion and integrity are separate from reliability: the original zero-failure criterion remains false.",
            "Every interrupted attempt is retained; unknown server scores, ranks and rating effects are not invented.",
            "Observed and uninstrumented transport runtimes are separate groups even when model and inference bytes match.",
            "The first two stopped campaigns and all inherited training, evaluation and validation evidence remain immutable.",
            "Ranked completion is not formal strength promotion or the 1000-ranked-game/seven-day leaderboard milestone."]}
    release["inherited_v1_fields"] = {key: copy.deepcopy(prior[key]) for key in updates if key in prior}
    release.update(updates)
    require(all(same_json(release[key], value) or same_json(release["inherited_v1_fields"].get(key), value)
                for key, value in prior.items()), "Historical v1 information was lost")
    return {"status": "ready_with_interruptions", "completed_games": 100,
            "attempts": fresh["totals"]["attempts"], "original_zero_failure_criterion_met": False}, release


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--verify-release", action="store_true")
    parser.add_argument("--registry", type=Path, help="Explicit append-only registry head; defaults to v1")
    args = parser.parse_args(argv)
    created_paths = []
    try:
        storage = Storage.from_env()
        targets = (storage.path(f"models/{RUN}/release-v2.json"), EXPERIMENT / "model-release-v2.json")
        if args.write:
            require(not any(path.exists() or path.is_symlink() for path in targets),
                    "Release v2 already exists and is immutable")
        status, release = prepare(storage, registry_path=args.registry)
        if release is None:
            print(json.dumps(dict(status, release_written=False), ensure_ascii=False))
            return 2 if args.write or args.verify_release else 0
        if args.verify_release:
            require(all(path.is_file() and not path.is_symlink() for path in targets), "Invalid published v2 copies")
            require(targets[0].read_bytes() == targets[1].read_bytes(), "Published v2 copies differ")
            existing = read_json(targets[0])
            release["created_at"] = existing["created_at"]
            require(same_json(existing, release), "Published series v2 differs from fresh evidence")
            status["status"] = "verified-existing"
        elif args.write:
            publish = load_base_helper().publish_new
            content = (json.dumps(release, ensure_ascii=False, indent=2) + "\n").encode()
            for target in targets:
                publish(target, content)
                created_paths.append(str(target))
            status["status"] = "published"
        status.update(release_id=release["release_id"], release_status=release["status"],
                      release_written=args.write, created_paths=created_paths,
                      paths=[str(path) for path in targets],
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
