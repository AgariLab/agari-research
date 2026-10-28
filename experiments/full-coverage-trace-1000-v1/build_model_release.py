#!/usr/bin/env python3
"""Verify existing PPO evidence and publish an immutable release-v1 manifest.

Run with the locked research environment and PYTHONDONTWRITEBYTECODE=1.
Default: verify inputs and print a short result, without writing. --write
creates two identical manifests only when neither destination exists.
--verify-release checks the already published manifests against current bytes.
No training, game connections, token access, or inference-source edits occur.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

from agari_kit.mjai import canonical_json
from agari_kit.storage import Storage
from agari_research.manifest import file_sha256, model_manifest

RUN = "full-coverage-trace-1000-v1"
TRAIN_COMMIT = "761b4522b5343b01413d640b58444b8a3fb7568e"
EVAL_COMMIT = "723b81090944ecc430c2be331ba1cba364ab41b6"
UPSTREAM_COMMIT = "479c1faeb33d082965eef8198f63261a79c0fce3"
EXPECTED_WEIGHTS = "613d6a20cf1057a9c6556153a5506fdc11d115f62aaf5de83e39143572bd9096"
EXPECTED_IMPLEMENTATION = "2f6e3618a1fa7812837b042e257d654c6e4d94f95764904002dbfef0bb9afedf"
EXPECTED_EVAL_SOURCE = "44bb7f08633036823aef7e61c0f7351fd9886ede798ec5ec2cd9b389835771cc"
EXPECTED_VALIDATE_SOURCE = "43eabb39e1518ec550f8692bce79224060f07aa280f12a254b990ccc2c3e7e21"
EXPERIMENT = Path(__file__).resolve().parent
REPO = EXPERIMENT.parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def artifact(path: Path, root: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
            "sha256": file_sha256(path)}


def source_at(commit: str) -> dict:
    """Recreate both historical digest algorithms from immutable local Git blobs."""
    prefix = "src/agari_research/"
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit, prefix], cwd=REPO, text=True
    ).splitlines()
    names = sorted(name for name in names if name.endswith(".py"))
    implementation, package = hashlib.sha256(), hashlib.sha256()
    files = {}
    for name in names:
        body = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=REPO)
        relative = name.removeprefix(prefix)
        implementation.update(relative.encode() + b"\0" + body)
        package.update(("0/" + relative).encode() + b"\0" + body + b"\0")
        files[relative] = hashlib.sha256(body).hexdigest()
    require(bool(files), f"Missing Python source at {commit}")
    return {"commit": commit, "python_files": len(files),
            "training_implementation_sha256": implementation.hexdigest(),
            "agent_package_source_sha256": package.hexdigest(), "file_sha256": files}


def validate_inputs(storage: Storage) -> dict:
    import torch
    from agari_kit.plugins import agent_provenance
    from agari_research.models.actor_critic import ActorCriticNetwork
    from agari_research.runtime import runtime_provenance

    root = storage.root
    storage.check(min_free_bytes=0)
    model_dir = storage.path(f"models/{RUN}")
    training_dir = storage.path(f"runs/{RUN}")
    offline_dir = storage.path("runs/trace-1000-candidate-eval-v1")
    online_dir = storage.path("runs/riichi-validation-v1")
    weights = model_dir / "ppo.pt"
    metadata = read_json(model_dir / "ppo.json")
    descriptor_path = model_dir / "ppo-agent.json"
    descriptor = read_json(descriptor_path)
    require(file_sha256(weights) == EXPECTED_WEIGHTS == metadata["weights_sha256"], "Wrong PPO weights")
    payload = torch.load(weights, map_location="cpu", weights_only=True)
    payload_metadata = {key: value for key, value in payload.items() if key != "state_dict"}
    require(dict(payload_metadata, filename="ppo.pt", weights_sha256=EXPECTED_WEIGHTS) == metadata,
            "Export JSON differs from actual weights payload")
    require(payload["kind"] == "ppo" and payload["completed_updates"] == 4, "Incomplete PPO export")
    require(metadata["model_config"] == metadata["config"]["model"], "Architecture configuration differs")
    require(metadata["sample_semantics_version"] == "seeded-trace-player-decisions-v1", "Wrong action labels")
    require(metadata["upstream_commit"] == UPSTREAM_COMMIT, "Wrong upstream revision")
    require(descriptor["factory"] == "agari_research.model_agent:ModelAgent", "Wrong Agent factory")
    require(descriptor["artifacts"]["weights_path"] == {"path": "ppo.pt", "sha256": EXPECTED_WEIGHTS},
            "Descriptor weight binding differs")
    require(descriptor["kwargs"] == {"device": "cpu", "torch_threads": 1}, "Unexpected inference configuration")
    with torch.random.fork_rng():
        model = ActorCriticNetwork(**metadata["model_config"])
    model.load_state_dict(payload["state_dict"], strict=True)
    require(all(torch.isfinite(value).all().item() for value in payload["state_dict"].values()),
            "Nonfinite model parameter or buffer")
    tensor_schema = {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                     for name, value in sorted(payload["state_dict"].items())}
    architecture = {"class": "agari_research.models.actor_critic:ActorCriticNetwork",
                    "parameters": metadata["model_config"],
                    "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "state_dict_tensors": len(tensor_schema),
                    "state_dict_schema_sha256": hashlib.sha256(canonical_json(tensor_schema).encode()).hexdigest(),
                    "strict_load_verified": True, "all_tensors_finite": True}

    train_source, eval_source = source_at(TRAIN_COMMIT), source_at(EVAL_COMMIT)
    require(train_source["training_implementation_sha256"] == EXPECTED_IMPLEMENTATION
            == metadata["implementation_sha256"], "Training commit does not reproduce implementation digest")
    require(eval_source["agent_package_source_sha256"] == EXPECTED_EVAL_SOURCE, "Wrong evaluation snapshot")
    require(train_source["agent_package_source_sha256"] == EXPECTED_VALIDATE_SOURCE, "Wrong validation package")
    current_runtime = runtime_provenance()
    require(current_runtime == metadata["runtime_provenance"], "Current verification runtime differs from export")
    current_agent = agent_provenance("@" + str(descriptor_path))
    require(current_agent["source"]["source_sha256"] == EXPECTED_VALIDATE_SOURCE, "Live package source changed")

    inventory = read_json(EXPERIMENT / "training/artifacts.json")
    for entry in inventory["artifacts"]:
        require(artifact(storage.path(entry["path"]), root) == entry,
                f"Training evidence changed: {entry['path']}")
    result = read_json(training_dir / "result.json")
    acceptance = read_json(EXPERIMENT / "training/acceptance.json")
    require(result["status"] == "complete" and result["progress"] == {"grp": 1133, "cql": 8217, "ppo": 4},
            "Training result is incomplete")
    require(acceptance["research_commit"] == TRAIN_COMMIT and all(acceptance["checks"].values()),
            "Historical training acceptance is incomplete")
    for key in ("dataset_sha256", "implementation_sha256", "training_plan_sha256", "training_coverage"):
        require(result[key] == metadata[key], f"Training result/export disagree: {key}")
    split_path = storage.path("datasets/manifests/selfplay-v1/split-v1.json")
    split = read_json(split_path)
    body = {key: value for key, value in split.items() if key != "dataset_sha256"}
    dataset_digest = hashlib.sha256(canonical_json(body).encode()).hexdigest()
    require(dataset_digest == split["dataset_sha256"] == metadata["dataset_sha256"], "Dataset canonical digest differs")
    partitions = split["partitions"]
    require({key: len(value) for key, value in partitions.items()} == {"train": 900, "validation": 50, "test": 50},
            "Unexpected split sizes")
    ordered = [row for part in ("train", "validation", "test") for row in partitions[part]]
    require({row["source"] for row in ordered} == {"agarilab-selfplay"}
            and {row["rules"] for row in ordered} == {"4p-red-half/tenhou"}
            and {row["validation"] for row in ordered} == {"engine-verified"}, "Unexpected dataset provenance or rules")
    require(len({row["game_id"] for row in ordered}) == len({row["sha256"] for row in ordered}) == 1000,
            "Dataset game ID or content overlap")
    require(all(a["played_at"] <= b["played_at"] for a, b in zip(ordered, ordered[1:])), "Split is not chronological")

    offline_manifest = read_json(offline_dir / "manifest.json")
    offline_report = read_json(offline_dir / "report.json")
    games = read_rows(offline_dir / "games.jsonl")
    require(offline_manifest["status"] == "complete" and offline_manifest["games_completed"] == len(games) == 1000,
            "Offline evaluation is incomplete")
    require(offline_manifest["seeds"] == list(range(600000, 600250)), "Unexpected evaluation seeds")
    require(Counter((row["seed"], row["rotation"]) for row in games)
            == Counter((seed, rotation) for seed in offline_manifest["seeds"] for rotation in range(4)),
            "Missing, duplicated or unrotated evaluation game")
    require(offline_manifest["lineup"][1:] == ["efficiency", "random", "tsumogiri"], "Opponent pool changed")
    for row in games:
        require(row["seats"] == [offline_manifest["lineup"][(seat-row["rotation"]) % 4] for seat in range(4)],
                "Incorrect seat rotation")
        order = sorted(range(4), key=lambda seat: (-row["scores"][seat], seat))
        require(row["ranks"] == [order.index(seat)+1 for seat in range(4)], "Stored ranks disagree with scores")
        require(row["illegal_actions"] == 0 and row["event_counts"]["end_game"] == 1, "Unhealthy offline game")
    rank_values = [row["ranks"][row["rotation"]] for row in games]
    recomputed = {"mean_rank": statistics.mean(rank_values),
                  "rank_rates": [rank_values.count(rank)/1000 for rank in range(1, 5)],
                  "mean_point_delta": statistics.mean(row["scores"][row["rotation"]]-25000 for row in games)}
    require(recomputed == offline_report["candidate"], "Offline outcome summary differs from game rows")
    events = Counter()
    for row in games:
        events.update(row["event_counts"])
    require(dict(events) == offline_report["event_counts"], "Offline event totals differ")
    for key in ("decisions", "candidate_decisions", "illegal_actions"):
        require(sum(row[key] for row in games) == offline_report[key], f"Offline total differs: {key}")
    require(offline_report["games"] == 1000 and offline_report["seed_groups"] == 250
            and offline_report["promotion_sample_size_met"] is False, "Wrong offline evaluation scope")
    recorded_agent = offline_manifest["lineup_provenance"][0]
    require(recorded_agent["source"]["source_sha256"] == EXPECTED_EVAL_SOURCE, "Offline inference source differs")

    online_manifest = read_json(online_dir / "manifest.json")
    online_report = read_json(online_dir / "report.json")
    online_games, online_events = read_rows(online_dir / "games.jsonl"), read_rows(online_dir / "events.jsonl")
    require(online_manifest["status"] == "complete" and online_manifest["mode"] == "validate", "Not completed validation")
    require(online_report["games_requested"] == online_report["games_completed"] == len(online_games) == 1,
            "Validation game count differs")
    require(online_report["validation_passed"] is True and online_report["degraded_games"] == 0, "Validation failed")
    event_counts = Counter(row["event"] for row in online_events)
    require(event_counts == Counter(start_game=1, decision=120, action_ack=120, end_game=1, validation_result=1),
            "Unexpected validation events")
    decisions = [row for row in online_events if row["event"] == "decision"]
    acks = [row for row in online_events if row["event"] == "action_ack"]
    ids = [row["request_id"] for row in decisions]
    require(len(set(ids)) == 120 and ids == [row["request_id"] for row in acks], "Validation request/ack mismatch")
    require(all(row["status"] == "accepted" for row in acks), "Nonaccepted validation action")
    require(online_events[-1]["event"] == "validation_result" and online_events[-1]["passed"] is True,
            "Missing final server verdict")
    require(online_report["counts"] == {"requests": 120, "replies_prepared": 120, "replies_sent": 120, "ack_accepted": 120},
            "Validation counters differ")
    game = online_games[0]
    require(game["status"] == "complete" and game["mode"] == "validate" and game["validation_passed"] is True
            and game["degraded"] is False and game["counts"] == online_report["counts"], "Invalid validation game row")
    require(next(row for row in online_events if row["event"] == "end_game")["scores"] == game["scores"],
            "Validation terminal scores differ")
    for provenance in (recorded_agent, online_manifest["agent_provenance"]):
        require(provenance["descriptor_sha256"] == file_sha256(descriptor_path)
                and provenance["descriptor"] == descriptor
                and provenance["artifacts"]["weights_path"]["sha256"] == EXPECTED_WEIGHTS,
                "An evaluated Agent used different artifacts")
    require(online_manifest["agent_provenance"]["source"]["source_sha256"] == EXPECTED_VALIDATE_SOURCE,
            "Validation source differs")
    for manifest in (offline_manifest, online_manifest):
        require(manifest["engine"] == current_runtime["engine"] and manifest["code"] == current_runtime["agari_kit"],
                "Recorded engine/kit identity differs")
    require(offline_manifest["rules"] == {"game_mode": "4p-red-half", "preset": "tenhou", "engine_version": "0.4.10"},
            "Offline rules differ")
    require(online_manifest["rules_expected"] == {"game_mode": "4p-red-east", "preset": "tenhou", "engine_version": "0.4.10"},
            "Validation rules differ")
    public_artifacts = {}
    for label, directory, public in (
            ("offline", offline_dir, EXPERIMENT / "evaluation"),
            ("validation", online_dir, REPO / "experiments/riichi-validation-20261028-v1")):
        names = ["manifest.json", "games.jsonl", "report.json"] + (["events.jsonl"] if label == "validation" else [])
        public_artifacts[label] = {}
        for name in names:
            require((directory/name).read_bytes() == (public/name).read_bytes(), f"Public {label} evidence copy differs: {name}")
            public_artifacts[label][name] = {"storage": artifact(directory/name, root), "repository": artifact(public/name, REPO)}

    release = model_manifest(weights=weights, code_revision=TRAIN_COMMIT,
        architecture="ActorCriticNetwork: Conv1d ResNet + channel attention, actor82 and scalar critic",
        encoding_version=metadata["encoding_version"], dataset_sha256=metadata["dataset_sha256"],
        evaluation_report=offline_dir/"report.json", upstream_revision=UPSTREAM_COMMIT)
    release.update({
        "release_id": RUN + "/release-v1", "model_run": RUN, "kind": "ppo",
        "scope": "Completed training, 1000 local hanchan, and one server validation East game only",
        "artifact_roots": {"storage": str(root), "repository": "https://github.com/AgariLab/agari-research"},
        "architecture_details": architecture, "sample_semantics_version": metadata["sample_semantics_version"],
        "weights": dict(release["weights"], artifact=artifact(weights, root)),
        "agent": {"descriptor": descriptor, "descriptor_artifact": artifact(descriptor_path, root),
                  "export_metadata_artifact": artifact(model_dir/"ppo.json", root),
                  "payload_metadata_equals_export_json": True},
        "dataset": {"name": "selfplay-v1", "canonical_sha256": dataset_digest,
                    "split_artifact": artifact(split_path, root),
                    "partitions": {key: len(value) for key, value in partitions.items()},
                    "whole_game_id_and_hash_disjoint": True, "chronological": True,
                    "test_content_opened_by_this_verification": False,
                    "source": "agarilab-selfplay", "rules": "4p-red-half/tenhou",
                    "usage_basis": sorted({row["usage_basis"] for row in ordered}),
                    "behavior_limit": "Four efficiency producers; no chi/pon/kan events"},
        "training": {"run": RUN, "source": train_source, "runtime_provenance": metadata["runtime_provenance"],
                     "training_plan_sha256": metadata["training_plan_sha256"], "progress": result["progress"],
                     "coverage": metadata["training_coverage"],
                     "artifacts": {name: artifact(training_dir/name, root) for name in
                         ("config.json", "dataset.json", "execution-plan.json", "result.json", "coverage.json", "validation.json", "metrics.jsonl")},
                     "checkpoint": artifact(storage.path(f"checkpoints/{RUN}/latest.pt"), root),
                     "historical_acceptance": artifact(EXPERIMENT/"training/acceptance.json", REPO),
                     "historical_artifact_inventory": artifact(EXPERIMENT/"training/artifacts.json", REPO),
                     "historical_inventory_entries_verified": len(inventory["artifacts"])},
        "evaluation": dict(release["evaluation"], run="trace-1000-candidate-eval-v1",
            rules=offline_manifest["rules"], opponents=offline_manifest["lineup"][1:],
            seeds={"first": 600000, "last": 600249, "groups": 250, "rotations_per_seed": 4},
            inference_source=eval_source, recorded_source=recorded_agent["source"],
            recorded_runtime={"engine": offline_manifest["engine"], "agari_kit": offline_manifest["code"],
                              "python": offline_manifest["python"]},
            artifacts=public_artifacts["offline"], report=offline_report,
            verification={"complete_unique_seed_rotations": True, "outcomes_and_event_totals_recomputed": True},
            limitation="Fixed weak opponent pool and previously used screening seeds; not formal promotion or strong-baseline reproduction"),
        "online_validation": {"run": "riichi-validation-v1", "endpoint": online_manifest["endpoint"],
            "rules": online_manifest["rules_expected"], "report": online_report, "game": game,
            "recorded_source": online_manifest["agent_provenance"]["source"],
            "source_commit_matched_by_bytes": TRAIN_COMMIT,
            "source_commit_note": "Verified package-byte match; the online manifest did not record an execution Git commit",
            "recorded_runtime": {"engine": online_manifest["engine"], "agari_kit": online_manifest["code"]},
            "artifacts": public_artifacts["validation"],
            "verification": {"server_passed_verdict": True, "unique_decisions_and_matching_accepted_acks": 120,
                             "event_counts": dict(event_counts), "terminal_scores_match": True},
            "limitation": "One East validation game against platform tsumogiri bots; transport evidence is not a full MJAI replay or ranked-strength result"},
        "verification": {"runtime_observed_now": current_runtime,
            "package_source_observed_now": current_agent["source"]["source_sha256"],
            "generator": artifact(Path(__file__).resolve(), REPO),
            "manifest_helper": artifact(REPO/"src/agari_research/manifest.py", REPO),
            "no_training_or_game_connections": True,
            "verification_scope": "Existing model tensors, source Git blobs, split metadata and completed run artifacts; no dataset replay or held-out game content"},
        "release_policy": {"immutable": True, "overwrite_allowed": False,
            "ranked_campaign_included": False,
            "next_version": "Add ranked acceptance only in a new release version after its independent completion audit"},
        "limitations": ["Training and local evaluation use half games; server validation uses an East game.",
            "Training and server validation match package 43eabb39; local evaluation used the distinct 723b810 snapshot / package 44bb7f08.",
            "Offline latency covers Agent.act only; validation transport timings have different boundaries.",
            "The running 100-game ranked campaign, formal 10000-game promotion and leaderboard milestones are outside this immutable release.",
            "Weights, raw third-party data and credentials are not included in the repository manifest."]})
    require(runtime_provenance() == current_runtime
            and agent_provenance("@" + str(descriptor_path))["source"] == current_agent["source"],
            "Runtime or inference package changed during verification")
    return release


def publish_new(path: Path, content: bytes) -> None:
    """Exclusive creation on volumes without hard links; never overwrite v1.

    Remove our own incomplete file on a write error. A machine crash may leave
    an incomplete new file, which --verify-release detects and never overwrites.
    """
    require(path.parent.is_dir(), f"Destination parent missing: {path.parent}")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--verify-release", action="store_true")
    args = parser.parse_args()
    storage = Storage.from_env()
    paths = (storage.path(f"models/{RUN}/release-v1.json"), EXPERIMENT/"model-release-v1.json")
    if args.write:
        require(not any(path.exists() for path in paths), "Release destination exists; v1 is immutable")
    release = validate_inputs(storage)
    if args.verify_release:
        require(paths[0].read_bytes() == paths[1].read_bytes(), "Published copies differ")
        existing = read_json(paths[0])
        release["created_at"] = existing["created_at"]
        require(existing == release, "Published release differs from independently checked evidence")
    elif args.write:
        content = (json.dumps(release, indent=2, ensure_ascii=False) + "\n").encode()
        for path in paths:
            publish_new(path, content)
    print(json.dumps({"status": "verified-existing" if args.verify_release else "published" if args.write else "inputs-verified",
        "release_id": release["release_id"], "weights_sha256": EXPECTED_WEIGHTS,
        "model_parameters": release["architecture_details"]["trainable_parameters"],
        "offline_games": release["evaluation"]["games"], "validation_passed": True,
        "ranked_included": False, "paths": [str(path) for path in paths],
        "release_sha256": file_sha256(paths[0]) if args.write or args.verify_release else None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
