"""Seal a metadata-only thin report for the fixed, completed teacher 1000 pilot.

Never collects, trains, opens dataset traces/MJAI, or selects a model. The
original frozen matrix rechecks preparation/qualification; this report retains
its legacy weak-test identity limitation. Use a new external runs/phase2-* path.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys

LAUNCH = "runs/phase2-pilot-1000-launch-v1"
PREPARATION = "runs/phase2-pilot-1000-v1"
TEACHER = "runs/phase2-mortal-s42-v1/agent.json"
QUALIFICATION = "runs/phase2-teacher-qualification-v1"
DATASETS = {"teacher": "phase2-teacher-1000-v1", "weak": "selfplay-v1"}
PROBE = "phase2-teacher-probe-v1"
PARTITIONS = ("train", "validation", "test")
MODELS = {f"{arm}-{seed}" for arm in "ABC" for seed in (2026, 2027, 2028)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def content_sha(value):
    return sha(canonical(value).encode())


class Metadata:
    """An explicit JSON metadata boundary, with byte snapshots and drift checks."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.bytes, self.thin = {}, set()

    def path(self, relative):
        path = Path(relative)
        require(not path.is_absolute() and ".." not in path.parts
                and path.suffix == ".json" and (path.parts[:2] == ("datasets", "manifests")
                or path.parts[:1] == ("runs",)), "Only storage-relative JSON metadata is allowed")
        resolved = (self.root / path).resolve()
        require(resolved.is_relative_to(self.root) and resolved == self.root / path,
                "Metadata paths must not escape or alias storage")
        return resolved

    def read(self, relative, *, expected=None, thin=True):
        data = self.path(relative).read_bytes()
        require(relative not in self.bytes or self.bytes[relative] == data, "Metadata changed during report")
        require(expected is None or sha(data) == expected, "Metadata SHA differs from frozen evidence")
        self.bytes[relative] = data
        if thin:
            self.thin.add(relative)
        return json.loads(data)

    def verify(self):
        require(all(self.path(path).read_bytes() == data for path, data in self.bytes.items()),
                "Metadata changed before publication")

    def sources(self):
        return [{"path": path, "bytes": len(self.bytes[path]), "sha256": sha(self.bytes[path])}
                for path in sorted(self.thin)]


def ready(metadata):
    result = metadata.read(LAUNCH + "/result.json")
    require(result.get("status") == "prepared-not-trained"
            and result.get("pilot", {}).get("status") == "ready"
            and result.get("qualification", {}).get("status") == "passed"
            and result.get("preparation", {}).get("jobs") == 9
            and result["preparation"].get("output") == str(metadata.root / PREPARATION),
            "Pilot must have a completed prepared-not-trained handoff")
    return result


class Context:
    def __init__(self, runtime_root, tools_root, storage_root):
        self.tools = Path(tools_root).resolve()
        script = self.tools / "scripts/phase2_evaluation_matrix.py"
        spec = importlib.util.spec_from_file_location("teacher_report_frozen_matrix", script)
        self.matrix = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.matrix)
        self.rt = self.matrix.Runtime(runtime_root)
        require(self.rt.storage.root.resolve() == storage_root, "Gate and frozen runtime storage differ")
        self.initial = self.binding()

    def binding(self):
        from agari_research.runtime import runtime_provenance
        from agari_research.training import implementation_sha256
        self.matrix.check_module_paths(self.rt.root)
        require(self.rt.runtime == runtime_provenance() and self.rt.implementation == implementation_sha256(),
                "Frozen runtime changed during report")
        return {"tools_checkout": self.matrix.checkout_identity(self.tools),
                "report_checkout": self.matrix.checkout_identity(Path(__file__).resolve().parents[1]),
                "runtime_checkouts": {repo: self.matrix.checkout_identity(self.rt.root / repo)
                                      for repo in ("agari-kit", "agari-research")},
                "runtime": self.rt.implementation_binding(),
                "entrypoint": {"path": str(Path(__file__).resolve()), "sha256": self.matrix.digest(__file__)},
                "python": {"path": str(Path(sys.executable).resolve()), "sha256": self.matrix.digest(sys.executable)}}

    def check(self):
        require(self.binding() == self.initial, "Report source/runtime identity changed")

    def preparation(self):
        return self.matrix.preparation(self.rt, PREPARATION)


def receipt(metadata, row, config, expected_hashes, *, thin=False, allow_legacy=False):
    path = row["validation_receipt"]
    require(Path(path).name.endswith(".verification.json"), "Expected a verification receipt")
    value = metadata.read(path, expected=row["validation_receipt_sha256"], thin=thin)
    require(value.get("validation") == "engine-verified" and value.get("method") == "seeded-action-trace-v1"
            and value.get("sha256") == row["sha256"] and value.get("rules") == config["rules"]
            and value.get("producer_code") == config["code"]
            and value.get("engine_version") == config["engine"]["version"], "Receipt validation/source identity differs")
    hashes = value.get("agent_provenance_sha256")
    require((allow_legacy and hashes is None) or hashes == expected_hashes,
            "Receipt must bind all four expected agent identities")
    if not allow_legacy:
        require(value.get("engine_binary_sha256") == config["engine"]["binary_sha256"],
                "Receipt engine binary differs from qualified teacher")
    return value


def source(metadata, name, manifest, protocol):
    entry = manifest["datasets"][name]
    parent = "datasets/manifests/" + DATASETS[name]
    require(str(Path(entry["split"]).parent) == parent, "Source split is outside its fixed acquisition")
    split = metadata.read(entry["split"])
    require(content_sha({key: value for key, value in split.items() if key != "dataset_sha256"})
            == split.get("dataset_sha256") == entry["sha256"], "Split content SHA differs")
    require(set(split["partitions"]) == set(PARTITIONS)
            and [len(split["partitions"][part]) for part in PARTITIONS] == [900, 50, 50], "Expected exactly 900/50/50 games")
    audit = metadata.read(f"{PREPARATION}/{name}-source-audit.json", expected=entry["source_audit_sha256"])
    require(audit["name"] == name and audit["games"] == 1000 and audit["test_partition_content_opened"] is False
            and audit["acquisition"] == parent + "/acquisition.json"
            and content_sha(audit["receipt_inventory"]) == audit["receipt_inventory_sha256"], "Frozen audit differs")
    acquisition = metadata.read(audit["acquisition"], expected=audit["acquisition_sha256"])
    config = acquisition["config"]
    expected_players = [manifest["teacher_provenance"]] * 4 if name == "teacher" else [{"kind": "builtin", "name": "efficiency"}] * 4
    hashes = [content_sha(player) for player in expected_players]
    start = 2630000 if name == "teacher" else 0
    require(acquisition["status"] == "complete" and config["games"] == 1000 and config["seed_start"] == start
            and config["source"] == "agarilab-selfplay" and config["agent_provenance"] == expected_players
            and config["agents"] == [player["name"] for player in expected_players], "Acquisition identity/cohort differs")
    require(config["engine"]["version"] == protocol["engine_version"]
            and config["rules"] == manifest["qualification"]["collection_identity"]["rules"],
            "Source engine version or registered rules differ")
    if name == "teacher":
        require({key: config[key] for key in ("code", "engine", "rules")}
                == manifest["qualification"]["collection_identity"], "Teacher acquisition differs from qualification")
    ids, digests, seeds, inventory, legacy = set(), set(), set(), [], 0
    train = {}
    for part in PARTITIONS:
        for row in split["partitions"][part]:
            require(row["game_id"] not in ids and row["sha256"] not in digests,
                    "Duplicate game or canonical content across partitions")
            require(str(Path(row["validation_receipt"]).parent) == parent, "Receipt belongs to another acquisition")
            value = receipt(metadata, row, config, hashes, allow_legacy=name == "weak")
            seed = value.get("seed")
            require(type(seed) is int and start <= seed < start + 1000 and seed not in seeds,
                    "Receipt seed is duplicate or outside the registered cohort")
            ids.add(row["game_id"])
            digests.add(row["sha256"])
            seeds.add(seed)
            inventory.append({"game_id": row["game_id"], "partition": part, "seed": seed,
                              "receipt": row["validation_receipt"], "sha256": row["validation_receipt_sha256"]})
            if part == "test" and value.get("agent_provenance_sha256") is None:
                legacy += 1
            if part == "train":
                train[row["game_id"]] = row["sha256"]
    require(inventory == audit["receipt_inventory"] and sorted(seeds) == audit["seeds"]
            and legacy == audit["legacy_test_games_without_per_game_policy_binding"], "Receipt inventory/legacy evidence differs")
    counts = metadata.read(f"{PREPARATION}/{name}-training-counts.json", expected=entry["counts_sha256"])
    require(len(counts) == len(train) and {row["game_id"] for row in counts} == set(train)
            and all(row["sha256"] == train[row["game_id"]]
                    and all(type(row[key]) is int and row[key] > 0 for key in ("cql_samples", "grp_samples")) for row in counts),
            "Training counts do not cover exactly the frozen train partition")
    policy = sum(row["cql_samples"] for row in counts)
    require(policy == entry["training_decisions"] and policy > 0, "Policy decision budget differs")
    return {"games": 1000, "partitions": {part: len(split["partitions"][part]) for part in PARTITIONS},
            "seed_start": min(seeds), "seed_end_inclusive": max(seeds), "dataset_sha256": split["dataset_sha256"],
            "receipt_inventory_sha256": audit["receipt_inventory_sha256"], "receipts_checked": len(inventory),
            "train_nonforced_policy_decisions": policy, "train_grp_samples": sum(row["grp_samples"] for row in counts),
            "legacy_test_games_without_per_game_policy_binding": legacy,
            "test_policy_identity_evidence": audit["test_policy_identity_evidence"]}, ids, digests, seeds


def probe_isolation(metadata, manifest, protocol, source_sets):
    require(protocol["seeds"]["teacher_probe"] == {"start": 2620000, "count": 16, "rotations": 1},
            "Wrong registered engineering probe cohort")
    parent = "datasets/manifests/" + PROBE
    acquisition = metadata.read(parent + "/acquisition.json")
    config = acquisition["config"]
    require(acquisition["status"] == "complete" and config["games"] == 16 and config["seed_start"] == 2620000
            and config["agent_provenance"] == [manifest["teacher_provenance"]] * 4
            and {key: config[key] for key in ("code", "engine", "rules")} == manifest["qualification"]["collection_identity"],
            "Engineering probe identity/cohort differs")
    ids, digests, seeds = set(), set(), set()
    inventory = []
    for index in range(16):
        row = metadata.read(f"{parent}/game-{index:06d}.json", thin=False)
        require(str(Path(row["validation_receipt"]).parent) == parent, "Probe receipt belongs to another acquisition")
        value = receipt(metadata, row, config, [content_sha(manifest["teacher_provenance"])] * 4)
        require(value["seed"] == 2620000 + index and row["game_id"] not in ids and row["sha256"] not in digests,
                "Probe game identity/cohort differs")
        ids.add(row["game_id"])
        digests.add(row["sha256"])
        seeds.add(value["seed"])
        inventory.append({"game_id": row["game_id"], "sha256": row["sha256"], "seed": value["seed"],
                          "receipt": row["validation_receipt"], "receipt_sha256": row["validation_receipt_sha256"]})
    groups = {**source_sets, "probe": (ids, digests, seeds)}
    checks = []
    for left, right in (("teacher", "weak"), ("teacher", "probe"), ("weak", "probe")):
        require(all(not groups[left][i] & groups[right][i] for i in range(3)), "Source/probe identifiers overlap")
        checks.append({"sources": [left, right], "game_id_overlap": 0, "recorded_canonical_sha256_overlap": 0, "seed_overlap": 0})
    reserved = {}
    for name, cohort in protocol["seeds"].items():
        if name in ("teacher_1000", "teacher_probe"):
            continue
        reserved[name] = dict(cohort)
        require(not set(range(cohort["start"], cohort["start"] + cohort["count"]))
                & (source_sets["teacher"][2] | source_sets["weak"][2] | seeds), "Reserved evaluation/large-cohort seeds overlap")
    return {"actual_recorded_metadata_identifier_comparisons": checks,
            "probe_receipt_inventory_sha256": content_sha(inventory), "probe_games": 16,
            "reserved_seed_ranges_checked": reserved,
            "evaluation_content_comparison_performed": False,
            "scope": "Recorded game IDs/canonical hashes for completed sources/probe; registered seed ranges only for evaluation and optional 10k cohort"}


def configurations(metadata, manifest, protocol, summaries):
    jobs = manifest["jobs"]
    require(len(jobs) == 9 and {f"{row['arm']}-{row['seed']}" for row in jobs} == MODELS, "Exactly nine registered jobs are required")
    updates = math.ceil(3 * max(row["train_nonforced_policy_decisions"] for row in summaries.values()) / 64)
    outputs = []
    for job in jobs:
        require(str(Path(job["config"]).parent) == PREPARATION + "/configs", "Config outside fixed preparation")
        config = metadata.read(job["config"], expected=job["config_sha256"])
        method = "cql" if job["arm"] == "C" else "bc"
        name = "weak" if job["arm"] == "A" else "teacher"
        require(job["method"] == method and job["split"] == manifest["datasets"][name]["split"]
                and job["max_wall_seconds"] == 172800 and config["seed"] == job["seed"] and config["device"] == "cpu"
                and config["model"] == protocol["training"]["model"] and "ppo" not in config
                and config["stages"] == (["grp", "cql"] if method == "cql" else ["bc"])
                and config["validation"]["max_samples"] is None
                and config["validation"]["interval_updates"] == math.ceil(updates / 3)
                and config[method]["steps"] == updates and config[method]["batch_size"] == 64,
                "Prepared CPU configuration or equal policy budget differs")
        grp_updates = 0
        if method == "cql":
            grp = config["grp"]
            require(type(grp["batch_size"]) is int and grp["batch_size"] > 0
                    and type(grp["epochs"]) is int and grp["epochs"] > 0, "Invalid GRP schedule")
            grp_updates = math.ceil(summaries["teacher"]["train_grp_samples"] / grp["batch_size"]) * grp["epochs"]
        outputs.append({"arm": job["arm"], "seed": job["seed"], "source": name, "method": method,
                        "policy_updates_U": updates, "grp_updates_G": grp_updates,
                        "config": job["config"], "config_sha256": job["config_sha256"], "run": job["run"]})
    return outputs


def build(metadata, context, output):
    handoff = ready(metadata)
    require(re.fullmatch(r"runs/phase2-[A-Za-z0-9_.-]{1,72}", output), "Choose a fresh runs/phase2-* report directory")
    destination = metadata.root / output
    require(not destination.exists() and destination.resolve().parent == metadata.root / "runs", "Report destination must be new and inside storage")
    manifest = metadata.read(PREPARATION + "/manifest.json")
    require(manifest["status"] == "prepared-not-trained" and manifest["test_partition_content_opened"] is False,
            "Preparation is not complete or opened held-out content")
    require(manifest["teacher_binding"]["descriptor"] == TEACHER
            and manifest["qualification"]["run"] == QUALIFICATION,
            "Teacher descriptor/qualification must use the fixed prior evidence paths")
    protocol = context.rt.protocol
    require(protocol["seeds"]["teacher_1000"] == {"start": 2630000, "count": 1000, "rotations": 1}, "Wrong registered pilot cohort")
    summaries, source_sets = {}, {}
    for name in ("weak", "teacher"):
        summary, *identifiers = source(metadata, name, manifest, protocol)
        summaries[name], source_sets[name] = summary, tuple(identifiers)
    isolation = probe_isolation(metadata, manifest, protocol, source_sets)
    jobs = configurations(metadata, manifest, protocol, summaries)
    # Paths are constrained above before the frozen gate follows metadata links.
    directory, checked, checked_jobs, _, _ = context.preparation()
    require(Path(directory) == metadata.root / PREPARATION and checked == manifest and set(checked_jobs) == MODELS,
            "Independent frozen preparation gate disagrees")
    require(handoff["preparation"]["training_decisions"] == {name: row["train_nonforced_policy_decisions"] for name, row in summaries.items()}
            and handoff["preparation"]["policy_updates_per_job"] == jobs[0]["policy_updates_U"], "Launch and preparation budgets differ")
    summary = {"schema_version": 1, "status": "verified-metadata-only", "dataset": DATASETS["teacher"],
               "preparation": PREPARATION, "datasets": summaries, "teacher_provenance": manifest["teacher_provenance"],
               "four_seat_teacher_provenance_sha256": [content_sha(manifest["teacher_provenance"])] * 4,
               "teacher_binding": manifest["teacher_binding"], "qualification": manifest["qualification"],
               "isolation": isolation, "jobs": jobs, "source_binding": context.initial,
               "read_boundary": {"dataset_traces_opened": False, "dataset_mjai_opened": False,
                   "test_partition_content_opened": False, "training_execution_checked": False,
                   "counts": "Frozen train-only counts: nonforced native choice decisions for U; round-by-seat GRP samples for separate G",
                   "legacy_weak": "Retains prior audit: legacy test per-seat policy identity rests on acquisition config; no new trace validation"}}
    context.check()
    metadata.verify()
    source_list = {"schema_version": 1, "storage_root": str(metadata.root), "artifacts": metadata.sources(),
                   "receipt_detail": "Full teacher/weak receipt inventory remains in the bound source-audits; probe metadata remains in its acquisition directory"}
    readme = ("# Teacher 1000 metadata report\n\n"
        "The completed pilot contains 1000 games, split by the frozen whole-game chronological policy into 900 train, 50 validation and 50 test games. "
        "All 1000 teacher receipts bind four identical qualified S42 agents, the registered seeds 2630000–2630999, producer, engine and rules.\n\n"
        f"Each of the nine CPU configurations uses U={jobs[0]['policy_updates_U']} policy updates. "
        f"The C configurations separately add G={next(job['grp_updates_G'] for job in jobs if job['arm'] == 'C')} GRP updates. "
        "U uses frozen train-only nonforced native choice counts; G uses train round-by-seat GRP samples. This report does not inspect training execution.\n\n"
        "Completed weak/probe sources were checked for overlapping recorded game IDs, canonical hashes and seeds. "
        "Evaluation isolation checks registered seed ranges only; future evaluation MJAI content was not compared. "
        "Legacy weak test receipts retain the original acquisition-only per-seat identity limitation.\n\n"
        "No dataset trace or MJAI was opened, including test content. The frozen preparation gate independently rechecks prior qualification evidence. "
        "summary.json records the identities and counts; sources.json binds external metadata; preparation-manifest.json preserves the original bytes. "
        "Original receipts, traces, models and counts remain on external storage.\n")
    payloads = {"sources.json": (canonical(source_list) + "\n").encode(),
                "preparation-manifest.json": metadata.bytes[PREPARATION + "/manifest.json"],
                "README.md": readme.encode()}
    summary["report_artifacts"] = {name: {"bytes": len(data), "sha256": sha(data)} for name, data in payloads.items()}
    # Publish the success summary last; a partial directory must never be reused.
    payloads["summary.json"] = (canonical(summary) + "\n").encode()
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in payloads.items():
        with (destination / name).open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    return {"status": summary["status"], "output": output, "summary_sha256": sha((destination / "summary.json").read_bytes())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--tools-root", type=Path, required=True, help="Original frozen matrix tools checkout")
    parser.add_argument("--output", required=True, help="New storage-relative runs/phase2-* directory")
    args = parser.parse_args(argv)
    metadata = Metadata(os.environ.get("AGARILAB_STORAGE_ROOT", "/Volumes/新加卷/AgariLab"))
    ready(metadata)  # No runtime/helper/dataset access until the completed handoff.
    context = Context(args.runtime_root, args.tools_root, metadata.root)
    print(canonical(build(metadata, context, args.output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
