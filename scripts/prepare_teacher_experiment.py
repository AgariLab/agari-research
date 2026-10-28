"""Verify frozen sources and register equal-budget phase-two training configs.

Run only after the teacher qualification and generation gates have passed.
This command verifies existing traces and writes configs; it does not train.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path

from agari_kit.collection import write_json
from agari_kit.plugins import agent_provenance
from agari_kit.storage import Storage
from agari_research.data import FrozenDataset
from agari_research.manifest import file_sha256
from agari_research.runtime import runtime_provenance
from agari_research.selection import agent_identity_sha256
from agari_research.training import implementation_sha256, validate_config

_spec = importlib.util.spec_from_file_location("phase2_contract", Path(__file__).with_name("phase2_contract.py"))
contract_tools = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(contract_tools)


def reject_overlapping_sources(weak: dict, teacher: dict) -> None:
    def identities(manifest, key):
        return {row[key] for rows in manifest["partitions"].values() for row in rows}
    for key in ("game_id", "sha256"):
        if identities(weak, key) & identities(teacher, key):
            raise ValueError(f"Weak and teacher sources overlap: {key}")


def training_configs(base: dict, weak_count: int, teacher_count: int, *,
                     weak_filter: dict, teacher_filter: dict, device: str, protocol=None) -> list[dict]:
    if any(type(count) is not int or count < 1 for count in (weak_count, teacher_count)):
        raise ValueError("Both frozen sources need positive policy decision counts")
    if protocol is None:
        protocol = contract_tools.load_protocol(Path(__file__).resolve().parents[1] /
                                               "experiments/phase2-teacher-student-v1/protocol.json")
    training = protocol["training"]
    batch_size = training["policy_batch_size"]
    if training["policy_updates"] != f"ceil(3 * max(weak_training_decisions, teacher_training_decisions) / {batch_size})":
        raise ValueError("Training budget formula differs from registered protocol")
    updates = math.ceil(3 * max(weak_count, teacher_count) / batch_size)
    interval = math.ceil(updates / 3)
    configurations = []
    for definition in training["arms"]:
        arm, method = definition["id"], definition["stages"][-1]
        for seed in training["seeds"]:
            config = copy.deepcopy(base)
            config.update(seed=seed, device=device, stages=definition["stages"],
                          policy_filter=weak_filter if arm == "A" else teacher_filter)
            config.pop("ppo", None)
            config["checkpoint_interval_updates"] = min(1000, updates)
            config["validation"] = {"batch_size": 128, "max_samples": None, "interval_updates": interval}
            if method == "bc":
                config["bc"] = {"steps": updates, "batch_size": batch_size,
                                "lr": base["cql"]["lr"], "weight_decay": base["cql"]["weight_decay"],
                                "eta_min": base["grp"]["eta_min"]}
                config.pop("grp", None)
                config.pop("cql", None)
            else:
                config["cql"].pop("epochs", None)
                config["cql"].update(steps=updates, batch_size=batch_size)
            validate_config(config)
            configurations.append({"arm": arm, "seed": seed, "method": method, "config": config})
    return configurations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weak-split", required=True, help="Storage-relative frozen split")
    parser.add_argument("--teacher-split", required=True, help="Storage-relative frozen split")
    parser.add_argument("--teacher-descriptor", type=Path, required=True)
    parser.add_argument("--qualification-run", required=True, help="Storage-relative completed teacher qualification directory")
    parser.add_argument("--run", required=True, help="New phase2-* preparation name")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    if not re.fullmatch(r"phase2-[A-Za-z0-9_.-]{1,72}", args.run):
        raise ValueError("Use a new simple phase2-* run name")
    storage = Storage.from_env()
    storage.check(probe=True)
    output = storage.path(f"runs/{args.run}")
    if output.exists():
        raise FileExistsError("Preparation is immutable; choose a new run name")
    source_root = Path(__file__).resolve().parents[1]
    protocol_path = source_root / "experiments/phase2-teacher-student-v1/protocol.json"
    protocol = contract_tools.load_protocol(protocol_path)
    base_path = source_root / protocol["training"]["model_reference_config"]
    base = json.loads(base_path.read_text())
    expected_model = protocol["training"]["model"]
    if base["model"] != expected_model:
        raise ValueError("Base model differs from the registered architecture")
    qualification = contract_tools.require_qualification(storage, storage.path(args.qualification_run),
                                                         protocol_path, args.teacher_descriptor)
    weak_filter = {"agent_identity_sha256": [agent_identity_sha256(agent_provenance("efficiency"))]}
    teacher_provenance, teacher_binding = contract_tools.validate_teacher(storage, args.teacher_descriptor, protocol)
    teacher_filter = {"agent_identity_sha256": [agent_identity_sha256(teacher_provenance)]}
    selections = {"weak": weak_filter, "teacher": teacher_filter}
    paths = {"weak": args.weak_split, "teacher": args.teacher_split}
    datasets = {name: FrozenDataset(storage, storage.path(path), policy_filter=selections[name])
                for name, path in paths.items()}
    reject_overlapping_sources(datasets["weak"].manifest, datasets["teacher"].manifest)
    source_audits = {name: contract_tools.audit_source(storage, dataset.manifest, name=name, protocol=protocol,
                                                      teacher_provenance=teacher_provenance,
                                                      collection_identity=qualification["collection_identity"])
                    for name, dataset in datasets.items()}
    if set(source_audits["weak"]["seeds"]) & set(source_audits["teacher"]["seeds"]):
        raise ValueError("Weak and teacher source seeds overlap")
    counts = {name: dataset.sample_counts("train") for name, dataset in datasets.items()}
    totals = {name: sum(row["cql_samples"] for row in rows) for name, rows in counts.items()}
    configurations = training_configs(base, totals["weak"], totals["teacher"],
        weak_filter=weak_filter, teacher_filter=teacher_filter, device=args.device, protocol=protocol)
    output.mkdir(parents=True, exist_ok=False)
    configs_dir = output / "configs"
    configs_dir.mkdir()
    jobs = []
    for entry in configurations:
        filename = f"{entry['arm']}-seed{entry['seed']}.json"
        path = configs_dir / filename
        write_json(path, entry["config"])
        jobs.append({key: value for key, value in entry.items() if key != "config"} | {
            "config": path.relative_to(storage.root).as_posix(), "config_sha256": file_sha256(path),
            "split": args.weak_split if entry["arm"] == "A" else args.teacher_split,
            "run": f"{args.run}-{entry['arm'].lower()}-{entry['seed']}",
            "max_wall_seconds": protocol["max_wall_seconds_per_invocation"]})
    for name, rows in counts.items():
        write_json(output / f"{name}-training-counts.json", rows)
        write_json(output / f"{name}-source-audit.json", source_audits[name])
    manifest = {"schema_version": 1, "status": "prepared-not-trained", "jobs": jobs,
                "protocol_sha256": file_sha256(protocol_path), "base_config_sha256": file_sha256(base_path),
                "implementation_sha256": implementation_sha256(), "runtime": runtime_provenance(),
                "datasets": {name: {"split": paths[name], "sha256": dataset.sha256,
                                     "policy_filter": selections[name], "training_decisions": totals[name],
                                     "counts_sha256": file_sha256(output / f"{name}-training-counts.json"),
                                     "source_audit_sha256": file_sha256(output / f"{name}-source-audit.json")}
                             for name, dataset in datasets.items()},
                "teacher_provenance": teacher_provenance,
                "teacher_binding": teacher_binding, "qualification": qualification,
                "contract_helper_sha256": file_sha256(Path(contract_tools.__file__)),
                "test_partition_content_opened": False,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), "jobs": len(jobs), "training_decisions": totals,
                      "policy_updates_per_job": configurations[0]["config"]["bc"]["steps"]}))


if __name__ == "__main__":
    main()
