"""Qualify real-data BC and GRP/CQL training/resume separately on CPU and MPS.

This is an execution script, not a synthetic benchmark. Use a small existing
frozen split after source/runtime freeze. All evidence is written to a new
phase2-* storage namespace; held-out test contents are never opened. Exact
resume is required within each device, not between different devices.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import resource
import shutil
import sys
import time
from pathlib import Path

import torch

from agari_kit.agent import Context, ensure_legal, make_agent
from agari_kit.collection import write_json
from agari_kit.engine import new_environment
from agari_kit.plugins import agent_provenance
from agari_kit.storage import Storage
from agari_research.data import FrozenDataset
from agari_research.manifest import file_sha256
from agari_research.planning import build_training_plan, selected_stages
from agari_research.runtime import runtime_provenance
from agari_research.selection import agent_identity_sha256
from agari_research.training import implementation_sha256, run_training, validate_config

from verify_resume import identical


def device_config(base, device, method, policy_filter, policy_count):
    config = copy.deepcopy(base)
    config.update(device=device, stages=["bc"] if method == "bc" else ["grp", "cql"])
    config.pop("ppo", None)
    config["policy_filter"] = policy_filter
    config["checkpoint_interval_updates"] = 100
    config["validation"] = {"batch_size": 128, "max_samples": None,
                            "interval_updates": math.ceil(policy_count / 64)}
    if method == "bc":
        config["bc"] = {"epochs": 2, "batch_size": 64, "lr": base["cql"]["lr"],
                        "weight_decay": base["cql"]["weight_decay"], "eta_min": base["grp"]["eta_min"]}
        config.pop("grp", None)
        config.pop("cql", None)
    else:
        for stage, batch_size in (("grp", 32), ("cql", 64)):
            config[stage].pop("steps", None)
            config[stage].update(epochs=2, batch_size=batch_size)
    validate_config(config)
    return config


def compare_checkpoints(storage, left_path, right_path):
    left = torch.load(left_path, map_location="cpu", weights_only=True)
    right = torch.load(right_path, map_location="cpu", weights_only=True)
    names = [left.pop("run_name"), right.pop("run_name")]
    for checkpoint in (left, right):
        for stage in selected_stages(checkpoint["config"]):
            if checkpoint["progress"][stage] != checkpoint["training_plan"]["stages"][stage]["target_updates"]:
                raise AssertionError("Resume comparison requires completed training budgets")
    tensors = identical(left, right)
    exports = []
    for stage in selected_stages(left["config"]):
        for prefix in ("", "best-"):
            paths = [storage.path(f"models/{name}/{prefix}{stage}.pt") for name in names]
            hashes = [file_sha256(path) for path in paths]
            if hashes[0] != hashes[1]:
                raise AssertionError(f"Equivalent checkpoint exports differ: {prefix}{stage}")
            exports.append({"kind": prefix + stage, "sha256": hashes[0]})
    return {"status": "exact-match", "runs": names, "tensors_compared": tensors,
            "checked": "all checkpoint fields except run_name, including models/optimizer/scheduler/RNG/cursors/best selection",
            "checkpoint_sha256": [file_sha256(left_path), file_sha256(right_path)], "exports": exports}


def public_agent_probe(storage, run, method):
    descriptor = storage.path(f"models/{run}/best-{method}-agent.json")
    agent = make_agent("@" + str(descriptor))
    try:
        observation = next(iter(new_environment(890000).reset().values()))
        context = Context(observation.player_id, 890000)
        agent.reset(context)
        action = ensure_legal(agent.act(observation, context), observation)
        return {"status": "legal", "scope": "one native public-Agent decision, no game or network",
                "descriptor_sha256": file_sha256(descriptor), "action": action.to_dict()}
    finally:
        close = getattr(agent, "close", None)
        if callable(close):
            close()


def finite_metrics(path):
    def check(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise AssertionError("Non-finite training or validation metric")
        if isinstance(value, dict):
            for item in value.values():
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        check(row)
    return {"updates_recorded": len(rows), "validation_records": sum("validation_loss" in row for row in rows)}


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, help="Storage-relative small frozen 90/5/5 split")
    parser.add_argument("--run", required=True, help="New phase2-* namespace, maximum 42 characters")
    parser.add_argument("--devices", nargs="+", choices=("cpu", "mps"), default=["cpu", "mps"])
    parser.add_argument("--policy-agent", help="Optional builtin or @descriptor whose recorded decisions are selected")
    parser.add_argument("--max-wall-seconds", type=float, default=172800,
                        help="Entire command's time budget; stop at recoverable training boundaries")
    args = parser.parse_args()
    if not re.fullmatch(r"phase2-[A-Za-z0-9_.-]{1,35}", args.run):
        parser.error("Use a new phase2-* name, maximum 42 characters")
    if len(args.devices) != len(set(args.devices)):
        parser.error("Devices must be unique")
    if not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds <= 0:
        parser.error("max-wall-seconds must be finite and positive")
    storage = Storage.from_env()
    storage.check(probe=True)
    output = storage.path(f"runs/{args.run}")
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    base_path = root / "configs/full-coverage-small.json"
    source = {"implementation_sha256": implementation_sha256(), "runtime": runtime_provenance()}
    report = {"schema_version": 1, "status": "running", "source": source,
              "script_sha256": file_sha256(Path(__file__)), "base_config_sha256": file_sha256(base_path),
              "split": args.split, "split_file_sha256": file_sha256(storage.path(args.split)),
              "max_wall_seconds": args.max_wall_seconds, "devices": args.devices, "cases": [],
              "test_partition_content_opened": False,
              "cross_device_tensor_equality_required": False}

    def save():
        report["command_elapsed_seconds"] = time.monotonic() - started
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["process_cumulative_peak_rss_bytes"] = peak if sys.platform == "darwin" else peak * 1024
        write_json(output / "report.json", report)

    def remaining():
        return args.max_wall_seconds - (time.monotonic() - started)

    def require_source():
        if source != {"implementation_sha256": implementation_sha256(), "runtime": runtime_provenance()}:
            raise RuntimeError("Source/runtime changed during device qualification")

    try:
        policy_filter = None if args.policy_agent is None else {
            "agent_identity_sha256": [agent_identity_sha256(agent_provenance(args.policy_agent))]}
        data_start = time.monotonic()
        dataset = FrozenDataset(storage, storage.path(args.split), policy_filter=policy_filter)
        counts = dataset.sample_counts("train")
        report.update(dataset_sha256=dataset.sha256, policy_filter=policy_filter,
                      preflight_elapsed_seconds=time.monotonic() - data_start,
                      training_games=len(counts), partition_sizes={key: len(rows) for key, rows in dataset.manifest["partitions"].items()})
        write_json(output / "training-counts.json", counts)
        report["training_counts_sha256"] = file_sha256(output / "training-counts.json")
        policy_count = sum(row["cql_samples"] for row in counts)
        base = json.loads(base_path.read_text())
        save()
        for device in args.devices:
            if device == "mps" and not torch.backends.mps.is_available():
                report["cases"].append({"device": device, "status": "unavailable", "qualified": False})
                save()
                continue
            for method in ("bc", "cql"):
                if remaining() <= 0:
                    report["status"] = "paused"
                    save()
                    return 2
                config = device_config(base, device, method, policy_filter, policy_count)
                config_path = output / f"{device}-{method}.json"
                write_json(config_path, config)
                plan = build_training_plan(config, dataset)
                name = f"{args.run}-{device}-{method}"
                case = {"device": device, "method": method, "status": "running", "invocations": [],
                        "config_sha256": file_sha256(config_path), "plan_sha256": plan["sha256"],
                        "deterministic_algorithms": device == "cpu"}
                report["cases"].append(case)
                # Each device starts with the mode used by a fresh standalone
                # training process. CPU's global deterministic flag must not
                # accidentally change the MPS qualification environment.
                torch.use_deterministic_algorithms(device == "cpu")

                def run(suffix, *, resume=None, stop_after_updates=None):
                    require_source()
                    if remaining() <= 0:
                        raise TimeoutError("Device qualification wall-time budget exhausted")
                    invocation = time.monotonic()
                    result = run_training(config, dataset, storage, run_name=name + suffix, resume=resume,
                                          stop_after_updates=stop_after_updates, max_wall_seconds=remaining())
                    elapsed = time.monotonic() - invocation
                    require_source()
                    case["invocations"].append({"run": name + suffix, "status": result["status"],
                                                "progress": result["progress"], "wall_seconds": elapsed})
                    save()
                    print(json.dumps({"device": device, "method": method, "run": name + suffix,
                                      "status": result["status"], "elapsed_seconds": elapsed}), flush=True)
                    if result["status"] == "paused":
                        raise TimeoutError("Training saved at a wall-time budget boundary")
                    return result

                try:
                    complete = run("-full")
                    epoch_updates = plan["stages"][method]["updates_per_epoch"]
                    if method == "bc":
                        boundaries = [max(1, epoch_updates - 1), epoch_updates + 1]
                    else:
                        grp = plan["stages"]["grp"]
                        boundaries = [max(1, grp["updates_per_epoch"] - 1), grp["target_updates"] + max(1, epoch_updates - 1)]
                    total = sum(row["target_updates"] for row in plan["stages"].values())
                    boundaries = sorted({value for value in boundaries if value < total})
                    checkpoint, previous = None, 0
                    case["interruptions"] = []
                    for index, boundary in enumerate(boundaries, 1):
                        partial = run("-resume", resume=checkpoint, stop_after_updates=boundary - previous)
                        if partial["status"] != "interrupted" or sum(partial["progress"].values()) != boundary:
                            raise AssertionError("Requested interruption was not preserved")
                        checkpoint = Path(partial["checkpoint"])
                        preserved = output / f"{device}-{method}-interruption-{index}.pt"
                        shutil.copyfile(checkpoint, preserved)
                        case["interruptions"].append({"global_update": boundary, "checkpoint": preserved.name,
                                                       "sha256": file_sha256(preserved)})
                        previous = boundary
                    resumed = run("-resume", resume=checkpoint)
                    case["resume"] = compare_checkpoints(storage, Path(complete["checkpoint"]), Path(resumed["checkpoint"]))
                    case["metrics"] = finite_metrics(storage.path(f"runs/{name}-full/metrics.jsonl"))
                    case["public_agent"] = public_agent_probe(storage, name + "-full", method)
                    case.update(status="passed", qualified=True)
                except TimeoutError as exc:
                    case.update(status="paused", qualified=False, reason=str(exc))
                    report["status"] = "paused"
                    save()
                    return 2
                except Exception as exc:
                    case.update(status="failed", qualified=False, error_type=type(exc).__name__, reason=str(exc))
                    require_source()
                save()
        require_source()
        report["status"] = "passed" if all(case.get("qualified") for case in report["cases"]) else "failed"
        save()
        return 0 if report["status"] == "passed" else 1
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, reason=str(exc))
        save()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
