"""Register nine-model development matrices and one independent final comparison.

The explicitly selected frozen runtime owns all inference and statistics. This
standalone orchestrator never trains, reads held-out dataset content, or opens
online connections. Only run/run-final start games; all writes require a CLI
invocation and remain on configured artifact storage.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

MODELS = tuple(f"{arm}-{seed}" for arm in "ABC" for seed in (2026, 2027, 2028))
POOLS = ("primary", "weak")
MODEL_FACTORY = "agari_research.model_agent:ModelAgent"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def content_digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_module_paths(root):
    """Fail if an already imported package silently came from a mutable checkout."""
    for name, module in tuple(sys.modules.items()):
        for package, repository in (("agari_kit", "agari-kit"), ("agari_research", "agari-research")):
            if name == package or name.startswith(package + "."):
                path = getattr(module, "__file__", None)
                require(path is not None and Path(path).resolve().is_relative_to(root / repository / "src" / package),
                        f"Loaded {name} outside the explicitly frozen runtime")


def checkout_identity(path):
    def git(*args):
        return subprocess.run(["git", "-C", str(path), *args], text=True, capture_output=True)
    head, branch, dirty, top = (git("rev-parse", "HEAD"), git("symbolic-ref", "--quiet", "HEAD"),
                               git("status", "--porcelain", "--untracked-files=all"), git("rev-parse", "--show-toplevel"))
    require(not head.returncode and branch.returncode == 1 and not dirty.returncode and not dirty.stdout.strip()
            and not top.returncode and Path(top.stdout.strip()).resolve() == path,
            "Runtime requires clean detached source worktrees")
    return {"path": str(path), "commit": head.stdout.strip()}


class Runtime:
    def __init__(self, root):
        self.root = Path(root).resolve()
        for repo in ("agari-kit", "agari-research"):
            require((self.root / repo / "src").is_dir(), "runtime-root must contain both frozen source checkouts")
        self.checkouts = {repo: checkout_identity(self.root / repo) for repo in ("agari-kit", "agari-research")}
        check_module_paths(self.root)
        sys.path[:0] = [str(self.root / repo / "src") for repo in ("agari-kit", "agari-research")]
        import agari_kit
        import agari_research
        from agari_kit.storage import Storage
        from agari_research.runtime import runtime_provenance
        from agari_research.training import implementation_sha256
        self.storage = Storage.from_env()
        self.research = self.root / "agari-research"
        self.contract = load_module("matrix_frozen_contract", self.research / "scripts/phase2_contract.py")
        self.comparator = self.contract.comparison_module()
        self.protocol_path = self.research / "experiments/phase2-teacher-student-v1/protocol.json"
        self.protocol = self.contract.load_protocol(self.protocol_path)
        self.runtime = runtime_provenance()
        self.implementation = implementation_sha256()
        check_module_paths(self.root)

    def path(self, relative):
        return self.storage.path(relative)

    def artifact(self, path):
        path = Path(path)
        return {"path": self.contract.storage_relative(self.storage, path), "sha256": digest(path)}

    def bound(self, entry):
        path = self.path(entry["path"])
        require(digest(path) == entry["sha256"], f"Bound evidence changed: {entry['path']}")
        return path

    def implementation_binding(self):
        return {"runtime_root": str(self.root), "runtime": self.runtime,
                "checkouts": self.checkouts,
                "research_implementation_sha256": self.implementation,
                "protocol_sha256": digest(self.protocol_path), "orchestrator_sha256": digest(__file__),
                "contract_helper_sha256": digest(self.contract.__file__),
                "comparator_sha256": digest(self.comparator.__file__)}


def publish(path, value):
    """Exclusive immutable small artifact; an interrupted write cannot look valid."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(value) + "\n"
    if path.exists():
        require(path.read_text() == data, f"Immutable artifact differs: {path}")
        return
    with path.open("x", encoding="utf-8") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def require_jobs(jobs):
    keys = [f"{job['arm']}-{job['seed']}" for job in jobs]
    require(len(keys) == 9 and set(keys) == set(MODELS), "Exactly all nine registered arm/seed jobs are required")
    require(len({job["run"] for job in jobs}) == 9, "Training run names must be distinct")
    return dict(zip(keys, jobs))


def preparation(rt, relative):
    directory = rt.path(relative)
    manifest = read(directory / "manifest.json")
    require(manifest.get("status") == "prepared-not-trained" and manifest.get("schema_version") == 1,
            "Expected a completed immutable preparation manifest")
    require(manifest["protocol_sha256"] == digest(rt.protocol_path)
            and manifest["base_config_sha256"] == digest(rt.research / rt.protocol["training"]["model_reference_config"])
            and manifest["script_sha256"] == digest(rt.research / "scripts/prepare_teacher_experiment.py")
            and manifest["contract_helper_sha256"] == digest(rt.contract.__file__)
            and manifest["implementation_sha256"] == rt.implementation and manifest["runtime"] == rt.runtime,
            "Preparation source, protocol or runtime differs from frozen runtime")
    teacher = rt.path(manifest["teacher_binding"]["descriptor"])
    qualified = rt.contract.require_qualification(rt.storage, rt.path(manifest["qualification"]["run"]),
                                                 rt.protocol_path, teacher)
    require(qualified == manifest["qualification"], "Preparation qualification binding changed")
    provenance, teacher_binding = rt.contract.validate_teacher(rt.storage, teacher, rt.protocol)
    require(provenance == manifest["teacher_provenance"] and teacher_binding == manifest["teacher_binding"],
            "Preparation teacher identity changed")
    jobs = require_jobs(manifest["jobs"])
    evidence = [rt.artifact(directory / "manifest.json")]
    audits, splits, counts = {}, {}, {}
    for name in ("weak", "teacher"):
        entry = manifest["datasets"][name]
        split_path = rt.path(entry["split"])
        split = read(split_path)
        require(content_digest({k: v for k, v in split.items() if k != "dataset_sha256"})
                == split.get("dataset_sha256") == entry["sha256"], "Prepared split content changed")
        from agari_kit.dataset import make_split
        require(make_split([row for part in ("train", "validation", "test") for row in split["partitions"][part]]) == split,
                "Prepared split does not follow the whole-game 90/5/5 contract")
        splits[name] = split
        for suffix, field in (("source-audit", "source_audit_sha256"), ("training-counts", "counts_sha256")):
            path = directory / f"{name}-{suffix}.json"
            require(digest(path) == entry[field], "Prepared source audit or training counts changed")
            evidence.append(rt.artifact(path))
        audit = read(directory / f"{name}-source-audit.json")
        require(audit["test_partition_content_opened"] is False
                and digest(rt.path(audit["acquisition"])) == audit["acquisition_sha256"]
                and content_digest(audit["receipt_inventory"]) == audit["receipt_inventory_sha256"],
                "Prepared acquisition or receipt inventory changed")
        rows = {(part, row["game_id"]): row for part in ("train", "validation", "test") for row in split["partitions"][part]}
        inventory = audit["receipt_inventory"]
        require(len(rows) == len(inventory) == audit["games"], "Source audit and split differ")
        for receipt in inventory:
            row = rows.pop((receipt["partition"], receipt["game_id"]))
            require(row["validation_receipt"] == receipt["receipt"]
                    and row["validation_receipt_sha256"] == receipt["sha256"]
                    and digest(rt.path(receipt["receipt"])) == receipt["sha256"], "Prepared receipt changed")
        require(not rows, "Source receipt coverage is incomplete")
        counts[name] = read(directory / f"{name}-training-counts.json")
        require({row["game_id"] for row in counts[name]} == {row["game_id"] for row in split["partitions"]["train"]}
                and sum(row["cql_samples"] for row in counts[name]) == entry["training_decisions"],
                "Prepared policy budget does not match complete training counts")
        evidence.extend([rt.artifact(split_path), rt.artifact(rt.path(audit["acquisition"]))])
        audits[name] = audit
    require(not set(audits["weak"]["seeds"]) & set(audits["teacher"]["seeds"]), "Source seeds overlap")
    weak_ids = {row["sha256"] for part in splits["weak"]["partitions"].values() for row in part}
    teacher_ids = {row["sha256"] for part in splits["teacher"]["partitions"].values() for row in part}
    require(not weak_ids & teacher_ids, "Source content overlaps")
    from agari_research.selection import agent_identity_sha256
    from agari_kit.plugins import agent_provenance
    filters = {"weak": {"agent_identity_sha256": [agent_identity_sha256(agent_provenance("efficiency"))]},
               "teacher": {"agent_identity_sha256": [agent_identity_sha256(provenance)]}}
    require(all(manifest["datasets"][name]["policy_filter"] == filters[name] for name in filters), "Prepared policy identity differs")
    expected_teacher = sorted(rt.contract.seed_range(rt.protocol["seeds"][f"teacher_{audits['teacher']['games']}"]))
    require(audits["teacher"]["seeds"] == expected_teacher, "Teacher seeds differ from registered source cohort")
    reserved = set().union(*(rt.contract.seed_range(rt.protocol["seeds"][name]) for name in rt.contract.EVALUATION_SEEDS))
    require(not reserved & (set(audits["weak"]["seeds"]) | set(audits["teacher"]["seeds"])), "Evaluation seeds leaked into training")
    generator = load_module("matrix_frozen_preparer", rt.research / "scripts/prepare_teacher_experiment.py")
    base = read(rt.research / rt.protocol["training"]["model_reference_config"])
    devices = {read(rt.path(job["config"]))["device"] for job in jobs.values()}
    require(len(devices) == 1, "Nine training jobs must use the same device")
    generated = generator.training_configs(base, manifest["datasets"]["weak"]["training_decisions"],
        manifest["datasets"]["teacher"]["training_decisions"], weak_filter=filters["weak"], teacher_filter=filters["teacher"],
        device=devices.pop(), protocol=rt.protocol)
    for item in generated:
        job = jobs[f"{item['arm']}-{item['seed']}"]
        require(read(rt.path(job["config"])) == item["config"] and job["max_wall_seconds"] == 172800,
                "Prepared configuration differs from the registered three-arm budget")
    return directory, manifest, jobs, audits, evidence


def training_contract(job, config, result, checkpoint, payload, metrics, *, dataset_sha256, implementation, runtime):
    """Pure metadata gate; file bytes and actual tensor loading are checked outside."""
    method = "cql" if job["arm"] == "C" else "bc"
    require(job["method"] == method and config["seed"] == job["seed"]
            and config["stages"] == (["grp", "cql"] if method == "cql" else ["bc"]), "Wrong training arm/seed")
    require(config.get("validation", {}).get("max_samples") is None, "Validation must use the entire partition")
    plan = checkpoint["training_plan"]
    require(result.get("status") == "complete" and result.get("stop_reason") is None
            and result["run"] == job["run"] == checkpoint["run_name"], "Training is not finished")
    require(checkpoint["config"] == config and result["progress"] == checkpoint["progress"]
            and set(checkpoint["progress"]) == set(config["stages"])
            and all(checkpoint["progress"][stage] == plan["stages"][stage]["target_updates"] for stage in config["stages"]),
            "Training completion does not match the prepared budget")
    for key, value in (("dataset_sha256", dataset_sha256), ("implementation_sha256", implementation)):
        require(result[key] == checkpoint[key] == payload[key] == value, f"Training binding differs: {key}")
    require(checkpoint["runtime_provenance"] == payload["runtime_provenance"] == runtime
            and payload["config"] == config and payload["policy_filter"] == config["policy_filter"]
            and payload["kind"] == method and payload["model_config"] == config["model"], "Model/config/runtime binding differs")
    require(result["training_plan_sha256"] == payload["training_plan_sha256"] == plan["sha256"]
            and content_digest({k: v for k, v in plan.items() if k != "sha256"}) == plan["sha256"], "Training plan digest differs")
    require(result["best_validation"] == checkpoint["best_validation"], "Result and checkpoint selected different models")
    require(set(row["stage"] for row in metrics) == set(config["stages"]), "Training metrics have missing or extra stages")
    for stage in config["stages"]:
        require([row["update"] for row in metrics if row["stage"] == stage]
                == list(range(1, checkpoint["progress"][stage] + 1)), "Duplicate/missing/out-of-order training updates")
        target = checkpoint["progress"][stage]
        interval = config["validation"].get("interval_updates")
        expected_checks = sorted(set(range(interval, target + 1, interval) if interval is not None else []) | {target})
        observed_checks = [row for row in metrics if row["stage"] == stage and "validation_loss" in row]
        require([row["update"] for row in observed_checks] == expected_checks,
                "Missing or unexpected registered validation checkpoint")
        require(all(row.get("validation_scope") == "full" and row.get("validation_sample_limit") is None
                    and row.get("validation_samples", 0) > 0 and math.isfinite(row["validation_loss"]) for row in observed_checks),
                "Full finite validation evidence is missing")
    selected = checkpoint["best_validation"][method]
    require(selected["metric"] == "validation_loss" and type(selected["update"]) is int
            and 1 <= selected["update"] <= checkpoint["progress"][method]
            and selected["artifact"] == f"best-{method}-u{selected['update']:09d}.pt", "Invalid best model selection")
    require(payload["completed_updates"] == selected["update"]
            and payload["validation_selection"] == {key: selected[key] for key in ("metric", "value", "update")},
            "Versioned model does not match checkpoint selection")
    validations = [row for row in metrics if row["stage"] == method and "validation_loss" in row]
    require(bool(validations) and all(row.get("validation_scope") == "full" and row.get("validation_sample_limit") is None
            and row.get("validation_samples", 0) > 0 and math.isfinite(row["validation_loss"]) for row in validations),
            "Full finite validation evidence is missing")
    require(len({row["update"] for row in validations}) == len(validations)
            and all(1 <= row["update"] <= checkpoint["progress"][method] for row in validations), "Duplicate/out-of-budget validation updates")
    best = min(validations, key=lambda row: (row["validation_loss"], row["update"]))
    require((best["update"], best["validation_loss"]) == (selected["update"], selected["value"]), "Selected model is not the recorded validation minimum")
    return selected


def verify_completed_job(rt, prepared, key, job):
    import torch
    name = "weak" if job["arm"] == "A" else "teacher"
    config_path = rt.path(job["config"])
    require(digest(config_path) == job["config_sha256"], "Prepared job config changed")
    config = read(config_path)
    entry = prepared["datasets"][name]
    require(job["split"] == entry["split"] and config["policy_filter"] == entry["policy_filter"], "Job changed source or policy filter")
    updates = math.ceil(3 * max(value["training_decisions"] for value in prepared["datasets"].values()) / 64)
    require(config[job["method"]]["steps"] == updates and config[job["method"]]["batch_size"] == 64,
            "Prepared equal policy update budget changed")
    run = rt.path("runs/" + job["run"])
    models = rt.path("models/" + job["run"])
    checkpoint_path = rt.path("checkpoints/" + job["run"] + "/latest.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    finished = read(run / "result.json")
    require(finished["checkpoint"] == str(checkpoint_path) and finished["models"] == str(models), "Result artifact paths differ")
    selected = checkpoint["best_validation"][job["method"]]
    weight = models / selected["artifact"]
    require(weight.parent == models and digest(weight) == selected["sha256"], "Best snapshot hash changed")
    payload = torch.load(weight, map_location="cpu", weights_only=True)
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_bytes().splitlines() if line.strip()]
    training_contract(job, config, finished, checkpoint, payload, metrics, dataset_sha256=entry["sha256"],
                      implementation=rt.implementation, runtime=rt.runtime)
    execution = read(run / "execution-plan.json")
    require(execution == {"plan": checkpoint["training_plan"], "runtime_provenance": rt.runtime,
                          "implementation_sha256": rt.implementation}
            and read(run / "config.json") == config
            and read(run / "dataset.json") == read(rt.path(job["split"])), "Run files differ from finished checkpoint")
    alias = models / f"best-{job['method']}.pt"
    descriptor = alias.with_name(alias.stem + "-agent.json")
    require(digest(alias) == selected["sha256"] and read(descriptor) == {
        "factory": MODEL_FACTORY, "kwargs": {"device": config["device"], "torch_threads": config["torch_threads"]},
        "artifacts": {"weights_path": {"path": alias.name, "sha256": selected["sha256"]}}}, "Best descriptor/alias differs from selected snapshot")
    from agari_research.planning import build_training_plan, coverage, expected_samples
    plan = checkpoint["training_plan"]
    split = read(rt.path(job["split"]))
    producer_engines = set()
    for partition in ("train", "validation"):
        for row in split["partitions"][partition]:
            receipt = read(rt.path(row["validation_receipt"]))
            require(digest(rt.path(row["validation_receipt"])) == row["validation_receipt_sha256"], "Training producer receipt changed")
            producer_engines.add((receipt["engine_version"], receipt["engine_binary_sha256"]))
    if any("epochs" in config[stage] for stage in config["stages"]):
        require(hashlib.sha256((canonical(plan["training_games"]) + "\n").encode()).hexdigest() == entry["counts_sha256"],
                "Epoch plan does not use prepared native sample counts")
    counted_dataset = SimpleNamespace(sha256=entry["sha256"], policy_filter=config["policy_filter"],
        producer_engines=[{"version": version, "binary_sha256": sha} for version, sha in sorted(producer_engines)],
        sample_counts=lambda part: plan["training_games"], records=lambda part: split["partitions"][part])
    require(build_training_plan(config, counted_dataset) == plan, "Finished plan differs from prepared config and counts")
    require(all(checkpoint["samples_seen"][stage] == expected_samples(plan["stages"][stage], checkpoint["progress"][stage], config[stage]["batch_size"])
                for stage in config["stages"]), "Finished sample coverage differs from budget")
    require(finished["training_coverage"] == {stage: coverage(plan, stage, checkpoint["samples_seen"][stage], config["seed"])
                                              for stage in config["stages"]}, "Finished coverage receipt differs")
    require(all(torch.isfinite(tensor).all().item() for tensor in payload["state_dict"].values()), "Non-finite model tensors")
    paths = [config_path, checkpoint_path, weight, alias, descriptor, run / "result.json", run / "metrics.jsonl",
             run / "execution-plan.json", run / "config.json", run / "dataset.json"]
    from agari_research.features import ENCODING_VERSION, UPSTREAM_COMMIT
    from agari_research.trace_samples import SAMPLE_SEMANTICS_VERSION
    for field, expected in (("encoding_version", ENCODING_VERSION), ("upstream_commit", UPSTREAM_COMMIT),
                            ("sample_semantics_version", SAMPLE_SEMANTICS_VERSION), ("engine_version", "0.4.10"),
                            ("policy_initialization", checkpoint["policy_initialization"])):
        require(payload[field] == checkpoint[field] == expected, f"Model/checkpoint semantics differ: {field}")
    exports = [(weight, payload), (alias, payload)]
    for stage in config["stages"]:
        final_path = models / f"{stage}.pt"
        final = torch.load(final_path, map_location="cpu", weights_only=True)
        require(final["kind"] == stage and final["completed_updates"] == checkpoint["progress"][stage]
                and final["config"] == config and final["dataset_sha256"] == entry["sha256"]
                and final["runtime_provenance"] == rt.runtime and final["implementation_sha256"] == rt.implementation
                and final["training_plan_sha256"] == plan["sha256"], "Final export does not match finished checkpoint")
        require(set(final["state_dict"]) == set(checkpoint["models"][stage])
                and all(torch.equal(tensor, checkpoint["models"][stage][name]) and torch.isfinite(tensor).all().item()
                        for name, tensor in final["state_dict"].items()), "Final tensors differ from finished checkpoint")
        exports.append((final_path, final))
    for path, content in exports:
        sha = digest(path)
        sidecar = path.with_suffix(".json")
        require(read(sidecar) == {**{name: value for name, value in content.items() if name != "state_dict"},
                                 "weights_sha256": sha, "filename": path.name}, "Export metadata differs from actual model payload")
        paths.extend([path, sidecar])
        if content["kind"] in ("bc", "cql"):
            agent_file = path.with_name(path.stem + "-agent.json")
            require(read(agent_file) == {"factory": MODEL_FACTORY,
                "kwargs": {"device": config["device"], "torch_threads": config["torch_threads"]},
                "artifacts": {"weights_path": {"path": path.name, "sha256": sha}}}, "Versioned/final descriptor differs from model")
            paths.append(agent_file)
    return {"arm": job["arm"], "seed": job["seed"], "method": job["method"], "best": selected,
                   "weights": rt.artifact(weight), "training_evidence": [rt.artifact(path) for path in dict.fromkeys(paths)],
                   "policy_initialization": checkpoint["policy_initialization"],
                   "inference_kwargs": {"device": "cpu", "torch_threads": 1}}


def training_models(rt, prepared, jobs):
    result = {key: verify_completed_job(rt, prepared, key, job) for key, job in jobs.items()}
    for seed in rt.protocol["training"]["seeds"]:
        initializations = [result[f"{arm}-{seed}"]["policy_initialization"] for arm in "ABC"]
        require(all(value["seed"] == seed for value in initializations)
                and len({value["q_state_sha256"] for value in initializations}) == 1,
                "Three arms must share the same Q initialization for each training seed")
    return result


def explicit_contract(rt, baseline, candidate):
    compare = rt.comparator
    require(compare.comparison_conditions(baseline) == compare.comparison_conditions(candidate), "Paired conditions differ")
    return {"schema_version": 1, "kind": "different-candidates", "primary_metric": "mean_rank",
            "hypothesis": "Registered student versus incumbent on the fixed primary/diagnostic opponent pool",
            "baseline_provenance_sha256": compare.identity_sha256(baseline["lineup_provenance"][0]),
            "candidate_provenance_sha256": compare.identity_sha256(candidate["lineup_provenance"][0]),
            "conditions_sha256": compare.identity_sha256(compare.comparison_conditions(baseline))}


def identity(rt, lineup, seeds):
    from agari_kit.evaluation_campaign import _identity
    result = dict(_identity(lineup), seeds=seeds)
    check_module_paths(rt.root)
    return result


def register(rt, args):
    require(re.fullmatch(r"phase2-[A-Za-z0-9_.-]{1,72}", args.run), "Use a new simple phase2-* matrix name")
    output = rt.path("runs/" + args.run)
    require(not output.exists(), "Matrix registration is immutable; select a new run")
    _, prepared, jobs, audits, evidence = preparation(rt, args.preparation)
    expected_games = 1000 if args.scale == "small" else 10000
    require(all(audit["games"] == expected_games for audit in audits.values()), "Selected scale differs from weak/teacher data")
    scale_from = None
    if args.scale == "large":
        require(args.scale_from is not None, "Large matrix requires an eligible small development matrix")
        previous = load_matrix(rt, args.scale_from)
        summary = development_report(rt, previous)
        require(previous["scale"] == "small" and summary["scale_10000_eligible"], "Small pilot did not pass the scale-up gate")
        publish(rt.path(args.scale_from) / "development-report.json", summary)
        scale_from = {"matrix": args.scale_from, "registration_sha256": digest(rt.path(args.scale_from) / "registration.json"),
                      "development_report": rt.artifact(rt.path(args.scale_from) / "development-report.json")}
    else:
        require(args.scale_from is None, "Small matrix cannot reuse a scale-up authorization")
    models = training_models(rt, prepared, jobs)
    from agari_kit.plugins import agent_provenance
    incumbent = "@" + str(rt.path(rt.protocol["incumbent"]["descriptor"]))
    incumbent_provenance = agent_provenance(incumbent)
    require(incumbent_provenance["artifacts"]["weights_path"]["sha256"] == rt.protocol["incumbent"]["weights_sha256"], "Incumbent weight differs")
    seeds = sorted(rt.contract.seed_range(rt.protocol["seeds"]["development"]))
    output.mkdir(parents=True, exist_ok=False)
    names = {"incumbent": incumbent}
    for key, entry in models.items():
        directory = output / "models" / key
        directory.mkdir(parents=True)
        target = directory / "weights.pt"
        with rt.bound(entry["weights"]).open("rb") as source, target.open("xb") as destination:
            shutil.copyfileobj(source, destination)
        require(digest(target) == entry["weights"]["sha256"], "Copied best model changed")
        descriptor = directory / "agent.json"
        publish(descriptor, {"factory": MODEL_FACTORY, "kwargs": entry["inference_kwargs"],
                             "artifacts": {"weights_path": {"path": "weights.pt", "sha256": digest(target)}}})
        names[key] = "@" + str(descriptor)
        entry["frozen_weights"], entry["descriptor"] = rt.artifact(target), rt.artifact(descriptor)
    opponents = {"teacher": "@" + str(rt.path(prepared["teacher_binding"]["descriptor"])), "incumbent": incumbent,
                 **{name: name for name in ("efficiency", "random", "tsumogiri")}}
    fixtures = {}
    for pool in POOLS:
        others = [opponents[name] for name in rt.protocol["evaluation"][pool + "_opponents"]]
        for key, candidate in names.items():
            fixtures[f"{pool}/{key}"] = identity(rt, [candidate, *others], seeds)
    contracts = {f"{pool}/{key}": explicit_contract(rt, fixtures[f"{pool}/incumbent"], fixtures[f"{pool}/{key}"])
                 for pool in POOLS for key in MODELS}
    registration = {"schema_version": 1, "kind": "phase2-evaluation-matrix-v1", "directory": "runs/" + args.run,
                    "scale": args.scale, "scale_from": scale_from, "implementation": rt.implementation_binding(),
                    "preparation": args.preparation, "preparation_evidence": evidence,
                    "models": models, "fixtures": fixtures, "contracts": contracts,
                    "selection": rt.protocol["evaluation"]["candidate_selection"],
                    "residual_exact_tie": "fail-closed; no undeclared tie-break",
                    "test_content_opened": False, "inference_device": "cpu", "torch_threads": 1}
    publish(output / "registration.json", registration)
    return {"status": "registered-not-played", "matrix": registration["directory"], "jobs": sorted(fixtures),
            "games_per_job": len(seeds) * 4, "development_games_total": len(fixtures) * len(seeds) * 4}


def load_matrix(rt, relative):
    matrix = read(rt.path(relative) / "registration.json")
    require(matrix.get("schema_version") == 1 and matrix.get("kind") == "phase2-evaluation-matrix-v1"
            and matrix["directory"] == relative and matrix["implementation"] == rt.implementation_binding(), "Matrix implementation or runtime changed")
    require(set(matrix["models"]) == set(MODELS)
            and set(matrix["fixtures"]) == {f"{pool}/{key}" for pool in POOLS for key in ("incumbent", *MODELS)}, "Incomplete registered matrix")
    _, prepared, jobs, audits, expected_evidence = preparation(rt, matrix["preparation"])
    require(matrix["preparation_evidence"] == expected_evidence, "Matrix preparation evidence was substituted")
    require(matrix["scale"] in ("small", "large") and all(audit["games"] == (1000 if matrix["scale"] == "small" else 10000)
                                                        for audit in audits.values()), "Matrix scale changed")
    verify_scale_authorization(rt, matrix)
    expected_models = training_models(rt, prepared, jobs)
    for entry in matrix["preparation_evidence"]:
        rt.bound(entry)
    for key, model in matrix["models"].items():
        require({name: value for name, value in model.items() if name not in ("frozen_weights", "descriptor")}
                == expected_models[key], "Registered model no longer matches the finished training job")
        for entry in [*model["training_evidence"], model["frozen_weights"], model["descriptor"]]:
            rt.bound(entry)
        require(digest(rt.path(model["frozen_weights"]["path"])) == model["best"]["sha256"]
                and read(rt.path(model["descriptor"]["path"])) == {"factory": MODEL_FACTORY, "kwargs": model["inference_kwargs"],
                    "artifacts": {"weights_path": {"path": "weights.pt", "sha256": model["best"]["sha256"]}}},
                "Frozen descriptor does not use the selected weights")
    incumbent = "@" + str(rt.path(rt.protocol["incumbent"]["descriptor"]))
    names = {"incumbent": incumbent, **{key: "@" + str(rt.path(model["descriptor"]["path"])) for key, model in matrix["models"].items()}}
    opponents = {"incumbent": incumbent, "teacher": "@" + str(rt.path(prepared["teacher_binding"]["descriptor"])),
                 **{name: name for name in ("efficiency", "random", "tsumogiri")}}
    seeds = sorted(rt.contract.seed_range(rt.protocol["seeds"]["development"]))
    for pool in POOLS:
        others = [opponents[name] for name in rt.protocol["evaluation"][pool + "_opponents"]]
        for key, candidate in names.items():
            require(identity(rt, [candidate, *others], seeds) == matrix["fixtures"][f"{pool}/{key}"],
                    "Matrix candidate, opponent pool, seeds or runtime identity changed")
        for key in MODELS:
            require(matrix["contracts"][f"{pool}/{key}"] == explicit_contract(rt, matrix["fixtures"][f"{pool}/incumbent"],
                    matrix["fixtures"][f"{pool}/{key}"]), "Registered comparison contract changed")
    return matrix


def verify_scale_authorization(rt, matrix):
    if matrix["scale"] == "small":
        require(matrix["scale_from"] is None, "Small matrix has an unexpected scale-up authorization")
        return
    authorization = matrix["scale_from"]
    require(isinstance(authorization, dict) and authorization["matrix"] != matrix["directory"], "Large matrix lacks its independent pilot gate")
    previous_root = rt.path(authorization["matrix"])
    previous_path = previous_root / "registration.json"
    require(digest(previous_path) == authorization["registration_sha256"], "Scale-up source registration changed")
    previous = read(previous_path)
    require(previous["scale"] == "small" and previous["scale_from"] is None
            and previous["implementation"] == matrix["implementation"], "Scale-up source is not the same frozen small experiment")
    report_path = rt.bound(authorization["development_report"])
    require(report_path == previous_root / "development-report.json", "Scale-up report belongs to another matrix")
    report = read(report_path)
    require(report["registration_sha256"] == authorization["registration_sha256"]
            and select_development(report["comparisons"])["scale_10000_eligible"], "Small pilot did not pass the scale-up gate")
    require(set(report["evidence"]) == set(previous["fixtures"]), "Scale-up source omitted a development campaign")
    for key, fixture in previous["fixtures"].items():
        require(verify_fixture(rt, previous_root / "development" / key, fixture) == report["evidence"][key],
                "Scale-up development campaign changed")


def verify_fixture(rt, path, expected):
    from agari_kit.evaluation_campaign import CONDITIONS, verify_campaign
    report = verify_campaign(path)
    manifest = read(path / "manifest.json")
    require(all(manifest.get(key) == expected[key] for key in (*CONDITIONS, "seeds")), "Campaign differs from registered matrix fixture")
    require(report["attempts"]["failed"] == 0 and report["attempts"]["unsealed"] == 0, "Failed or unsealed attempts disqualify the gate")
    return {name: digest(path / name) for name in ("plan.json", "completion.json", "manifest.json", "games.jsonl")}


def select_development(comparisons):
    require(set(comparisons) == {f"{pool}/{key}" for pool in POOLS for key in MODELS}, "All nine models in both pools must be reported")
    require(all(report["games_per_model"] == 1000 for report in comparisons.values()), "Development needs 1000 games per model/pool")
    def order(key):
        metrics = comparisons["primary/" + key]["candidate"]["metrics"]
        values = (metrics["mean_rank"], metrics["rank_4_rate"], -metrics["mean_point_delta"])
        require(all(math.isfinite(value) for value in values), "Non-finite development metric")
        return values
    ordered = sorted(MODELS, key=order)
    require(order(ordered[0]) != order(ordered[1]), "Exact unresolved development tie; no undeclared tie-break")
    eligible = [key for key in MODELS if key[0] in "BC" and comparisons["primary/" + key]["ci95"]["mean_rank"][1] < 0]
    return {"winner": ordered[0], "primary_order": ordered, "scale_10000_eligible": bool(eligible),
            "scale_10000_candidates": eligible,
            "scope": "Development selection and conditional data expansion only; no promotion claim",
            "multiplicity": "Nine candidates reported; individual development CIs are exploratory, final inference uses independent seeds"}


def development_report(rt, matrix):
    root = rt.path(matrix["directory"])
    evidence = {key: verify_fixture(rt, root / "development" / key, fixture) for key, fixture in matrix["fixtures"].items()}
    comparisons = {f"{pool}/{key}": rt.comparator.compare(root / "development" / pool / "incumbent",
                    root / "development" / pool / key, contract=matrix["contracts"][f"{pool}/{key}"])
                   for pool in POOLS for key in MODELS}
    return {"schema_version": 1, "status": "complete", "registration_sha256": digest(root / "registration.json"),
            "evidence": evidence, "comparisons": comparisons, **select_development(comparisons)}


def execute(rt, fixture, output, *, resume, max_wall_seconds, invocation_started):
    require(math.isfinite(max_wall_seconds) and 0 < max_wall_seconds <= 172800, "Invocation budget must be within 48 hours")
    require(identity(rt, fixture["lineup"], fixture["seeds"]) == fixture, "Runtime changed before execution")
    remaining = max_wall_seconds - (time.monotonic() - invocation_started)
    if remaining <= 0:
        return {"status": "paused", "reason": "preflight_wall_time_budget", "games_started": 0}
    from agari_kit.evaluation_campaign import evaluate_campaign
    return evaluate_campaign(fixture["seeds"], fixture["lineup"], output, resume=resume, max_wall_seconds=remaining)


def final_reservation(rt):
    seeds = sorted(rt.contract.seed_range(rt.protocol["seeds"]["final_promotion"]))
    return rt.path("runs/phase2-final-reservation-" + content_digest(seeds)[:16] + ".json")


def freeze_final(rt, matrix):
    root = rt.path(matrix["directory"])
    require(not (root / "final").exists(), "Final evaluation already exists; cannot select after seeing final seeds")
    report = development_report(rt, matrix)
    require(not (matrix["scale"] == "small" and report["scale_10000_eligible"]),
            "Eligible small pilot must proceed to the large matrix before consuming final seeds")
    publish(root / "development-report.json", report)
    winner = report["winner"]
    baseline = matrix["fixtures"]["primary/incumbent"]
    candidate = matrix["fixtures"]["primary/" + winner]
    seeds = sorted(rt.contract.seed_range(rt.protocol["seeds"]["final_promotion"]))
    require(not set(seeds) & set(baseline["seeds"]), "Final and development seeds overlap")
    fixtures = {side: identity(rt, fixture["lineup"], seeds) for side, fixture in (("incumbent", baseline), ("candidate", candidate))}
    frozen = {"schema_version": 1, "matrix": matrix["directory"], "scale": matrix["scale"], "winner": winner,
              "registration_sha256": digest(root / "registration.json"), "development_sha256": digest(root / "development-report.json"),
              "selected_model": matrix["models"][winner], "fixtures": fixtures,
              "contract": explicit_contract(rt, fixtures["incumbent"], fixtures["candidate"])}
    reservation = {"matrix": matrix["directory"], "freeze_sha256": content_digest(frozen), "seeds": seeds}
    # Shared exclusive reservation prevents switching small/large matrices after
    # inspecting any final comparison. Existing matching reservations are resumable.
    publish(final_reservation(rt), reservation)
    publish(root / "final-freeze.json", frozen)
    return {"status": "frozen-not-played", "winner": winner, "scale": matrix["scale"], "games_per_model": len(seeds) * 4}


def frozen_final(rt, matrix):
    root = rt.path(matrix["directory"])
    frozen = read(root / "final-freeze.json")
    require(frozen["matrix"] == matrix["directory"] and frozen["scale"] == matrix["scale"]
            and frozen["registration_sha256"] == digest(root / "registration.json")
            and frozen["development_sha256"] == digest(root / "development-report.json")
            and frozen["selected_model"] == matrix["models"][frozen["winner"]], "Frozen candidate binding changed")
    report = development_report(rt, matrix)
    require(report == read(root / "development-report.json") and report["winner"] == frozen["winner"], "Frozen development selection changed")
    seeds = sorted(rt.contract.seed_range(rt.protocol["seeds"]["final_promotion"]))
    require(read(final_reservation(rt)) == {"matrix": matrix["directory"], "freeze_sha256": content_digest(frozen), "seeds": seeds},
            "Final seed reservation belongs to another freeze")
    for side, key in (("incumbent", "incumbent"), ("candidate", frozen["winner"])):
        require(frozen["fixtures"][side] == identity(rt, matrix["fixtures"]["primary/" + key]["lineup"], seeds), "Final fixture differs from frozen primary pool")
    return frozen


def final_report(rt, matrix):
    frozen = frozen_final(rt, matrix)
    root = rt.path(matrix["directory"]) / "final"
    evidence = {side: verify_fixture(rt, root / side, fixture) for side, fixture in frozen["fixtures"].items()}
    comparison = rt.comparator.compare(root / "incumbent", root / "candidate", contract=frozen["contract"])
    passed = comparison["games_per_model"] == 10000 and comparison["ci95"]["mean_rank"][1] < 0
    return {"schema_version": 1, "status": "promoted" if passed else "not-promoted", "comparison": comparison,
            "evidence": evidence, "freeze_sha256": digest(root.parent / "final-freeze.json"),
            "scope": "Independent fixed-primary-pool evaluation; no online ranking claim"}


def plan_akochan(rt, matrix, descriptor):
    report = development_report(rt, matrix)
    winner = report["winner"]
    from agari_kit.plugins import agent_provenance
    name = "@" + str(Path(descriptor).resolve())
    provenance = agent_provenance(name)
    require(provenance["factory"] == "agari_kit.opponents.akochan:AkochanAgent", "Expected the Akochan adapter")
    settings = rt.protocol["seeds"]["akochan_exploratory"]
    first, total, start = settings["initial_count"], settings["count"], settings["start"]
    require((first, total) == (16, 64), "Unsupported Akochan exploration budget")
    candidates = {"incumbent": matrix["fixtures"]["primary/incumbent"]["lineup"][0],
                  "candidate": matrix["fixtures"]["primary/" + winner]["lineup"][0]}
    phases = {}
    for phase, offset, count in (("initial", 0, first), ("extension", first, total-first)):
        seeds = list(range(start+offset, start+offset+count))
        phases[phase] = {side: identity(rt, [candidate, name, "efficiency", "tsumogiri"], seeds) for side, candidate in candidates.items()}
    plan = {"schema_version": 1, "status": "commands-only-not-played", "winner": winner,
            "development_sha256": content_digest(report), "phases": phases,
            "contracts": {phase: explicit_contract(rt, sides["incumbent"], sides["candidate"]) for phase, sides in phases.items()},
            "note": "16 initial seed groups plus 48 disjoint extension groups = 64 total; continue after the initial paired campaigns pass engineering checks.",
            "pooled_report": "Not implemented: verify both campaigns and reject overlap before pooling paired groups; never average their confidence intervals.",
            "selection_use": "Diagnostic only; does not change the primary-pool selection or final gate",
            "commands": []}
    for phase, sides in phases.items():
        for side, fixture in sides.items():
            run = matrix["directory"].removeprefix("runs/") + f"/akochan/{phase}/{side}"
            plan["commands"].append({"phase": phase, "side": side, "argv": [sys.executable, "-B", "-m", "agari_kit.cli", "evaluate",
                "--candidate", fixture["lineup"][0], "--opponents", *fixture["lineup"][1:], "--seed-start", str(fixture["seeds"][0]),
                "--seed-count", str(len(fixture["seeds"])), "--run", run, "--campaign", "--max-wall-seconds", "172800"],
                "PYTHONPATH": os.pathsep.join(str(rt.root / repo / "src") for repo in ("agari-kit", "agari-research"))})
    publish(rt.path(matrix["directory"]) / "akochan-plan.json", plan)
    return {"status": plan["status"], "winner": winner, "initial_games_per_model": first*4, "extension_games_per_model": (total-first)*4,
            "pooled_64_group_report": "not-implemented"}


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    registration = commands.add_parser("register")
    registration.add_argument("--preparation", required=True)
    registration.add_argument("--run", required=True)
    registration.add_argument("--scale", choices=("small", "large"), required=True)
    registration.add_argument("--scale-from")
    for name in ("run", "report-development", "freeze-final", "run-final", "report-final", "plan-akochan"):
        command = commands.add_parser(name)
        command.add_argument("--matrix", required=True)
        if name in ("run", "run-final"):
            command.add_argument("--resume", action="store_true")
            command.add_argument("--max-wall-seconds", type=float, default=172800)
        if name == "run":
            command.add_argument("--job", choices=[f"{pool}/{key}" for pool in POOLS for key in ("incumbent", *MODELS)], required=True)
        elif name == "run-final":
            command.add_argument("--side", choices=("incumbent", "candidate"), required=True)
        elif name == "plan-akochan":
            command.add_argument("--akochan-descriptor", type=Path, required=True)
    args = parser.parse_args()
    if args.command in ("run", "run-final"):
        require(math.isfinite(args.max_wall_seconds) and 0 < args.max_wall_seconds <= 172800,
                "Invocation budget must be within 48 hours")
    rt = Runtime(args.runtime_root)
    if args.command == "register":
        result = register(rt, args)
    else:
        matrix = load_matrix(rt, args.matrix)
        root = rt.path(args.matrix)
        if args.command == "run":
            result = execute(rt, matrix["fixtures"][args.job], root / "development" / args.job,
                             resume=args.resume, max_wall_seconds=args.max_wall_seconds, invocation_started=started)
        elif args.command == "report-development":
            report = development_report(rt, matrix)
            publish(root / "development-report.json", report)
            result = {key: report[key] for key in ("status", "winner", "primary_order", "scale_10000_eligible", "scale_10000_candidates")}
        elif args.command == "freeze-final":
            result = freeze_final(rt, matrix)
        elif args.command == "run-final":
            frozen = frozen_final(rt, matrix)
            result = execute(rt, frozen["fixtures"][args.side], root / "final" / args.side,
                             resume=args.resume, max_wall_seconds=args.max_wall_seconds, invocation_started=started)
        elif args.command == "report-final":
            report = final_report(rt, matrix)
            publish(root / "final-report.json", report)
            result = {key: report[key] for key in ("status", "scope")}
        else:
            result = plan_akochan(rt, matrix, args.akochan_descriptor)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
