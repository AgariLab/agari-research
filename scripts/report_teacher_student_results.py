"""Seal a thin report after all nine models and twenty development campaigns finish.

Use the original frozen b9ee tools and paired runtime, from a clean detached
checkout of this script. Never trains, evaluates, expands data, opens dataset
traces/test actions, or accesses final evaluation results. No partial reports.
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

MODELS = tuple(f"{arm}-{seed}" for arm in "ABC" for seed in (2026, 2027, 2028))
POOLS = ("primary", "weak")
FIXTURES = {f"{pool}/{key}" for pool in POOLS for key in ("incumbent", *MODELS)}
TOOLS_COMMIT = "b9ee0715c8e1c16ea811e3e37f2a8ebd89814671"
TEACHER = "runs/phase2-mortal-s42-v1/agent.json"
QUALIFICATION = "runs/phase2-teacher-qualification-v1"
LEGALITY = "runs/phase2-mortal-s42-protocol-qualification-v1/report.json"
COSTS = ("invocation_wall_seconds", "child_wall_seconds", "user_seconds", "system_seconds", "max_rss_bytes")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def run_path(value):
    require(isinstance(value, str) and re.fullmatch(r"runs/phase2-[A-Za-z0-9_.-]{1,100}", value),
            "Expected one storage-relative runs/phase2-* directory")
    return value


class Evidence:
    """Hash external artifacts; retain bytes only for small JSON and selected rows."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.files, self.directories = {}, {}

    def path(self, relative):
        path = Path(relative)
        require(isinstance(relative, str) and not path.is_absolute() and ".." not in path.parts
                and path.parts and "final" not in path.parts
                and (path.parts[0] in ("runs", "models", "checkpoints")
                     or path.parts[:2] == ("datasets", "manifests") and path.suffix == ".json"),
                "Artifact path is outside the metadata/development boundary")
        resolved = (self.root / path).resolve()
        require(resolved == self.root / path and resolved.is_relative_to(self.root),
                "Artifact path escapes or aliases storage")
        return resolved

    def bind(self, relative, expected=None):
        path = self.path(relative)
        entry = {"path": relative, "sha256": digest(path), "bytes": path.stat().st_size}
        require(expected is None or entry["sha256"] == expected, "Bound artifact SHA differs: " + relative)
        require(relative not in self.files or self.files[relative] == entry, "Artifact changed during report")
        self.files[relative] = entry
        return entry

    def read(self, relative, expected=None):
        data = self.path(relative).read_bytes()
        entry = self.bind(relative, expected)
        require(sha(data) == entry["sha256"], "Artifact changed while reading")
        return json.loads(data)

    def inventory(self, relative):
        path = self.path(relative)
        # Only names/types, never reads unknown files or follows their symlinks.
        value = None if not path.exists() else sorted(
            (str(item.relative_to(path)), "symlink" if item.is_symlink() else "directory" if item.is_dir() else "file")
            for item in path.rglob("*") if item.name != "job.lock")
        require(relative not in self.directories or self.directories[relative] == value,
                "Attempt inventory changed during report")
        self.directories[relative] = value
        return value

    def no_symlinks(self, relative):
        """Inspect directory entries only, before frozen helpers open leaf files."""
        root = self.path(relative)
        for directory, directories, files in os.walk(root, followlinks=False):
            require(all(not (Path(directory) / name).is_symlink() for name in directories + files),
                    "Artifact tree contains a symlink: " + relative)

    def verify(self):
        for relative, entry in tuple(self.files.items()):
            self.bind(relative, entry["sha256"])
        for relative in tuple(self.directories):
            self.inventory(relative)


def ready(evidence, relative):
    """Completion and pointer boundary before helpers or any development scores."""
    run_path(relative)
    matrix = evidence.read(relative + "/registration.json")
    require(matrix.get("directory") == relative and matrix.get("scale") in ("small", "large")
            and set(matrix.get("models", {})) == set(MODELS)
            and set(matrix.get("fixtures", {})) == FIXTURES, "Incomplete registered nine-model matrix")
    prep = run_path(matrix["preparation"])
    manifest = evidence.read(prep + "/manifest.json")
    evidence.no_symlinks(prep)
    require(manifest.get("status") == "prepared-not-trained"
            and manifest.get("test_partition_content_opened") is False
            and manifest["teacher_binding"]["descriptor"] == TEACHER
            and manifest["qualification"]["run"] == QUALIFICATION,
            "Preparation or qualified teacher path differs")
    evidence.path(TEACHER)
    qualification = evidence.read(QUALIFICATION + "/registration.json")
    require(qualification["legality"]["path"] == LEGALITY, "Legality evidence must use its prior fixed path")
    evidence.path(LEGALITY)
    evidence.no_symlinks(QUALIFICATION)
    jobs = {f"{job['arm']}-{job['seed']}": job for job in manifest["jobs"]}
    require(len(manifest["jobs"]) == len(jobs) == 9 and set(jobs) == set(MODELS), "Exactly nine prepared jobs required")
    for key, job in jobs.items():
        run_path("runs/" + job["run"])
        require(str(Path(job["config"]).parent) == prep + "/configs", "Job config is outside its preparation")
        evidence.path(job["config"])
        evidence.path(f"checkpoints/{job['run']}/latest.pt")
        evidence.path(f"models/{job['run']}")
        model = matrix["models"][key]
        require(model["frozen_weights"]["path"] == f"{relative}/models/{key}/weights.pt"
                and model["descriptor"]["path"] == f"{relative}/models/{key}/agent.json",
                "Matrix model artifacts must stay in their registered model directory")
        evidence.path(model["frozen_weights"]["path"])
        evidence.path(model["descriptor"]["path"])
        for tree in (f"runs/{job['run']}", f"checkpoints/{job['run']}", f"models/{job['run']}",
                     f"runs/{Path(prep).name}-execution/{key}"):
            evidence.no_symlinks(tree)
        result = evidence.read(f"runs/{job['run']}/result.json")
        require(result.get("status") == "complete", "All nine training jobs must be complete")
    for key in FIXTURES:
        require(evidence.path(f"{relative}/development/{key}/completion.json").is_file(),
                "All twenty development campaigns must be sealed")
    evidence.no_symlinks(relative + "/development")
    # The original helper follows these source metadata links. Constrain them
    # before invoking it; no dataset trace_path/mjai_path is ever dereferenced.
    for name in ("weak", "teacher"):
        entry = manifest["datasets"][name]
        split_path = Path(entry["split"])
        require(split_path.parts[:2] == ("datasets", "manifests") and len(split_path.parts) == 4
                and split_path.suffix == ".json", "Split must be acquisition metadata")
        parent = str(split_path.parent)
        split = evidence.read(entry["split"])
        audit = evidence.read(f"{prep}/{name}-source-audit.json")
        require(audit["acquisition"] == parent + "/acquisition.json", "Acquisition path differs")
        evidence.path(audit["acquisition"])
        for row in [row for part in split["partitions"].values() for row in part]:
            require(str(Path(row["validation_receipt"]).parent) == parent
                    and row["validation_receipt"].endswith(".verification.json"), "Receipt outside acquisition metadata")
            evidence.path(row["validation_receipt"])
        for row in audit["receipt_inventory"]:
            require(str(Path(row["receipt"]).parent) == parent and row["receipt"].endswith(".verification.json"),
                    "Audit receipt outside acquisition metadata")
            evidence.path(row["receipt"])
        require(all(job["split"] == entry["split"] for job in jobs.values()
                    if ("weak" if job["arm"] == "A" else "teacher") == name), "Job source differs from preparation")
    if matrix["scale"] == "large":
        authorization = matrix["scale_from"]
        previous = run_path(authorization["matrix"])
        require(previous != relative and authorization["development_report"]["path"] == previous + "/development-report.json",
                "Large matrix must refer to its registered small development report")
        old = evidence.read(previous + "/registration.json", authorization["registration_sha256"])
        require(old["scale"] == "small" and old["scale_from"] is None and set(old["fixtures"]) == FIXTURES,
                "Invalid prior small matrix boundary")
        evidence.no_symlinks(previous + "/development")
        evidence.bind(previous + "/development-report.json", authorization["development_report"]["sha256"])
    else:
        require(matrix["scale_from"] is None, "Small matrix cannot carry scale-up authorization")
    return matrix, manifest, jobs


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Context:
    def __init__(self, runtime_root, tools_root, storage_root):
        sys.dont_write_bytecode = True  # Importing frozen helpers must not create caches there.
        self.tools = Path(tools_root).resolve()
        self.matrix = load("results_frozen_matrix", self.tools / "scripts/phase2_evaluation_matrix.py")
        self.runner = load("results_frozen_runner", self.tools / "scripts/run_prepared_training.py")
        self.rt = self.matrix.Runtime(runtime_root)
        require(self.rt.storage.root.resolve() == storage_root, "Report and runtime storage differ")
        self.initial = self.binding()

    def binding(self):
        from agari_research.runtime import runtime_provenance
        from agari_research.training import implementation_sha256
        self.matrix.check_module_paths(self.rt.root)
        tools = self.matrix.checkout_identity(self.tools)
        require(tools["commit"] == TOOLS_COMMIT, "Report requires the original b9ee matrix/runner tools")
        require(self.rt.runtime == runtime_provenance() and self.rt.implementation == implementation_sha256(),
                "Runtime identity changed")
        return {"tools_checkout": tools,
                "report_checkout": self.matrix.checkout_identity(Path(__file__).resolve().parents[1]),
                "runtime_checkouts": {name: self.matrix.checkout_identity(self.rt.root / name)
                                      for name in ("agari-kit", "agari-research")},
                "implementation": self.rt.implementation_binding(),
                "scripts": {"report": digest(__file__), "matrix": digest(self.matrix.__file__),
                            "runner": digest(self.runner.__file__)},
                "python": {"path": str(Path(sys.executable).resolve()), "sha256": digest(sys.executable)}}

    def check(self):
        require(self.binding() == self.initial, "Report source or runtime changed")


def counter(value):
    n, correct = value["samples"], value["correct"]
    require(type(n) is int and type(correct) is int and 0 <= correct <= n, "Invalid diagnostic counter")
    accuracy = value["accuracy"]
    require(accuracy is None if n == 0 else type(accuracy) in (int, float)
            and math.isclose(accuracy, correct / n, rel_tol=1e-6, abs_tol=1e-7), "Diagnostic accuracy differs from counts")
    return n, correct


def validation_row(path, method, best):
    matches = []
    with Path(path).open("rb") as stream:
        for raw in stream:
            if raw.strip():
                row = json.loads(raw)
                if row.get("stage") == method and row.get("update") == best["update"]:
                    matches.append((row, raw))
    require(len(matches) == 1, "Selected update must have exactly one original metrics row")
    row, raw = matches[0]
    require(row.get("validation_scope") == "full" and row.get("validation_sample_limit") is None
            and row.get("validation_loss") == best["value"] and best["metric"] == "validation_loss",
            "Selected validation differs from the frozen best choice")
    diagnostics = row["validation_policy_diagnostics"]
    require(diagnostics.get("status") == "complete" and diagnostics.get("metric") == "encoded_action_accuracy"
            and diagnostics.get("scope") == "full" and diagnostics.get("sample_limit") is None,
            "Full policy diagnostics are required")
    for kind in ("choice", "forced"):
        expected = counter(diagnostics[kind])
        values = [counter(group[kind]) for group in diagnostics["by_action_type"].values()]
        require(tuple(sum(value[index] for value in values) for index in (0, 1)) == expected,
                "Action-type diagnostic totals differ")
    require(diagnostics["choice"]["samples"] == row["validation_samples"] > 0
            and math.isclose(row["validation_accuracy"], diagnostics["choice"]["accuracy"], rel_tol=1e-6, abs_tol=1e-7),
            "Validation and choice diagnostic denominators differ")
    return {key: value for key, value in row.items() if key.startswith("validation_")}, raw


def classify_attempt(started, finished, prior, method, targets):
    """Historical checkpoint SHAs are identities, never hashes of today's latest.pt."""
    result = {"stage": "unattributed", "start_progress": None, "end_progress": None, "planned_grp_boundary": False}
    if not started or not finished or "process" not in finished:
        return result
    end = finished.get("cli_result", {}).get("progress")
    start = None
    if started.get("resume") is False:
        start = dict.fromkeys(targets, 0)
    elif finished.get("resume_checkpoint"):
        candidates = [item.get("cli_result", {}).get("progress") for item in prior
                      if item.get("latest_checkpoint") == finished["resume_checkpoint"]]
        if candidates and candidates[0] is not None and all(value == candidates[0] for value in candidates):
            start = candidates[0]
    result.update(start_progress=start, end_progress=end)
    if method == "bc":
        result["stage"] = "bc"
        return result
    if not start or not end or set(start) != set(targets) or set(end) != set(targets):
        return result
    if any(type(start[s]) is not int or type(end[s]) is not int or not 0 <= start[s] <= end[s] <= targets[s]
           for s in targets):
        return result
    grp, cql = end["grp"] - start["grp"], end["cql"] - start["cql"]
    if grp > 0 and cql > 0:
        result["stage"] = "mixed-grp-cql"
    elif grp > 0 and cql == 0:
        result["stage"] = "grp"
    elif cql > 0 and grp == 0 and start["grp"] == targets["grp"]:
        result["stage"] = "cql"
    result["planned_grp_boundary"] = (start == {"grp": 0, "cql": 0}
        and end == {"grp": targets["grp"], "cql": 0} and started.get("stop_after_updates") == targets["grp"]
        and finished.get("status") == "interrupted" and finished.get("stop_reason") == "update_limit"
        and finished.get("exit_code") == 2 and finished["process"].get("exit_code") == 0)
    return result


def summarize_costs(rows):
    summary = {}
    for key in COSTS:
        values = [row["costs"].get(key) for row in rows if row["costs"].get(key) is not None]
        require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in values),
                "Invalid recorded attempt cost")
        summary[key] = {"observed_max" if key == "max_rss_bytes" else "observed_sum":
                        (max(values) if key == "max_rss_bytes" else sum(values)) if values else None,
                        "known_attempts": len(values), "unknown_attempts": len(rows) - len(values)}
    return summary


def training_attempts(evidence, context, directory, prepared, key, job, prep_evidence, targets):
    execution = f"runs/{directory.name}-execution/{key}"
    inventory = evidence.inventory(execution)
    if inventory is None:
        return {"directory": execution, "status": "not-recorded", "attempts": [],
                "costs": summarize_costs([]), "batch_wall_seconds": None}
    binding_path = execution + "/binding.json"
    binding = evidence.read(binding_path)
    expected = context.runner.execution_binding(context.rt, context.matrix, directory, prepared, job, prep_evidence)
    require(binding == expected, "Original runner execution binding differs")
    attempts, prior = [], []
    for attempt in sorted(evidence.path(execution).iterdir()):
        if not attempt.name.startswith("attempt-"):
            continue
        require(attempt.is_dir() and not attempt.is_symlink() and re.fullmatch(r"attempt-\d{4,}", attempt.name),
                "Malformed runner attempt directory")
        relative = execution + "/" + attempt.name
        artifacts, documents, document_errors = {}, {}, {}
        for name in ("started.json", "command.json", "finished.json", "stdout.log", "stderr.log"):
            path = relative + "/" + name
            if evidence.path(path).exists():
                artifacts[name] = evidence.bind(path)
                if name.endswith(".json"):
                    try:
                        documents[name] = evidence.read(path)
                    except json.JSONDecodeError as exc:
                        document_errors[name] = str(exc)
        started, finished = documents.get("started.json"), documents.get("finished.json")
        if started:
            require(started["job_id"] == key and started["binding_sha256"] == evidence.files[binding_path]["sha256"]
                    and started["targets"] == targets, "Attempt start belongs to another job/budget")
        command = documents.get("command.json")
        if command and started:
            budget = command["child_max_wall_seconds"]
            require(type(budget) in (int, float) and math.isfinite(budget) and 0 < budget <= started["max_wall_seconds"],
                    "Child command budget differs from invocation")
            argv = [binding["python"]["path"], "-B", "-u", "-m", "agari_research.cli", "run", "--config",
                    str(evidence.path(job["config"])), "--split", job["split"], "--run", job["run"],
                    "--max-wall-seconds", str(budget)]
            # sys.executable may be the venv symlink; binding records its resolved binary.
            observed = list(command["argv"])
            require(Path(observed[0]).resolve() == Path(binding["python"]["path"]), "Child Python differs")
            observed[0] = binding["python"]["path"]
            if started["resume"]:
                argv.extend(["--resume", f"checkpoints/{job['run']}/latest.pt"])
            if started["stop_after_updates"] is not None:
                argv.extend(["--stop-after-updates", str(started["stop_after_updates"])])
            require(observed == argv and command["cwd"] == str(context.rt.root / "agari-research")
                    and command["pythonpath"] == os.pathsep.join(str(context.rt.root / repo / "src") for repo in ("agari-kit", "agari-research")),
                    "Recorded child command differs from prepared invocation")
        if finished:
            require(started is not None and finished["job_id"] == key and finished["attempt"] == relative
                    and finished["started_sha256"] == artifacts["started.json"]["sha256"], "Finished attempt binding differs")
            logs = {item["path"]: item["sha256"] for item in finished["logs"]}
            require(logs == {item["path"]: item["sha256"] for name, item in artifacts.items() if name.endswith(".log")},
                    "Finished attempt log binding differs")
            if "cli_result" in finished:
                require("stdout.log" in artifacts and context.runner.stdout_result(attempt / "stdout.log") == finished["cli_result"],
                        "Historical CLI snapshot differs from its original stdout")
            for field, path in (("resume_checkpoint", f"checkpoints/{job['run']}/latest.pt"),
                                ("latest_checkpoint", f"checkpoints/{job['run']}/latest.pt"),
                                ("training_result", f"runs/{job['run']}/result.json")):
                if field in finished:
                    require(finished[field]["path"] == path and re.fullmatch(r"[0-9a-f]{64}", finished[field]["sha256"]),
                            "Historical artifact identity belongs to another job")
        process = (finished or {}).get("process", {})
        costs = {"invocation_wall_seconds": (finished or {}).get("invocation_wall_seconds"),
                 "child_wall_seconds": process.get("wall_seconds"), "user_seconds": process.get("user_seconds"),
                 "system_seconds": process.get("system_seconds"), "max_rss_bytes": process.get("max_rss_bytes")}
        row = {"attempt": relative, "status": (finished or {}).get("status", "unsealed"),
               "stop_reason": (finished or {}).get("stop_reason"), "exit_code": (finished or {}).get("exit_code"),
               "error": (finished or {}).get("error"), "costs": costs, "artifacts": artifacts,
               "started": started, "command": command, "process": process or None, "document_errors": document_errors,
               "historical_snapshots": {field: finished[field] for field in ("resume_checkpoint", "latest_checkpoint", "training_result")
                                        if finished and field in finished},
               **classify_attempt(started, finished, prior, job["method"], targets)}
        attempts.append(row)
        if finished:
            prior.append(finished)
    return {"directory": execution, "status": "recorded", "attempts": attempts, "costs": summarize_costs(attempts),
            "costs_by_stage": {stage: summarize_costs([row for row in attempts if row["stage"] == stage])
                               for stage in sorted({row["stage"] for row in attempts})},
            "batch_wall_seconds": None,
            "scope": "All original runner attempts, including failures/unsealed receipts; inspect and outer batch-driver overhead are not recorded here"}


def training_rows(evidence, context, matrix, prepared, jobs, prep_evidence):
    rows, excerpts = [], {}
    for key in MODELS:
        job, model = jobs[key], matrix["models"][key]
        for entry in model["training_evidence"] + [model["frozen_weights"], model["descriptor"]]:
            evidence.bind(entry["path"], entry["sha256"])
        result = evidence.read(f"runs/{job['run']}/result.json")
        plan = evidence.read(f"runs/{job['run']}/execution-plan.json")["plan"]
        config = evidence.read(job["config"], job["config_sha256"])
        targets = {stage: value["target_updates"] for stage, value in plan["stages"].items()}
        validation, raw = validation_row(evidence.path(f"runs/{job['run']}/metrics.jsonl"), job["method"], model["best"])
        excerpt = f"selected-validation/{key}.jsonl"
        excerpts[excerpt] = raw
        name = "weak" if job["arm"] == "A" else "teacher"
        counts = evidence.read(f"{matrix['preparation']}/{name}-training-counts.json")
        versioned_descriptor = str(Path(model["weights"]["path"]).with_name(Path(model["weights"]["path"]).stem + "-agent.json"))
        rows.append({"job": key, "arm": job["arm"], "seed": job["seed"], "source": name, "method": job["method"],
            "run": job["run"], "config": {"path": job["config"], "sha256": job["config_sha256"]},
            "device": config["device"], "dataset_sha256": result["dataset_sha256"],
            "policy_initialization": model["policy_initialization"],
            "policy_updates_U": targets[job["method"]], "grp_updates_G": targets.get("grp", 0),
            "progress": result["progress"], "training_plan_sha256": plan["sha256"],
            "stage_budgets": plan["stages"], "training_coverage": result["training_coverage"],
            "source_counts": {"train_games": len(counts), "train_nonforced_policy_decisions": sum(row["cql_samples"] for row in counts),
                              "train_grp_samples": sum(row["grp_samples"] for row in counts)},
            "best": model["best"], "weights": model["weights"],
            "versioned_descriptor": evidence.bind(versioned_descriptor), "evaluation_descriptor": model["descriptor"],
            "validation": validation, "validation_original_line": {"path": excerpt, "sha256": sha(raw), "bytes": len(raw)},
            "execution": training_attempts(evidence, context, evidence.path(matrix["preparation"]), prepared,
                                            key, job, prep_evidence, targets)})
    return rows, excerpts


def campaign_rows(evidence, matrix, report):
    rows = []
    for pool in POOLS:
        for key in ("incumbent", *MODELS):
            fixture = pool + "/" + key
            root = matrix["directory"] + "/development/" + fixture
            for name, expected in report["evidence"][fixture].items():
                require(name in ("plan.json", "completion.json", "manifest.json", "games.jsonl"), "Unexpected campaign evidence")
                evidence.bind(root + "/" + name, expected)
            receipt = evidence.read(root + "/completion.json")
            for name in ("report.json", "candidate-latencies.json"):
                evidence.bind(root + "/" + name, receipt["sha256"][name])
            source = evidence.read(root + "/report.json")
            require(source["status"] == "complete" and source["games"] == 1000 and source["seed_groups"] == 250
                    and source["illegal_actions"] == 0 and source["attempts"]["failed"] == source["attempts"]["unsealed"] == 0,
                    "Expected a complete zero-failure development campaign")
            comparison = None if key == "incumbent" else report["comparisons"][fixture]
            rows.append({"pool": pool, "job": key, "report": evidence.files[root + "/report.json"],
                "games": source["games"], "seed_groups": source["seed_groups"], "missing_games": 1000 - source["games"],
                "metrics": source["candidate"], "single_campaign_ci95": source["ci95"],
                "paired_difference": comparison["difference"] if comparison else None,
                "paired_ci95": comparison["ci95"] if comparison else None,
                "paired_direction": "candidate minus same-pool incumbent" if comparison else None,
                "candidate_diagnostics": source["candidate_diagnostics"],
                "latency": {name: source[name] for name in ("candidate_decisions", "candidate_latency_ms_mean",
                    "candidate_latency_ms_p95", "candidate_latency_ms_max", "latency_scope")},
                "all_seats_decisions": source["decisions"], "illegal_actions": source["illegal_actions"], "attempts": source["attempts"],
                "cost": {"recorded_game_loop_sum_seconds": source["elapsed_seconds"],
                         "full_command_wall_seconds": None, "full_command_max_rss_bytes": None,
                         "newly_executed_games_for_this_scale": None,
                         "scope": "Sealed campaign sample; matrix does not bind full-command costs or baseline reuse history"}})
    return rows


def readme(summary):
    campaigns = {(row["pool"], row["job"]): row for row in summary["development"]}
    lines = ["# Teacher/student development results", "",
        f"Scale: {summary['scale']}. Matrix: `{summary['matrix']}`. All nine completed models and twenty sealed campaigns were rechecked by the original frozen gates.", "",
        "| Job | U | G | Best update | Full choice accuracy | Primary mean rank | Primary paired rank CI | Weak mean rank |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |"]
    for row in summary["training"]:
        primary, weak = (campaigns[(pool, row["job"])] for pool in POOLS)
        lines.append(f"| {row['job']} | {row['policy_updates_U']} | {row['grp_updates_G']} | {row['best']['update']} | "
                     f"{row['validation']['validation_policy_diagnostics']['choice']['accuracy']:.6f} | "
                     f"{primary['metrics']['mean_rank']:.6f} | {primary['paired_ci95']['mean_rank']} | {weak['metrics']['mean_rank']:.6f} |")
    lines += ["", f"The registered development rule selects **{summary['selection']['winner']}**. "
              f"Its existing conditional scale gate is `{summary['selection']['scale_10000_eligible']}`; this report performs no expansion or final evaluation.", "",
        "Selection uses primary-pool mean rank, fourth rate, then point delta, exactly as registered. Validation loss selects each method's own checkpoint; BC and CQL losses are not comparable rankings. A uses weak-source validation while B/C use teacher-source validation: choice accuracy is a diagnostic for its own source, not a shared-validation-set ranking. All three training seeds and both pools remain separate; no CI or percentile is averaged across them. This is fixed-pool development evidence, not a global strength, Elo, final-test, or promotion claim.", "",
        "summary.json retains all nine selected full-validation diagnostics (choice, forced, action type), U/G, source counts and coverage. selected-validation/ contains the exact nine original metrics rows, including their original training fields; they are excerpts, not newly evaluated predictions. Policy updates have a fixed sampling budget, not an assertion of complete dataset coverage. Legacy weak test identity limitations remain in the bound preparation audits; no dataset trace, MJAI or test actions are read.", "",
        "Every original runner attempt is retained, including planned GRP update-limit stops, budget pauses, failures and unsealed attempts. Historical latest.pt/result.json SHAs identify bytes at that attempt; these mutable paths are not compared with today's bytes. Unknown costs are null, never zero. Runner invocation wall includes preflight and completion verification but is recorded before final log hashing; child wall includes native preflight/restoration. CPU and peak RSS come from wait4 for the training child. Peak RSS is not summed. Sums of invocation/child durations include overlaps and are not batch elapsed time; inspect and outer-driver overhead are not bound here.", "",
        "Each development row describes 1000 sealed hanchan in 250 complete paired seed groups. Paired differences are candidate minus that pool's single incumbent; incumbent rows have no paired CI. Single-campaign CIs are distinct from paired CIs. hands counts recorded kyoku, not hanchan or decisions; only the original four behavior rates are copied. deal_in_payments counts payment events, not points; meld_events counts events, not distinct melds.", "",
        "Latency is milliseconds over all candidate act calls, including forced/pass. The original scope string is preserved: its exclusion of transport is imprecise because act includes local IPC, feature encoding and any lazy work inside act; external constructor/reset, legal checks and env.step are outside this timer. Campaign elapsed time sums completed game loops after reset and excludes initialization and merge/preflight. Full-command wall/RSS are unknown because the matrix does not bind them. Large-scale controls may reuse small-scale artifacts: twenty sealed samples do not imply twenty thousand newly run games or incremental cost. Zero illegal actions in completed rows is interpreted together with the sealed zero-failed/unsealed-attempt gate.", "",
        "sources.json binds originals left on external storage; source and artifact hashes are checked before publication. Existing winner/scale decisions are copied from the independently recomputed development report. No final cohort results are opened.", ""]
    return "\n".join(lines)


def build(evidence, context, relative, output):
    matrix, prepared, jobs = ready(evidence, relative)
    run_path(output)
    destination = evidence.path(output)
    require(not destination.exists(), "Report output must be a new directory")
    require(context.matrix.load_matrix(context.rt, relative) == matrix, "Frozen matrix gate disagrees")
    directory, checked, checked_jobs, audits, prep_evidence = context.matrix.preparation(context.rt, matrix["preparation"])
    require(checked == prepared and checked_jobs == jobs, "Frozen preparation gate disagrees")
    for entry in prep_evidence:
        evidence.bind(entry["path"], entry["sha256"])
    report = context.matrix.development_report(context.rt, matrix)
    require(evidence.read(relative + "/development-report.json") == report, "Saved development report differs from original gate")
    training, excerpts = training_rows(evidence, context, matrix, prepared, jobs, prep_evidence)
    development = campaign_rows(evidence, matrix, report)
    attempts = [attempt for row in training for attempt in row["execution"]["attempts"]]
    summary = {"schema_version": 1, "status": "complete-development-report", "matrix": relative,
        "scale": matrix["scale"], "scale_from": matrix["scale_from"], "preparation": matrix["preparation"],
        "selection": {key: report[key] for key in ("winner", "primary_order", "scale_10000_eligible", "scale_10000_candidates", "scope", "multiplicity")},
        "training": training, "development": development,
        "training_cost_totals": {"recorded_attempts": len(attempts), "costs": summarize_costs(attempts),
            "jobs_without_recorded_execution": [row["job"] for row in training if row["execution"]["status"] == "not-recorded"],
            "scope": "Observed original-runner costs only; duration sums include concurrency overlap and are not full experiment or batch wall time"},
        "source_audit_limitations": {name: {key: value for key, value in audit.items() if key in
            ("legacy_test_games_without_per_game_policy_binding", "test_policy_identity_evidence", "test_partition_content_opened")}
            for name, audit in audits.items()},
        "source_binding": context.initial,
        "read_boundary": {"dataset_traces_opened": False, "dataset_mjai_opened": False, "test_partition_content_opened": False,
                          "final_results_opened": False, "qualification_engineering_evidence": "Original qualification gate rechecks existing engineering artifact hashes"}}
    # Re-run the original transitive gates: these also detect campaign group and
    # source-receipt changes not individually duplicated into the thin report.
    ready(evidence, relative)  # Recheck leaf aliases before helpers reopen files.
    require(context.matrix.load_matrix(context.rt, relative) == matrix
            and context.matrix.development_report(context.rt, matrix) == report, "Frozen evidence changed during report")
    context.check()
    evidence.verify()
    sources = {"schema_version": 1, "storage_root": str(evidence.root),
               "artifacts": [evidence.files[key] for key in sorted(evidence.files)],
               "attempt_directory_inventories": evidence.directories,
               "historical_hash_scope": "Historical checkpoint/result identities are inside original finished receipts, not current artifact bindings"}
    payloads = {**excerpts, "sources.json": (canonical(sources) + "\n").encode(), "README.md": readme(summary).encode()}
    summary["report_artifacts"] = {name: {"sha256": sha(data), "bytes": len(data)} for name, data in payloads.items()}
    payloads["summary.json"] = (canonical(summary) + "\n").encode()
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in payloads.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    return {"status": summary["status"], "output": output, "summary_sha256": digest(destination / "summary.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--tools-root", type=Path, required=True, help="Original frozen b9ee tools checkout")
    parser.add_argument("--matrix", required=True, help="Registered completed small or large matrix")
    parser.add_argument("--output", required=True, help="New external runs/phase2-* report directory")
    args = parser.parse_args(argv)
    evidence = Evidence(os.environ.get("AGARILAB_STORAGE_ROOT", "/Volumes/新加卷/AgariLab"))
    ready(evidence, args.matrix)
    context = Context(args.runtime_root, args.tools_root, evidence.root)
    print(canonical(build(evidence, context, args.matrix, args.output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
