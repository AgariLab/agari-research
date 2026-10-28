"""One finite small development matrix after a real completed training batch.

The original frozen matrix CLI owns registration, games and statistics. This
driver only waits, dispatches its twenty fixed campaigns, and seals receipts.
No resume, large-data collection, final reservation or Akochan is automatic.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

MODELS = tuple(f"{arm}-{seed}" for arm in "ABC" for seed in (2026, 2027, 2028))
JOBS = tuple(f"{pool}/{key}" for pool in ("primary", "weak") for key in ("incumbent", *MODELS))
PROBES = JOBS[:2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Context:
    def __init__(self, runtime_root, tools_root, training_tools_root):
        self.training_tools = Path(training_tools_root).resolve()
        self.batch = load(self.training_tools / "scripts/phase2_training_batch.py", "development_training_batch")
        self.training = self.batch.Context(runtime_root, tools_root)
        self.matrix, self.rt, self.runner = self.training.matrix, self.training.rt, self.training.runner
        self.tools = self.training.tools

    def path(self, relative):
        return self.rt.path(relative)

    def artifact(self, path):
        return self.rt.artifact(path)

    def binding(self, training_run):
        directory = self.path(training_run)
        registration = self.matrix.read(directory / "registration.json")
        require(registration["kind"] == "phase2-nine-training-batch-v1"
                and registration["directory"] == training_run
                and registration["job_order"] == list(self.batch.JOBS)
                and registration["preparation"] == self.batch.PREPARATION,
                "Expected the registered nine-job small training batch")
        self.training.check(registration)
        started = self.matrix.read(directory / "started.json")
        require(started["registration"] == self.artifact(directory / "registration.json")
                and started["pid"] == started["process"]["pid"], "Training start receipt differs")
        original = started["process"]
        argv = original["cmdline"]
        require(str(self.training_tools / "scripts/phase2_training_batch.py") in argv
                and argv.count("--run") == 1 and argv[argv.index("--run") + 1] == training_run
                and "run" in argv and Path(original["exe"]).resolve() == Path(sys.executable).resolve(),
                "Training start does not identify the registered executable and run")
        return {"self_checkout": self.matrix.checkout_identity(Path(__file__).resolve().parents[1]),
                "training_checkout": self.matrix.checkout_identity(self.training_tools),
                "files": {str(Path(__file__).resolve()): self.matrix.digest(__file__),
                          str(Path(self.batch.__file__).resolve()): self.matrix.digest(self.batch.__file__)},
                "training_binding": registration["binding"],
                "training_registration": self.artifact(directory / "registration.json"),
                "training_started": self.artifact(directory / "started.json"),
                "training_process": original,
                "runtime": self.rt.implementation_binding(),
                "python": {"path": str(Path(sys.executable).resolve()), "sha256": self.matrix.digest(sys.executable)}}

    def check(self, registration):
        require(self.binding(registration["training_batch"]) == registration["binding"],
                "Registered development inputs changed")


def training_status(context, registration):
    """Wait for the original PID/create-time to exit, then verify all 15 steps."""
    batch, matrix = context.batch, context.matrix
    original = registration["binding"]["training_process"]
    current = batch.process_identity(original["pid"])
    if current is not None and current["create_time"] == original["create_time"]:
        require(current == original, "Training process identity changed")
        return None
    root = context.path(registration["training_batch"])
    result = matrix.read(root / "result.json")
    require(result.get("status") == "complete" and result["completed_jobs"] == sorted(batch.JOBS)
            and result["registration_sha256"] == matrix.digest(root / "registration.json")
            and result["automatic_budget_resumes"] == 0 and result["evaluation_started"] is False,
            "Training batch did not complete all nine registered jobs")
    events = [json.loads(line) for line in (root / "events.jsonl").read_bytes().splitlines() if line.strip()]
    finished = [row for row in events if row["event"] == "step-finished"]
    starts = [row for row in events if row["event"] == "step-started"]
    require(len(finished) == len(starts) == 15, "Training needs exactly its fifteen original steps")
    require(sorted((row["job"], row["stage"]) for row in starts)
            == sorted((row["job"], row["stage"]) for row in finished), "Training start/finish events differ")
    cleanups = [row for row in events if row["event"] == "post-exit-cleanup"]
    require(len(cleanups) == 15 and all(row["evidence"]["status"] == "cleaned"
            and not row["evidence"]["signals"] and not row["evidence"]["orphaned_descendants"]
            and not row["evidence"]["forced_termination"] for row in cleanups), "Training left live processes after completion")
    require(not any(row["event"] in ("step-rejected", "dispatch-rejected", "batch-error", "operator-cleanup", "final-cleanup")
                    for row in events), "Training contains a rejected or interrupted step")
    states, evidence = {}, []
    step_paths = set()
    for row in finished:
        path = context.rt.bound(row["receipt"])
        require(path.name == "finished.json" and path.parent.parent == root / "steps"
                and path not in step_paths, "Duplicate or foreign training step")
        step_paths.add(path)
        receipt = matrix.read(path)
        key, stage = receipt["job"], receipt["stage"]
        require(key in batch.JOBS and row["job"] == key and row["stage"] == stage
                and row["returncode"] == receipt["returncode"], "Training event differs from step receipt")
        expected_stage = states.get(key, {}).get("next", "inspect" if key.startswith("C-") else "bc")
        require(not states.get(key, {}).get("done") and stage == expected_stage, "Training stages were repeated or reordered")
        for name in ("started", "stdout", "stderr"):
            require(context.rt.bound(receipt[name]) == path.parent / {"started": "started.json", "stdout": "stdout.log", "stderr": "stderr.log"}[name],
                    "Training output belongs to another step")
        start = matrix.read(path.parent / "started.json")
        budget = float(start["argv"][start["argv"].index("--max-wall-seconds") + 1])
        batch.limits(1, 1, budget)
        batch.limits(1, 1, start["remaining_batch_seconds"])
        targets = states.get(key, {}).get("targets")
        require(start["job"] == key and start["stage"] == stage and start["remaining_batch_seconds"] <= budget
                and start["registration_sha256"] == result["registration_sha256"]
                and start["argv"] == batch.command_for(context.training, key, stage, budget,
                                                       targets["grp"] if targets else None),
                "Training command differs from the registered runner")
        stdout = matrix.read(path.parent / "stdout.log")
        if stage != "inspect":
            original_receipt = context.rt.bound(receipt["runner_finished"])
            require(stdout["attempt"].startswith(f"runs/{Path(batch.PREPARATION).name}-execution/{key}/attempt-")
                    and original_receipt == context.path(stdout["attempt"]) / "finished.json"
                    and matrix.read(original_receipt) == stdout and stdout["job_id"] == key
                    and stdout["exit_code"] == receipt["returncode"], "Original runner receipt differs from stdout")
            evidence.append(receipt["runner_finished"])
        transition = batch.step_transition(key, stage, receipt["returncode"], stdout, targets)
        require("halt" not in transition, "Training paused rather than completing")
        states.setdefault(key, {}).update(transition)
        evidence.extend([row["receipt"], receipt["started"], receipt["stdout"], receipt["stderr"]])
    require(set(states) == set(batch.JOBS) and all(state.get("done") for state in states.values()),
            "Training step ledger does not finish all nine jobs")
    require({path for path in (root / "steps").glob("*/finished.json")} == step_paths
            and len(list((root / "steps").iterdir())) == 15, "Unreported training steps remain")
    prepared, jobs, prepared_binding = context.training.prepared()
    require(matrix.read(root / "prepared.json")["preparation"] == prepared_binding,
            "Training preparation handoff changed")
    completed_models = context.rt.bound(result["completed_models"])
    require(completed_models == root / "completed-models.json"
            and matrix.read(completed_models) == matrix.training_models(context.rt, prepared, jobs),
            "Training completed-model evidence differs from the current nine models")
    evidence.append(result["completed_models"])
    evidence.extend(context.artifact(root / name) for name in ("result.json", "events.jsonl", "prepared.json"))
    return {"original_process_exited": True, "evidence": evidence, "preparation": prepared_binding}


def register(context, args):
    context.batch.limits(args.max_concurrent, args.min_available_memory_bytes, args.max_wall_seconds)
    for name in (args.run, args.matrix_name):
        require(re.fullmatch(r"phase2-[A-Za-z0-9_.-]{1,72}", name), "Use fresh simple phase2-* names")
    require(args.run != args.matrix_name, "Driver and matrix directories must differ")
    output = context.path("runs/" + args.run)
    require(not output.exists() and not context.path("runs/" + args.matrix_name).exists(), "Driver and matrix must be fresh")
    registration = {"schema_version": 1, "kind": "phase2-development-batch-v1", "created_at": context.batch.stamp(),
        "directory": "runs/" + args.run, "matrix": "runs/" + args.matrix_name,
        "training_batch": args.training_batch, "binding": context.binding(args.training_batch),
        "job_order": list(JOBS), "max_concurrent": args.max_concurrent,
        "min_available_memory_bytes": args.min_available_memory_bytes, "max_wall_seconds": args.max_wall_seconds,
        "scope": "Twenty small development campaigns, then scale decision; no automatic resume, expansion, final or Akochan"}
    terminal = training_status(context, registration)
    registration["training_terminal_at_registration"] = terminal
    context.check(registration)
    context.matrix.publish(output / "registration.json", registration)
    return {"status": "registered-not-started", "directory": registration["directory"]}


def command_for(context, registration, action, *, key=None, remaining=None):
    command = [sys.executable, "-B", "-u", str(context.tools / "scripts/phase2_evaluation_matrix.py"),
               "--runtime-root", str(context.rt.root), action]
    if action == "register":
        command += ["--preparation", context.batch.PREPARATION, "--run", Path(registration["matrix"]).name, "--scale", "small"]
    else:
        require(action in ("run", "report-development"), "Unsupported development action")
        command += ["--matrix", registration["matrix"]]
        if action == "run":
            require(key in JOBS, "Only the fixed twenty campaigns are allowed")
            context.batch.limits(1, 1, remaining)
            command += ["--job", key, "--max-wall-seconds", str(remaining)]
    return command


def campaign_status(code, result):
    require(code == 0, "Campaign process failed")
    if result.get("status") == "paused":
        return "paused-campaign"
    require(result.get("status") == "complete", "Campaign did not report complete")
    return "complete"


def sample_processes(batch, active):
    """Sample aggregate tree RSS; shared pages may count twice, peaks may be missed."""
    import psutil
    batch.discover_owned(active)
    for step in active.values():
        total = 0
        for process in batch.owned_processes({step["key"]: step}).values():
            try:
                if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                    total += process.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        step["rss_samples"] += 1
        step["sampled_peak_tree_rss_bytes"] = max(step["sampled_peak_tree_rss_bytes"], total)


def run_batch(context, args, invocation_started):
    matrix, batch = context.matrix, context.batch
    output = context.path(args.run)
    registration = matrix.read(output / "registration.json")
    require(registration["kind"] == "phase2-development-batch-v1" and registration["directory"] == args.run
            and registration["job_order"] == list(JOBS), "Unsupported development registration")
    batch.limits(registration["max_concurrent"], registration["min_available_memory_bytes"], registration["max_wall_seconds"])
    context.check(registration)
    remaining = lambda: registration["max_wall_seconds"] - (time.monotonic() - invocation_started)
    active, completed, probe_rss = {}, {}, {}
    stop, terminal, registered, report = None, None, None, None
    interrupted, handlers = [None], {}
    number, memory_wait = 0, None

    def event(name, **fields):
        with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(matrix.canonical({"at": batch.stamp(), "event": name, **fields}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def inputs():
        context.check(registration)
        if terminal:
            for artifact in terminal["evidence"]:
                context.rt.bound(artifact)
        if registered:
            require(matrix.load_matrix(context.rt, registration["matrix"]) == registered, "Registered matrix changed")

    def launch(key, command, memory=None):
        nonlocal number
        number += 1
        path = output / "steps" / f"{number:04d}-{key.replace('/', '-')}"
        path.mkdir(parents=True, exist_ok=False)
        matrix.publish(path / "started.json", {"at": batch.stamp(), "key": key, "argv": command,
            "remaining_batch_seconds": remaining(), "memory": memory,
            "registration_sha256": matrix.digest(output / "registration.json")})
        env = dict(os.environ)
        env.update(context.rt.storage.environment())
        env.update(PYTHONPATH=os.pathsep.join(str(context.rt.root / repo / "src") for repo in ("agari-kit", "agari-research")),
                   PYTHONDONTWRITEBYTECODE="1")
        stdout, stderr = (path / "stdout.log").open("xb"), (path / "stderr.log").open("xb")
        try:
            process = subprocess.Popen(command, cwd=context.rt.root / "agari-research", env=env, stdout=stdout, stderr=stderr,
                                       start_new_session=True, pass_fds=(lock_fd,))
        except BaseException as exc:
            stdout.close()
            stderr.close()
            matrix.publish(path / "finished.json", {"returncode": None, "key": key,
                "error": {"type": type(exc).__name__, "message": str(exc)},
                **{name: context.artifact(path / file) for name, file in
                   (("started", "started.json"), ("stdout", "stdout.log"), ("stderr", "stderr.log"))}})
            raise
        active[key] = {"key": key, "path": path, "process": process, "stdout": stdout, "stderr": stderr,
                       "started": time.monotonic(), "rss_samples": 0, "sampled_peak_tree_rss_bytes": 0}
        batch.register_owned(active)
        matrix.publish(path / "launched.json", {"process": batch.process_identity(process.pid),
                                                "started": context.artifact(path / "started.json")})
        event("step-started", key=key, pid=process.pid)

    def finish(key):
        step = active[key]
        code = step["process"].wait()
        step["stdout"].close()
        step["stderr"].close()
        path = step["path"]
        receipt = {"schema_version": 1, "key": key, "returncode": code,
            "wall_seconds": time.monotonic() - step["started"], "rss_samples": step["rss_samples"],
            "sampled_peak_tree_rss_bytes": step["sampled_peak_tree_rss_bytes"],
            "rss_scope": "Sampled sum of tracked command and descendant RSS; shared pages may count twice; not an OS maximum",
            **{name: context.artifact(path / file) for name, file in
               (("started", "started.json"), ("launched", "launched.json"), ("stdout", "stdout.log"), ("stderr", "stderr.log"))}}
        value = None
        try:
            receipt["cleanup"] = batch.interrupt_owned({key: step})
            require(not receipt["cleanup"]["signals"] and not receipt["cleanup"]["orphaned_descendants"]
                    and not receipt["cleanup"]["forced_termination"], "Exited command left live descendants")
            value = matrix.read(path / "stdout.log")
            if key in JOBS and campaign_status(code, value) == "complete":
                receipt["sealed"] = matrix.verify_fixture(context.rt, context.path(registration["matrix"]) / "development" / key,
                                                         registered["fixtures"][key])
        except Exception as exc:
            receipt["error"] = {"type": type(exc).__name__, "message": str(exc)}
        matrix.publish(path / "finished.json", receipt)
        step["finished_recorded"] = True
        event("step-finished", key=key, returncode=code, receipt=context.artifact(path / "finished.json"))
        if "cleanup" in receipt:
            del active[key]
        require("error" not in receipt, "Command output, cleanup or sealed fixture failed verification")
        return code, value, receipt

    def serial(action):
        nonlocal stop
        if remaining() <= 0 or interrupted[0] is not None:
            stop = "paused-wall-time" if interrupted[0] is None else "interrupted-by-operator"
            return None
        inputs()
        if remaining() <= 0 or interrupted[0] is not None:
            stop = "paused-wall-time" if interrupted[0] is None else "interrupted-by-operator"
            return None
        launch(action, command_for(context, registration, action))
        while active[action]["process"].poll() is None:
            sample_processes(batch, active)
            if remaining() <= 0 or interrupted[0] is not None:
                stop = "paused-wall-time" if interrupted[0] is None else "interrupted-by-operator"
                event("control-cleanup", evidence=batch.interrupt_owned(active))
                break
            time.sleep(.5)
        code, value, receipt = finish(action)
        if stop:
            return None
        require(code == 0 and isinstance(value, dict), "Matrix control command failed or returned no object")
        inputs()
        return value

    with context.runner.job_lock(output / "batch.lock") as lock_fd:
        require(not (output / "started.json").exists(), "A development batch cannot restart or renew its budget")
        matrix.publish(output / "started.json", {"at": batch.stamp(), "pid": os.getpid(), "process": batch.process_identity(os.getpid()),
                                                "registration": context.artifact(output / "registration.json")})
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, lambda signum, frame: interrupted.__setitem__(0, signum))
        try:
            while terminal is None:
                if interrupted[0] is not None or remaining() <= 0:
                    stop = "interrupted-by-operator" if interrupted[0] is not None else "paused-wall-time"
                    break
                inputs()
                terminal = training_status(context, registration)
                if terminal is None:
                    time.sleep(max(0, min(5, remaining())))
            if stop is None:
                original_terminal = registration["training_terminal_at_registration"]
                require(original_terminal is None or original_terminal == terminal, "Training terminal evidence changed after registration")
                matrix.publish(output / "training-ready.json", terminal)
                require(not context.path(registration["matrix"]).exists(), "Matrix must still be fresh")
                value = serial("register")
                if value is not None:
                    require(value["status"] == "registered-not-played" and value["matrix"] == registration["matrix"]
                            and value["jobs"] == sorted(JOBS) and value["games_per_job"] == 1000
                            and value["development_games_total"] == 20000, "Unexpected small matrix registration")
                    registered = matrix.load_matrix(context.rt, registration["matrix"])
                    require(registered["scale"] == "small" and set(registered["fixtures"]) == set(JOBS), "Wrong matrix scope")
                    matrix.publish(output / "matrix-ready.json", {"registration": context.artifact(context.path(registration["matrix"]) / "registration.json")})
            while (stop is None and len(completed) < len(JOBS)) or active:
                sample_processes(batch, active)
                if interrupted[0] is not None:
                    stop = "interrupted-by-operator"
                    event("operator-cleanup", evidence=batch.interrupt_owned(active))
                if remaining() <= 0 and stop is None:
                    stop = "paused-wall-time"
                for key in list(active):
                    if active[key]["process"].poll() is None:
                        continue
                    try:
                        code, value, receipt = finish(key)
                        state = campaign_status(code, value)
                        if state != "complete":
                            stop = stop or state
                        else:
                            completed[key] = receipt["sealed"]
                            if key in PROBES:
                                rss = receipt["sampled_peak_tree_rss_bytes"]
                                require(type(rss) is int and rss > 0, "Serial probe lacks a real RSS sample")
                                probe_rss[key] = rss
                        if stop is None:
                            inputs()
                    except Exception as exc:
                        stop = stop or "failed"
                        event("step-rejected", key=key, error_type=type(exc).__name__, error=str(exc))
                        if key in active:
                            raise  # Failed cleanup retains its tracked identities for final cleanup.
                if stop is None and memory_wait != tuple(active):
                    try:
                        for key in JOBS:
                            if interrupted[0] is not None:
                                stop = "interrupted-by-operator"
                                break
                            if key in completed or key in active:
                                continue
                            probing = len(probe_rss) < len(PROBES)
                            if probing and (active or key not in PROBES):
                                break
                            if len(active) >= (1 if probing else registration["max_concurrent"]):
                                break
                            inputs()
                            if interrupted[0] is not None:
                                stop = "interrupted-by-operator"
                                break
                            allowed, memory = batch.memory_gate(len(active), max(probe_rss.values(), default=0),
                                registration["min_available_memory_bytes"], batch.available_memory())
                            memory["serial_probe_sampled_peak_tree_rss_bytes"] = memory.pop("first_a_max_rss_bytes")
                            event("memory-check", key=key, allowed=allowed, snapshot=memory)
                            if not allowed:
                                if not active:
                                    stop = "paused-memory-needs-resource-review"
                                else:
                                    memory_wait = tuple(active)
                                break
                            memory_wait = None
                            if remaining() <= 0:
                                stop = "paused-wall-time"
                                break
                            launch(key, command_for(context, registration, "run", key=key, remaining=remaining()), memory)
                    except Exception as exc:
                        stop = "failed"
                        event("dispatch-rejected", error_type=type(exc).__name__, error=str(exc))
                if (stop is not None and not active) or len(completed) == len(JOBS):
                    break
                time.sleep(.5)
            if stop is None and len(completed) == len(JOBS):
                value = serial("report-development")
                if value is not None:
                    inputs()
                    expected = matrix.development_report(context.rt, registered)
                    path = context.path(registration["matrix"]) / "development-report.json"
                    require(matrix.read(path) == expected and value == {key: expected[key] for key in
                            ("status", "winner", "primary_order", "scale_10000_eligible", "scale_10000_candidates")},
                            "Development report differs from the complete sealed matrix")
                    inputs()
                    report = context.artifact(path)
        except BaseException as exc:
            stop = stop or "failed"
            event("batch-error", error_type=type(exc).__name__, error=str(exc))
        finally:
            if active:
                try:
                    event("final-cleanup", evidence=batch.interrupt_owned(active))
                except Exception as exc:
                    stop = "failed-cleanup"
                    event("cleanup-failed", error_type=type(exc).__name__, error=str(exc))
                for key in list(active):
                    try:
                        if active[key]["process"].poll() is None:
                            stop = "failed-cleanup"
                            event("unreaped-process", key=key, pid=active[key]["process"].pid)
                            active[key]["stdout"].close()
                            active[key]["stderr"].close()
                        elif active[key].get("finished_recorded"):
                            del active[key]
                        else:
                            finish(key)
                    except Exception as exc:
                        event("cleanup-result-error", key=key, error=str(exc))
            result = {"schema_version": 1, "status": "complete" if report and stop is None else stop or "failed",
                "completed_jobs": sorted(completed), "sealed": completed, "development_report": report,
                "probe_sampled_peak_tree_rss_bytes": probe_rss, "wall_seconds": time.monotonic() - invocation_started,
                "finished_at": batch.stamp(), "registration_sha256": matrix.digest(output / "registration.json"),
                "automatic_resumes": 0, "large_collection_started": False, "final_started": False, "akochan_started": False,
                "next_step": "Review the completed development report and scale cost; no automatic continuation"}
            matrix.publish(output / "result.json", result)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
        return result


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--tools-root", type=Path, required=True, help="Original frozen b9ee matrix/runner repository")
    parser.add_argument("--training-tools-root", type=Path, required=True, help="Frozen repository containing the actual training batch driver")
    commands = parser.add_subparsers(dest="action", required=True)
    new = commands.add_parser("register")
    new.add_argument("--training-batch", required=True, help="Storage-relative real training batch directory")
    new.add_argument("--matrix-name", required=True, help="Fresh phase2-* small matrix name")
    new.add_argument("--run", required=True, help="Fresh phase2-* development driver name")
    new.add_argument("--max-concurrent", type=int, choices=(1, 2), default=1)
    new.add_argument("--min-available-memory-bytes", type=int, default=2147483648)
    new.add_argument("--max-wall-seconds", type=float, default=172800)
    execute = commands.add_parser("run")
    execute.add_argument("--run", required=True)
    args = parser.parse_args(argv)
    context = Context(args.runtime_root, args.tools_root, args.training_tools_root)
    result = register(context, args) if args.action == "register" else run_batch(context, args, started)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0 if result["status"] in ("complete", "registered-not-started") else 2 if result["status"].startswith("paused-") else 1


if __name__ == "__main__":
    raise SystemExit(main())
