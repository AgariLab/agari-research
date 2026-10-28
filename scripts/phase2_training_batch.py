"""Finite nine-job training handoff; no collection or evaluation is started.

Register from a clean detached tools checkout, then run that one immutable batch.
The original runner retains all training/resume semantics and invocation receipts.
This driver never resumes a budget pause and cannot be restarted under the same
registration. Only the registered C GRP boundary permits an automatic resume.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

JOBS = tuple(f"{arm}-{seed}" for seed in (2026, 2027, 2028) for arm in "ABC")
PILOT = "runs/phase2-pilot-1000-launch-v1"
PREPARATION = "runs/phase2-pilot-1000-v1"
DATASET = "phase2-teacher-1000-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stamp():
    return datetime.now(timezone.utc).isoformat()


def limits(concurrent, memory, wall):
    require(type(concurrent) is int and concurrent in (1, 2), "max-concurrent must be 1 or 2")
    require(type(memory) is int and memory > 0, "Memory threshold must be positive bytes")
    require(type(wall) in (int, float) and math.isfinite(wall) and 0 < wall <= 172800,
            "Whole batch, including waiting, must fit within 48 hours")


def memory_gate(active_count, first_rss, floor, snapshot):
    require(type(snapshot["available"]) is int and 0 <= snapshot["available"] <= snapshot["total"], "Invalid available-memory sample")
    threshold = max(floor, (2 if active_count else 1) * (first_rss or 0))
    return snapshot["available"] >= threshold, dict(snapshot, threshold_bytes=threshold,
            active_jobs=active_count, first_a_max_rss_bytes=first_rss)


def available_memory():
    import psutil
    value = psutil.virtual_memory()
    return {"at": stamp(), "total": value.total, "available": value.available, "percent": value.percent}


def process_identity(pid):
    import psutil
    try:
        process = psutil.Process(pid)
        if process.status() == psutil.STATUS_ZOMBIE:
            return None
        return {"pid": process.pid, "create_time": process.create_time(), "exe": process.exe(), "cmdline": process.cmdline()}
    except psutil.NoSuchProcess:
        return None


class Context:
    def __init__(self, runtime_root, tools_root):
        self.tools = Path(tools_root).resolve()
        self.matrix = load(self.tools / "scripts/phase2_evaluation_matrix.py", "batch_matrix")
        self.runner = load(self.tools / "scripts/run_prepared_training.py", "batch_runner")
        self.rt = self.matrix.Runtime(runtime_root)
        self.own = Path(__file__).resolve().parents[1]

    def path(self, relative):
        return self.rt.path(relative)

    def artifact(self, path):
        return self.rt.artifact(path)

    def binding(self, pilot):
        matrix = self.matrix
        launch_path = pilot / "launch.json"
        launch = matrix.read(launch_path)
        require(launch["dataset"] == DATASET and launch["preparation"] == Path(PREPARATION).name
                and launch["runtime_root"] == str(self.rt.root) and launch["tools_root"] == str(self.tools)
                and Path(launch["python"]).resolve() == Path(sys.executable).resolve(), "Pilot handoff identity differs")
        expected = {str(pilot / "driver.py"), str(self.tools / "scripts/collect_teacher_pilot.py"),
                    *(str(self.rt.research / name) for name in ("experiments/phase2-teacher-student-v1/protocol.json",
                      "experiments/phase2-teacher-student-v1/qualify_teacher.py", "scripts/phase2_contract.py", "scripts/prepare_teacher_experiment.py"))}
        require(set(launch["sources"]) == expected and launch["python_sha256"] == matrix.digest(sys.executable),
                "Pilot source list or interpreter differs")
        require({repo: matrix.checkout_identity(self.rt.root / repo) for repo in ("agari-kit", "agari-research")}
                == self.rt.checkouts, "Frozen runtime checkout identity changed")
        files = dict(launch["sources"])
        files.update({str(Path(__file__).resolve()): matrix.digest(__file__),
                      str(self.tools / "scripts/run_prepared_training.py"): matrix.digest(self.runner.__file__),
                      str(self.tools / "scripts/phase2_evaluation_matrix.py"): matrix.digest(self.matrix.__file__)})
        require(all(matrix.digest(path) == expected for path, expected in files.items()), "Bound pilot/tool source changed")
        self.runner.refresh_runtime(self.rt)
        return {"self_checkout": matrix.checkout_identity(self.own),
                "tools_checkout": matrix.checkout_identity(self.tools),
                "runtime": self.rt.implementation_binding(), "files": files,
                "pilot_launch": self.artifact(launch_path),
                "python": {"path": str(Path(sys.executable).resolve()), "sha256": matrix.digest(sys.executable)}}

    def check(self, registration):
        require(self.binding(self.path(registration["pilot_launch"])) == registration["binding"],
                "Registered batch/tool/runtime inputs changed")

    def prepared(self):
        directory, manifest, jobs, audits, evidence = self.matrix.preparation(self.rt, PREPARATION)
        require(set(jobs) == set(JOBS) and all(audit["games"] == 1000 for audit in audits.values()), "Expected fixed nine 1000-game jobs")
        require(all(self.matrix.read(self.path(job["config"]))["device"] == "cpu" for job in jobs.values()), "This batch requires nine CPU jobs")
        return manifest, jobs, {"evidence": evidence,
            "configs": {key: self.artifact(self.path(job["config"])) for key, job in jobs.items()},
            "manifest": self.artifact(directory / "manifest.json")}


def register(context, args):
    limits(args.max_concurrent, args.min_available_memory_bytes, args.max_wall_seconds)
    require(args.pilot_launch == PILOT, "Only the existing registered 1000-game pilot handoff is in scope")
    require(re.fullmatch(r"phase2-[A-Za-z0-9_.-]{1,72}", args.run), "Use a new simple phase2-* batch name")
    output = context.path("runs/" + args.run)
    require(not output.exists(), "Batch registration is immutable; choose a new run")
    pilot = context.path(args.pilot_launch)
    binding = context.binding(pilot)
    launch = context.matrix.read(pilot / "launch.json")
    process = process_identity(launch["pid"])
    if process is not None:
        require(str(pilot / "driver.py") in process["cmdline"]
                and Path(process["exe"]).resolve() == Path(launch["python"]).resolve(), "Pilot PID belongs to a different process")
    else:
        require((pilot / "result.json").is_file(), "Pilot exited without a terminal handoff result")
    registration = {"schema_version": 1, "kind": "phase2-nine-training-batch-v1", "created_at": stamp(),
                    "directory": "runs/" + args.run, "pilot_launch": args.pilot_launch,
                    "preparation": PREPARATION, "job_order": list(JOBS), "binding": binding,
                    "pilot_process": process, "pilot_pid": launch["pid"],
                    "max_concurrent": args.max_concurrent, "min_available_memory_bytes": args.min_available_memory_bytes,
                    "max_wall_seconds": args.max_wall_seconds,
                    "scope": "Nine prepared CPU training jobs; no development, expansion or final evaluation"}
    context.matrix.publish(output / "registration.json", registration)
    return {"status": "registered-not-started", "directory": registration["directory"]}


def pilot_status(context, registration):
    """Only the original process ending plus a successful terminal handoff opens the gate."""
    pilot = context.path(registration["pilot_launch"])
    current = process_identity(registration["pilot_pid"])
    original = registration["pilot_process"]
    alive = original is not None and current is not None and current["create_time"] == original["create_time"]
    if alive:
        require(original is not None and current == original, "Pilot process identity changed")
        return None
    result_path = pilot / "result.json"
    require(result_path.is_file(), "Pilot process ended without a terminal result")
    result = context.matrix.read(result_path)
    require(result.get("status") == "prepared-not-trained", "Pilot terminal status is not prepared-not-trained")
    require(result["pilot"]["status"] == "ready" and result["preparation"]["jobs"] == 9
            and result["preparation"]["output"] == str(context.path(PREPARATION)), "Pilot terminal preparation differs")
    events_path = pilot / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_bytes().splitlines() if line.strip()]
    require(events and events[-1]["event"] == "handoff-finished" and events[-1]["status"] == "prepared-not-trained", "Pilot terminal event is missing")
    steps = [event for event in events if event["event"] == "step-finished"]
    require([step["step"] for step in steps] == ["qualification", "pilot", "preparation"]
            and all(step["returncode"] == 0 for step in steps), "Pilot step execution did not finish successfully")
    return {"result": context.artifact(result_path), "events": context.artifact(events_path),
            "original_process_exited": True, "pid_reused": current is not None}


def command_for(context, key, stage, remaining, grp_target=None):
    command = [sys.executable, "-B", "-u", str(context.tools / "scripts/run_prepared_training.py"),
               "--runtime-root", str(context.rt.root), "--preparation", PREPARATION,
               "--job", key, "--max-wall-seconds", str(remaining)]
    if stage == "inspect":
        command.append("--inspect")
    elif stage == "grp":
        require(type(grp_target) is int and grp_target > 0, "C requires the inspected GRP target")
        command.extend(["--stop-after-updates", str(grp_target)])
    elif stage == "cql":
        command.append("--resume")
    return command


def step_transition(key, stage, code, result, targets=None):
    """A C GRP stop is the sole automatic-resume exception; no budget stop qualifies."""
    if result.get("stop_reason") in ("wall_time_budget", "preflight_wall_time_budget"):
        require(code == 2 and result.get("status") == "paused", "Unexpected wall-time return status")
        return {"halt": "paused-wall-time"}
    if stage == "inspect":
        require(code == 0 and result.get("status") == "inspected-not-started"
                and set(result["targets"]) == {"grp", "cql"}
                and all(type(value) is int and value > 0 for value in result["targets"].values()), "C inspection failed")
        return {"next": "grp", "targets": result["targets"]}
    if stage == "grp":
        require(code == 2 and result.get("status") == "interrupted" and result.get("stop_reason") == "update_limit"
                and result["cli_result"]["progress"] == {"grp": targets["grp"], "cql": 0}, "C did not stop exactly at the registered GRP boundary")
        return {"next": "cql"}
    require(stage in ("bc", "cql") and code == 0 and result.get("status") == "complete", "Training step did not complete normally")
    return {"done": True}


class _OwnedProcessTracker:
    """Keep original psutil identities after a runner exits or a child reparents."""
    def __init__(self, process):
        import psutil
        self.process = process
        # Register immediately after Popen, before polling/reaping the runner.
        # Never reconstruct this root from a bare PID during later cleanup.
        self.root = psutil.Process(process.pid)
        self.root_key = (self.root.pid, self.root.create_time())
        self.owned = {self.root_key: self.root}

    def discover(self):
        import psutil
        for parent in tuple(self.owned.values()):
            if not _owned_alive(parent):
                continue
            try:
                children = parent.children(recursive=True)
            except psutil.NoSuchProcess:
                continue
            for child in children:
                try:
                    key = (child.pid, child.create_time())
                    if key[1] >= self.root_key[1]:
                        self.owned.setdefault(key, child)
                except psutil.NoSuchProcess:
                    pass


def _owned_alive(process):
    import psutil
    try:
        # is_running also compares PID creation identity, unlike pid_exists.
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def register_owned(active):
    for step in active.values():
        if "tracker" not in step:
            step["tracker"] = _OwnedProcessTracker(step["process"])
        elif step["tracker"].process is not step["process"]:
            raise ValueError("A step cannot replace its registered runner process")


def discover_owned(active):
    register_owned(active)
    owned = {}
    for step in active.values():
        tracker = step["tracker"]
        tracker.discover()
        owned.update(tracker.owned)
    return owned


def owned_processes(active):
    """Compatibility view; identities accumulate, including reparented children."""
    return discover_owned(active)


def _owned_signal(process, sig, signals):
    import psutil
    if not _owned_alive(process):
        return
    try:
        # psutil rechecks PID reuse immediately before signalling this object.
        process.send_signal(sig)
        signals.append({"pid": process.pid, "create_time": process.create_time(), "signal": int(sig)})
    except psutil.NoSuchProcess:
        pass


def _wait_owned(active, timeout, *, descendants_only=False):
    deadline = time.monotonic() + timeout
    roots = {step["tracker"].root_key for step in active.values()}
    while True:
        owned = discover_owned(active)
        # Only Popen may reap its direct child; psutil.wait_procs must not race
        # Popen.wait and turn an unknown exit code into an apparent success.
        root_done = all(step["process"].poll() is not None for step in active.values())
        descendants = [process for key, process in owned.items() if key not in roots and _owned_alive(process)]
        if not descendants and (descendants_only or root_done):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(.05, remaining))


def interrupt_owned(active, *, grace_seconds=10., terminate_seconds=5., kill_seconds=5.):
    """Cancel only recorded process identities, leaving the runner time to seal.

    A stopped runner cannot spawn between the cancellation snapshot and signal.
    If it has no live training child, kill it while stopped; this can leave an
    unsealed original attempt, which the caller must retain and mark failed.
    Otherwise resume with SIGINT pending, letting the frozen runner forward it,
    reap training with wait4, and publish its own unchanged failure receipt.
    """
    import psutil
    for timeout in (grace_seconds, terminate_seconds, kill_seconds):
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError("Cleanup timeouts must be finite and nonnegative")
    register_owned(active)
    signals, forced_before_child, suspended = [], [], []
    trackers = [step["tracker"] for step in active.values()]
    discover_owned(active)
    orphaned = [{"pid": process.pid, "create_time": key[1]} for tracker in trackers
                if not _owned_alive(tracker.root) for key, process in tracker.owned.items()
                if key != tracker.root_key and _owned_alive(process)]
    try:
        for tracker in trackers:
            root = tracker.root
            if not _owned_alive(root):
                continue
            _owned_signal(root, signal.SIGSTOP, signals)
            suspended.append(root)
            deadline = time.monotonic() + max(1., kill_seconds)
            while _owned_alive(root) and root.status() != psutil.STATUS_STOPPED:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Owned runner did not stop for a safe cancellation snapshot")
                time.sleep(.01)
        discover_owned(active)
        for tracker in trackers:
            root = tracker.root
            if root not in suspended:
                continue
            children = [process for key, process in tracker.owned.items()
                        if key != tracker.root_key and _owned_alive(process)]
            if children:
                _owned_signal(root, signal.SIGINT, signals)
                _owned_signal(root, signal.SIGCONT, signals)
            else:
                # Do not resume a runner in its pre-spawn window: the frozen
                # runner only catches KeyboardInterrupt once it reaches wait4.
                _owned_signal(root, signal.SIGKILL, signals)
                forced_before_child.append({"pid": root.pid, "create_time": tracker.root_key[1]})
            suspended.remove(root)
        if not _wait_owned(active, grace_seconds):
            roots = {tracker.root_key for tracker in trackers}
            for key, process in discover_owned(active).items():
                if key not in roots:
                    _owned_signal(process, signal.SIGTERM, signals)
            if not _wait_owned(active, terminate_seconds, descendants_only=True):
                for key, process in discover_owned(active).items():
                    if key not in roots:
                        _owned_signal(process, signal.SIGKILL, signals)
                if not _wait_owned(active, kill_seconds, descendants_only=True):
                    raise RuntimeError("Owned training descendants survived forced cancellation")
            # Training is gone. Keep the wrapper alive briefly so its wait4 and
            # original stdout/stderr/finished.json remain authoritative.
            if not _wait_owned(active, terminate_seconds):
                for tracker in trackers:
                    _owned_signal(tracker.root, signal.SIGTERM, signals)
                if not _wait_owned(active, terminate_seconds):
                    for tracker in trackers:
                        _owned_signal(tracker.root, signal.SIGKILL, signals)
                    if not _wait_owned(active, kill_seconds):
                        raise RuntimeError("Owned runner survived forced cancellation")
        for step in active.values():
            step["process"].wait(timeout=0)
        return {"status": "cleaned", "signals": signals,
                "tracked_identities": [{"pid": pid, "create_time": created} for pid, created in owned_processes(active)],
                "forced_before_child": forced_before_child,
                "orphaned_descendants": orphaned,
                "forced_termination": any(row["signal"] in (signal.SIGTERM, signal.SIGKILL) for row in signals),
                "original_finished_may_be_missing": bool(forced_before_child),
                "runner_exit_codes": [step["process"].returncode for step in active.values()]}
    finally:
        # An inspection/permission error must not silently leave our runner
        # suspended. Preserve the error; the batch must stop, never dispatch on.
        original = sys.exception()
        for root in suspended:
            try:
                _owned_signal(root, signal.SIGCONT, signals)
            except BaseException as cleanup:
                if original is None:
                    raise
                original.add_note(f"Owned runner resume also failed: {cleanup}")


def run_batch(context, args, invocation_started):
    matrix = context.matrix
    output = context.path(args.run)
    registration = matrix.read(output / "registration.json")
    require(registration["kind"] == "phase2-nine-training-batch-v1" and registration["directory"] == args.run
            and registration["job_order"] == list(JOBS), "Unsupported batch registration")
    limits(registration["max_concurrent"], registration["min_available_memory_bytes"], registration["max_wall_seconds"])
    context.check(registration)
    remaining = lambda: registration["max_wall_seconds"] - (time.monotonic() - invocation_started)
    active, states, completed = {}, {key: {"next": "inspect" if key.startswith("C-") else "bc"} for key in JOBS}, {}
    stranded = {}
    completed_models = None
    stop = None
    interrupted = [None]
    first_rss = None
    memory_wait = None
    number = 0
    original_handlers = {}
    def signal_stop(signum, frame):
        interrupted[0] = signum
    def event(name, **fields):
        with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(matrix.canonical({"at": stamp(), "event": name, **fields}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    def inputs(prepared_binding=None):
        context.check(registration)
        if prepared_binding is not None:
            require(context.prepared()[2] == prepared_binding, "Prepared training inputs changed")
            for name in ("result", "events"):
                require(matrix.digest(context.path(pilot[name]["path"])) == pilot[name]["sha256"], "Pilot terminal evidence changed")
    def finish(step):
        process = step["process"]
        code = process.wait()
        step["stdout"].close()
        step["stderr"].close()
        path = step["path"]
        receipt = {"schema_version": 1, "job": step["job"], "stage": step["stage"], "returncode": code,
                   "started": context.artifact(path / "started.json"), "wall_seconds": time.monotonic() - step["started"],
                   "stdout": context.artifact(path / "stdout.log"), "stderr": context.artifact(path / "stderr.log")}
        result = None
        try:
            result = matrix.read(path / "stdout.log")
            if step["stage"] != "inspect":
                prefix = f"runs/{Path(PREPARATION).name}-execution/{step['job']}/attempt-"
                require(isinstance(result.get("attempt"), str) and result["attempt"].startswith(prefix), "Runner receipt belongs to another job")
                require(result.get("job_id") == step["job"] and result.get("exit_code") == code, "Runner result identity or exit code differs")
                original = context.path(result["attempt"] + "/finished.json")
                require(matrix.read(original) == result, "Runner stdout differs from original finished receipt")
                receipt["runner_finished"] = context.artifact(original)
        except Exception as exc:
            receipt["result_error"] = {"type": type(exc).__name__, "message": str(exc)}
        matrix.publish(path / "finished.json", receipt)
        event("step-finished", job=step["job"], stage=step["stage"], returncode=code, receipt=context.artifact(path / "finished.json"))
        require("result_error" not in receipt, "Step produced no verifiable runner result")
        return code, result

    with context.runner.job_lock(output / "batch.lock") as lock_fd:
        require(not (output / "started.json").exists(), "A batch cannot restart or renew its budget; inspect the preserved terminal evidence")
        matrix.publish(output / "started.json", {"at": stamp(), "pid": os.getpid(), "process": process_identity(os.getpid()),
                                                 "registration": context.artifact(output / "registration.json")})
        for sig in (signal.SIGINT, signal.SIGTERM):
            original_handlers[sig] = signal.signal(sig, signal_stop)
        try:
            event("waiting-for-pilot", pilot_launch=registration["pilot_launch"], remaining_batch_seconds=remaining())
            while True:
                if interrupted[0] is not None:
                    stop = "interrupted-by-operator"
                    break
                if remaining() <= 0:
                    stop = "paused-wall-time"
                    break
                inputs()
                pilot = pilot_status(context, registration)
                if pilot is not None:
                    break
                time.sleep(min(30, remaining()))
            if stop is None:
                prepared, jobs, prepared_binding = context.prepared()
                for job in jobs.values():
                    require(not any(context.path(f"{prefix}/{job['run']}").exists() for prefix in ("runs", "models", "checkpoints")),
                            "This batch requires fresh jobs; existing work needs explicit review")
                matrix.publish(output / "prepared.json", {"pilot": pilot, "preparation": prepared_binding})
                while len(completed) < len(JOBS) or active:
                    discover_owned(active)
                    if interrupted[0] is not None:
                        stop = "interrupted-by-operator"
                        event("operator-cleanup", evidence=interrupt_owned(active))
                    if remaining() <= 0 and stop is None:
                        stop = "paused-wall-time"
                    for key, step in list(active.items()):
                        if step["process"].poll() is None:
                            continue
                        cleanup_complete = False
                        try:
                            cleanup = interrupt_owned({key: step})
                            cleanup_complete = True
                            event("post-exit-cleanup", job=key, evidence=cleanup)
                            code, result = finish(step)
                            if stop is None:
                                require(cleanup.get("status") == "cleaned" and not cleanup.get("signals")
                                        and not cleanup.get("orphaned_descendants") and not cleanup.get("forced_termination"),
                                        "Normally exited runner left descendants requiring cleanup")
                                inputs(prepared_binding)
                                transition = step_transition(key, step["stage"], code, result, states[key].get("targets"))
                                if "halt" in transition:
                                    stop = transition["halt"]
                                elif transition.get("done"):
                                    completed[key] = matrix.verify_completed_job(context.rt, prepared, key, jobs[key])
                                    states[key] = {"done": True}
                                    if key == JOBS[0]:
                                        first_rss = result["process"]["max_rss_bytes"]
                                        require(type(first_rss) is int and first_rss > 0, "First A job lacks a real measured RSS")
                                        event("first-a-probe-complete", max_rss_bytes=first_rss)
                                else:
                                    states[key].update(transition)
                        except Exception as exc:
                            if not cleanup_complete:
                                stranded[key] = step
                            stop = stop or "failed"
                            event("step-rejected", job=key, error_type=type(exc).__name__, error=str(exc))
                        finally:
                            del active[key]
                    if stop is None and memory_wait != tuple(active):
                        try:
                            for key in JOBS:
                                if interrupted[0] is not None:
                                    stop = "interrupted-by-operator"
                                    break
                                if key in active or key in completed or (first_rss is None and key != JOBS[0]):
                                    continue
                                if len(active) >= (registration["max_concurrent"] if first_rss is not None else 1):
                                    break
                                inputs(prepared_binding)
                                if interrupted[0] is not None:
                                    stop = "interrupted-by-operator"
                                    break
                                allowed, memory = memory_gate(len(active), first_rss, registration["min_available_memory_bytes"], available_memory())
                                event("memory-check", job=key, allowed=allowed, snapshot=memory)
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
                                number += 1
                                stage = states[key]["next"]
                                path = output / "steps" / f"{number:04d}-{key}-{stage}"
                                path.mkdir(parents=True, exist_ok=False)
                                command = command_for(context, key, stage, remaining(), states[key].get("targets", {}).get("grp"))
                                matrix.publish(path / "started.json", {"at": stamp(), "job": key, "stage": stage,
                                    "argv": command, "memory": memory, "remaining_batch_seconds": remaining(),
                                    "registration_sha256": matrix.digest(output / "registration.json")})
                                env = dict(os.environ)
                                env.update(context.rt.storage.environment())
                                env.update(PYTHONPATH=os.pathsep.join(str(context.rt.root / repo / "src") for repo in ("agari-kit", "agari-research")), PYTHONDONTWRITEBYTECODE="1")
                                stdout, stderr = (path / "stdout.log").open("xb"), (path / "stderr.log").open("xb")
                                step_started = time.monotonic()
                                try:
                                    process = subprocess.Popen(command, cwd=context.rt.root / "agari-research", env=env,
                                        stdout=stdout, stderr=stderr, start_new_session=True, pass_fds=(lock_fd,))
                                except BaseException as exc:
                                    stdout.close()
                                    stderr.close()
                                    matrix.publish(path / "finished.json", {"schema_version": 1, "job": key, "stage": stage,
                                        "returncode": None, "error_type": type(exc).__name__, "error": str(exc),
                                        "wall_seconds": time.monotonic() - step_started,
                                        "started": context.artifact(path / "started.json"),
                                        "stdout": context.artifact(path / "stdout.log"), "stderr": context.artifact(path / "stderr.log")})
                                    raise
                                active[key] = {"process": process, "job": key, "stage": stage, "path": path,
                                               "stdout": stdout, "stderr": stderr, "started": step_started}
                                register_owned(active)
                                discover_owned(active)
                                event("step-started", job=key, stage=stage, pid=process.pid)
                        except Exception as exc:
                            stop = "failed"
                            event("dispatch-rejected", error_type=type(exc).__name__, error=str(exc))
                    if stop is not None and not active:
                        break
                    if len(completed) == len(JOBS):
                        break
                    time.sleep(.5)
                if stop is None and len(completed) == len(JOBS):
                    inputs(prepared_binding)
                    models = matrix.training_models(context.rt, prepared, jobs)
                    matrix.publish(output / "completed-models.json", models)
                    completed_models = context.artifact(output / "completed-models.json")
        except BaseException as exc:
            stop = stop or "failed"
            event("batch-error", error_type=type(exc).__name__, error=str(exc))
        finally:
            active.update(stranded)
            if active:
                try:
                    event("final-cleanup", evidence=interrupt_owned(active))
                except Exception as exc:
                    stop = "failed-cleanup"
                    event("cleanup-failed", error_type=type(exc).__name__, error=str(exc))
                for key, step in list(active.items()):
                    try:
                        if step["process"].poll() is None:
                            stop = "failed-cleanup"
                            event("unreaped-process", job=key, pid=step["process"].pid)
                            step["stdout"].close()
                            step["stderr"].close()
                        else:
                            finish(step)
                    except Exception as exc:
                        event("cleanup-result-error", job=key, error=str(exc))
            result = {"schema_version": 1, "status": "complete" if len(completed) == 9 and stop is None else stop or "failed",
                      "completed_jobs": sorted(completed), "first_a_max_rss_bytes": first_rss,
                      "wall_seconds": time.monotonic() - invocation_started, "finished_at": stamp(),
                      "registration_sha256": matrix.digest(output / "registration.json"),
                      "completed_models": completed_models,
                      "automatic_budget_resumes": 0, "evaluation_started": False}
            matrix.publish(output / "result.json", result)
            for sig, handler in original_handlers.items():
                signal.signal(sig, handler)
        return result


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--tools-root", type=Path, required=True, help="Original frozen runner/matrix/pilot repository")
    subs = parser.add_subparsers(dest="action", required=True)
    new = subs.add_parser("register")
    new.add_argument("--pilot-launch", default=PILOT)
    new.add_argument("--run", required=True, help="New phase2-* batch name")
    new.add_argument("--max-concurrent", type=int, default=1, choices=(1, 2))
    new.add_argument("--min-available-memory-bytes", type=int, default=2147483648)
    new.add_argument("--max-wall-seconds", type=float, default=172800)
    execute = subs.add_parser("run")
    execute.add_argument("--run", required=True, help="Storage-relative registered batch directory")
    args = parser.parse_args(argv)
    context = Context(args.runtime_root, args.tools_root)
    result = register(context, args) if args.action == "register" else run_batch(context, args, started)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0 if result["status"] in ("complete", "registered-not-started") else 2 if result["status"].startswith("paused-") else 1


if __name__ == "__main__":
    raise SystemExit(main())
