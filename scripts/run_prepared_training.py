"""Run one registered phase-two job with immutable invocation receipts.

The adjacent evaluation helper owns preparation and completed-model gates. Both
scripts belong in the pinned tools checkout; only --runtime-root supplies package
code. No collection, evaluation games, or held-out actions are opened here.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_matrix():
    path = Path(__file__).with_name("phase2_evaluation_matrix.py")
    spec = importlib.util.spec_from_file_location("training_matrix_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def budget(value):
    require(type(value) in (int, float) and math.isfinite(value) and 0 < value <= 172800,
            "max_wall_seconds must be finite, positive and no more than 48 hours")
    return value


def target_updates(config, counts, stage):
    settings = config[stage]
    if "steps" in settings:
        return settings["steps"]
    column = "cql_samples" if stage == "bc" else stage + "_samples"
    return math.ceil(sum(row[column] for row in counts) / settings["batch_size"]) * settings["epochs"]


def resume_contract(rt, matrix, directory, prepared, job, config, checkpoint):
    """Cheap pre-launch binding gate; the CLI also rebuilds the native plan.

    The CLI remains authoritative for optimizer, RNG, encoding, sample semantics
    and selected-snapshot restoration. A stale result.json after a crash does not
    override the actual checkpoint. Completed jobs use the full shared gate.
    """
    name = "weak" if job["arm"] == "A" else "teacher"
    entry = prepared["datasets"][name]
    for key, value in (("schema_version", 1), ("config", config), ("run_name", job["run"]),
                       ("dataset_sha256", entry["sha256"]), ("implementation_sha256", rt.implementation),
                       ("runtime_provenance", rt.runtime), ("policy_filter", entry["policy_filter"])):
        require(checkpoint.get(key) == value, f"Resume checkpoint differs: {key}")
    plan = checkpoint["training_plan"]
    require(matrix.content_digest({key: value for key, value in plan.items() if key != "sha256"}) == plan["sha256"]
            and plan["dataset_sha256"] == entry["sha256"] and plan["policy_filter"] == entry["policy_filter"],
            "Resume training plan binding changed")
    counts = matrix.read(directory / f"{name}-training-counts.json")
    stages = config["stages"]
    require(set(plan["stages"]) == set(checkpoint["progress"]) == set(stages), "Resume stages changed")
    from agari_research.planning import expected_samples
    for stage in stages:
        target = target_updates(config, counts, stage)
        update = checkpoint["progress"][stage]
        require(plan["stages"][stage]["target_updates"] == target
                and type(update) is int and 0 <= update <= target, "Resume update budget changed")
        require(checkpoint["samples_seen"][stage] == expected_samples(plan["stages"][stage], update,
                                                                     config[stage]["batch_size"]),
                "Resume sample cursor changed")
    from agari_research.training import initialize_q, q_initialization_sha256
    require(checkpoint["policy_initialization"] == {
        "seed": config["seed"], "q_state_sha256": q_initialization_sha256(initialize_q(config))},
        "Resume Q initialization changed")
    metrics_path = rt.path(f"runs/{job['run']}/metrics.jsonl")
    metrics = [json.loads(line) for line in metrics_path.read_bytes().splitlines() if line.strip()] if metrics_path.exists() else []
    require(all(row["stage"] in stages for row in metrics)
            and all([row["update"] for row in metrics if row["stage"] == stage]
                    == list(range(1, checkpoint["progress"][stage] + 1)) for stage in stages),
            "Metrics do not end at this checkpoint; do not silently trim or replay a crashed attempt")


def refresh_runtime(rt):
    from agari_research.runtime import runtime_provenance
    from agari_research.training import implementation_sha256
    require(rt.runtime == runtime_provenance() and rt.implementation == implementation_sha256(),
            "Frozen runtime files changed during this invocation")


@contextmanager
def job_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("This training job is already running") from exc
        # The child inherits this descriptor, retaining exclusion if this wrapper
        # dies while training continues. Never unlink a lock file on completion.
        yield stream.fileno()


def new_attempt(directory):
    directory.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        path = directory / f"attempt-{number:04d}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            number += 1


def run_child(command, env, attempt, lock_fd, *, cwd=None):
    """wait4 reports this child, not cumulative usage from previous invocations."""
    started = time.monotonic()
    with (attempt / "stdout.log").open("xb") as stdout, (attempt / "stderr.log").open("xb") as stderr:
        process = subprocess.Popen(command, env=env, cwd=cwd, stdout=stdout, stderr=stderr,
                                   start_new_session=True, pass_fds=(lock_fd,))
        interrupted = False
        while True:
            try:
                _, status, usage = os.wait4(process.pid, 0)
                break
            except KeyboardInterrupt:
                interrupted = True
                try:
                    os.killpg(process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass  # It exited between the interrupt and signal; reap below.
        process.returncode = os.waitstatus_to_exitcode(status)
    return {"exit_code": process.returncode, "interrupted_by_operator": interrupted,
            "wall_seconds": time.monotonic() - started,
            "user_seconds": usage.ru_utime, "system_seconds": usage.ru_stime,
            "max_rss_bytes": usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024),
            "usage_scope": "training child, including native dataset preflight and checkpoint restoration"}


def stdout_result(path):
    """The frozen CLI prints one pretty-printed JSON object, often many lines."""
    with path.open(encoding="utf-8") as stream:
        result = json.load(stream)
    require(isinstance(result, dict), "Training stdout did not contain a result object")
    return result


def execution_binding(rt, matrix, directory, prepared, job, evidence):
    return {"schema_version": 1, "kind": "phase2-training-execution-v1", "job": job,
            "preparation": rt.artifact(directory / "manifest.json"), "preparation_evidence": evidence,
            "config": rt.artifact(rt.path(job["config"])), "split": rt.artifact(rt.path(job["split"])),
            "implementation": rt.implementation_binding(),
            "runner_sha256": matrix.digest(__file__), "matrix_helper_sha256": matrix.digest(matrix.__file__),
            "python": {"path": str(Path(sys.executable).resolve()), "sha256": matrix.digest(sys.executable)},
            "storage_root": str(rt.storage.root), "test_partition_content_opened": False}


def execute_job(rt, matrix, args, invocation_started):
    directory, prepared, jobs, _, evidence = matrix.preparation(rt, args.preparation)
    require(args.job in jobs, "Job must be one of the nine registered arm/seed identifiers")
    job = jobs[args.job]
    config_path = rt.path(job["config"])
    require(matrix.digest(config_path) == job["config_sha256"], "Prepared configuration hash changed")
    config = matrix.read(config_path)
    name = "weak" if job["arm"] == "A" else "teacher"
    counts = matrix.read(directory / f"{name}-training-counts.json")
    targets = {stage: target_updates(config, counts, stage) for stage in config["stages"]}
    if args.inspect:
        return {"status": "inspected-not-started", "job": job, "targets": targets,
                "runtime_root": str(rt.root), "device": config["device"], "test_partition_content_opened": False}, 0
    maximum = budget(args.max_wall_seconds)
    require(maximum <= job["max_wall_seconds"], "Invocation exceeds the registered wall-time ceiling")
    require(args.stop_after_updates is None or type(args.stop_after_updates) is int and args.stop_after_updates > 0,
            "stop_after_updates must be a positive invocation update count")
    binding = execution_binding(rt, matrix, directory, prepared, job, evidence)
    # This sibling directory must not pre-create the fresh training run directory.
    execution = rt.path(f"runs/{directory.name}-execution/{args.job}")
    with job_lock(execution / "job.lock") as lock_fd:
        matrix.publish(execution / "binding.json", binding)
        attempt = new_attempt(execution)
        started_record = {"schema_version": 1, "job_id": args.job, "binding_sha256": matrix.digest(execution / "binding.json"),
                          "started_unix_seconds": time.time(), "max_wall_seconds": maximum,
                          "resume": args.resume, "stop_after_updates": args.stop_after_updates,
                          "targets": targets, "preflight_seconds": time.monotonic() - invocation_started}
        matrix.publish(attempt / "started.json", started_record)
        receipt = {"schema_version": 1, "job_id": args.job, "attempt": str(attempt.relative_to(rt.storage.root)),
                   "started_sha256": matrix.digest(attempt / "started.json"), "status": "failed"}
        code = 1
        try:
            checkpoint_path = rt.path(f"checkpoints/{job['run']}/latest.pt")
            result_path = rt.path(f"runs/{job['run']}/result.json")
            if result_path.exists() and matrix.read(result_path).get("status") == "complete":
                receipt.update(status="already-complete", completed=matrix.verify_completed_job(rt, prepared, args.job, job))
                code = 0
            else:
                if args.resume:
                    require(checkpoint_path.is_file(), "Explicit resume requires the prepared job's latest checkpoint")
                    import torch
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                    resume_contract(rt, matrix, directory, prepared, job, config, checkpoint)
                    receipt["resume_checkpoint"] = rt.artifact(checkpoint_path)
                else:
                    require(not any(rt.path(f"{prefix}/{job['run']}").exists()
                                    for prefix in ("runs", "checkpoints", "models")),
                            "Training artifacts already exist; use explicit --resume after inspecting the checkpoint")
                remaining = maximum - (time.monotonic() - invocation_started)
                if remaining <= 0:
                    receipt.update(status="paused", stop_reason="preflight_wall_time_budget")
                    code = 2
                else:
                    command = [sys.executable, "-B", "-u", "-m", "agari_research.cli", "run",
                               "--config", str(config_path), "--split", job["split"], "--run", job["run"],
                               "--max-wall-seconds", str(remaining)]
                    if args.resume:
                        command.extend(["--resume", str(checkpoint_path.relative_to(rt.storage.root))])
                    if args.stop_after_updates is not None:
                        command.extend(["--stop-after-updates", str(args.stop_after_updates)])
                    env = dict(os.environ)
                    env.update(rt.storage.environment())
                    env.update(PYTHONPATH=os.pathsep.join(str(rt.root / repo / "src") for repo in ("agari-kit", "agari-research")),
                               PYTHONDONTWRITEBYTECODE="1")
                    matrix.publish(attempt / "command.json", {"argv": command, "pythonpath": env["PYTHONPATH"],
                                                              "cwd": str(rt.root / "agari-research"),
                                                              "child_max_wall_seconds": remaining})
                    receipt["process"] = run_child(command, env, attempt, lock_fd, cwd=rt.root / "agari-research")
                    require(receipt["process"]["exit_code"] == 0, "Training process failed; inspect this attempt's stderr.log")
                    result = stdout_result(attempt / "stdout.log")
                    require(matrix.read(result_path) == {key: value for key, value in result.items()
                                                       if key != "command_elapsed_seconds"},
                            "CLI output differs from its saved training result")
                    # Detect changed preparation/runtime evidence across a long invocation.
                    refresh_runtime(rt)
                    after_directory, after_prepared, after_jobs, _, after_evidence = matrix.preparation(rt, args.preparation)
                    require(execution_binding(rt, matrix, after_directory, after_prepared, after_jobs[args.job], after_evidence)
                            == binding, "Training inputs changed during this invocation")
                    if result.get("status") == "complete":
                        receipt.update(status="complete", completed=matrix.verify_completed_job(rt, prepared, args.job, job))
                        code = 0
                    else:
                        require(result.get("status") in ("paused", "interrupted"), "Unexpected training result status")
                        require(result.get("stop_reason") in ("wall_time_budget", "update_limit"),
                                "Unrecognized safe-stop reason")
                        import torch
                        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                        resume_contract(rt, matrix, directory, prepared, job, config, checkpoint)
                        require(result["progress"] == checkpoint["progress"], "Saved safe-stop cursor differs from CLI result")
                        receipt.update(status=result["status"], stop_reason=result["stop_reason"])
                        code = 2
                    receipt["cli_result"] = result
                    if checkpoint_path.exists():
                        receipt["latest_checkpoint"] = rt.artifact(checkpoint_path)
                    if result_path.exists():
                        receipt["training_result"] = rt.artifact(result_path)
        except Exception as exc:
            receipt.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
            code = 1
        finally:
            receipt.update(invocation_wall_seconds=time.monotonic() - invocation_started,
                           finished_unix_seconds=time.time(), exit_code=code)
            receipt["logs"] = [rt.artifact(attempt / name) for name in ("stdout.log", "stderr.log") if (attempt / name).exists()]
            matrix.publish(attempt / "finished.json", receipt)
        return receipt, code


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--preparation", required=True, help="Storage-relative prepared directory, runs/phase2-...")
    parser.add_argument("--job", required=True, help="Exactly one arm/seed, e.g. A-2026")
    parser.add_argument("--resume", action="store_true", help="Explicitly restore this job's latest.pt")
    parser.add_argument("--max-wall-seconds", type=float, default=172800)
    parser.add_argument("--stop-after-updates", type=int, help="Additional updates in this invocation, across stages")
    parser.add_argument("--inspect", action="store_true", help="Read-only source gate and target-update report")
    args = parser.parse_args(argv)
    budget(args.max_wall_seconds)
    matrix = load_matrix()
    rt = matrix.Runtime(args.runtime_root)
    result, code = execute_job(rt, matrix, args, started)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
