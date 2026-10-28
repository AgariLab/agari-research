"""Invocation ownership, budgets and receipts, without running formal training."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location("training_runner_test", Path(__file__).parents[1] / "scripts/run_prepared_training.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
matrix = runner.load_matrix()


class Runtime:
    def __init__(self, root):
        self.root = root / "frozen"
        self.storage = SimpleNamespace(root=root, environment=lambda: {"AGARILAB_STORAGE_ROOT": str(root)})
        self.runtime = {"runtime": "test"}
        self.implementation = "implementation"

    def path(self, relative):
        return self.storage.root / relative

    def artifact(self, path):
        return {"path": str(path.relative_to(self.storage.root)), "sha256": matrix.digest(path)}

    def implementation_binding(self):
        return {"runtime": self.runtime, "runtime_root": str(self.root), "implementation": self.implementation}


class InvocationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rt = Runtime(self.root)
        self.directory = self.root / "runs/phase2-prepared"
        self.directory.mkdir(parents=True)
        self.config = {"device": "cpu", "stages": ["bc"], "bc": {"steps": 3, "batch_size": 64}}
        config_path = self.directory / "A-seed2026.json"
        config_path.write_text(json.dumps(self.config))
        split_path = self.root / "split.json"
        split_path.write_text("{}")
        self.job = {"arm": "A", "seed": 2026, "method": "bc", "run": "phase2-prepared-a-2026",
                    "config": str(config_path.relative_to(self.root)), "config_sha256": matrix.digest(config_path),
                    "split": "split.json", "max_wall_seconds": 172800}
        self.prepared = {"jobs": [self.job], "datasets": {"weak": {"sha256": "dataset", "policy_filter": None}}}
        (self.directory / "manifest.json").write_text(json.dumps(self.prepared))
        (self.directory / "weak-training-counts.json").write_text('[{"cql_samples":64,"grp_samples":10}]')
        self.args = SimpleNamespace(preparation="runs/phase2-prepared", job="A-2026", inspect=False,
                                    max_wall_seconds=60, resume=False, stop_after_updates=None)
        self.evidence = [self.rt.artifact(self.directory / "manifest.json")]
        self.preparation = patch.object(matrix, "preparation", return_value=(
            self.directory, self.prepared, {"A-2026": self.job}, {}, self.evidence)).start()
        self.addCleanup(patch.stopall)
        self.completed = patch.object(matrix, "verify_completed_job", return_value={"best": {"update": 2}}).start()
        self.child = patch.object(runner, "run_child", side_effect=self.complete_child).start()
        patch.object(runner, "refresh_runtime").start()

    @property
    def execution(self):
        return self.root / "runs/phase2-prepared-execution/A-2026"

    def complete_child(self, command, env, attempt, lock_fd, **kwargs):
        result = {"status": "complete", "run": self.job["run"]}
        run = self.rt.path("runs/" + self.job["run"])
        run.mkdir(parents=True, exist_ok=True)
        (run / "result.json").write_text(json.dumps(result))
        (attempt / "stdout.log").write_text(json.dumps(result | {"command_elapsed_seconds": .01}, indent=2))
        (attempt / "stderr.log").write_text("diagnostic\n")
        return {"exit_code": 0, "wall_seconds": .02, "user_seconds": .01, "system_seconds": .005, "max_rss_bytes": 1234}

    def invoke(self, started=None):
        return runner.execute_job(self.rt, matrix, self.args, time.monotonic() if started is None else started)

    def test_read_only_inspection_returns_real_targets_without_execution_artifacts(self):
        self.args.inspect = True
        result, code = self.invoke()
        self.assertEqual((code, result["targets"]), (0, {"bc": 3}))
        self.assertFalse(self.execution.exists())
        self.child.assert_not_called()

    def test_whole_invocation_budget_expired_before_child_is_a_recorded_pause(self):
        result, code = self.invoke(time.monotonic() - 61)
        self.assertEqual((code, result["status"], result["stop_reason"]), (2, "paused", "preflight_wall_time_budget"))
        self.child.assert_not_called()
        self.assertFalse(self.rt.path("runs/" + self.job["run"]).exists())
        self.assertTrue((self.execution / "attempt-0001/finished.json").is_file())

    def test_exact_command_and_immutable_logs_are_bound_to_frozen_runtime(self):
        result, code = self.invoke(time.monotonic() - 2)
        self.assertEqual((code, result["status"]), (0, "complete"))
        command, env, attempt, _ = self.child.call_args.args
        self.assertEqual(command[:6], [sys.executable, "-B", "-u", "-m", "agari_research.cli", "run"])
        self.assertEqual(env["PYTHONPATH"], os.pathsep.join(str(self.rt.root / repo / "src") for repo in ("agari-kit", "agari-research")))
        self.assertEqual(self.child.call_args.kwargs["cwd"], self.rt.root / "agari-research")
        self.assertLess(float(command[command.index("--max-wall-seconds") + 1]), 59)
        self.assertEqual(self.completed.call_args.args, (self.rt, self.prepared, "A-2026", self.job))
        first_bytes = (attempt / "finished.json").read_bytes()
        self.args.max_wall_seconds = 30  # Execution budget is outside the immutable model identity.
        again, code = self.invoke()
        self.assertEqual((code, again["status"]), (0, "already-complete"))
        self.assertEqual((attempt / "finished.json").read_bytes(), first_bytes)
        self.assertEqual(self.child.call_count, 1)
        self.assertEqual(len(result["logs"]), 2)
        self.assertGreater(result["invocation_wall_seconds"], 2)

    def test_completed_claim_still_requires_actual_weights_and_shared_gate(self):
        self.invoke()
        self.completed.side_effect = ValueError("Missing final weights")
        result, code = self.invoke()
        self.assertEqual((code, result["status"]), (1, "failed"))
        self.assertIn("Missing final", result["error"]["message"])
        self.assertEqual(self.child.call_count, 1)

    def test_existing_partial_run_requires_explicit_resume_and_checkpoint(self):
        self.rt.path("runs/" + self.job["run"]).mkdir()
        result, code = self.invoke()
        self.assertEqual(code, 1)
        self.assertIn("explicit --resume", result["error"]["message"])
        self.args.resume = True
        result, code = self.invoke()
        self.assertEqual(code, 1)
        self.assertIn("latest checkpoint", result["error"]["message"])
        self.child.assert_not_called()

    def test_legal_interrupted_attempt_then_explicit_resume_completes(self):
        import torch
        checkpoint_path = self.rt.path("checkpoints/" + self.job["run"] + "/latest.pt")
        self.args.stop_after_updates = 1
        def interrupted(command, env, attempt, lock_fd, **kwargs):
            self.complete_child(command, env, attempt, lock_fd)
            checkpoint_path.parent.mkdir(parents=True)
            torch.save({"progress": {"bc": 1}}, checkpoint_path)
            result = {"status": "interrupted", "stop_reason": "update_limit", "progress": {"bc": 1}}
            self.rt.path("runs/" + self.job["run"] + "/result.json").write_text(json.dumps(result))
            (attempt / "stdout.log").write_text(json.dumps(result, indent=2))
            return {"exit_code": 0}
        self.child.side_effect = interrupted
        with patch.object(runner, "resume_contract") as gate:
            result, code = self.invoke()
            self.assertEqual((code, result["status"]), (2, "interrupted"))
            self.assertIn("--stop-after-updates", self.child.call_args.args[0])
            self.args.resume, self.args.stop_after_updates = True, None
            self.child.side_effect = self.complete_child
            result, code = self.invoke()
            self.assertEqual((code, result["status"]), (0, "complete"))
            self.assertIn("--resume", self.child.call_args.args[0])
            self.assertEqual(gate.call_count, 2)
        self.assertTrue((self.execution / "attempt-0001/finished.json").is_file())
        self.assertTrue((self.execution / "attempt-0002/finished.json").is_file())

    def test_changed_config_and_changed_execution_binding_fail_before_training(self):
        self.rt.path(self.job["config"]).write_text("{}")
        with self.assertRaisesRegex(ValueError, "configuration hash"):
            self.invoke()
        self.child.assert_not_called()
        self.rt.path(self.job["config"]).write_text(json.dumps(self.config))
        self.invoke()
        with patch.object(self.rt, "implementation_binding", return_value={"changed": True}), self.assertRaisesRegex(ValueError, "Immutable artifact differs"):
            self.invoke()

    def test_nonzero_exit_keeps_logs_and_receipt_without_completion_claim(self):
        def failed(*args, **kwargs):
            result = self.complete_child(*args, **kwargs)
            return result | {"exit_code": 9}
        self.child.side_effect = failed
        result, code = self.invoke()
        self.assertEqual((code, result["status"]), (1, "failed"))
        self.assertEqual(result["process"]["exit_code"], 9)
        self.assertEqual(len(result["logs"]), 2)
        self.completed.assert_not_called()


class PrimitiveTests(unittest.TestCase):
    def test_interrupt_race_with_already_exited_child_still_reaps_and_records(self):
        process = SimpleNamespace(pid=12345, returncode=None)
        usage = SimpleNamespace(ru_utime=.1, ru_stime=.02, ru_maxrss=10)
        with tempfile.TemporaryDirectory() as directory, patch.object(runner.subprocess, "Popen", return_value=process), patch.object(
                runner.os, "wait4", side_effect=[KeyboardInterrupt, (12345, 0, usage)]) as wait, patch.object(
                runner.os, "killpg", side_effect=ProcessLookupError):
            result = runner.run_child(["python"], {}, Path(directory), 1)
        self.assertEqual(wait.call_count, 2)
        self.assertEqual(result["exit_code"], 0)
        self.assertTrue(result["interrupted_by_operator"])

    def test_invalid_budget_types_and_values_are_rejected(self):
        for value in (True, False, 0, -1, float("inf"), float("nan"), 172801, "60"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.budget(value)
        self.assertEqual(runner.budget(172800), 172800)

    def test_grp_partial_batch_budget_and_step_budget(self):
        config = {"grp": {"epochs": 2, "batch_size": 32}, "cql": {"steps": 500, "batch_size": 64}}
        counts = [{"grp_samples": 33}, {"grp_samples": 32}]
        self.assertEqual(runner.target_updates(config, counts, "grp"), 6)
        self.assertEqual(runner.target_updates(config, counts, "cql"), 500)

    def test_wait4_captures_only_child_stdout_stderr_exit_and_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with runner.job_lock(root / "job.lock") as fd:
                with self.assertRaisesRegex(ValueError, "already running"):
                    with runner.job_lock(root / "job.lock"):
                        pass
                result = runner.run_child([sys.executable, "-B", "-c",
                    'import json, sys; print(json.dumps({"status":"paused"}, indent=2)); print("stderr preserved", file=sys.stderr); sys.exit(7)'],
                    dict(os.environ), root, fd)
            self.assertEqual(result["exit_code"], 7)
            self.assertGreater(result["wall_seconds"], 0)
            self.assertGreater(result["max_rss_bytes"], 0)
            self.assertEqual(runner.stdout_result(root / "stdout.log"), {"status": "paused"})
            self.assertEqual((root / "stderr.log").read_text(), "stderr preserved\n")
            with runner.job_lock(root / "job.lock"):
                pass  # Existing lock inode is reusable after the previous owner exits.


class ResumeTests(unittest.TestCase):
    def test_partial_checkpoint_bound_to_registered_config_source_and_initialization(self):
        from agari_research.training import initialize_q, q_initialization_sha256
        config = json.loads((Path(__file__).parents[1] / "configs/full-coverage-small.json").read_text())
        config.update(stages=["bc"], policy_filter=None, bc={"steps": 3, "batch_size": 64})
        plan = {"dataset_sha256": "dataset", "policy_filter": None,
                "stages": {"bc": {"mode": "steps", "target_updates": 3}}}
        plan["sha256"] = matrix.content_digest(plan)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weak-training-counts.json").write_text('[{"cql_samples":64}]')
            rt = Runtime(root)
            job = {"arm": "A", "run": "registered"}
            prepared = {"datasets": {"weak": {"sha256": "dataset", "policy_filter": None}}}
            checkpoint = {"schema_version": 1, "config": config, "run_name": "registered", "dataset_sha256": "dataset",
                          "implementation_sha256": rt.implementation, "runtime_provenance": rt.runtime, "policy_filter": None,
                          "training_plan": plan, "progress": {"bc": 1}, "samples_seen": {"bc": 64},
                          "policy_initialization": {"seed": config["seed"], "q_state_sha256": q_initialization_sha256(initialize_q(config))}}
            metrics = rt.path("runs/registered/metrics.jsonl")
            metrics.parent.mkdir(parents=True)
            metrics.write_text('{"stage":"bc","update":1}\n')
            runner.resume_contract(rt, matrix, root, prepared, job, config, checkpoint)
            for field, value in (("config", {}), ("run_name", "other"), ("runtime_provenance", {}),
                                 ("dataset_sha256", "other"), ("progress", {"bc": 4}),
                                 ("samples_seen", {"bc": 63}), ("policy_initialization", {})):
                changed = copy.deepcopy(checkpoint)
                changed[field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    runner.resume_contract(rt, matrix, root, prepared, job, config, changed)
            metrics.write_text('{"stage":"bc","update":1}\n{"stage":"bc","update":2}\n')
            with self.assertRaisesRegex(ValueError, "Metrics do not end"):
                runner.resume_contract(rt, matrix, root, prepared, job, config, checkpoint)


if __name__ == "__main__":
    unittest.main()
