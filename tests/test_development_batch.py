"""Finite development scheduling and original training handoff, without matches."""
import importlib.util
import json
from pathlib import Path
import signal
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


def module(name, filename):
    path = Path(__file__).parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


dev = module("development_test", "phase2_development_batch.py")
batch = module("development_training_test", "phase2_training_batch.py")
matrix = module("development_matrix_test", "phase2_evaluation_matrix.py")
runner = module("development_runner_test", "run_prepared_training.py")
CLEAN = {"status": "cleaned", "signals": [], "orphaned_descendants": [], "forced_termination": False}


class FakeProcess:
    serial = 0

    def __init__(self, code):
        self.code = self.returncode = code
        FakeProcess.serial += 1
        self.pid = 20000 + FakeProcess.serial

    def poll(self):
        return self.code

    def wait(self, **kwargs):
        return self.code


class SchedulingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "runs/phase2-dev-batch-test"
        self.matrix_path = self.root / "runs/phase2-dev-matrix-test"
        self.registration = {"schema_version": 1, "kind": "phase2-development-batch-v1",
            "directory": "runs/phase2-dev-batch-test", "matrix": "runs/phase2-dev-matrix-test",
            "training_batch": "runs/phase2-training-batch-test", "binding": {},
            "job_order": list(dev.JOBS), "max_concurrent": 1, "min_available_memory_bytes": 2 * 1024**3,
            "max_wall_seconds": 60, "training_terminal_at_registration": None}
        matrix.publish(self.output / "registration.json", self.registration)
        artifact = lambda path: {"path": str(path.relative_to(self.root)), "sha256": matrix.digest(path)}
        def bound(entry):
            path = self.root / entry["path"]
            dev.require(matrix.digest(path) == entry["sha256"], "Bound artifact changed")
            return path
        self.context = SimpleNamespace(matrix=matrix, batch=batch, runner=runner,
            tools=self.root / "old-tools", path=lambda path: self.root / path, artifact=artifact,
            check=Mock(), rt=SimpleNamespace(root=self.root / "runtime", bound=bound,
                storage=SimpleNamespace(environment=lambda: {})))
        self.context.binding = Mock(return_value={})
        self.arguments = SimpleNamespace(run=self.registration["directory"])
        self.registered = {"scale": "small", "fixtures": {key: {} for key in dev.JOBS}}
        self.report = {"status": "complete", "winner": "B-2026", "primary_order": list(dev.MODELS),
                       "scale_10000_eligible": True, "scale_10000_candidates": ["B-2026"], "evidence": {}}
        self.commands = []
        self.modify_result = None
        self.addCleanup(patch.stopall)
        patch.object(dev, "training_status", return_value={"evidence": [], "preparation": {}}).start()
        patch.object(batch, "register_owned").start()
        patch.object(batch, "discover_owned").start()
        self.cleanup = patch.object(batch, "interrupt_owned", return_value=CLEAN).start()
        patch.object(batch, "process_identity", side_effect=lambda pid: {"pid": pid, "create_time": 10}).start()
        patch.object(batch, "available_memory", return_value={"total": 16 * 1024**3, "available": 8 * 1024**3}).start()
        def sample(helper, active):
            for step in active.values():
                step["rss_samples"] += 1
                step["sampled_peak_tree_rss_bytes"] = 512 * 1024**2
        patch.object(dev, "sample_processes", side_effect=sample).start()
        patch.object(dev.time, "sleep", return_value=None).start()
        self.launch = patch.object(dev.subprocess, "Popen", side_effect=self.child).start()
        self.load_matrix = patch.object(matrix, "load_matrix", return_value=self.registered).start()
        self.verify = patch.object(matrix, "verify_fixture", side_effect=lambda rt, path, expected: {"completion.json": path.name + "-sealed"}).start()
        self.report_gate = patch.object(matrix, "development_report", return_value=self.report).start()

    def child(self, command, **kwargs):
        self.commands.append(command)
        action = command[command.index("--runtime-root") + 2]
        code = 0
        if action == "register":
            matrix.publish(self.matrix_path / "registration.json", self.registered)
            result = {"status": "registered-not-played", "matrix": self.registration["matrix"],
                      "jobs": sorted(dev.JOBS), "games_per_job": 1000, "development_games_total": 20000}
        elif action == "report-development":
            self.assertEqual(len(self.verify.call_args_list), 20)
            matrix.publish(self.matrix_path / "development-report.json", self.report)
            result = {key: self.report[key] for key in ("status", "winner", "primary_order", "scale_10000_eligible", "scale_10000_candidates")}
        else:
            self.assertEqual(action, "run")
            result = {"status": "complete"}
        if self.modify_result:
            code, result = self.modify_result(command, code, result)
        kwargs["stdout"].write(json.dumps(result).encode())
        kwargs["stdout"].flush()
        kwargs["stderr"].write(b"original matrix diagnostic\n")
        kwargs["stderr"].flush()
        return FakeProcess(code)

    def run_batch(self, started=None):
        return dev.run_batch(self.context, self.arguments, time.monotonic() if started is None else started)

    def update_registration(self, **values):
        self.registration.update(values)
        (self.output / "registration.json").write_text(matrix.canonical(self.registration) + "\n")

    def campaigns(self):
        return [command for command in self.commands if "--job" in command]

    def events(self):
        return [json.loads(line) for line in (self.output / "events.jsonl").read_bytes().splitlines()]

    def test_exact_twenty_complete_before_report_and_no_other_actions(self):
        result = self.run_batch()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["completed_jobs"], sorted(dev.JOBS))
        self.assertEqual([command[command.index("--job") + 1] for command in self.campaigns()], list(dev.JOBS))
        self.assertEqual(len(self.commands), 22)
        self.assertEqual(self.verify.call_count, 20)
        self.report_gate.assert_called_once()
        self.assertEqual(result["development_report"]["sha256"], matrix.digest(self.matrix_path / "development-report.json"))
        for command in self.commands:
            self.assertEqual(command[3], str(self.context.tools / "scripts/phase2_evaluation_matrix.py"))
            self.assertNotIn("--resume", command)
            self.assertFalse(any(action in command for action in ("freeze-final", "run-final", "plan-akochan")))
        for path in (self.output / "steps").iterdir():
            receipt = matrix.read(path / "finished.json")
            self.assertEqual(receipt["stdout"], self.context.artifact(path / "stdout.log"))
            self.assertEqual((path / "stderr.log").read_bytes(), b"original matrix diagnostic\n")
        original = (self.output / "result.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "cannot restart"):
            self.run_batch()
        self.assertEqual((self.output / "result.json").read_bytes(), original)

    def test_two_serial_rss_probes_then_at_most_two(self):
        self.update_registration(max_concurrent=2)
        result = self.run_batch()
        self.assertEqual(result["status"], "complete")
        active, done, peak = set(), set(), 0
        for row in self.events():
            if row["event"] == "step-started" and row["key"] in dev.JOBS:
                if not set(dev.PROBES) <= done:
                    self.assertFalse(active)
                    self.assertIn(row["key"], dev.PROBES)
                active.add(row["key"])
                peak = max(peak, len(active))
            elif row["event"] == "step-finished" and row["key"] in dev.JOBS:
                active.remove(row["key"])
                done.add(row["key"])
        self.assertEqual(peak, 2)
        self.assertEqual(set(result["probe_sampled_peak_tree_rss_bytes"]), set(dev.PROBES))

    def test_zero_exit_paused_stops_without_resume_or_report(self):
        self.modify_result = lambda command, code, result: (0, {"status": "paused", "reason": "wall_time_budget"}) if "--job" in command else (code, result)
        result = self.run_batch()
        self.assertEqual(result["status"], "paused-campaign")
        self.assertEqual(len(self.campaigns()), 1)
        self.verify.assert_not_called()
        self.report_gate.assert_not_called()

    def test_failure_stops_dispatch_but_seals_already_started_neighbor(self):
        self.update_registration(max_concurrent=2)
        self.modify_result = lambda command, code, result: (3, result) if "primary/A-2027" in command else (code, result)
        result = self.run_batch()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(self.campaigns()), 4)
        self.assertIn("primary/A-2028", result["completed_jobs"])
        self.assertNotIn("primary/A-2027", result["completed_jobs"])
        self.report_gate.assert_not_called()

    def test_forged_complete_stdout_does_not_override_failed_seal(self):
        self.verify.side_effect = ValueError("unsealed attempt")
        result = self.run_batch()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["completed_jobs"], [])
        self.assertEqual(len(self.campaigns()), 1)

    def test_source_change_stops_before_another_campaign(self):
        def check(registration):
            if len(self.campaigns()) >= 1:
                raise ValueError("Bound runtime changed")
        self.context.check.side_effect = check
        result = self.run_batch()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(self.campaigns()), 1)

    def test_budget_includes_waiting_and_preflight(self):
        with patch.object(dev, "training_status", return_value=None):
            result = self.run_batch(time.monotonic() - 61)
        self.assertEqual(result["status"], "paused-wall-time")
        self.launch.assert_not_called()

    def test_training_failed_terminal_launches_nothing(self):
        with patch.object(dev, "training_status", side_effect=ValueError("training failed")):
            result = self.run_batch()
        self.assertEqual(result["status"], "failed")
        self.launch.assert_not_called()

    def test_insufficient_idle_memory_preserves_registered_matrix_but_starts_no_games(self):
        with patch.object(batch, "available_memory", return_value={"total": 16 * 1024**3, "available": 1024**3}):
            result = self.run_batch()
        self.assertEqual(result["status"], "paused-memory-needs-resource-review")
        self.assertTrue((self.matrix_path / "registration.json").exists())
        self.assertEqual(self.campaigns(), [])

    def test_second_memory_denial_degrades_to_serial(self):
        self.update_registration(max_concurrent=2)
        samples = iter([8, 8, 8, 1])
        with patch.object(batch, "available_memory", side_effect=lambda: {"total": 16 * 1024**3, "available": next(samples, 8) * 1024**3}):
            result = self.run_batch()
        self.assertEqual(result["status"], "complete")
        denied = [row for row in self.events() if row["event"] == "memory-check" and not row["allowed"]]
        self.assertEqual(len(denied), 1)
        self.assertEqual(denied[0]["snapshot"]["active_jobs"], 1)

    def test_operator_signal_during_preflight_prevents_dispatch(self):
        handlers = {}
        def signal_handler(number, handler):
            handlers[number] = handler
            return signal.SIG_DFL
        def check(registration):
            if len(self.commands) == 1:
                handlers[signal.SIGINT](signal.SIGINT, None)
        self.context.check.side_effect = check
        with patch.object(dev.signal, "signal", side_effect=signal_handler):
            result = self.run_batch()
        self.assertEqual(result["status"], "interrupted-by-operator")
        self.assertEqual(self.campaigns(), [])

    def test_complete_matrix_with_orphan_cannot_pass(self):
        self.cleanup.return_value = dict(CLEAN, orphaned_descendants=[{"pid": 1}])
        result = self.run_batch()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(self.campaigns(), [])

    def test_null_registration_stdout_cannot_start_games(self):
        self.modify_result = lambda command, code, result: (0, None)
        result = self.run_batch()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(self.campaigns(), [])

    def test_missing_or_changed_final_report_cannot_pass(self):
        self.modify_result = lambda command, code, result: (code, dict(result, winner="C-2028")) if "report-development" in command else (code, result)
        result = self.run_batch()
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["development_report"])

    def test_spawn_failure_keeps_original_command_and_terminal_failure(self):
        self.launch.side_effect = OSError("spawn failed")
        result = self.run_batch()
        self.assertEqual(result["status"], "failed")
        step = next((self.output / "steps").iterdir())
        receipt = matrix.read(step / "finished.json")
        self.assertIsNone(receipt["returncode"])
        self.assertEqual(receipt["started"], self.context.artifact(step / "started.json"))

    def test_failed_cleanup_keeps_tracker_for_final_retry(self):
        tracked = []
        def cleanup(active):
            tracked.append({key: step["process"].pid for key, step in active.items()})
            if len(tracked) == 1:
                raise RuntimeError("cleanup inspection failed")
            return CLEAN
        self.cleanup.side_effect = cleanup
        result = self.run_batch()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(self.cleanup.call_count, 2)
        self.assertEqual(tracked[0], tracked[1])
        self.assertTrue(any(row["event"] == "final-cleanup" for row in self.events()))

    def test_repeated_cleanup_error_still_preserves_terminal_failure(self):
        self.cleanup.side_effect = RuntimeError("cleanup inspection failed")
        result = self.run_batch()
        self.assertEqual(result["status"], "failed-cleanup")
        self.assertEqual(matrix.read(self.output / "result.json")["status"], "failed-cleanup")


class HandoffTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "runs/phase2-training-real"
        self.registration = {"training_batch": "runs/phase2-training-real", "binding": {"training_process": {
            "pid": 100, "create_time": 10, "exe": "/python", "cmdline": ["training.py"]}}}
        def artifact(path):
            return {"path": str(path.relative_to(self.root)), "sha256": matrix.digest(path)}
        def bound(entry):
            path = self.root / entry["path"]
            dev.require(matrix.digest(path) == entry["sha256"], "Bound receipt changed")
            return path
        self.context = SimpleNamespace(batch=batch, matrix=matrix, path=lambda path: self.root / path, artifact=artifact,
            rt=SimpleNamespace(bound=bound), training=SimpleNamespace(tools=self.root / "tools", rt=SimpleNamespace(root=self.root / "runtime"),
                prepared=Mock(return_value=({}, {}, {"frozen": "preparation"}))))
        self.addCleanup(patch.stopall)
        self.process = patch.object(batch, "process_identity", return_value=None).start()
        self.models = patch.object(matrix, "training_models", return_value={}).start()
        self.events = []
        matrix.publish(self.directory / "registration.json", {"kind": "frozen"})
        reg_sha = matrix.digest(self.directory / "registration.json")
        number = 0
        for key in batch.JOBS:
            for stage in (("inspect", "grp", "cql") if key.startswith("C-") else ("bc",)):
                number += 1
                path = self.directory / "steps" / f"{number:04d}-{key}-{stage}"
                start = {"job": key, "stage": stage, "registration_sha256": reg_sha, "remaining_batch_seconds": 119.9,
                    "argv": batch.command_for(self.context.training, key, stage, 120.0, 7 if stage in ("grp", "cql") else None)}
                matrix.publish(path / "started.json", start)
                code = 2 if stage == "grp" else 0
                result = {"status": "inspected-not-started", "targets": {"grp": 7, "cql": 20}} if stage == "inspect" else {
                    "status": "interrupted" if stage == "grp" else "complete", "job_id": key, "exit_code": code,
                    "attempt": f"runs/{Path(batch.PREPARATION).name}-execution/{key}/attempt-{number:04d}",
                    "stop_reason": "update_limit" if stage == "grp" else None,
                    "cli_result": {"progress": {"grp": 7, "cql": 0}}}
                matrix.publish(path / "stdout.log", result)
                (path / "stderr.log").write_text("original\n")
                receipt = {"job": key, "stage": stage, "returncode": code,
                    **{name: artifact(path / file) for name, file in (("started", "started.json"), ("stdout", "stdout.log"), ("stderr", "stderr.log"))}}
                if stage != "inspect":
                    original = self.root / result["attempt"] / "finished.json"
                    matrix.publish(original, result)
                    receipt["runner_finished"] = artifact(original)
                matrix.publish(path / "finished.json", receipt)
                self.events += [{"event": "step-started", "job": key, "stage": stage},
                    {"event": "post-exit-cleanup", "job": key, "evidence": CLEAN},
                    {"event": "step-finished", "job": key, "stage": stage, "returncode": code, "receipt": artifact(path / "finished.json")}]
        self.write_events()
        matrix.publish(self.directory / "prepared.json", {"preparation": {"frozen": "preparation"}})
        matrix.publish(self.directory / "completed-models.json", {})
        matrix.publish(self.directory / "result.json", {"status": "complete", "completed_jobs": sorted(batch.JOBS),
            "registration_sha256": reg_sha, "automatic_budget_resumes": 0, "evaluation_started": False,
            "completed_models": artifact(self.directory / "completed-models.json")})

    def write_events(self):
        (self.directory / "events.jsonl").write_text("\n".join(matrix.canonical(row) for row in self.events) + "\n")

    def test_real_complete_handoff_rechecks_all_steps_and_shared_nine_model_gate(self):
        result = dev.training_status(self.context, self.registration)
        self.assertTrue(result["original_process_exited"])
        self.models.assert_called_once()
        self.assertEqual(len(result["evidence"]), 76)  # 15*4 step files + 12 runner receipts + four terminal files.

    def test_matching_live_process_waits_even_if_result_already_appeared(self):
        self.process.return_value = self.registration["binding"]["training_process"]
        self.assertIsNone(dev.training_status(self.context, self.registration))
        self.models.assert_not_called()

    def test_reused_pid_is_not_mistaken_for_original_live_process(self):
        self.process.return_value = dict(self.registration["binding"]["training_process"], create_time=999)
        self.assertIsNotNone(dev.training_status(self.context, self.registration))

    def test_same_pid_ctime_with_different_argv_is_rejected(self):
        self.process.return_value = dict(self.registration["binding"]["training_process"], cmdline=["different.py"])
        with self.assertRaisesRegex(ValueError, "identity changed"):
            dev.training_status(self.context, self.registration)

    def test_failed_or_missing_ninth_training_job_rejected(self):
        path = self.directory / "result.json"
        value = matrix.read(path)
        value["completed_jobs"].pop()
        path.write_text(matrix.canonical(value))
        with self.assertRaisesRegex(ValueError, "all nine"):
            dev.training_status(self.context, self.registration)

    def test_changed_original_stdout_rejected(self):
        (next((self.directory / "steps").iterdir()) / "stdout.log").write_text('{"status":"complete"}')
        with self.assertRaisesRegex(ValueError, "receipt changed"):
            dev.training_status(self.context, self.registration)

    def test_extra_failed_attempt_cannot_be_hidden_behind_complete_result(self):
        self.events.append({"event": "step-rejected"})
        self.write_events()
        with self.assertRaisesRegex(ValueError, "rejected"):
            dev.training_status(self.context, self.registration)

    def test_shared_completed_model_gate_is_required(self):
        self.models.side_effect = ValueError("best descriptor changed")
        with self.assertRaisesRegex(ValueError, "descriptor changed"):
            dev.training_status(self.context, self.registration)

    def test_completed_model_receipt_must_match_actual_best_models(self):
        self.models.return_value = {"A-2026": "substituted"}
        with self.assertRaisesRegex(ValueError, "completed-model evidence"):
            dev.training_status(self.context, self.registration)


if __name__ == "__main__":
    unittest.main()
