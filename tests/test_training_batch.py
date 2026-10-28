"""Finite scheduling and fail-closed handoffs using synthetic runner receipts."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "scripts" / filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


batch = module("batch_test", "phase2_training_batch.py")
matrix = module("batch_matrix_test", "phase2_evaluation_matrix.py")
runner = module("batch_runner_test", "run_prepared_training.py")


class FakeProcess:
    next_pid = 10000

    def __init__(self, code):
        self.code = code
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1

    def poll(self):
        return self.code

    def wait(self):
        return self.code


class BatchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "runs/phase2-training-batch-test"
        self.output.mkdir(parents=True)
        self.registration = {"kind": "phase2-nine-training-batch-v1", "directory": "runs/phase2-training-batch-test",
            "job_order": list(batch.JOBS), "pilot_launch": batch.PILOT, "max_concurrent": 1,
            "min_available_memory_bytes": 2 * 1024**3, "max_wall_seconds": 60, "binding": {}}
        matrix.publish(self.output / "registration.json", self.registration)
        artifact = lambda path: {"path": str(path.relative_to(self.root)), "sha256": matrix.digest(path)}
        pilot = self.root / batch.PILOT
        pilot.mkdir(parents=True)
        (pilot / "result.json").write_text('{}\n')
        (pilot / "events.jsonl").write_text('{}\n')
        self.pilot = {"result": artifact(pilot / "result.json"), "events": artifact(pilot / "events.jsonl")}
        jobs = {key: {"arm": key[0], "seed": int(key[2:]), "run": "phase2-" + key.lower()} for key in batch.JOBS}
        self.context = SimpleNamespace(matrix=matrix, runner=runner, tools=self.root / "tools", path=lambda name: self.root / name,
            artifact=artifact, check=Mock(), prepared=Mock(return_value=({}, jobs, {"input": "fixed"})),
            rt=SimpleNamespace(root=self.root / "runtime", storage=SimpleNamespace(environment=lambda: {})))
        self.arguments = SimpleNamespace(run=self.registration["directory"])
        self.commands = []
        self.modify_result = None
        self.addCleanup(patch.stopall)
        patch.object(batch, "pilot_status", return_value=self.pilot).start()
        patch.object(batch, "register_owned", create=True).start()
        patch.object(batch, "discover_owned", create=True).start()
        self.cleanup = patch.object(batch, "interrupt_owned", return_value={"status": "cleaned", "signals": [],
                                    "orphaned_descendants": [], "forced_termination": False}).start()
        patch.object(batch, "available_memory", return_value={"total": 16 * 1024**3, "available": 8 * 1024**3}).start()
        patch.object(batch.time, "sleep", return_value=None).start()
        self.launch = patch.object(batch.subprocess, "Popen", side_effect=self.child).start()
        self.completed = patch.object(matrix, "verify_completed_job", return_value={"verified": True}).start()
        self.final_models = patch.object(matrix, "training_models", return_value={key: {"verified": True} for key in batch.JOBS}).start()

    def child(self, command, **kwargs):
        self.commands.append(command)
        key = command[command.index("--job") + 1]
        if "--inspect" in command:
            result = {"status": "inspected-not-started", "targets": {"grp": 7, "cql": 30}}
            code = 0
        else:
            if "--stop-after-updates" in command:
                self.assertEqual(command[command.index("--stop-after-updates") + 1], "7")
                result = {"status": "interrupted", "stop_reason": "update_limit",
                          "cli_result": {"status": "interrupted", "stop_reason": "update_limit", "progress": {"grp": 7, "cql": 0}}}
                code = 2
            else:
                result = {"status": "complete", "process": {"max_rss_bytes": 400 * 1024**2}}
                code = 0
            result["attempt"] = f"runs/{Path(batch.PREPARATION).name}-execution/{key}/attempt-{len(self.commands):04d}"
        if self.modify_result:
            code, result = self.modify_result(command, code, result)
        if "attempt" in result:
            result.update(job_id=key, exit_code=code)
            path = self.root / result["attempt"] / "finished.json"
            matrix.publish(path, result)
        kwargs["stdout"].write(json.dumps(result).encode())
        kwargs["stdout"].flush()
        kwargs["stderr"].write(b"original diagnostic\n")
        kwargs["stderr"].flush()
        return FakeProcess(code)

    def execute(self, started=None):
        return batch.run_batch(self.context, self.arguments, time.monotonic() if started is None else started)

    def events(self):
        return [json.loads(line) for line in (self.output / "events.jsonl").read_text().splitlines()]

    def update_registration(self, **values):
        self.registration.update(values)
        (self.output / "registration.json").write_text(matrix.canonical(self.registration) + "\n")

    def test_all_nine_jobs_and_exact_c_boundaries_use_original_runner(self):
        result = self.execute()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["completed_jobs"], sorted(batch.JOBS))
        self.assertEqual(len(self.commands), 15)  # Six BC plus three inspect/GRP/CQL triples.
        self.assertEqual(self.completed.call_count, 9)
        self.final_models.assert_called_once()
        self.assertEqual(result["completed_models"]["sha256"], matrix.digest(self.output / "completed-models.json"))
        for seed in (2026, 2027, 2028):
            commands = [command for command in self.commands if f"C-{seed}" in command]
            self.assertIn("--inspect", commands[0])
            self.assertIn("--stop-after-updates", commands[1])
            self.assertIn("--resume", commands[2])
        for path in sorted((self.output / "steps").iterdir()):
            receipt = matrix.read(path / "finished.json")
            self.assertEqual(receipt["stdout"]["sha256"], matrix.digest(path / "stdout.log"))
            self.assertEqual((path / "stderr.log").read_bytes(), b"original diagnostic\n")
            if receipt["stage"] != "inspect":
                self.assertEqual(receipt["runner_finished"]["sha256"], matrix.digest(self.context.path(receipt["runner_finished"]["path"])))
        old = (self.output / "result.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "cannot restart"):
            self.execute()
        self.assertEqual((self.output / "result.json").read_bytes(), old)

    def test_first_a_is_alone_then_at_most_two_and_every_dispatch_has_real_memory_sample(self):
        self.update_registration(max_concurrent=2)
        result = self.execute()
        self.assertEqual(result["status"], "complete")
        running, peak, first_finished = set(), 0, False
        for event in self.events():
            if event["event"] == "step-started":
                if not first_finished:
                    self.assertEqual(event["job"], "A-2026")
                running.add(event["job"])
                peak = max(peak, len(running))
            elif event["event"] == "step-finished":
                running.remove(event["job"])
                first_finished |= event["job"] == "A-2026"
        self.assertEqual(peak, 2)
        self.assertEqual(len([row for row in self.events() if row["event"] == "memory-check"]), 15)

    def test_insufficient_second_job_memory_degrades_to_serial_and_does_not_pause_batch(self):
        self.update_registration(max_concurrent=2)
        samples = iter([8 * 1024**3, 8 * 1024**3, 1024**3])
        with patch.object(batch, "available_memory", side_effect=lambda: {"total": 16 * 1024**3, "available": next(samples, 8 * 1024**3)}):
            result = self.execute()
        self.assertEqual(result["status"], "complete")
        denied = [row for row in self.events() if row["event"] == "memory-check" and not row["allowed"]]
        self.assertEqual(len(denied), 1)
        self.assertEqual(denied[0]["snapshot"]["active_jobs"], 1)

    def test_insufficient_idle_memory_pauses_without_spinning_or_launching(self):
        with patch.object(batch, "available_memory", return_value={"total": 16 * 1024**3, "available": 1024**3}):
            result = self.execute()
        self.assertEqual(result["status"], "paused-memory-needs-resource-review")
        self.launch.assert_not_called()

    def test_wall_time_pause_is_never_automatically_resumed(self):
        self.modify_result = lambda command, code, result: (2, result | {"status": "paused", "stop_reason": "wall_time_budget"})
        result = self.execute()
        self.assertEqual(result["status"], "paused-wall-time")
        self.assertEqual(len(self.commands), 1)
        self.assertNotIn("--resume", self.commands[0])

    def test_bad_grp_boundary_stops_before_cql_and_other_jobs(self):
        def changed(command, code, result):
            if "--stop-after-updates" in command:
                result["cli_result"]["progress"]["cql"] = 1
            return code, result
        self.modify_result = changed
        result = self.execute()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(self.commands), 4)
        self.assertFalse(any("--resume" in command for command in self.commands))

    def test_shared_model_gate_failure_stops_before_dispatching_second_job(self):
        self.completed.side_effect = ValueError("Final tensor binding failed")
        result = self.execute()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(self.commands), 1)

    def test_final_whole_matrix_reverification_can_reject_early_model_mutation(self):
        self.final_models.side_effect = ValueError("Previously completed model changed")
        result = self.execute()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(self.completed.call_count, 9)
        self.assertIsNone(result["completed_models"])
        self.assertFalse((self.output / "completed-models.json").exists())

    def test_normal_exit_that_needed_orphan_cleanup_is_rejected(self):
        self.cleanup.return_value = {"status": "cleaned", "signals": [{"pid": 9, "signal": 15}],
                                     "orphaned_descendants": [{"pid": 9}], "forced_termination": True}
        result = self.execute()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(self.commands), 1)
        self.assertEqual(len(list((self.output / "steps").glob("*/finished.json"))), 1)
        self.completed.assert_not_called()

    def test_cleanup_error_preserves_tracker_for_final_cleanup_and_writes_failure(self):
        self.cleanup.side_effect = [PermissionError("test denied"), {"status": "cleaned", "signals": []}]
        result = self.execute()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(self.cleanup.call_count, 2)
        self.assertEqual(len(self.commands), 1)

    def test_failed_job_stops_dispatch_while_existing_other_step_finishes(self):
        self.update_registration(max_concurrent=2)
        def fail_b(command, code, result):
            return (1, result | {"status": "failed"}) if "B-2026" in command else (code, result)
        self.modify_result = fail_b
        result = self.execute()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(self.commands), 3)  # A, B, C inspect; never C GRP or next seed.
        self.assertEqual(len(list((self.output / "steps").glob("*/finished.json"))), 3)
        self.assertEqual(result["completed_jobs"], ["A-2026"])

    def test_runner_finished_receipt_cannot_be_substituted(self):
        def substituted(command, **kwargs):
            child = self.child(command, **kwargs)
            key = command[command.index("--job") + 1]
            path = self.root / f"runs/{Path(batch.PREPARATION).name}-execution/{key}/attempt-0001/finished.json"
            path.write_text('{"status":"complete","substituted":true}')
            return child
        self.launch.side_effect = substituted
        result = self.execute()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(self.commands), 1)
        self.completed.assert_not_called()

    def test_spawn_failure_preserves_step_receipt_and_terminal_failure(self):
        self.launch.side_effect = OSError("test executable failure")
        result = self.execute()
        self.assertEqual(result["status"], "failed")
        receipts = list((self.output / "steps").glob("*/finished.json"))
        self.assertEqual(len(receipts), 1)
        self.assertIsNone(matrix.read(receipts[0])["returncode"])
        self.assertEqual(matrix.read(receipts[0])["error_type"], "OSError")

    def test_changed_preparation_stops_at_next_boundary(self):
        original = self.context.prepared.return_value
        self.context.prepared.side_effect = lambda: original if not self.commands else (original[0], original[1], {"input": "changed"})
        result = self.execute()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(self.commands), 1)

    def test_budget_includes_waiting_before_pilot_is_ready(self):
        with patch.object(batch, "pilot_status", return_value=None):
            result = self.execute(time.monotonic() - 61)
        self.assertEqual(result["status"], "paused-wall-time")
        self.launch.assert_not_called()


class GatesTests(unittest.TestCase):
    def test_memory_thresholds_use_future_available_and_first_probe_rss(self):
        sample = {"total": 16 * 1024**3, "available": 3 * 1024**3}
        self.assertTrue(batch.memory_gate(0, 2 * 1024**3, 2 * 1024**3, sample)[0])
        allowed, evidence = batch.memory_gate(1, 2 * 1024**3, 2 * 1024**3, sample)
        self.assertFalse(allowed)
        self.assertEqual(evidence["threshold_bytes"], 4 * 1024**3)

    def test_invalid_limits_and_false_wall_return_are_rejected(self):
        for values in ((3, 1, 60), (True, 1, 60), (1, 0, 60), (1, 1, 172801), (1, 1, float("nan"))):
            with self.subTest(values=values), self.assertRaises(ValueError):
                batch.limits(*values)
        with self.assertRaises(ValueError):
            batch.step_transition("A-2026", "bc", 1, {"status": "paused", "stop_reason": "wall_time_budget"})

    def test_pilot_requires_original_process_exit_and_successful_terminal_events(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pilot = root / batch.PILOT
            pilot.mkdir(parents=True)
            original = {"pid": 123, "create_time": 100, "exe": "/python", "cmdline": ["driver.py"]}
            registration = {"pilot_launch": batch.PILOT, "pilot_pid": 123, "pilot_process": original}
            context = SimpleNamespace(matrix=matrix, path=lambda relative: root / relative,
                artifact=lambda path: {"path": str(path.relative_to(root)), "sha256": matrix.digest(path)})
            with patch.object(batch, "process_identity", return_value=original):
                self.assertIsNone(batch.pilot_status(context, registration))
            result = {"status": "prepared-not-trained", "pilot": {"status": "ready"},
                      "preparation": {"jobs": 9, "output": str(root / batch.PREPARATION)}}
            (pilot / "result.json").write_text(json.dumps(result))
            events = [{"event": "step-finished", "step": name, "returncode": 0} for name in ("qualification", "pilot", "preparation")]
            events.append({"event": "handoff-finished", "status": "prepared-not-trained"})
            (pilot / "events.jsonl").write_text("\n".join(json.dumps(row) for row in events))
            with patch.object(batch, "process_identity", return_value=None):
                self.assertTrue(batch.pilot_status(context, registration)["original_process_exited"])
                result["status"] = "pilot-requires-resume"
                (pilot / "result.json").write_text(json.dumps(result))
                with self.assertRaisesRegex(ValueError, "not prepared-not-trained"):
                    batch.pilot_status(context, registration)


if __name__ == "__main__":
    unittest.main()
