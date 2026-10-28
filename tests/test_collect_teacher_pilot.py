"""Pilot safety gates with temporary storage; no teacher, engine games or network."""
from dataclasses import asdict
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agari_kit.agent import Rules
from agari_kit.storage import Storage

SCRIPT = Path(__file__).parents[1] / "scripts/collect_teacher_pilot.py"
spec = importlib.util.spec_from_file_location("pilot_entrypoint_test", SCRIPT)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class PilotTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.storage = Storage(self.root / "artifacts")
        self.storage.root.mkdir()
        frozen = self.root / "frozen"
        helper = frozen / "agari-research/scripts/phase2_contract.py"
        helper.parent.mkdir(parents=True)
        helper.write_text("synthetic helper identity only\n")
        protocol = frozen / "agari-research/protocol.json"
        protocol.write_text("{}\n")
        self.identity = {"code": {"source_sha256": "a" * 64},
                         "engine": {"version": "0.4.10", "binary_sha256": "b" * 64},
                         "rules": asdict(Rules())}
        descriptor = self.storage.root / "teacher.json"
        descriptor.write_text("{}\n")
        self.provenance = {"name": "@" + str(descriptor), "factory": "fixture:Teacher"}
        contract = SimpleNamespace(
            load_protocol=Mock(return_value={"seeds": {"teacher_1000": {"start": 2630000, "count": 1000, "rotations": 1}}}),
            require_qualification=Mock(return_value={"collection_identity": self.identity, "qualification_sha256": "c" * 64}),
            content_sha256=lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest())
        self.runtime = SimpleNamespace(root=frozen, protocol_path=protocol, contract=contract,
            Storage=SimpleNamespace(from_env=lambda: self.storage), Rules=Rules,
            source_fingerprint=Mock(return_value=self.identity["code"]),
            engine_identity=Mock(return_value=self.identity["engine"]),
            agent_provenance=Mock(return_value=self.provenance), collect_selfplay=Mock(),
            binding={"entrypoint_sha256": pilot.digest(SCRIPT), "helper_sha256": pilot.digest(helper),
                     "protocol_sha256": pilot.digest(protocol), "root": str(frozen)})
        self.args = SimpleNamespace(runtime_root=frozen, teacher_descriptor=descriptor,
            qualification_run="runs/qualified", name="phase2-pilot-test", max_wall_seconds=20.)
        self.output = self.storage.root / "runs/phase2-pilot-test-collection"
        self.clock = [0.]
        for target, kwargs in (
            ("load_runtime", {"return_value": self.runtime}),
            ("require_module_paths", {"return_value": {}}),
            ("split_with_budget", {"return_value": {"games": 1000, "sha256": "d" * 64}}),
        ):
            mock = patch.object(pilot, target, **kwargs).start()
            self.addCleanup(patch.stopall)
            setattr(self, target, mock)
        patch.object(Storage, "check", return_value={"write_verified": True}).start()
        patch.object(pilot.time, "monotonic", side_effect=lambda: self.clock[0]).start()

    def qualify_after(self, seconds):
        def qualified(*args):
            self.clock[0] += seconds
            return {"collection_identity": self.identity, "qualification_sha256": "c" * 64}
        self.runtime.contract.require_qualification.side_effect = qualified

    def collect_as(self, status, seconds=0):
        def collect(*args, **kwargs):
            self.clock[0] += seconds
            registration = json.loads((self.output / "registration.json").read_text())
            path = self.storage.path(f"datasets/manifests/{self.args.name}/acquisition.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"status": status, "config": registration["collection"]}))
            return {"status": status, "games": 1000}
        self.runtime.collect_selfplay.side_effect = collect

    def results(self):
        return [json.loads(path.read_text()) for path in sorted((self.output / "invocations").glob("*/result.json"))]

    def test_qualification_failure_never_registers_or_collects(self):
        self.runtime.contract.require_qualification.side_effect = ValueError("independent gate failed")
        with self.assertRaisesRegex(ValueError, "gate failed"):
            pilot.run(self.args)
        self.runtime.collect_selfplay.assert_not_called()
        self.split_with_budget.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_wrong_qualified_source_and_unregistered_cohort_fail_before_collection(self):
        self.runtime.source_fingerprint.return_value = {"source_sha256": "different"}
        with self.assertRaisesRegex(ValueError, "qualified identity"):
            pilot.run(self.args)
        self.runtime.collect_selfplay.assert_not_called()
        self.runtime.source_fingerprint.return_value = self.identity["code"]
        self.runtime.contract.load_protocol.return_value["seeds"]["teacher_1000"]["start"] = 99
        with self.assertRaisesRegex(ValueError, "registered 1000"):
            pilot.run(self.args)
        self.runtime.collect_selfplay.assert_not_called()

    def test_invalid_budget_is_rejected_before_loading_runtime(self):
        for value in (0, -1, 172801, float("nan"), float("inf"), True):
            with self.subTest(value=value):
                self.args.max_wall_seconds = value
                with self.assertRaisesRegex(ValueError, "budget"):
                    pilot.run(self.args)
        self.load_runtime.assert_not_called()

    def test_preflight_exhaustion_does_not_call_collector(self):
        self.qualify_after(21)
        result = pilot.run(self.args)
        self.assertEqual((result["status"], result["stage"]), ("paused", "preflight-budget"))
        self.runtime.collect_selfplay.assert_not_called()
        self.split_with_budget.assert_not_called()
        self.assertEqual(self.results()[0]["status"], "paused")

    def test_budget_deduction_and_paused_collection_never_split(self):
        self.qualify_after(5)
        self.collect_as("paused", seconds=16)
        result = pilot.run(self.args)
        kwargs = self.runtime.collect_selfplay.call_args.kwargs
        self.assertEqual(kwargs["max_wall_seconds"], 15)
        self.assertEqual((kwargs["games"], kwargs["seed_start"]), (1000, 2630000))
        self.assertEqual(kwargs["agents"], [self.provenance["name"]] * 4)
        self.assertEqual(result["status"], "paused")
        self.split_with_budget.assert_not_called()

    def test_complete_after_budget_waits_for_split_and_resume_does_not_recollect(self):
        self.collect_as("complete", seconds=21)
        result = pilot.run(self.args)
        self.assertEqual(result["status"], "complete-awaiting-split")
        self.split_with_budget.assert_not_called()
        original = (self.output / "registration.json").read_bytes()
        previous = next((self.output / "invocations").glob("*/result.json"))
        previous_bytes = previous.read_bytes()
        self.runtime.collect_selfplay.reset_mock()
        result = pilot.run(self.args)
        self.assertEqual(result["status"], "ready")
        self.runtime.collect_selfplay.assert_not_called()
        self.split_with_budget.assert_called_once()
        self.assertEqual((self.output / "registration.json").read_bytes(), original)
        self.assertEqual(previous.read_bytes(), previous_bytes)
        self.assertEqual(len(self.results()), 2)

    def test_existing_ready_split_is_reverified_on_reentry(self):
        self.collect_as("complete")
        self.assertEqual(pilot.run(self.args)["status"], "ready")
        self.runtime.collect_selfplay.reset_mock()
        self.assertEqual(pilot.run(self.args)["status"], "ready")
        self.runtime.collect_selfplay.assert_not_called()
        self.assertEqual(self.split_with_budget.call_count, 2)

    def test_changed_registration_identity_never_resumes_collection(self):
        self.collect_as("paused")
        pilot.run(self.args)
        self.runtime.collect_selfplay.reset_mock()
        self.runtime.binding["entrypoint_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "registration identity changed"):
            pilot.run(self.args)
        self.runtime.collect_selfplay.assert_not_called()
        self.assertEqual(len(self.results()), 1)

    def test_partial_registration_and_unregistered_data_fail_closed(self):
        self.output.mkdir(parents=True)
        with self.assertRaises(FileNotFoundError):
            pilot.run(self.args)
        self.runtime.collect_selfplay.assert_not_called()
        self.output.rmdir()
        self.storage.path(f"datasets/raw/{self.args.name}").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "refusing adoption"):
            pilot.run(self.args)
        self.runtime.collect_selfplay.assert_not_called()

    def test_split_timeout_is_awaiting_but_other_errors_are_failed(self):
        self.collect_as("complete")
        self.split_with_budget.return_value = None
        self.assertEqual(pilot.run(self.args)["status"], "complete-awaiting-split")
        self.split_with_budget.side_effect = RuntimeError("split subprocess failed")
        with self.assertRaisesRegex(RuntimeError, "subprocess failed"):
            pilot.run(self.args)
        self.assertEqual(self.results()[-1]["status"], "failed")

    def test_identity_drift_during_split_cannot_be_reported_ready(self):
        self.collect_as("complete")
        def changed(*args):
            self.runtime.source_fingerprint.return_value = {"source_sha256": "changed during split"}
            return {"games": 1000, "sha256": "f" * 64}
        self.split_with_budget.side_effect = changed
        with self.assertRaisesRegex(ValueError, "changed during invocation"):
            pilot.run(self.args)
        self.assertEqual(self.results()[0]["status"], "failed")


class BoundaryTests(unittest.TestCase):
    def test_split_receipts_must_match_qualified_code_engine_and_rules(self):
        import agari_kit
        import agari_kit.collection as collection
        import agari_kit.dataset as dataset
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "receipts").mkdir()
            qualified = {"code": {"source_sha256": "a" * 64},
                         "engine": {"version": "0.4.10", "binary_sha256": "b" * 64},
                         "rules": asdict(Rules())}
            base = {"agent_provenance_sha256": ["c" * 64] * 4, "producer_code": qualified["code"],
                    "engine_version": "0.4.10", "engine_binary_sha256": "b" * 64, "rules": qualified["rules"]}
            records = []
            for index in range(1000):
                relative = f"receipts/{index}.json"
                (root / relative).write_text(json.dumps(dict(base, seed=2630000 + index)))
                records.append({"validation_receipt": relative})
            source = Mock()
            source.__enter__ = Mock(return_value=SimpleNamespace(games=lambda: records))
            source.__exit__ = Mock(return_value=False)
            argv = ["split", str(root), "catalog.sqlite", "split-v1.json", "c" * 64, json.dumps(qualified)]
            with patch.object(agari_kit, "__file__", str(root / "agari-kit/src/agari_kit/__init__.py")), \
                    patch.object(Storage, "from_env", return_value=Storage(root)), \
                    patch.object(dataset, "Catalog", return_value=source), \
                    patch.object(collection, "verified_split", return_value={"games": 1000}) as split, \
                    patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                exec(pilot.SPLIT_CODE, {})
                split.assert_called_once()
                for field, changed in (("producer_code", {}), ("engine_binary_sha256", "changed"),
                                       ("engine_version", "other"), ("rules", {})):
                    split.reset_mock()
                    (root / "receipts/0.json").write_text(json.dumps(dict(base, seed=2630000, **{field: changed})))
                    with self.subTest(field=field), self.assertRaisesRegex(ValueError, "qualified collection identity"):
                        exec(pilot.SPLIT_CODE, {})
                    split.assert_not_called()

    def test_wrong_actual_import_path_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrong = SimpleNamespace(__file__=str(root / "mutable/agari_kit/__init__.py"))
            with patch.object(pilot.importlib, "import_module", return_value=wrong):
                with self.assertRaisesRegex(ValueError, "Wrong agari_kit module source"):
                    pilot.require_module_paths(root)

    def test_attached_or_dirty_runtime_checkout_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for branch_code, dirty in ((0, ""), (1, " M src/policy.py\n")):
                replies = [SimpleNamespace(returncode=0, stdout="a" * 40),
                           SimpleNamespace(returncode=0, stdout=str(root)),
                           SimpleNamespace(returncode=branch_code, stdout="refs/heads/main"),
                           SimpleNamespace(returncode=0, stdout=dirty)]
                with patch.object(pilot.subprocess, "run", side_effect=replies):
                    with self.assertRaisesRegex(ValueError, "detached paired"):
                        pilot.checkout_identity(root)

    def test_split_timeout_terminates_waits_then_kills_and_reaps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = Mock()
            process.wait.side_effect = [subprocess.TimeoutExpired("split", 1),
                                       subprocess.TimeoutExpired("split", 2), 0]
            with patch.object(pilot.subprocess, "Popen", return_value=process) as start:
                result = pilot.split_with_budget(SimpleNamespace(root=root), Storage(root),
                                                 "phase2-test", "f" * 64, {}, root, 1)
            self.assertIsNone(result)
            process.terminate.assert_called_once()
            process.kill.assert_called_once()
            self.assertEqual(process.wait.call_count, 3)
            command = start.call_args.args[0]
            self.assertEqual(command[:3], [sys.executable, "-B", "-c"])
            environment = start.call_args.kwargs["env"]
            self.assertEqual(environment["PYTHONPATH"], f"{root}/agari-kit/src:{root}/agari-research/src")

    def test_split_nonzero_exit_raises_instead_of_reporting_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = Mock()
            process.wait.return_value = 1
            with patch.object(pilot.subprocess, "Popen", return_value=process):
                with self.assertRaisesRegex(RuntimeError, "split verification failed"):
                    pilot.split_with_budget(SimpleNamespace(root=root), Storage(root),
                                            "phase2-test", "f" * 64, {}, root, 1)
            process.terminate.assert_not_called()

    def test_split_keyboard_interrupt_reaps_and_preserves_original_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = Mock()
            original = KeyboardInterrupt("user cancelled")
            process.wait.side_effect = [original, 0]
            with patch.object(pilot.subprocess, "Popen", return_value=process):
                with self.assertRaises(KeyboardInterrupt) as caught:
                    pilot.split_with_budget(SimpleNamespace(root=root), Storage(root),
                                            "phase2-test", "f" * 64, {}, root, 1)
            self.assertIs(caught.exception, original)
            process.terminate.assert_called_once()
            self.assertEqual(process.wait.call_count, 2)


if __name__ == "__main__":
    unittest.main()
