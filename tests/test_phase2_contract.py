"""Small synthetic evidence tests; never launch teacher workers or play games."""
import copy
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from agari_kit.agent import Rules
from agari_kit.storage import Storage


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


gates = module("phase2_contract_test", "scripts/phase2_contract.py")
prepare = module("phase2_prepare_test", "scripts/prepare_teacher_experiment.py")
PROTOCOL_PATH = Path(__file__).parents[1] / "experiments/phase2-teacher-student-v1/protocol.json"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class ProtocolTests(unittest.TestCase):
    def test_registered_contract_and_equal_policy_budgets(self):
        protocol = gates.load_protocol(PROTOCOL_PATH)
        base = json.loads((PROTOCOL_PATH.parents[2] / protocol["training"]["model_reference_config"]).read_text())
        selection = {"agent_identity_sha256": ["a" * 64]}
        jobs = prepare.training_configs(base, 1001, 2003, weak_filter=selection,
                                        teacher_filter=selection, device="cpu", protocol=protocol)
        self.assertEqual({(job["arm"], job["seed"]) for job in jobs},
                         {(arm, seed) for arm in "ABC" for seed in (2026, 2027, 2028)})
        for job in jobs:
            config = job["config"]
            self.assertEqual((config[job["method"]]["steps"], config[job["method"]]["batch_size"]), (94, 64))
            self.assertNotIn("ppo", config)

    def test_protocol_overlap_and_unsupported_budget_change_fail(self):
        original = json.loads(PROTOCOL_PATH.read_text())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "protocol.json"
            changed = copy.deepcopy(original)
            changed["seeds"]["teacher_1000"]["start"] = changed["seeds"]["development"]["start"]
            write(path, changed)
            with self.assertRaisesRegex(ValueError, "overlap"):
                gates.load_protocol(path)
            changed = copy.deepcopy(original)
            changed["training"]["policy_batch_size"] = 128
            write(path, changed)
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                gates.load_protocol(path)


class TeacherIdentityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.storage = Storage(self.root)
        self.descriptor = self.root / "bundle/agent.json"
        self.manifest_path = self.root / "bundle/runtime-manifest.json"
        self.weight = self.root / "bundle/source/weights/s42-inference.pth"
        self.weight.parent.mkdir(parents=True)
        self.weight.write_bytes(b"synthetic model bytes, never loaded")
        self.protocol = json.loads(PROTOCOL_PATH.read_text())
        self.protocol["teacher"].update(sha256=gates.file_sha256(self.weight), bytes=self.weight.stat().st_size)
        published = self.protocol["teacher"]
        self.identity = {"name": "Mortal-S42", "format": "pytorch-state-dicts", "file": published["weights"],
                         "sha256": published["sha256"], "bytes": published["bytes"], "is_oracle": False,
                         "observation_version": 4, "observation_shape": [1012, 34], "action_space": 46,
                         "resnet": {"conv_channels": 192, "num_blocks": 40}}
        self.identity_path = self.root / "bundle/source/model-identity.json"
        write(self.identity_path, self.identity)
        self.manifest = {"schema_version": 1, "kind": "mortal-runtime-v1", "checkout": "source",
                         "upstream": {"repo": published["repository"], "revision": published["revision"],
                                      "mortal_commit": "0cff2b52982be5b1163aa9a62fb01f03ce91e0d2"},
                         "files": {"source/" + published["weights"]: published["sha256"],
                                   "source/model-identity.json": gates.file_sha256(self.identity_path)}}
        write(self.manifest_path, self.manifest)
        self.provenance = {"factory": gates.TEACHER_FACTORY, "artifacts": {"manifest_path": {"path": str(self.manifest_path)}}}

    def validate(self):
        with patch.object(gates, "agent_provenance", return_value=self.provenance), patch.object(
                gates, "managed_agents", return_value=nullcontext()) as constructor:
            result = gates.validate_teacher(self.storage, self.descriptor, self.protocol)
            constructor.assert_called_once()
            return result

    def test_binds_actual_model_and_runs_constructor_verification_without_reset(self):
        provenance, binding = self.validate()
        self.assertEqual(provenance, self.provenance)
        self.assertEqual(binding["model_identity_sha256"], gates.file_sha256(self.identity_path))

    def test_rejects_substitute_factory_checkout_repository_or_oracle_identity(self):
        self.provenance["factory"] = "another:Agent"
        with self.assertRaisesRegex(ValueError, "exact Mortal"):
            self.validate()
        self.provenance["factory"] = gates.TEACHER_FACTORY
        for field, value in (("kind", "different"), ("checkout", "other"),
                             ("upstream", dict(self.manifest["upstream"], repo="other/S42"))):
            write(self.manifest_path, dict(self.manifest, **{field: value}))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "publication"):
                self.validate()
        write(self.manifest_path, self.manifest)
        write(self.identity_path, dict(self.identity, is_oracle=True))
        with self.assertRaisesRegex(ValueError, "model-identity"):
            self.validate()

    def test_weight_mutation_cannot_pass_with_stale_manifest(self):
        self.weight.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "content hash"):
            self.validate()


class LegalityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.storage = Storage(self.root)
        self.teacher = {"kind": "external", "name": "@教师.json"}
        self.engine, self.rules = {"binary_sha256": "engine"}, asdict(Rules())
        common = {"status": "passed", "teacher_provenance": self.teacher,
                  "teacher_provenance_after": self.teacher, "engine": self.engine, "rules": self.rules}
        self.native = dict(common, case_status={name: True for name in gates.NATIVE_CASES})
        write(self.root / "native.json", self.native)
        checks, games = [], []
        def check(name, relative):
            checks.append({"name": name, "status": "passed", "evidence_path": relative,
                           "evidence_sha256": gates.file_sha256(self.root / relative)})
        for name in sorted(gates.NATIVE_CASES):
            check(name, "native.json")
        for seat in range(4):
            receipt = {"validation": "engine-verified"}
            for suffix, field in (("mjai.jsonl.gz", "mjai_file_sha256"), ("trace.jsonl.gz", "trace_sha256")):
                relative = f"seat-{seat}-{suffix}"
                (self.root / relative).write_bytes(f"synthetic fixture {relative}".encode())
                receipt[field] = gates.file_sha256(self.root / relative)
                check(relative, relative)
            games.append({"seat": seat, "illegal_actions": 0, "validation": receipt})
        write(self.root / "games.json", dict(common, games=games))
        check("four-seat-native-half-games", "games.json")
        self.report = {"schema_version": 1, **common, "teacher_provenance_sha256": gates.content_sha256(self.teacher), "checks": checks}
        write(self.root / "legality.json", self.report)

    def verify(self):
        return gates.verify_legality(self.storage, "legality.json", self.teacher, self.engine, self.rules)

    def test_verifies_underlying_scenarios_and_four_seat_raw_hashes(self):
        result = self.verify()
        self.assertEqual(len(result["checks"]), 16)
        self.assertEqual(result["sha256"], gates.file_sha256(self.root / "legality.json"))

    def test_missing_scenario_or_tampered_underlying_evidence_fails(self):
        write(self.root / "legality.json", dict(self.report, checks=self.report["checks"][:-1]))
        with self.assertRaisesRegex(ValueError, "complete scenario"):
            self.verify()
        write(self.root / "legality.json", self.report)
        (self.root / "seat-0-trace.jsonl.gz").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "evidence hash"):
            self.verify()

    def test_passed_summary_cannot_override_failed_scenario(self):
        self.native["case_status"][next(iter(gates.NATIVE_CASES))] = False
        write(self.root / "native.json", self.native)
        for check in self.report["checks"]:
            if check["evidence_path"] == "native.json":
                check["evidence_sha256"] = gates.file_sha256(self.root / "native.json")
        write(self.root / "legality.json", self.report)
        with self.assertRaisesRegex(ValueError, "did not pass"):
            self.verify()


class QualificationTests(unittest.TestCase):
    def test_recomputes_gate_and_does_not_trust_a_passed_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identities = {name: {key: f"fixture-{key}" for key in gates.CONDITIONS} | {"seeds": [1, 2]}
                          for name in ("baseline", "teacher")}
            registration = {"identities": identities, "contract": {"registered": "contract"}, "legality": {"path": "legality.json"}}
            write(root / "registration.json", registration)
            write(root / "comparison-contract.json", registration["contract"])
            for name in identities:
                write(root / name / "manifest.json", identities[name])
            comparator = Mock()
            comparator.compare.return_value = {"games_per_model": 1000, "ci95": {"mean_rank": [-1., -.1]}}
            sealed = {"attempts": {"failed": 0, "unsealed": 0}}
            with patch.object(gates, "verify_campaign", return_value=sealed) as verify, patch.object(
                    gates, "registration_for", return_value=(registration, {}, comparator)):
                result = gates.qualification_result(root, registration, comparator)
                self.assertEqual(result["status"], "passed")
                self.assertEqual(verify.call_count, 2)
                comparator.compare.assert_called_with(root / "baseline", root / "teacher", contract=registration["contract"])
                write(root / "qualification.json", result)
                gates.require_qualification(Storage(root), root, PROTOCOL_PATH, root / "teacher.json")
                comparator.compare.return_value = {"games_per_model": 1000, "ci95": {"mean_rank": [-1., .01]}}
                with self.assertRaisesRegex(ValueError, "recomputed"):
                    gates.require_qualification(Storage(root), root, PROTOCOL_PATH, root / "teacher.json")
                comparator.compare.return_value = result["comparison"]
                write(root / "teacher/manifest.json", dict(identities["teacher"], seeds=[2, 3]))
                with self.assertRaisesRegex(ValueError, "registered fixtures"):
                    gates.qualification_result(root, registration, comparator)

    def test_failed_attempts_disqualify_even_with_improving_ci(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = {key: "same" for key in gates.CONDITIONS} | {"seeds": [1]}
            registration = {"identities": {"baseline": identity, "teacher": identity}, "contract": {}, "legality": {}}
            write(root / "registration.json", registration)
            for name in registration["identities"]:
                write(root / name / "manifest.json", identity)
            comparator = Mock()
            comparator.compare.return_value = {"games_per_model": 1000, "ci95": {"mean_rank": [-1., -.1]}}
            with patch.object(gates, "verify_campaign", return_value={"attempts": {"failed": 1, "unsealed": 0}}):
                self.assertEqual(gates.qualification_result(root, registration, comparator)["status"], "not-qualified")


class SourceAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        cls.storage = Storage(cls.root)
        cls.protocol = gates.load_protocol(PROTOCOL_PATH)
        cls.teacher = {"kind": "external", "name": "@teacher.json", "factory": gates.TEACHER_FACTORY}
        cls.collection_identity = {"code": {"package": "agari-kit", "source_sha256": "a" * 64, "version": "0.1.0"},
                                   "engine": {"version": "0.4.10", "binary_sha256": "e" * 64}, "rules": asdict(Rules())}
        cls.manifests = {name: cls.make_source(name) for name in ("teacher", "weak")}

    @classmethod
    def make_source(cls, name):
        parent = cls.root / "datasets/manifests" / ("phase2-teacher" if name == "teacher" else "selfplay-v1")
        start = cls.protocol["seeds"]["teacher_1000"]["start"] if name == "teacher" else 0
        players = [cls.teacher] * 4 if name == "teacher" else [{"kind": "builtin", "name": "efficiency"}] * 4
        code = {"package": "agari-kit", "source_sha256": "a" * 64, "version": "0.1.0"}
        config = {"source": "agarilab-selfplay", "games": 1000, "seed_start": start,
                  "rules": asdict(Rules()), "agent_provenance": players,
                  "agents": [player["name"] for player in players], "code": code}
        if name == "teacher":
            config["engine"] = cls.collection_identity["engine"]
        write(parent / "acquisition.json", {"status": "complete", "config": config})
        result = {"partitions": {"train": [], "validation": [], "test": []}}
        for index in range(1000):
            partition = "train" if index < 900 else "validation" if index < 950 else "test"
            header = {"type": "agari_trace", "schema_version": 1, "seed": start + index,
                      **{key: config[key] for key in ("rules", "agents", "agent_provenance", "code")}}
            trace_path = parent / f"game-{index:06d}.trace.jsonl"
            if partition != "test":
                write(trace_path, header)
            receipt = {"seed": start + index, "validation": "engine-verified", "method": "seeded-action-trace-v1",
                       "rules": config["rules"], "sha256": gates.content_sha256([name, index]),
                       "engine_version": "0.4.10", "producer_code": code}
            if name == "teacher":
                receipt["agent_provenance_sha256"] = [gates.content_sha256(player) for player in players]
                receipt["engine_binary_sha256"] = cls.collection_identity["engine"]["binary_sha256"]
            receipt_path = parent / f"game-{index:06d}.verification.json"
            write(receipt_path, receipt)
            result["partitions"][partition].append({"game_id": f"{name}-{index}", "sha256": receipt["sha256"],
                "trace_path": trace_path.relative_to(cls.root).as_posix(),
                "validation_receipt": receipt_path.relative_to(cls.root).as_posix(),
                "validation_receipt_sha256": gates.file_sha256(receipt_path)})
        return result

    def audit(self, name, manifest=None, protocol=None):
        return gates.audit_source(self.storage, manifest or self.manifests[name], name=name,
                                  protocol=protocol or self.protocol, teacher_provenance=self.teacher,
                                  collection_identity=self.collection_identity)

    @contextmanager
    def alter_receipt(self, partition, index, change):
        manifest = copy.deepcopy(self.manifests["teacher"])
        row = manifest["partitions"][partition][index]
        path = self.storage.path(row["validation_receipt"])
        original = path.read_bytes()
        payload = json.loads(original)
        change(payload)
        write(path, payload)
        row["validation_receipt_sha256"] = gates.file_sha256(path)
        try:
            yield manifest
        finally:
            path.write_bytes(original)

    def test_all_teacher_seats_and_seeds_bound_without_opening_test_traces(self):
        # Test trace paths deliberately do not exist: auditing them must not read content.
        result = self.audit("teacher")
        self.assertEqual(result["games"], 1000)
        self.assertEqual(result["legacy_test_games_without_per_game_policy_binding"], 0)
        self.assertFalse(result["test_partition_content_opened"])
        old = self.audit("weak")
        self.assertEqual(old["legacy_test_games_without_per_game_policy_binding"], 50)

    def test_test_receipt_cannot_hide_mixed_players_missing_binding_or_eval_seed(self):
        changes = [lambda receipt: receipt.pop("agent_provenance_sha256"),
                   lambda receipt: receipt["agent_provenance_sha256"].__setitem__(2, "f" * 64),
                   lambda receipt: receipt.update(engine_binary_sha256="f" * 64),
                   lambda receipt: receipt.update(seed=self.protocol["seeds"]["development"]["start"]),
                   lambda receipt: receipt.update(seed=self.protocol["seeds"]["teacher_1000"]["start"])]
        for change in changes:
            with self.subTest(change=change), self.alter_receipt("test", 0, change) as manifest:
                with self.assertRaises(ValueError):
                    self.audit("teacher", manifest)

    def test_actual_training_header_must_match_four_seat_receipt(self):
        row = self.manifests["teacher"]["partitions"]["train"][0]
        path = self.storage.path(row["trace_path"])
        original = path.read_bytes()
        header = json.loads(original)
        header["agent_provenance"][0] = {"kind": "builtin", "name": "random"}
        try:
            write(path, header)
            with self.assertRaisesRegex(ValueError, "training trace header"):
                self.audit("teacher")
        finally:
            path.write_bytes(original)

    def test_wrong_source_size_and_cross_source_content_are_rejected(self):
        short = copy.deepcopy(self.manifests["teacher"])
        short["partitions"]["test"].pop()
        with self.assertRaisesRegex(ValueError, "exactly"):
            self.audit("teacher", short)
        overlapping = copy.deepcopy(self.manifests["teacher"])
        overlapping["partitions"]["train"][0]["game_id"] = self.manifests["weak"]["partitions"]["test"][0]["game_id"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            prepare.reject_overlapping_sources(self.manifests["weak"], overlapping)


if __name__ == "__main__":
    unittest.main()
