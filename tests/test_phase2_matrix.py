"""Synthetic matrix gates only: no external artifacts, teachers or games."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location("matrix_gate_test", Path(__file__).parents[1] / "scripts/phase2_evaluation_matrix.py")
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)


def comparisons():
    rows = {}
    for pool in matrix.POOLS:
        for index, key in enumerate(matrix.MODELS):
            metrics = {"mean_rank": 2.0 + index / 20, "rank_4_rate": .2, "mean_point_delta": 0.0}
            rows[f"{pool}/{key}"] = {"games_per_model": 1000, "candidate": {"metrics": metrics},
                                     "ci95": {"mean_rank": [-.2, .1]}}
    return rows


class SelectionTests(unittest.TestCase):
    def test_all_nine_models_both_pools_required_and_weak_pool_cannot_pick_winner(self):
        rows = comparisons()
        rows["weak/C-2028"]["candidate"]["metrics"]["mean_rank"] = 1.0
        result = matrix.select_development(rows)
        self.assertEqual(result["winner"], "A-2026")
        self.assertEqual(len(result["primary_order"]), 9)
        rows.pop("primary/C-2028")
        with self.assertRaisesRegex(ValueError, "All nine"):
            matrix.select_development(rows)

    def test_scale_gate_requires_teacher_arm_improvement_in_primary_pool(self):
        rows = comparisons()
        rows["primary/A-2026"]["ci95"]["mean_rank"] = [-.3, -.1]
        rows["weak/B-2027"]["ci95"]["mean_rank"] = [-.3, -.1]
        self.assertFalse(matrix.select_development(rows)["scale_10000_eligible"])
        rows["primary/B-2027"]["ci95"]["mean_rank"] = [-.3, -.1]
        self.assertEqual(matrix.select_development(rows)["scale_10000_candidates"], ["B-2027"])

    def test_registered_tie_order_and_no_arbitrary_exact_tie(self):
        rows = comparisons()
        a = rows["primary/A-2026"]["candidate"]["metrics"]
        b = rows["primary/B-2026"]["candidate"]["metrics"]
        b.update(a, rank_4_rate=.1)
        self.assertEqual(matrix.select_development(rows)["winner"], "B-2026")
        a.update(b, mean_point_delta=100.)
        self.assertEqual(matrix.select_development(rows)["winner"], "A-2026")
        b.update(a)
        with self.assertRaisesRegex(ValueError, "unresolved"):
            matrix.select_development(rows)

    def test_incomplete_samples_cannot_be_selected(self):
        rows = comparisons()
        rows["weak/C-2028"]["games_per_model"] = 996
        with self.assertRaisesRegex(ValueError, "1000 games"):
            matrix.select_development(rows)

    def test_three_arms_cannot_use_different_q_initialization_for_one_seed(self):
        rt = SimpleNamespace(protocol={"training": {"seeds": [2026, 2027, 2028]}})
        jobs = {key: {"arm": key[0], "seed": int(key[2:])} for key in matrix.MODELS}
        def verified(rt, prepared, key, job):
            return {"policy_initialization": {"seed": job["seed"], "q_state_sha256": str(job["seed"])}}
        with patch.object(matrix, "verify_completed_job", side_effect=verified):
            self.assertEqual(len(matrix.training_models(rt, {}, jobs)), 9)
        def changed(rt, prepared, key, job):
            result = verified(rt, prepared, key, job)
            if key == "B-2026":
                result["policy_initialization"]["q_state_sha256"] = "different"
            return result
        with patch.object(matrix, "verify_completed_job", side_effect=changed), self.assertRaisesRegex(ValueError, "same Q initialization"):
            matrix.training_models(rt, {}, jobs)


class ScaleAuthorizationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.rt = SimpleNamespace(path=lambda relative: self.root / relative)
        def bound(entry):
            path = self.root / entry["path"]
            matrix.require(matrix.digest(path) == entry["sha256"], "bound report changed")
            return path
        self.rt.bound = bound
        self.small = self.root / "runs/phase2-small"
        self.fixtures = {f"{pool}/{key}": {} for pool in matrix.POOLS for key in ("incumbent", *matrix.MODELS)}
        self.previous = {"scale": "small", "scale_from": None, "implementation": {"frozen": "code"}, "fixtures": self.fixtures}
        matrix.publish(self.small / "registration.json", self.previous)
        rows = comparisons()
        rows["primary/B-2026"]["ci95"]["mean_rank"] = [-.2, -.1]
        self.report = {"registration_sha256": matrix.digest(self.small / "registration.json"), "comparisons": rows,
                       "evidence": {key: {"hash": "sealed"} for key in self.fixtures}}
        matrix.publish(self.small / "development-report.json", self.report)
        self.large = {"directory": "runs/phase2-large", "scale": "large", "implementation": {"frozen": "code"}, "scale_from": {
            "matrix": "runs/phase2-small", "registration_sha256": self.report["registration_sha256"], "development_report": {
                "path": "runs/phase2-small/development-report.json", "sha256": matrix.digest(self.small / "development-report.json")}}}

    def test_large_reentry_rechecks_small_campaign_evidence(self):
        with patch.object(matrix, "verify_fixture", return_value={"hash": "sealed"}) as verify:
            matrix.verify_scale_authorization(self.rt, self.large)
            self.assertEqual(verify.call_count, 20)
        with patch.object(matrix, "verify_fixture", return_value={"hash": "changed"}), self.assertRaisesRegex(ValueError, "campaign changed"):
            matrix.verify_scale_authorization(self.rt, self.large)

    def test_changed_registration_or_lost_pilot_eligibility_blocks_large_reentry(self):
        registration = self.small / "registration.json"
        original = registration.read_bytes()
        registration.write_text("{}")
        with self.assertRaisesRegex(ValueError, "registration changed"):
            matrix.verify_scale_authorization(self.rt, self.large)
        registration.write_bytes(original)
        self.report["comparisons"]["primary/B-2026"]["ci95"]["mean_rank"] = [-.2, .1]
        path = self.small / "development-report.json"
        path.write_text(json.dumps(self.report))
        self.large["scale_from"]["development_report"]["sha256"] = matrix.digest(path)
        with self.assertRaisesRegex(ValueError, "did not pass"):
            matrix.verify_scale_authorization(self.rt, self.large)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.job = {"arm": "B", "seed": 2026, "method": "bc", "run": "teacher-b-2026"}
        self.config = {"seed": 2026, "stages": ["bc"], "validation": {"max_samples": None, "interval_updates": 3},
                       "policy_filter": {"agent_identity_sha256": ["teacher"]}, "model": {"shape": "small"}}
        plan = {"stages": {"bc": {"target_updates": 10}}}
        plan["sha256"] = matrix.content_digest(plan)
        self.selected = {"metric": "validation_loss", "value": 1.0, "update": 6,
                         "artifact": "best-bc-u000000006.pt", "sha256": "weights"}
        self.checkpoint = {"training_plan": plan, "config": self.config, "progress": {"bc": 10},
                           "run_name": self.job["run"], "dataset_sha256": "split", "implementation_sha256": "impl",
                           "runtime_provenance": {"runtime": "frozen"}, "best_validation": {"bc": self.selected}}
        self.result = {"status": "complete", "stop_reason": None, "run": self.job["run"], "progress": {"bc": 10},
                       "dataset_sha256": "split", "implementation_sha256": "impl", "training_plan_sha256": plan["sha256"],
                       "best_validation": {"bc": self.selected}}
        self.payload = {"dataset_sha256": "split", "implementation_sha256": "impl", "runtime_provenance": {"runtime": "frozen"},
                        "config": self.config, "policy_filter": self.config["policy_filter"], "kind": "bc", "model_config": self.config["model"],
                        "training_plan_sha256": plan["sha256"], "completed_updates": 6,
                        "validation_selection": {key: self.selected[key] for key in ("metric", "value", "update")}}
        self.metrics = [{"stage": "bc", "update": step} for step in range(1, 11)]
        for step, loss in ((3, 2.), (6, 1.), (9, 1.2), (10, 1.5)):
            self.metrics[step-1].update(validation_loss=loss, validation_scope="full", validation_sample_limit=None, validation_samples=100)

    def verify(self):
        return matrix.training_contract(self.job, self.config, self.result, self.checkpoint, self.payload, self.metrics,
                                         dataset_sha256="split", implementation="impl", runtime={"runtime": "frozen"})

    def test_earlier_validation_best_is_valid_only_after_whole_budget_finished(self):
        self.assertEqual(self.verify()["update"], 6)
        self.result["status"] = "paused"
        with self.assertRaisesRegex(ValueError, "not finished"):
            self.verify()

    def test_wrong_split_runtime_or_training_budget_fails(self):
        for target, field, value in ((self.payload, "dataset_sha256", "other"),
                                      (self.payload, "runtime_provenance", {"runtime": "changed"}),
                                      (self.checkpoint, "progress", {"bc": 9})):
            original = target[field]
            try:
                target[field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    self.verify()
            finally:
                target[field] = original

    def test_claimed_best_cannot_ignore_lower_validation_loss_or_bounded_validation(self):
        self.metrics[-1]["validation_loss"] = .5
        with self.assertRaisesRegex(ValueError, "minimum"):
            self.verify()
        self.metrics[-1]["validation_loss"] = 1.5
        self.metrics[2]["validation_scope"] = "bounded"
        with self.assertRaisesRegex(ValueError, "Full finite"):
            self.verify()

    def test_duplicate_validation_after_resume_is_not_silently_accepted(self):
        self.metrics.append(copy.deepcopy(self.metrics[-1]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.verify()

    def test_missing_scheduled_or_final_validation_is_rejected(self):
        for index in (5, 9):
            original = self.metrics[index]
            try:
                self.metrics[index] = {"stage": "bc", "update": index + 1}
                with self.subTest(update=index+1), self.assertRaisesRegex(ValueError, "validation checkpoint"):
                    self.verify()
            finally:
                self.metrics[index] = original


class RuntimeAndBudgetTests(unittest.TestCase):
    def test_loaded_mutable_module_is_rejected_instead_of_silently_reused(self):
        root = Path("/frozen")
        with patch.dict(sys.modules, {"agari_kit": SimpleNamespace(__file__="/mutable/agari-kit/src/agari_kit/__init__.py")}, clear=True):
            with self.assertRaisesRegex(ValueError, "outside"):
                matrix.check_module_paths(root)
        with patch.dict(sys.modules, {"agari_kit": SimpleNamespace(__file__="/frozen/agari-kit/src/agari_kit/__init__.py")}, clear=True):
            matrix.check_module_paths(root)

    def test_preflight_exhaustion_never_starts_campaign(self):
        fixture = {"lineup": ["a"] * 4, "seeds": [1, 2]}
        with patch.object(matrix, "identity", return_value=fixture), patch.object(matrix.time, "monotonic", return_value=100.):
            report = matrix.execute(None, fixture, Path("/unused"), resume=False, max_wall_seconds=10, invocation_started=0)
        self.assertEqual(report["games_started"], 0)

    def test_remaining_budget_excludes_preflight_time(self):
        fixture = {"lineup": ["a"] * 4, "seeds": [1, 2]}
        with patch.object(matrix, "identity", return_value=fixture), patch.object(matrix.time, "monotonic", return_value=100.), patch(
                "agari_kit.evaluation_campaign.evaluate_campaign", return_value={"status": "paused"}) as run:
            matrix.execute(None, fixture, Path("/unused"), resume=True, max_wall_seconds=80, invocation_started=50)
        self.assertEqual(run.call_args.kwargs["max_wall_seconds"], 30)


class CompletedArtifactTests(unittest.TestCase):
    """Exercise actual local tensor/JSON bindings without training a model."""
    def setUp(self):
        import torch
        from agari_research.features import ENCODING_VERSION, UPSTREAM_COMMIT
        from agari_research.trace_samples import SAMPLE_SEMANTICS_VERSION
        from agari_research.planning import build_training_plan, coverage
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.rt = SimpleNamespace(path=lambda relative: self.root / relative, runtime={"runtime": "fixed"}, implementation="implementation")
        self.rt.artifact = lambda path: {"path": path.relative_to(self.root).as_posix(), "sha256": matrix.digest(path)}
        self.run_name = "phase2-test-b-2026"
        self.run = self.root / "runs" / self.run_name
        self.models = self.root / "models" / self.run_name
        self.checkpoint_path = self.root / "checkpoints" / self.run_name / "latest.pt"
        self.config = {"schema_version": 1, "seed": 2026, "device": "cpu", "torch_threads": 1, "stages": ["bc"],
                       "validation": {"max_samples": None}, "policy_filter": {"agent_identity_sha256": ["a" * 64]},
                       "model": {"synthetic": True}, "bc": {"steps": 1, "batch_size": 64}}
        self.config_path = self.root / "config.json"
        matrix.publish(self.config_path, self.config)
        receipt = self.root / "receipt.json"
        matrix.publish(receipt, {"engine_version": "0.4.10", "engine_binary_sha256": "e" * 64})
        row = {"game_id": "game", "validation_receipt": "receipt.json", "validation_receipt_sha256": matrix.digest(receipt)}
        split = {"partitions": {"train": [row], "validation": [], "test": []}}
        matrix.publish(self.root / "split.json", split)
        self.job = {"arm": "B", "seed": 2026, "method": "bc", "run": self.run_name, "config": "config.json",
                    "config_sha256": matrix.digest(self.config_path), "split": "split.json"}
        self.prepared = {"datasets": {name: {"training_decisions": 1, "sha256": "d" * 64,
            "policy_filter": self.config["policy_filter"], "split": "split.json", "counts_sha256": "unused"} for name in ("weak", "teacher")}}
        dataset = SimpleNamespace(sha256="d" * 64, policy_filter=self.config["policy_filter"],
            producer_engines=[{"version": "0.4.10", "binary_sha256": "e" * 64}])
        plan = build_training_plan(self.config, dataset)
        initial = {"seed": 2026, "q_state_sha256": "q" * 64}
        common = {"schema_version": 1, "config": self.config, "dataset_sha256": "d" * 64,
                  "implementation_sha256": self.rt.implementation, "runtime_provenance": self.rt.runtime,
                  "encoding_version": ENCODING_VERSION, "upstream_commit": UPSTREAM_COMMIT,
                  "sample_semantics_version": SAMPLE_SEMANTICS_VERSION, "engine_version": "0.4.10",
                  "policy_filter": self.config["policy_filter"], "policy_initialization": initial}
        selection = {"metric": "validation_loss", "value": 2., "update": 1}
        self.payload = dict(common, kind="bc", model_config=self.config["model"], completed_updates=1,
                            training_plan_sha256=plan["sha256"], state_dict={"weight": torch.ones(2)}, validation_selection=selection)
        self.best = self.models / "best-bc-u000000001.pt"
        self.export(self.best, self.payload)
        self.alias = self.models / "best-bc.pt"
        self.alias.write_bytes(self.best.read_bytes())
        self.export_metadata(self.alias, self.payload)
        self.final = self.models / "bc.pt"
        self.final_payload = {key: value for key, value in self.payload.items() if key != "validation_selection"}
        self.export(self.final, self.final_payload)
        chosen = dict(selection, artifact=self.best.name, sha256=matrix.digest(self.best))
        self.checkpoint = dict(common, training_plan=plan, progress={"bc": 1}, samples_seen={"bc": 64}, run_name=self.run_name,
                               best_validation={"bc": chosen}, models={"bc": self.payload["state_dict"]})
        self.checkpoint_path.parent.mkdir(parents=True)
        torch.save(self.checkpoint, self.checkpoint_path)
        finished = {"status": "complete", "stop_reason": None, "run": self.run_name, "progress": {"bc": 1},
                    "checkpoint": str(self.checkpoint_path), "models": str(self.models), "dataset_sha256": "d" * 64,
                    "implementation_sha256": self.rt.implementation, "training_plan_sha256": plan["sha256"],
                    "best_validation": {"bc": chosen}, "training_coverage": {"bc": coverage(plan, "bc", 64, 2026)}}
        matrix.publish(self.run / "result.json", finished)
        matrix.publish(self.run / "config.json", self.config)
        matrix.publish(self.run / "dataset.json", split)
        matrix.publish(self.run / "execution-plan.json", {"plan": plan, "runtime_provenance": self.rt.runtime,
                                                         "implementation_sha256": self.rt.implementation})
        matrix.publish(self.run / "metrics.jsonl", {"stage": "bc", "update": 1, "validation_loss": 2.,
            "validation_scope": "full", "validation_sample_limit": None, "validation_samples": 1})

    def export_metadata(self, path, payload):
        sha = matrix.digest(path)
        matrix.publish(path.with_suffix(".json"), {**{key: value for key, value in payload.items() if key != "state_dict"},
                                                    "weights_sha256": sha, "filename": path.name})
        matrix.publish(path.with_name(path.stem + "-agent.json"), {"factory": matrix.MODEL_FACTORY,
            "kwargs": {"device": "cpu", "torch_threads": 1}, "artifacts": {"weights_path": {"path": path.name, "sha256": sha}}})

    def export(self, path, payload):
        import torch
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        self.export_metadata(path, payload)

    def verify(self):
        return matrix.verify_completed_job(self.rt, self.prepared, "B-2026", self.job)

    def test_real_tensor_payload_alias_and_all_metadata_are_bound(self):
        result = self.verify()
        self.assertEqual(result["best"]["sha256"], matrix.digest(self.best))
        evidence = {row["path"] for row in result["training_evidence"]}
        self.assertIn(self.final.relative_to(self.root).as_posix(), evidence)
        self.assertIn(self.best.with_name(self.best.stem + "-agent.json").relative_to(self.root).as_posix(), evidence)

    def test_final_tensor_or_best_alias_cannot_be_substituted(self):
        import torch
        original = self.final.read_bytes()
        changed = dict(self.final_payload, state_dict={"weight": torch.zeros(2)})
        torch.save(changed, self.final)
        with self.assertRaisesRegex(ValueError, "Final tensors"):
            self.verify()
        self.final.write_bytes(original)
        self.alias.write_bytes(b"wrong alias")
        with self.assertRaisesRegex(ValueError, "descriptor/alias"):
            self.verify()

    def test_best_sidecar_cannot_claim_a_different_selected_update(self):
        path = self.best.with_suffix(".json")
        value = matrix.read(path)
        value["completed_updates"] = 99
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "Export metadata"):
            self.verify()

    def test_training_log_replayed_update_is_detected_with_real_bundle(self):
        path = self.run / "metrics.jsonl"
        path.write_bytes(path.read_bytes() * 2)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.verify()


class FinalGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.protocol = self.root / "protocol.json"
        self.protocol.write_text("{}")
        self.rt = SimpleNamespace(path=lambda relative: self.root / relative, protocol_path=self.protocol,
                                  protocol={"seeds": {"final_promotion": {"start": 10000, "count": 2500}}},
                                  contract=SimpleNamespace(seed_range=lambda config: set(range(config["start"], config["start"] + config["count"]))))
        self.registration = {"directory": "runs/phase2-test", "scale": "small", "models": {"A-2026": {"frozen": "weights"}},
                             "fixtures": {"primary/incumbent": {"lineup": ["old", "teacher", "old", "efficiency"], "seeds": [100, 101]},
                                          "primary/A-2026": {"lineup": ["new", "teacher", "old", "efficiency"], "seeds": [100, 101]}}}
        matrix.publish(self.root / self.registration["directory"] / "registration.json", self.registration)
        self.report = {"winner": "A-2026", "scale_10000_eligible": False}

    def freeze(self):
        with patch.object(matrix, "development_report", return_value=self.report), patch.object(
                matrix, "identity", side_effect=lambda rt, lineup, seeds: {"lineup": lineup, "seeds": seeds}), patch.object(
                matrix, "explicit_contract", return_value={"contract": "bound"}):
            return matrix.freeze_final(self.rt, self.registration)

    def test_eligible_small_pilot_cannot_consume_final_seeds_before_large_matrix(self):
        self.report["scale_10000_eligible"] = True
        with self.assertRaisesRegex(ValueError, "large matrix"):
            self.freeze()
        self.assertFalse(matrix.final_reservation(self.rt).exists())

    def test_final_freeze_is_immutable_and_reserves_seeds_across_matrices(self):
        self.assertEqual(self.freeze()["games_per_model"], 10000)
        self.assertEqual(self.freeze()["status"], "frozen-not-played")
        self.registration["directory"] = "runs/phase2-another"
        matrix.publish(self.root / self.registration["directory"] / "registration.json", self.registration)
        with self.assertRaisesRegex(ValueError, "Immutable artifact differs"):
            self.freeze()

    def test_existing_final_output_blocks_late_selection(self):
        (self.root / self.registration["directory"] / "final").mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.freeze()

    def test_final_requires_exact_registered_size_and_negative_ci_upper_bound(self):
        self.rt.comparator = Mock()
        frozen = {"fixtures": {"incumbent": {}, "candidate": {}}, "contract": {}}
        (self.root / self.registration["directory"] / "final-freeze.json").write_text("{}")
        with patch.object(matrix, "frozen_final", return_value=frozen), patch.object(matrix, "verify_fixture", return_value={}):
            for games, upper, expected in ((1000, -.1, "not-promoted"), (10000, 0., "not-promoted"), (10000, -.001, "promoted")):
                self.rt.comparator.compare.return_value = {"games_per_model": games, "ci95": {"mean_rank": [-.2, upper]}}
                self.assertEqual(matrix.final_report(self.rt, self.registration)["status"], expected)


if __name__ == "__main__":
    unittest.main()
