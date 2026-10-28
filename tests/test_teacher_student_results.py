"""Synthetic report artifacts only; no dataset actions, models or live scores."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).parents[1] / "scripts/report_teacher_student_results.py"
spec = importlib.util.spec_from_file_location("student_results_test", SCRIPT)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def counts(n, correct):
    return {"samples": n, "correct": correct, "accuracy": correct / n if n else None}


def validation(method="bc"):
    return {"stage": method, "update": 2, "loss": 0.4,
        "validation_loss": 0.5, "validation_scope": "full", "validation_sample_limit": None,
        "validation_samples": 3, "validation_accuracy": 2 / 3, "validation_cross_entropy": 0.5,
        "validation_policy_diagnostics": {"status": "complete", "metric": "encoded_action_accuracy",
            "scope": "full", "sample_limit": None, "choice": counts(3, 2), "forced": counts(2, 2),
            "by_action_type": {"DISCARD": {"choice": counts(3, 2), "forced": counts(0, 0)},
                               "PASS": {"choice": counts(0, 0), "forced": counts(2, 2)}}}}


class ResultsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.prep, self.matrix_path, self.output = "runs/phase2-prepared", "runs/phase2-development", "runs/phase2-report"
        self.manifest = {"status": "prepared-not-trained", "test_partition_content_opened": False,
                         "teacher_binding": {"descriptor": report.TEACHER},
                         "qualification": {"run": report.QUALIFICATION}, "datasets": {}, "jobs": []}
        self.write(report.QUALIFICATION + "/registration.json", {"legality": {"path": report.LEGALITY}})
        self.audits = {}
        for source in ("weak", "teacher"):
            parent = f"datasets/manifests/{source}"
            receipt = parent + "/game.verification.json"
            split = {"partitions": {"train": [{"validation_receipt": receipt}], "validation": [], "test": []}}
            split["partitions"]["train"][0]["trace_path"] = "datasets/raw/DO-NOT-OPEN.trace.jsonl.gz"
            self.write(parent + "/split.json", split)
            self.manifest["datasets"][source] = {"split": parent + "/split.json", "training_decisions": 64}
            self.audits[source] = {"acquisition": parent + "/acquisition.json", "receipt_inventory": [{"receipt": receipt}],
                                  "test_partition_content_opened": False,
                                  "legacy_test_games_without_per_game_policy_binding": 1 if source == "weak" else 0}
            self.write(f"{self.prep}/{source}-source-audit.json", self.audits[source])
            self.write(f"{self.prep}/{source}-training-counts.json", [{"cql_samples": 64, "grp_samples": 32}])
        self.matrix = {"directory": self.matrix_path, "preparation": self.prep, "scale": "small", "scale_from": None,
                       "models": {}, "fixtures": dict.fromkeys(report.FIXTURES, {})}
        self.jobs = {}
        for key in report.MODELS:
            arm, seed = key.split("-")
            method = "cql" if arm == "C" else "bc"
            stages = {"grp": {"target_updates": 1}, "cql": {"target_updates": 2}} if arm == "C" else {"bc": {"target_updates": 2}}
            job = {"arm": arm, "seed": int(seed), "method": method, "run": "phase2-" + key,
                   "split": self.manifest["datasets"]["weak" if arm == "A" else "teacher"]["split"],
                   "config": f"{self.prep}/configs/{key}.json"}
            job["config_sha256"] = self.write(job["config"], {"device": "cpu"})
            self.jobs[key] = job
            self.manifest["jobs"].append(job)
            self.write(f"runs/{job['run']}/result.json", {"status": "complete", "dataset_sha256": "a" * 64,
                "progress": {stage: value["target_updates"] for stage, value in stages.items()}, "training_coverage": {"fixture": True}})
            self.write(f"runs/{job['run']}/execution-plan.json", {"plan": {"stages": stages, "sha256": "b" * 64}})
            self.raw(f"runs/{job['run']}/metrics.jsonl", (json.dumps(validation(method), indent=None) + "\n").encode())
            weight = self.raw(f"models/{job['run']}/best-{method}-u000000002.pt", b"synthetic-no-tensors")
            self.raw(f"{self.matrix_path}/models/{key}/weights.pt", b"synthetic-no-tensors")
            self.write(f"models/{job['run']}/best-{method}-u000000002-agent.json", {"fixture": True})
            descriptor = self.write(f"{self.matrix_path}/models/{key}/agent.json", {"fixture": True})
            self.matrix["models"][key] = {"best": {"metric": "validation_loss", "value": 0.5, "update": 2,
                "artifact": f"best-{method}-u000000002.pt", "sha256": weight},
                "weights": {"path": f"models/{job['run']}/best-{method}-u000000002.pt", "sha256": weight},
                "frozen_weights": {"path": f"{self.matrix_path}/models/{key}/weights.pt", "sha256": weight},
                "descriptor": {"path": f"{self.matrix_path}/models/{key}/agent.json", "sha256": descriptor},
                "policy_initialization": {"seed": int(seed), "q_state_sha256": "c" * 64},
                "training_evidence": [{"path": f"runs/{job['run']}/metrics.jsonl", "sha256": report.digest(self.root / f"runs/{job['run']}/metrics.jsonl")}]}
        self.development = {"status": "complete", "evidence": {}, "comparisons": {}, "winner": "B-2026",
            "primary_order": list(report.MODELS), "scale_10000_eligible": True,
            "scale_10000_candidates": ["B-2026"], "scope": "original scope", "multiplicity": "original caution"}
        for fixture in sorted(report.FIXTURES):
            root = f"{self.matrix_path}/development/{fixture}"
            campaign = {"status": "complete", "games": 1000, "seed_groups": 250,
                "candidate": {"mean_rank": 2.4, "rank_rates": [.25] * 4, "mean_point_delta": 100},
                "ci95": {"mean_rank": [2.3, 2.5]}, "candidate_diagnostics": {"hands": 12345, "wins": 3000, "win_rate": 3000 / 12345},
                "candidate_decisions": 87654, "candidate_latency_ms_mean": 1.2, "candidate_latency_ms_p95": 2.4,
                "candidate_latency_ms_max": 123, "latency_scope": "original scope",
                "decisions": 999999, "illegal_actions": 0, "attempts": {"total": 250, "failed": 0, "unsealed": 0},
                "elapsed_seconds": 200.5}
            hashes = {"report.json": self.write(root + "/report.json", campaign),
                      "candidate-latencies.json": self.write(root + "/candidate-latencies.json", [1.2]),
                      "manifest.json": self.write(root + "/manifest.json", {}),
                      "games.jsonl": self.raw(root + "/games.jsonl", b"{}\n")}
            self.development["evidence"][fixture] = {"plan.json": self.write(root + "/plan.json", {}),
                "completion.json": self.write(root + "/completion.json", {"sha256": hashes}),
                "manifest.json": hashes["manifest.json"], "games.jsonl": hashes["games.jsonl"]}
            if not fixture.endswith("/incumbent"):
                self.development["comparisons"][fixture] = {"difference": {"mean_rank": -.1}, "ci95": {"mean_rank": [-.2, -.01]}}
        self.save()
        self.context = SimpleNamespace(rt=SimpleNamespace(root=Path("/frozen")), initial={"fixture": True}, check=Mock())
        self.context.matrix = SimpleNamespace(
            load_matrix=Mock(side_effect=lambda *_: copy.deepcopy(self.matrix)),
            preparation=Mock(side_effect=lambda *_: (self.root / self.prep, copy.deepcopy(self.manifest), copy.deepcopy(self.jobs), self.audits, [])),
            development_report=Mock(side_effect=lambda *_: copy.deepcopy(self.development)))
        self.context.runner = SimpleNamespace(execution_binding=Mock(return_value={"fixture": True}),
                                             stdout_result=lambda path: json.loads(path.read_bytes()))

    def raw(self, path, data):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return report.sha(data)

    def write(self, path, value):
        return self.raw(path, (json.dumps(value, indent=2) + "\n").encode())

    def save(self):
        self.write(self.prep + "/manifest.json", self.manifest)
        self.write(self.matrix_path + "/registration.json", self.matrix)
        self.write(self.matrix_path + "/development-report.json", self.development)

    def build(self):
        return report.build(report.Evidence(self.root), self.context, self.matrix_path, self.output)

    def test_complete_report_keeps_nine_raw_rows_twenty_campaigns_and_original_selection(self):
        result = self.build()
        self.assertEqual(result["status"], "complete-development-report")
        summary = json.loads((self.root / self.output / "summary.json").read_bytes())
        self.assertEqual(len(summary["training"]), 9)
        self.assertEqual(len(summary["development"]), 20)
        self.assertEqual(summary["selection"]["winner"], self.development["winner"])
        self.assertEqual(self.context.matrix.load_matrix.call_count, 2)
        self.assertEqual(self.context.matrix.development_report.call_count, 2)
        for row in summary["training"]:
            self.assertEqual((self.root / self.output / row["validation_original_line"]["path"]).read_bytes(),
                             (self.root / f"runs/{row['run']}/metrics.jsonl").read_bytes())
            self.assertIsNone(row["execution"]["costs"]["child_wall_seconds"]["observed_sum"])
        for row in summary["development"]:
            self.assertIsNone(row["cost"]["full_command_wall_seconds"])
            if row["job"] == "incumbent":
                self.assertIsNone(row["paired_ci95"])
            else:
                self.assertEqual(row["paired_ci95"]["mean_rank"], [-.2, -.01])
                self.assertEqual(row["single_campaign_ci95"]["mean_rank"], [2.3, 2.5])

    def test_incomplete_training_or_campaign_stops_before_helpers(self):
        for target in ("training", "campaign"):
            with self.subTest(target=target):
                if target == "training":
                    self.write("runs/phase2-A-2026/result.json", {"status": "paused"})
                else:
                    self.write("runs/phase2-A-2026/result.json", {"status": "complete"})
                    (self.root / self.matrix_path / "development/weak/C-2028/completion.json").unlink()
                with self.assertRaisesRegex(ValueError, "must be"):
                    self.build()
                self.context.matrix.load_matrix.assert_not_called()
                self.assertFalse((self.root / self.output).exists())

    def test_main_refuses_partial_results_before_loading_frozen_context(self):
        self.write("runs/phase2-A-2026/result.json", {"status": "paused"})
        with patch.dict(os.environ, {"AGARILAB_STORAGE_ROOT": str(self.root)}), patch.object(report, "Context") as context:
            with self.assertRaisesRegex(ValueError, "training jobs must be complete"):
                report.main(["--runtime-root", "/unused", "--tools-root", "/unused", "--matrix", self.matrix_path,
                             "--output", self.output])
            context.assert_not_called()

    def test_no_dataset_action_or_final_path_is_opened(self):
        original = Path.open
        def guarded(path, *args, **kwargs):
            self.assertNotIn("DO-NOT-OPEN", str(path))
            self.assertNotIn("final", path.parts)
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guarded):
            self.build()

    def test_redirected_metadata_is_rejected_before_helper(self):
        changes = [("teacher", lambda: self.manifest["teacher_binding"].update(descriptor="datasets/raw/secret.json")),
                   ("prep", lambda: self.matrix.update(preparation="datasets/raw/secret.json")),
                   ("config", lambda: self.manifest["jobs"][0].update(config="runs/phase2-foreign/config.json"))]
        old_manifest, old_matrix = copy.deepcopy(self.manifest), copy.deepcopy(self.matrix)
        for name, mutate in changes:
            with self.subTest(name=name):
                self.manifest, self.matrix = copy.deepcopy(old_manifest), copy.deepcopy(old_matrix)
                mutate()
                self.save()
                with self.assertRaises(ValueError):
                    self.build()
                self.context.matrix.load_matrix.assert_not_called()

    def test_receipt_redirected_to_test_actions_is_rejected(self):
        path = self.manifest["datasets"]["teacher"]["split"]
        self.write(path, {"partitions": {"test": [{"validation_receipt": "datasets/raw/test.json"}]}})
        with self.assertRaisesRegex(ValueError, "Receipt outside"):
            self.build()
        self.context.matrix.load_matrix.assert_not_called()

    def test_metadata_receipt_leaf_alias_is_rejected_before_helpers(self):
        receipt = self.root / "datasets/manifests/teacher/game.verification.json"
        receipt.symlink_to(self.root / "datasets/raw/DO-NOT-OPEN.json")
        with self.assertRaisesRegex(ValueError, "aliases storage"):
            self.build()
        self.context.matrix.load_matrix.assert_not_called()

    def test_frozen_weight_or_descriptor_redirect_fails_before_helpers(self):
        original = copy.deepcopy(self.matrix)
        for field in ("frozen_weights", "descriptor"):
            with self.subTest(field=field):
                self.matrix = copy.deepcopy(original)
                self.matrix["models"]["A-2026"][field]["path"] = "datasets/raw/test.json"
                self.save()
                with self.assertRaisesRegex(ValueError, "registered model directory"):
                    self.build()
                self.context.matrix.load_matrix.assert_not_called()

    def test_leaf_symlinks_are_rejected_without_opening_dataset_target_or_context(self):
        target = self.root / "datasets/raw/DO-NOT-OPEN.trace.jsonl.gz"
        for relative in ("runs/phase2-A-2026/metrics.jsonl",
                         self.matrix_path + "/development/primary/incumbent/plan.json"):
            with self.subTest(path=relative):
                leaf = self.root / relative
                original = leaf.read_bytes()
                leaf.unlink()
                leaf.symlink_to(target)
                try:
                    with patch.dict(os.environ, {"AGARILAB_STORAGE_ROOT": str(self.root)}), patch.object(report, "Context") as context:
                        with self.assertRaisesRegex(ValueError, "symlink"):
                            report.main(["--runtime-root", "/unused", "--tools-root", "/unused", "--matrix", self.matrix_path,
                                         "--output", self.output])
                        context.assert_not_called()
                    self.context.matrix.load_matrix.assert_not_called()
                finally:
                    leaf.unlink()
                    leaf.write_bytes(original)

    def test_original_gate_failure_and_saved_report_difference_produce_no_report(self):
        self.context.matrix.load_matrix.side_effect = ValueError("original gate failure")
        with self.assertRaisesRegex(ValueError, "original gate failure"):
            self.build()
        self.context.matrix.load_matrix.side_effect = lambda *_: copy.deepcopy(self.matrix)
        saved = copy.deepcopy(self.development)
        saved["winner"] = "A-2026"
        self.write(self.matrix_path + "/development-report.json", saved)
        with self.assertRaisesRegex(ValueError, "Saved development"):
            self.build()
        self.assertFalse((self.root / self.output).exists())

    def test_large_keeps_existing_authorization_and_does_not_claim_new_games(self):
        old = "runs/phase2-small"
        old_registration = {"scale": "small", "scale_from": None, "fixtures": dict.fromkeys(report.FIXTURES, {})}
        self.matrix.update(scale="large", scale_from={"matrix": old,
            "registration_sha256": self.write(old + "/registration.json", old_registration),
            "development_report": {"path": old + "/development-report.json", "sha256": self.write(old + "/development-report.json", {})}})
        self.save()
        self.build()
        value = json.loads((self.root / self.output / "summary.json").read_bytes())
        self.assertEqual(value["scale_from"], self.matrix["scale_from"])
        self.assertTrue(all(row["cost"]["newly_executed_games_for_this_scale"] is None for row in value["development"]))

    def test_large_cannot_redirect_prior_matrix_fixtures_to_final(self):
        old = "runs/phase2-small"
        self.matrix.update(scale="large", scale_from={"matrix": old,
            "registration_sha256": self.write(old + "/registration.json", {"scale": "small", "scale_from": None, "fixtures": {"../../final": {}}}),
            "development_report": {"path": old + "/development-report.json", "sha256": "a" * 64}})
        self.save()
        with self.assertRaisesRegex(ValueError, "prior small matrix"):
            self.build()

    def test_source_or_artifact_drift_blocks_publication(self):
        self.context.check.side_effect = lambda: self.write("runs/phase2-A-2026/result.json", {"status": "changed"})
        with self.assertRaisesRegex(ValueError, "SHA differs"):
            self.build()
        self.assertFalse((self.root / self.output).exists())

    def test_new_execution_attempt_after_snapshot_blocks_publication(self):
        self.context.check.side_effect = lambda: self.write(f"runs/{Path(self.prep).name}-execution/A-2026/attempt-0001/started.json", {})
        with self.assertRaisesRegex(ValueError, "inventory changed"):
            self.build()
        self.assertFalse((self.root / self.output).exists())

    def test_existing_output_is_never_overwritten(self):
        self.raw(self.output + "/marker", b"retain")
        with self.assertRaisesRegex(ValueError, "new directory"):
            self.build()
        self.assertEqual((self.root / self.output / "marker").read_bytes(), b"retain")

    def test_action_counter_or_validation_scope_drift_is_rejected(self):
        row = validation()
        best = self.matrix["models"]["A-2026"]["best"]
        path = self.root / "synthetic.jsonl"
        mutations = [lambda x: x.update(validation_scope="limited"),
                     lambda x: x["validation_policy_diagnostics"]["forced"].update(correct=1),
                     lambda x: x["validation_policy_diagnostics"]["by_action_type"]["PASS"].update(forced=counts(1, 1)),
                     lambda x: x.update(validation_accuracy=.9)]
        for mutate in mutations:
            changed = copy.deepcopy(row)
            mutate(changed)
            path.write_text(json.dumps(changed) + "\n")
            with self.assertRaises(ValueError):
                report.validation_row(path, "bc", best)
        row["validation_accuracy"] += 1e-8  # original torch float aggregation tolerance
        path.write_text(json.dumps(row) + "\n")
        report.validation_row(path, "bc", best)

    def test_all_attempts_and_historical_overwritten_hashes_remain_visible(self):
        key, job = "C-2026", self.jobs["C-2026"]
        execution = f"runs/{Path(self.prep).name}-execution/{key}"
        binding_sha = self.write(execution + "/binding.json", {"fixture": True})
        old = {"path": f"checkpoints/{job['run']}/latest.pt", "sha256": "d" * 64}
        self.raw(old["path"], b"new checkpoint bytes deliberately differ from historic SHA")
        end = {"progress": {"grp": 1, "cql": 0}}
        for index, status in enumerate(("interrupted", "failed", "paused"), 1):
            attempt = execution + f"/attempt-{index:04d}"
            started = {"job_id": key, "binding_sha256": binding_sha, "targets": {"grp": 1, "cql": 2},
                       "resume": index != 1, "stop_after_updates": 1 if index == 1 else None}
            start_sha = self.write(attempt + "/started.json", started)
            stdout_sha = self.write(attempt + "/stdout.log", end)
            finished = {"job_id": key, "attempt": attempt, "started_sha256": start_sha, "status": status,
                "exit_code": 2 if status != "failed" else 1, "stop_reason": "update_limit" if index == 1 else "wall_time_budget",
                "logs": [{"path": attempt + "/stdout.log", "sha256": stdout_sha}], "invocation_wall_seconds": 10,
                "cli_result": end, "latest_checkpoint": old}
            if index == 1:
                finished["process"] = {"exit_code": 0, "wall_seconds": 8, "user_seconds": 12, "system_seconds": 1, "max_rss_bytes": 999}
            self.write(attempt + "/finished.json", finished)
        self.raw(execution + "/attempt-0004/finished.json", b'{"partial":')
        history = report.training_attempts(report.Evidence(self.root), self.context, self.root / self.prep, self.manifest,
                                          key, job, [], {"grp": 1, "cql": 2})
        self.assertEqual([row["status"] for row in history["attempts"]], ["interrupted", "failed", "paused", "unsealed"])
        self.assertTrue(history["attempts"][0]["planned_grp_boundary"])
        self.assertEqual(history["attempts"][0]["historical_snapshots"]["latest_checkpoint"], old)
        self.assertEqual(history["costs"]["invocation_wall_seconds"], {"observed_sum": 30, "known_attempts": 3, "unknown_attempts": 1})
        self.assertEqual(history["costs"]["child_wall_seconds"]["unknown_attempts"], 3)
        self.assertIsNone(history["batch_wall_seconds"])

    def test_started_or_log_tamper_is_rejected(self):
        job = self.jobs["A-2026"]
        execution = f"runs/{Path(self.prep).name}-execution/A-2026"
        binding = self.write(execution + "/binding.json", {"fixture": True})
        self.write(execution + "/attempt-0001/started.json", {"job_id": "A-2026", "binding_sha256": "bad", "targets": {"bc": 2}})
        with self.assertRaisesRegex(ValueError, "start belongs"):
            report.training_attempts(report.Evidence(self.root), self.context, self.root / self.prep, self.manifest,
                                     "A-2026", job, [], {"bc": 2})
        attempt = execution + "/attempt-0001"
        start_sha = self.write(attempt + "/started.json", {"job_id": "A-2026", "binding_sha256": binding, "targets": {"bc": 2}})
        old = self.raw(attempt + "/stdout.log", b"original")
        self.write(attempt + "/finished.json", {"job_id": "A-2026", "attempt": attempt, "started_sha256": start_sha,
            "status": "failed", "logs": [{"path": attempt + "/stdout.log", "sha256": old}]})
        self.raw(attempt + "/stdout.log", b"substituted")
        with self.assertRaisesRegex(ValueError, "log binding differs"):
            report.training_attempts(report.Evidence(self.root), self.context, self.root / self.prep, self.manifest,
                                     "A-2026", job, [], {"bc": 2})

    def test_original_child_argv_is_retained_and_cross_job_mutation_rejected(self):
        key, job = "A-2026", self.jobs["A-2026"]
        execution = f"runs/{Path(self.prep).name}-execution/{key}"
        binding = {"python": {"path": str(Path(sys.executable).resolve())}}
        self.context.runner.execution_binding.return_value = binding
        binding_sha = self.write(execution + "/binding.json", binding)
        attempt = execution + "/attempt-0001"
        started = {"job_id": key, "binding_sha256": binding_sha, "targets": {"bc": 2},
                   "max_wall_seconds": 172800, "resume": True, "stop_after_updates": None}
        start_sha = self.write(attempt + "/started.json", started)
        command = {"argv": [sys.executable, "-B", "-u", "-m", "agari_research.cli", "run", "--config",
            str(self.root / job["config"]), "--split", job["split"], "--run", job["run"], "--max-wall-seconds", "100.0",
            "--resume", f"checkpoints/{job['run']}/latest.pt"], "cwd": "/frozen/agari-research",
            "pythonpath": "/frozen/agari-kit/src:/frozen/agari-research/src", "child_max_wall_seconds": 100.0}
        self.write(attempt + "/command.json", command)
        self.write(attempt + "/finished.json", {"job_id": key, "attempt": attempt, "started_sha256": start_sha,
            "status": "failed", "exit_code": 1, "logs": [], "process": {"exit_code": 1, "wall_seconds": 12}})
        def history():
            return report.training_attempts(report.Evidence(self.root), self.context, self.root / self.prep, self.manifest,
                                            key, job, [], {"bc": 2})
        self.assertEqual(history()["attempts"][0]["command"], command)
        command["argv"][command["argv"].index("--run") + 1] = "phase2-another-job"
        self.write(attempt + "/command.json", command)
        with self.assertRaisesRegex(ValueError, "Recorded child command"):
            history()


if __name__ == "__main__":
    unittest.main()
