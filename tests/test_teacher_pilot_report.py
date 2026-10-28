"""Synthetic metadata only: no actual dataset, teacher, qualification or training."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).parents[1] / "scripts/report_teacher_pilot.py"
spec = importlib.util.spec_from_file_location("teacher_pilot_report_test", SCRIPT)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class MetadataReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.output = "runs/phase2-report-test"
        self.teacher = {"kind": "external", "name": "@synthetic-teacher", "factory": "fixture:Teacher"}
        self.identity = {"rules": {"game_mode": "4p-red-half", "rule": "tenhou"},
                         "code": {"source_sha256": "a" * 64},
                         "engine": {"version": "0.4.10", "binary_sha256": "b" * 64}}
        self.protocol = {"engine_version": "0.4.10", "training": {"model": {"in_channels": 74}},
            "seeds": {"teacher_1000": {"start": 2630000, "count": 1000, "rotations": 1},
                      "teacher_probe": {"start": 2620000, "count": 16, "rotations": 1},
                      "development": {"start": 2660000, "count": 250, "rotations": 4},
                      "final_promotion": {"start": 2680000, "count": 2500, "rotations": 4}}}
        self.manifest = {"status": "prepared-not-trained", "test_partition_content_opened": False,
            "teacher_provenance": self.teacher, "teacher_binding": {"descriptor": report.TEACHER, "weights_sha256": "c" * 64},
            "qualification": {"run": report.QUALIFICATION, "collection_identity": self.identity,
                              "qualification_sha256": "d" * 64}, "datasets": {}, "jobs": []}
        self.splits, self.audits = {}, {}
        for name, start in (("weak", 0), ("teacher", 2630000)):
            parent = "datasets/manifests/" + report.DATASETS[name]
            players = [self.teacher] * 4 if name == "teacher" else [{"kind": "builtin", "name": "efficiency"}] * 4
            config = self.config(start, 1000, players)
            acquisition_sha = self.write(parent + "/acquisition.json", {"status": "complete", "config": config})
            split = {"schema_version": 1, "policy": "whole-game-time-90-5-5", "partitions": {part: [] for part in report.PARTITIONS}}
            inventory, counts = [], []
            for index in range(1000):
                part = "train" if index < 900 else "validation" if index < 950 else "test"
                row = self.make_game(parent, index, start + index, config, legacy=name == "weak")
                split["partitions"][part].append(row)
                inventory.append({"game_id": row["game_id"], "partition": part, "seed": start + index,
                                  "receipt": row["validation_receipt"], "sha256": row["validation_receipt_sha256"]})
                if part == "train":
                    counts.append({"game_id": row["game_id"], "sha256": row["sha256"],
                                   "cql_samples": 64 if name == "teacher" else 32, "grp_samples": 8})
            split["dataset_sha256"] = report.content_sha(split)
            split_path = parent + "/split-v1.json"
            self.write(split_path, split)
            audit = {"name": name, "games": 1000, "seeds": list(range(start, start + 1000)),
                     "acquisition": parent + "/acquisition.json", "acquisition_sha256": acquisition_sha,
                     "receipt_inventory": inventory, "receipt_inventory_sha256": report.content_sha(inventory),
                     "legacy_test_games_without_per_game_policy_binding": 50 if name == "weak" else 0,
                     "test_partition_content_opened": False,
                     "test_policy_identity_evidence": "acquisition config only for legacy receipts; per-seat receipt hashes otherwise"}
            self.manifest["datasets"][name] = {"split": split_path, "sha256": split["dataset_sha256"],
                "training_decisions": sum(row["cql_samples"] for row in counts),
                "counts_sha256": self.write(f"{report.PREPARATION}/{name}-training-counts.json", counts),
                "source_audit_sha256": self.write(f"{report.PREPARATION}/{name}-source-audit.json", audit)}
            self.splits[name], self.audits[name] = split, audit
        parent = "datasets/manifests/" + report.PROBE
        config = self.config(2620000, 16, [self.teacher] * 4)
        self.write(parent + "/acquisition.json", {"status": "complete", "config": config})
        for index in range(16):
            row = self.make_game(parent, index, 2620000 + index, config)
            self.write(f"{parent}/game-{index:06d}.json", row)
        for arm in "ABC":
            for seed in (2026, 2027, 2028):
                method = "cql" if arm == "C" else "bc"
                config = {"seed": seed, "device": "cpu", "model": self.protocol["training"]["model"],
                          "stages": ["grp", "cql"] if arm == "C" else ["bc"],
                          "validation": {"max_samples": None, "interval_updates": 900},
                          method: {"steps": 2700, "batch_size": 64}}
                if arm == "C":
                    config["grp"] = {"batch_size": 32, "epochs": 1}
                path = f"{report.PREPARATION}/configs/{arm}-seed{seed}.json"
                self.manifest["jobs"].append({"arm": arm, "seed": seed, "method": method, "config": path,
                    "config_sha256": self.write(path, config), "run": f"prepared-{arm}-{seed}", "max_wall_seconds": 172800,
                    "split": self.manifest["datasets"]["weak" if arm == "A" else "teacher"]["split"]})
        self.save_manifest()
        self.handoff = {"status": "prepared-not-trained", "qualification": {"status": "passed"},
            "pilot": {"status": "ready"}, "preparation": {"jobs": 9, "output": str(self.root / report.PREPARATION),
                "training_decisions": {"weak": 28800, "teacher": 57600}, "policy_updates_per_job": 2700}}
        self.write(report.LAUNCH + "/result.json", self.handoff)
        self.context = SimpleNamespace(rt=SimpleNamespace(protocol=self.protocol), initial={"synthetic": True}, check=Mock())
        self.context.preparation = Mock(side_effect=lambda: (self.root / report.PREPARATION, copy.deepcopy(self.manifest),
                {f"{row['arm']}-{row['seed']}": row for row in self.manifest["jobs"]}, {}, []))

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(value, indent=2) + "\n").encode()
        path.write_bytes(data)
        return report.sha(data)

    def config(self, start, count, players):
        return {**copy.deepcopy(self.identity), "source": "agarilab-selfplay", "games": count, "seed_start": start,
                "agents": [player["name"] for player in players], "agent_provenance": players}

    def make_game(self, parent, index, seed, config, legacy=False):
        digest = report.sha(f"{parent}/{index}".encode())
        path = f"{parent}/game-{index:06d}.verification.json"
        value = {"validation": "engine-verified", "method": "seeded-action-trace-v1", "seed": seed, "sha256": digest,
                 "rules": config["rules"], "producer_code": config["code"], "engine_version": "0.4.10",
                 "engine_binary_sha256": config["engine"]["binary_sha256"], "scores": [99999] * 4}
        if not legacy:
            value["agent_provenance_sha256"] = [report.content_sha(player) for player in config["agent_provenance"]]
        return {"game_id": "selfplay:" + digest, "sha256": digest, "played_at": f"synthetic-{index}",
                "validation_receipt": path, "validation_receipt_sha256": self.write(path, value),
                "trace_path": "datasets/raw/DO-NOT-OPEN", "mjai_path": "datasets/mjai/DO-NOT-OPEN"}

    def save_manifest(self):
        return self.write(report.PREPARATION + "/manifest.json", self.manifest)

    def save_source(self, name):
        split, audit = self.splits[name], self.audits[name]
        entry = self.manifest["datasets"][name]
        split["dataset_sha256"] = report.content_sha({key: value for key, value in split.items() if key != "dataset_sha256"})
        entry["sha256"] = split["dataset_sha256"]
        self.write(entry["split"], split)
        audit["receipt_inventory_sha256"] = report.content_sha(audit["receipt_inventory"])
        entry["source_audit_sha256"] = self.write(f"{report.PREPARATION}/{name}-source-audit.json", audit)
        self.save_manifest()

    def run_report(self):
        return report.build(report.Metadata(self.root), self.context, self.output)

    def assert_rejected(self, pattern):
        with self.assertRaisesRegex(ValueError, pattern):
            self.run_report()
        self.assertFalse((self.root / self.output).exists())

    def test_completed_metadata_report_is_thin_exact_and_preserves_legacy_boundary(self):
        original_read = Path.read_bytes
        seen = []
        def guard(path):
            seen.append(path)
            self.assertNotIn("raw", path.parts)
            self.assertNotIn("mjai", path.parts)
            return original_read(path)
        with patch.object(Path, "read_bytes", guard):
            value = self.run_report()
        self.assertEqual(value["status"], "verified-metadata-only")
        self.context.preparation.assert_called_once()
        directory = self.root / self.output
        self.assertEqual({path.name for path in directory.iterdir()}, {"summary.json", "sources.json", "README.md", "preparation-manifest.json"})
        self.assertEqual((directory / "preparation-manifest.json").read_bytes(), (self.root / report.PREPARATION / "manifest.json").read_bytes())
        summary = json.loads((directory / "summary.json").read_bytes())
        self.assertEqual(summary["datasets"]["teacher"]["receipts_checked"], 1000)
        self.assertEqual(summary["datasets"]["weak"]["legacy_test_games_without_per_game_policy_binding"], 50)
        self.assertEqual({job["policy_updates_U"] for job in summary["jobs"]}, {2700})
        self.assertEqual({job["grp_updates_G"] for job in summary["jobs"] if job["arm"] == "C"}, {225})
        self.assertFalse(summary["isolation"]["evaluation_content_comparison_performed"])
        self.assertFalse(summary["read_boundary"]["training_execution_checked"])
        for name, entry in summary["report_artifacts"].items():
            data = (directory / name).read_bytes()
            self.assertEqual(entry, {"bytes": len(data), "sha256": report.sha(data)})
        self.assertNotIn("99999", (directory / "summary.json").read_text())
        sources = json.loads((directory / "sources.json").read_bytes())["artifacts"]
        self.assertLess(len(sources), 30)
        self.assertTrue(all(not entry["path"].endswith("verification.json") for entry in sources))
        self.assertLess(sum(path.stat().st_size for path in directory.iterdir()), 60000)

    def test_incomplete_handoff_reads_only_result_and_never_loads_runtime(self):
        self.handoff["status"] = "collecting"
        self.write(report.LAUNCH + "/result.json", self.handoff)
        with patch.dict("os.environ", AGARILAB_STORAGE_ROOT=str(self.root)), patch.object(report, "Context") as context:
            with self.assertRaisesRegex(ValueError, "completed prepared"):
                report.main(["--runtime-root", "unused", "--tools-root", "unused", "--output", self.output])
            context.assert_not_called()
        metadata = report.Metadata(self.root)
        with self.assertRaises(ValueError):
            report.build(metadata, self.context, self.output)
        self.assertEqual(set(metadata.bytes), {report.LAUNCH + "/result.json"})
        self.context.preparation.assert_not_called()

    def test_independent_qualification_preparation_gate_cannot_be_skipped(self):
        self.context.preparation.side_effect = ValueError("qualification failed")
        self.assert_rejected("qualification failed")

    def test_four_seat_identity_engine_and_seed_are_checked_even_with_updated_hashes(self):
        row = self.splits["teacher"]["partitions"]["test"][0]
        path = row["validation_receipt"]
        original = json.loads((self.root / path).read_bytes())
        for field, value, pattern in (("agent_provenance_sha256", ["wrong"] * 4, "four expected"),
                                       ("engine_binary_sha256", "wrong", "engine binary"),
                                       ("seed", 2680000, "registered cohort")):
            with self.subTest(field=field):
                altered = dict(original, **{field: value})
                row["validation_receipt_sha256"] = self.write(path, altered)
                self.audits["teacher"]["receipt_inventory"][950]["sha256"] = row["validation_receipt_sha256"]
                self.save_source("teacher")
                self.assert_rejected(pattern)
        self.context.preparation.assert_not_called()

    def test_receipt_bytes_cannot_change_behind_inventory(self):
        path = self.splits["teacher"]["partitions"]["train"][0]["validation_receipt"]
        self.write(path, {"different": True})
        self.assert_rejected("SHA differs")

    def test_cross_partition_duplicate_cannot_be_relabelled_as_complete(self):
        self.splits["teacher"]["partitions"]["test"][0] = self.splits["teacher"]["partitions"]["train"][0]
        self.save_source("teacher")
        self.assert_rejected("Duplicate game")

    def test_raw_path_is_rejected_before_frozen_helper_follows_it(self):
        row = self.splits["teacher"]["partitions"]["test"][0]
        row["validation_receipt"] = "datasets/raw/test.verification.json"
        self.save_source("teacher")
        self.assert_rejected("another acquisition")
        self.context.preparation.assert_not_called()
        with self.assertRaisesRegex(ValueError, "JSON metadata"):
            report.Metadata(self.root).read("datasets/raw/test.verification.json")

    def test_qualification_and_teacher_paths_cannot_redirect_helper_to_dataset_content(self):
        for parent, field in (("teacher_binding", "descriptor"), ("qualification", "run")):
            with self.subTest(field=field):
                original = self.manifest[parent][field]
                self.manifest[parent][field] = "datasets/raw/hidden.json"
                self.save_manifest()
                self.assert_rejected("fixed prior evidence paths")
                self.context.preparation.assert_not_called()
                self.manifest[parent][field] = original

    def test_configuration_budget_cannot_change_even_with_updated_file_sha(self):
        job = self.manifest["jobs"][0]
        config = json.loads((self.root / job["config"]).read_bytes())
        config["bc"]["steps"] += 1
        job["config_sha256"] = self.write(job["config"], config)
        self.save_manifest()
        self.assert_rejected("policy budget differs")

    def test_complete_game_counts_require_positive_native_samples(self):
        path = report.PREPARATION + "/teacher-training-counts.json"
        counts = json.loads((self.root / path).read_bytes())
        counts[0]["grp_samples"] = 0
        self.manifest["datasets"]["teacher"]["counts_sha256"] = self.write(path, counts)
        self.save_manifest()
        self.assert_rejected("Training counts")

    def test_recorded_probe_content_overlap_and_reserved_seed_overlap_fail(self):
        self.protocol["seeds"]["development"]["start"] = 2630999
        self.assert_rejected("Reserved evaluation")
        self.protocol["seeds"]["development"]["start"] = 2660000
        path = f"datasets/manifests/{report.PROBE}/game-000000.json"
        row = json.loads((self.root / path).read_bytes())
        teacher_row = self.splits["teacher"]["partitions"]["test"][0]
        row.update(game_id=teacher_row["game_id"], sha256=teacher_row["sha256"])
        receipt_path = row["validation_receipt"]
        value = json.loads((self.root / receipt_path).read_bytes())
        value["sha256"] = row["sha256"]
        row["validation_receipt_sha256"] = self.write(receipt_path, value)
        self.write(path, row)
        self.assert_rejected("identifiers overlap")

    def test_metadata_and_runtime_drift_cannot_publish_success(self):
        def mutate():
            self.write(report.PREPARATION + "/teacher-training-counts.json", [])
        self.context.check.side_effect = mutate
        self.assert_rejected("changed before publication")
        self.context.preparation.assert_called_once()

    def test_runtime_source_failure_and_existing_output_are_fail_closed(self):
        self.context.check.side_effect = ValueError("runtime drift")
        self.assert_rejected("runtime drift")
        self.context.check.side_effect = None
        self.run_report()
        original = (self.root / self.output / "summary.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "must be new"):
            self.run_report()
        self.assertEqual((self.root / self.output / "summary.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
