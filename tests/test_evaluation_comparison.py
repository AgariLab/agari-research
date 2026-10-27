import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "compare_evaluations", Path(__file__).parents[1] / "scripts/compare_evaluations.py")
comparison = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(comparison)


def fixture(root, name, ranks_by_seed):
    directory = root / name
    directory.mkdir()
    lineup = [name, "efficiency", "random", "tsumogiri"]
    manifest = {"status": "complete", "seeds": list(ranks_by_seed), "lineup": lineup,
                "lineup_provenance": [{"kind": "builtin", "name": agent} for agent in lineup],
                "games_expected": len(ranks_by_seed) * 4, "games_completed": len(ranks_by_seed) * 4,
                "rules": {"preset": "tenhou"}, "engine": {"binary_sha256": "a" * 64},
                "code": {"source_sha256": "b" * 64}, "python": "same-python"}
    manifest["lineup_provenance"][0] = {
        "kind": "external", "name": name, "factory": "fixed_model:Agent",
        "source": {"package": "fixed_model", "module": "fixed_model", "files": 1, "source_sha256": "d" * 64},
        "descriptor": {"kwargs": {"device": "cpu", "torch_threads": 1}},
        "artifacts": {"weights_path": {"path": name + ".pt", "sha256": name}},
    }
    rows = []
    for seed, wanted_rank in ranks_by_seed.items():
        for rotation in range(4):
            order = [seat for seat in range(4) if seat != rotation]
            order.insert(wanted_rank - 1, rotation)
            rows.append({"seed": seed, "rotation": rotation,
                         "seats": [lineup[(seat - rotation) % 4] for seat in range(4)],
                         "scores": [40000 - order.index(seat) * 10000 for seat in range(4)],
                         "ranks": [order.index(seat) + 1 for seat in range(4)], "illegal_actions": 0})
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "games.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return directory


class EvaluationComparisonTests(unittest.TestCase):
    def test_paired_effect_preserved_despite_different_seed_difficulty(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = fixture(root, "old", {10: 2, 11: 3, 12: 4})
            new = fixture(root, "new", {10: 1, 11: 2, 12: 3})
            report = comparison.compare(old, new)
            self.assertEqual(report["difference"]["mean_rank"], -1)
            self.assertEqual(report["ci95"]["mean_rank"], [-1, -1])
            self.assertEqual(report["ci95"]["mean_point_delta"], [10000, 10000])
            self.assertFalse(report["promotion_sample_size_met"])

    def test_correlated_rotations_are_resampled_as_whole_seed_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = fixture(root, "old", {10: 2, 11: 2})
            new = fixture(root, "new", {10: 1, 11: 3})
            report = comparison.compare(old, new)
            self.assertEqual(report["difference"]["mean_rank"], 0)
            # Resampling eight independent seats would give an artificially narrow CI.
            self.assertEqual(report["ci95"]["mean_rank"], [-1, 1])

    def test_rejects_partial_duplicate_and_mismatched_experiments(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = fixture(root, "old", {10: 2, 11: 3})
            new = fixture(root, "new", {10: 1, 11: 2})
            manifest_path, rows_path = new / "manifest.json", new / "games.jsonl"
            original_manifest, original_rows = manifest_path.read_text(), rows_path.read_text()
            for field, value in (("status", "running"), ("engine", {"binary_sha256": "c" * 64}),
                                 ("lineup_provenance", [{"name": "changed"}] * 4)):
                with self.subTest(field=field):
                    manifest = json.loads(original_manifest)
                    manifest[field] = value
                    manifest_path.write_text(json.dumps(manifest))
                    with self.assertRaises(ValueError):
                        comparison.compare(old, new)
            manifest_path.write_text(original_manifest)
            rows = original_rows.splitlines()
            for altered in (rows[:-1], rows[:-1] + [rows[0]]):
                rows_path.write_text("\n".join(altered) + "\n")
                with self.assertRaises(ValueError):
                    comparison.compare(old, new)
            rows_path.write_text(original_rows)
            wrong = json.loads(rows[0])
            wrong["ranks"] = [4, 3, 2, 1]
            rows_path.write_text("\n".join([json.dumps(wrong), *rows[1:]]) + "\n")
            with self.assertRaises(ValueError):
                comparison.compare(old, new)

    def test_rejects_candidate_inference_changes_while_allowing_new_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = fixture(root, "old", {10: 2, 11: 3})
            new = fixture(root, "new", {10: 1, 11: 2})
            self.assertEqual(comparison.compare(old, new)["difference"]["mean_rank"], -1)
            path = new / "manifest.json"
            original = path.read_text()
            for field in ("source", "factory", "kwargs"):
                with self.subTest(field=field):
                    manifest = json.loads(original)
                    candidate = manifest["lineup_provenance"][0]
                    if field == "source":
                        candidate["source"]["source_sha256"] = "e" * 64
                    elif field == "factory":
                        candidate["factory"] = "another_model:Agent"
                    else:
                        candidate["descriptor"]["kwargs"]["torch_threads"] = 2
                    path.write_text(json.dumps(manifest))
                    with self.assertRaisesRegex(ValueError, "Candidate inference"):
                        comparison.compare(old, new)


if __name__ == "__main__":
    unittest.main()
