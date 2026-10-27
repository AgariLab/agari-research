"""Full-epoch coverage, tail batches and exact resume regression tests."""

import copy
import contextlib
import importlib.util
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from agari_kit.mjai import DataError
from agari_kit.dataset import make_split
from agari_kit.mjai import canonical_json
from agari_kit.storage import Storage
from agari_kit.engine import engine_identity
from agari_research.data import BatchStream, EpochBatchStream, FrozenDataset
from agari_research.planning import build_training_plan, coverage, expected_samples
from agari_research.training import atomic_torch_save, run_training, validate_config, validation_metrics

ROOT = Path(__file__).parents[1]
CONFIG = json.loads((ROOT / "configs/smoke.json").read_text())
spec = importlib.util.spec_from_file_location("verify_resume", ROOT / "scripts/verify_resume.py")
verify_resume = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify_resume)


class TinyDataset:
    """Five deterministic offline samples; accessing test fails immediately."""
    sha256 = "a" * 64
    manifest = {"dataset_sha256": sha256}

    def records(self, partition, seed=None):
        if partition not in ("train", "validation"):
            raise AssertionError("Test partition must stay unopened")
        rows = [{"game_id": "first", "sha256": "1" * 64, "offset": 0, "count": 2},
                {"game_id": "second", "sha256": "2" * 64, "offset": 2, "count": 3}]
        if seed is not None:
            random.Random(seed).shuffle(rows)
        return rows

    def sample_counts(self, partition):
        return [{"game_id": row["game_id"], "sha256": row["sha256"],
                 "grp_samples": row["count"], "cql_samples": row["count"]}
                for row in self.records(partition)]

    def grp_game_samples(self, record):
        for index in range(record["offset"], record["offset"] + record["count"]):
            yield torch.arange(20).float() / 20 + index / 10, torch.tensor(index % 4)

    def grp_samples(self, partition, seed=None):
        for record in self.records(partition, seed):
            yield from self.grp_game_samples(record)

    def cql_game_samples(self, record, grp, config):
        grp.eval()
        for features, rank in self.grp_game_samples(record):
            with torch.inference_mode():
                reward = float((grp(features[None]).softmax(1) * torch.tensor(config["pts_weight"])).sum()) - 3
            x = torch.arange(74 * 34).float().reshape(74, 34) / (74 * 34) + float(features[0])
            mask = torch.arange(82) < 5
            yield x, rank, mask, torch.tensor(reward), rank

    def cql_samples(self, partition, grp, config, seed=None):
        for record in self.records(partition, seed):
            yield from self.cql_game_samples(record, grp, config)


def epoch_config():
    config = copy.deepcopy(CONFIG)
    for stage in ("grp", "cql"):
        config[stage].pop("steps")
        config[stage].update(epochs=2, batch_size=2)
    config["validation"] = {"batch_size": 2, "max_samples": None}
    config["checkpoint_interval_updates"] = 5
    return config


class CoverageTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_tail_is_kept_and_every_update_boundary_resumes_exactly(self):
        for counts in ({"a": 2, "b": 3}, {"a": 1}, {"a": 2, "b": 2}):
            records = [{"game_id": key} for key in counts]
            def factory(record):
                for index in range(counts[record["game_id"]]):
                    yield (torch.tensor(ord(record["game_id"]) * 10 + index),)
            for size in (1, 2, 4, 8):
                state = random.getstate()
                stream = EpochBatchStream(records, counts, factory, size, 2, seed=19)
                batches, seen = [], [0]
                while True:
                    try:
                        batches.append(stream.next("cpu")[0].tolist())
                        seen.append(stream.seen)
                    except StopIteration:
                        break
                n = sum(counts.values())
                self.assertEqual([len(row) for row in batches], ([size] * (n // size) + ([n % size] if n % size else [])) * 2)
                self.assertEqual(sorted(sum(batches[:len(batches) // 2], [])), sorted(sum(batches[len(batches) // 2:], [])))
                for boundary, cursor in enumerate(seen):
                    resumed = EpochBatchStream(records, counts, factory, size, 2, seen=cursor, seed=19)
                    remaining = [resumed.next("cpu")[0].tolist() for _ in batches[boundary:]]
                    self.assertEqual(remaining, batches[boundary:])
                    with self.assertRaises(StopIteration):
                        resumed.next("cpu")
                self.assertEqual(random.getstate(), state)

    def test_count_mismatch_fails_instead_of_silently_dropping_or_padding(self):
        for actual in (1, 3):
            stream = EpochBatchStream([{"game_id": "a"}], {"a": 2},
                lambda _: ((torch.tensor(i),) for i in range(actual)), 2, 1)
            with self.assertRaises(DataError):
                stream.next("cpu")
        with self.assertRaises(DataError):
            EpochBatchStream([], {}, lambda _: [], 2, 1)

    def test_verification_and_native_counting_never_open_test_content(self):
        records = [{"game_id": str(index), "sha256": f"{index:064x}",
                    "played_at": f"2026-01-01T00:00:{index:02d}+00:00", "source": "agarilab-selfplay",
                    "trace_path": f"{index}.trace.jsonl",
                    "rules": "4p-red-half/tenhou", "usage_basis": "test", "validation": "engine-verified"}
                   for index in range(20)]
        manifest = make_split(records)
        visited = []
        def verify(storage, record, *, replay):
            self.assertTrue(replay)
            self.assertNotIn(record, manifest["partitions"]["test"])
            visited.append(record)
            return {"engine_version": "0.4.10", "engine_binary_sha256": engine_identity()["binary_sha256"],
                    "method": "seeded-action-trace-v1", "producer_code": {
                        "package": "agari-kit", "version": "0.1.0", "source_sha256": "a" * 64}}
        with tempfile.TemporaryDirectory() as directory, patch("agari_research.data.verify_record", side_effect=verify):
            storage = Storage(Path(directory))
            split = storage.path("split.json")
            split.write_text(canonical_json(manifest))
            dataset = FrozenDataset(storage, split)
            self.assertEqual(visited, manifest["partitions"]["train"] + manifest["partitions"]["validation"])
            with self.assertRaises(DataError):
                dataset.sample_counts("test")

    def test_plan_command_creates_evidence_directory_without_training(self):
        from agari_research.cli import main
        with tempfile.TemporaryDirectory() as directory:
            storage = Storage(Path(directory))
            config = storage.path("config.json")
            config.write_text(json.dumps(epoch_config()))
            with patch("sys.argv", ["agari-train", "plan", "--config", str(config), "--split", "split.json",
                                    "--output", "runs/new-preflight/plan.json"]), patch(
                    "agari_kit.storage.Storage.from_env", return_value=storage), patch(
                    "agari_research.data.FrozenDataset", return_value=TinyDataset()), patch(
                    "agari_research.training.run_training", side_effect=AssertionError("Plan must not train")), contextlib.redirect_stdout(io.StringIO()):
                main()
            evidence = json.loads(storage.path("runs/new-preflight/plan.json").read_text())
            self.assertEqual(evidence["plan"]["stages"]["cql"]["target_samples"], 10)
            self.assertFalse(storage.path("checkpoints/new-preflight/latest.pt").exists())

    def test_plan_and_coverage_record_complete_epochs_and_partial_games(self):
        config, dataset = epoch_config(), TinyDataset()
        plan = build_training_plan(config, dataset)
        self.assertEqual(plan["stages"]["grp"], {"mode": "epochs", "epochs": 2, "samples_per_epoch": 5,
            "updates_per_epoch": 3, "target_updates": 6, "target_samples": 10, "last_batch_samples": 1})
        self.assertEqual([expected_samples(plan["stages"]["grp"], n, 2) for n in range(7)], [0, 2, 4, 5, 7, 9, 10])
        actual = coverage(plan, "grp", 7, config["seed"], include_games=True)
        self.assertEqual((actual["complete_epochs"], actual["samples_in_epoch"], actual["unique_games_seen"]), (1, 2, 2))
        self.assertEqual(sum(row["samples_seen"] for row in actual["games"]), 7)
        completed = coverage(plan, "grp", 10, config["seed"], include_games=True)
        self.assertTrue(all(row["samples_seen"] == row["samples_per_epoch"] * 2 for row in completed["games"]))
        self.assertEqual(plan, build_training_plan(config, dataset))

    def test_steps_configuration_and_cross_epoch_batches_remain_supported(self):
        validate_config(CONFIG)
        plan = build_training_plan(CONFIG, TinyDataset())
        self.assertEqual(plan["training_games"], [])
        self.assertEqual(plan["stages"]["grp"]["target_updates"], 4)
        stream = BatchStream(lambda _: ((torch.tensor(i),) for i in range(3)), 2)
        self.assertEqual(stream.next("cpu")[0].tolist(), [0, 1])
        self.assertEqual(stream.next("cpu")[0].tolist(), [2, 0])
        invalid = copy.deepcopy(CONFIG)
        invalid["grp"]["epochs"] = 1
        with self.assertRaises(ValueError):
            validate_config(invalid)
        for field, value in (("checkpoint_interval_updates", 0), ("validation", {"max_samples": 0})):
            invalid = dict(CONFIG, **{field: value})
            with self.assertRaises(ValueError):
                validate_config(invalid)

    def test_full_validation_weights_partial_batch_by_samples_and_can_be_bounded(self):
        logits = torch.tensor([[5., 0., 0., 0.], [0., 5., 0., 0.], [8., 0., 0., 0.]])
        labels = torch.tensor([0, 1, 2])
        class Dataset:
            def grp_samples(self, partition):
                self.partition = partition
                assert partition == "validation"
                return iter(zip(logits, labels))
        config = dict(CONFIG, validation={"batch_size": 2, "max_samples": None})
        metrics = validation_metrics("grp", Dataset(), {"grp": torch.nn.Identity()}, config)
        self.assertEqual(metrics["validation_samples"], 3)
        self.assertEqual(metrics["validation_scope"], "full")
        self.assertAlmostEqual(metrics["validation_loss"], float(F.cross_entropy(logits, labels)), places=6)
        self.assertAlmostEqual(metrics["validation_accuracy"], 2 / 3)
        config["validation"]["max_samples"] = 2
        bounded = validation_metrics("grp", Dataset(), {"grp": torch.nn.Identity()}, config)
        self.assertEqual(bounded["validation_samples"], 2)
        self.assertEqual(bounded["validation_scope"], "bounded")
        self.assertEqual(bounded["validation_accuracy"], 1)

    def test_full_training_resume_crosses_epochs_with_single_sample_tails(self):
        config, dataset = epoch_config(), TinyDataset()
        saved = []
        def recording_save(payload, destination):
            if destination.name == "latest.pt":
                saved.append((payload["run_name"], sum(payload["progress"].values())))
            atomic_torch_save(payload, destination)
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ"), patch(
                "agari_research.training.atomic_torch_save", side_effect=recording_save):
            storage = Storage(Path(directory))
            complete = run_training(config, dataset, storage, run_name="complete")
            partial = run_training(config, dataset, storage, run_name="resume", stop_after_updates=4)
            checkpoint = storage.path("checkpoints/resume/latest.pt")
            interrupted = torch.load(checkpoint, weights_only=True)
            self.assertEqual(interrupted["progress"]["grp"], 4)
            self.assertEqual(interrupted["samples_seen"]["grp"], 7)
            self.assertEqual(partial["status"], "interrupted")
            # Continue across the second GRP epoch and the first CQL epoch.
            run_training(config, dataset, storage, run_name="resume", resume=checkpoint, stop_after_updates=5)
            interrupted = torch.load(checkpoint, weights_only=True)
            self.assertEqual(interrupted["progress"], {"grp": 6, "cql": 3, "ppo": 0})
            self.assertEqual(interrupted["samples_seen"], {"grp": 10, "cql": 5})
            final = run_training(config, dataset, storage, run_name="resume", resume=checkpoint)
            self.assertEqual(final["status"], "complete")
            self.assertEqual(complete["progress"], {"grp": 6, "cql": 6, "ppo": 1})
            left = torch.load(storage.path("checkpoints/complete/latest.pt"), weights_only=True)
            right = torch.load(checkpoint, weights_only=True)
            left.pop("run_name")
            right.pop("run_name")
            self.assertGreater(verify_resume.identical(left, right), 200)
            self.assertEqual([step for name, step in saved if name == "complete"], [3, 5, 6, 9, 10, 12, 13])
            self.assertIn(("resume", 4), saved)  # Explicit interruption is durable outside interval 5.
            evidence = json.loads(storage.path("runs/resume/coverage.json").read_text())
            for stage in ("grp", "cql"):
                self.assertEqual(evidence["stages"][stage]["complete_epochs"], 2)
                self.assertEqual(evidence["stages"][stage]["unique_games_seen"], 2)
                self.assertTrue(all(row["samples_seen"] == 2 * row["samples_per_epoch"]
                                    for row in evidence["stages"][stage]["games"]))
            metrics = [json.loads(line) for line in storage.path("runs/complete/metrics.jsonl").read_text().splitlines()]
            for stage in ("grp", "cql"):
                rows = [row for row in metrics if row["stage"] == stage]
                self.assertEqual([row["batch_samples"] for row in rows], [2, 2, 1, 2, 2, 1])
                self.assertEqual(rows[-1]["validation_samples"], 5)
                self.assertEqual(rows[-1]["validation_scope"], "full")
            legacy = torch.load(checkpoint, weights_only=True)
            legacy.pop("sample_semantics_version")
            old_checkpoint = storage.path("legacy.pt")
            atomic_torch_save(legacy, old_checkpoint)
            with self.assertRaisesRegex(ValueError, "sample_semantics_version"):
                run_training(config, dataset, storage, run_name="resume", resume=old_checkpoint)


if __name__ == "__main__":
    unittest.main()
