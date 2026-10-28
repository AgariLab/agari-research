"""BC objectives, isolated stages, identity selection and resumable budgets."""

import copy
import contextlib
import gzip
import json
import io
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from agari_kit.agent import Context
from agari_kit.collection import file_sha256, write_json
from agari_kit.engine import new_environment
from agari_kit.mjai import DataError
from agari_kit.storage import Storage
from agari_research.data import FrozenDataset
from agari_research.features import policy_features
from agari_research.losses import bc_loss
from agari_research.model_agent import ModelAgent
from agari_research.models.grp_model import RankPredictor
from agari_research.planning import build_training_plan, selected_stages
from agari_research.selection import agent_identity_sha256, normalize_policy_filter
from agari_research.training import initialize_q, policy_diagnostics, q_initialization_sha256, run_training, validate_config, validation_metrics

from test_coverage import CONFIG, TinyDataset, verify_resume
from test_trace_labels import native_record


class BehaviorDataset(TinyDataset):
    def bc_game_samples(self, record):
        for features, action in self.grp_game_samples(record):
            x = torch.arange(74 * 34).float().reshape(74, 34) / (74 * 34) + float(features[0])
            yield x, action, torch.arange(82) < 5

    def bc_samples(self, partition, seed=None):
        for record in self.records(partition, seed):
            yield from self.bc_game_samples(record)


def bc_config():
    return {key: copy.deepcopy(value) for key, value in CONFIG.items()
            if key not in ("grp", "cql", "ppo", "pts_weight")} | {
        "stages": ["bc"], "bc": {"epochs": 2, "batch_size": 2, "lr": 0.0005,
                                    "weight_decay": 0.1, "eta_min": 1e-7},
        "validation": {"batch_size": 2, "max_samples": None}, "checkpoint_interval_updates": 100}


def external_provenance():
    return {"kind": "external", "name": "@/old/teacher.json", "descriptor_path": "/old/teacher.json",
            "descriptor_sha256": "d" * 64, "factory": "teacher.agent:Agent",
            "descriptor": {"factory": "teacher.agent:Agent", "kwargs": {"temperature": 0},
                           "artifacts": {"weights": {"path": "model.pt", "sha256": "a" * 64}}},
            "artifacts": {"weights": {"path": "/old/model.pt", "declared_path": "model.pt",
                                      "declared_sha256": "a" * 64, "sha256": "a" * 64}},
            "source": {"package": "teacher", "module": "teacher.agent", "module_path": "/old/agent.py",
                       "roots": ["/old"], "files": 2, "source_sha256": "b" * 64}}


class BehaviorTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_bc_masks_illegal_actions_and_has_expected_gradient(self):
        q = torch.tensor([[1., 3., 999.]], requires_grad=True)
        loss, metrics = bc_loss(q, torch.tensor([0]), torch.tensor([[True, True, False]]))
        self.assertAlmostEqual(loss.item(), math.log1p(math.exp(2)), places=6)
        self.assertEqual(metrics["accuracy"].item(), 0)
        loss.backward()
        self.assertLess(q.grad[0, 0].item(), 0)
        self.assertGreater(q.grad[0, 1].item(), 0)
        self.assertEqual(q.grad[0, 2].item(), 0)
        with self.assertRaisesRegex(ValueError, "BC actions must be legal"):
            bc_loss(q, torch.tensor([2]), torch.tensor([[True, True, False]]))

    def test_bc_has_same_q_initialization_and_freezes_only_unused_heads(self):
        config = bc_config()
        torch.manual_seed(99)
        before = torch.get_rng_state().clone()
        bc = initialize_q(config, behavior_cloning=True)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        torch.rand(100)
        q = initialize_q(config)
        self.assertEqual(q_initialization_sha256(bc), q_initialization_sha256(q))
        for name, parameter in bc.named_parameters():
            self.assertEqual(parameter.requires_grad, not name.startswith(("v_head.", "aux_head.")))

    def test_stages_are_explicit_and_require_their_dependencies(self):
        self.assertEqual(selected_stages(CONFIG), ("grp", "cql", "ppo"))
        validate_config(bc_config())  # No GRP, reward or PPO settings required.
        reduced = {key: value for key, value in CONFIG.items() if key != "ppo"} | {"stages": ["grp", "cql"]}
        validate_config(reduced)
        for stages in ([], ["ppo"], ["cql"], ["bc", "cql"], ["grp", "grp"], "bc"):
            with self.subTest(stages=stages), self.assertRaises(ValueError):
                validate_config(bc_config() | {"stages": stages})

    def test_bc_only_trains_exports_and_restores_exactly_with_frozen_heads(self):
        config, dataset = bc_config(), BehaviorDataset()
        initial = initialize_q(config, behavior_cloning=True).state_dict()
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ"), patch(
                "agari_research.training.collect_rollout", side_effect=AssertionError("BC must not run PPO")), patch(
                "agari_research.training.RankPredictor", side_effect=AssertionError("BC must not initialize GRP")):
            storage = Storage(Path(directory))
            full = run_training(config, dataset, storage, run_name="full")
            partial = run_training(config, dataset, storage, run_name="resumed", stop_after_updates=2)
            checkpoint = storage.path("checkpoints/resumed/latest.pt")
            self.assertEqual(partial["status"], "interrupted")
            self.assertEqual(partial["training_coverage"]["bc"]["samples_seen"], 4)
            final = run_training(config, dataset, storage, run_name="resumed", resume=checkpoint)
            self.assertEqual(final["status"], "complete")
            left = torch.load(Path(full["checkpoint"]), weights_only=True)
            right = torch.load(checkpoint, weights_only=True)
            left.pop("run_name")
            right.pop("run_name")
            self.assertGreater(verify_resume.identical(left, right), 50)
            self.assertEqual(set(left["models"]), {"bc"})
            self.assertEqual(set(left["optimizers"]), {"bc"})
            self.assertEqual(final["progress"], {"bc": 6})
            export = storage.path("models/full/bc.pt")
            payload = torch.load(export, weights_only=True)
            for name in initial:
                if name.startswith(("v_head.", "aux_head.")):
                    self.assertTrue(torch.equal(initial[name], payload["state_dict"][name]), name)
            self.assertFalse(torch.equal(initial["a_head.weight"], payload["state_dict"]["a_head.weight"]))
            self.assertEqual(payload["kind"], "bc")
            self.assertTrue(storage.path("models/full/bc-agent.json").is_file())
            self.assertFalse(storage.path("models/full/grp.pt").exists())
            self.assertFalse(storage.path("models/full/ppo.pt").exists())
            agent = ModelAgent(export)
            observation = next(iter(new_environment(19).reset().values()))
            action = agent.act(observation, Context(observation.player_id, 19))
            self.assertIn(action.to_dict(), [value.to_dict() for value in observation.legal_actions()])
            rows = [json.loads(line) for line in storage.path("runs/full/metrics.jsonl").read_text().splitlines()]
            self.assertEqual([row["batch_samples"] for row in rows], [2, 2, 1, 2, 2, 1])
            self.assertEqual(rows[-1]["validation_samples"], 5)
            self.assertIn("validation_cross_entropy", rows[-1])

    def test_grp_cql_only_skips_ppo_and_matches_bc_initialization(self):
        config = copy.deepcopy(CONFIG)
        config.pop("ppo")
        config["stages"] = ["grp", "cql"]
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ"), patch(
                "agari_research.training.collect_rollout", side_effect=AssertionError("CQL must not run PPO")):
            storage = Storage(Path(directory))
            result = run_training(config, BehaviorDataset(), storage, run_name="cql")
            self.assertEqual(result["status"], "complete")
            self.assertEqual(set(result["progress"]), {"grp", "cql"})
            payload = torch.load(storage.path("models/cql/cql.pt"), weights_only=True)
            self.assertEqual(payload["policy_initialization"]["q_state_sha256"],
                             q_initialization_sha256(initialize_q(bc_config())))
            self.assertFalse(storage.path("models/cql/ppo.pt").exists())

    def test_wall_budget_saves_an_update_boundary_and_resumes_without_limit_in_contract(self):
        config, dataset = bc_config(), BehaviorDataset()
        clock = [0.0]
        def timed_loss(*args, **kwargs):
            result = bc_loss(*args, **kwargs)
            clock[0] = 3.0
            return result
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ"):
            storage = Storage(Path(directory))
            with patch("agari_research.training.time", SimpleNamespace(monotonic=lambda: clock[0])), patch(
                    "agari_research.training.bc_loss", side_effect=timed_loss):
                paused = run_training(config, dataset, storage, run_name="timed", max_wall_seconds=2)
            self.assertEqual((paused["status"], paused["stop_reason"], paused["progress"]),
                             ("paused", "wall_time_budget", {"bc": 1}))
            checkpoint = Path(paused["checkpoint"])
            state = torch.load(checkpoint, weights_only=True)
            self.assertNotIn("max_wall_seconds", state["config"])
            resumed = run_training(config, dataset, storage, run_name="timed", resume=checkpoint)
            complete = run_training(config, dataset, storage, run_name="untimed")
            self.assertEqual(resumed["status"], "complete")
            a = torch.load(checkpoint, weights_only=True)
            b = torch.load(Path(complete["checkpoint"]), weights_only=True)
            a.pop("run_name")
            b.pop("run_name")
            self.assertGreater(verify_resume.identical(a, b), 50)

    def test_elapsed_setup_budget_saves_before_first_update(self):
        ticks = iter([0., 2., 2.])
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ"), patch(
                "agari_research.training.time", SimpleNamespace(monotonic=lambda: next(ticks))), patch(
                "agari_research.training.bc_loss", side_effect=AssertionError("Budget expired before update")):
            result = run_training(bc_config(), BehaviorDataset(), Storage(Path(directory)),
                                  run_name="already-expired", max_wall_seconds=1)
            self.assertEqual(result["status"], "paused")
            self.assertEqual(result["progress"], {"bc": 0})
            self.assertTrue(Path(result["checkpoint"]).is_file())

    def test_resume_rejects_changed_selector_stage_and_budget(self):
        config, dataset = bc_config(), BehaviorDataset()
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ"):
            storage = Storage(Path(directory))
            result = run_training(config, dataset, storage, run_name="strict", stop_after_updates=1)
            checkpoint = Path(result["checkpoint"])
            changed = copy.deepcopy(config)
            changed["bc"]["epochs"] = 3
            with self.assertRaisesRegex(ValueError, "config"):
                run_training(changed, dataset, storage, run_name="strict", resume=checkpoint)
            changed = copy.deepcopy(CONFIG) | {"stages": ["grp", "cql"]}
            with self.assertRaisesRegex(ValueError, "config"):
                run_training(changed, dataset, storage, run_name="strict", resume=checkpoint)
            selection = {"agent_identity_sha256": ["a" * 64]}
            with self.assertRaisesRegex(ValueError, "policy_filter"):
                run_training(config | {"policy_filter": selection}, dataset, storage, run_name="strict", resume=checkpoint)
            dataset.policy_filter = selection
            with self.assertRaisesRegex(ValueError, "config"):
                run_training(config | {"policy_filter": selection}, dataset, storage, run_name="strict", resume=checkpoint)

    def test_best_checkpoint_can_precede_final_and_its_selection_survives_resume(self):
        config = bc_config()
        config["validation"]["interval_updates"] = 2
        def scripted_validation(values):
            values = iter(values)
            def validate(*args, **kwargs):
                metrics = validation_metrics(*args, **kwargs)
                metrics["validation_loss"] = next(values)
                return metrics
            return validate
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ"):
            storage = Storage(Path(directory))
            with patch("agari_research.training.validation_metrics", side_effect=scripted_validation([1., 3., 4.])):
                complete = run_training(config, BehaviorDataset(), storage, run_name="selection-full")
            with patch("agari_research.training.validation_metrics", side_effect=scripted_validation([1., 3.])):
                interrupted = run_training(config, BehaviorDataset(), storage, run_name="selection-resume", stop_after_updates=4)
            self.assertEqual(interrupted["best_validation"]["bc"]["update"], 2)
            alias = storage.path("models/selection-resume/best-bc.pt")
            alias.write_bytes(b"alias is derived; restore it from the checkpoint-bound snapshot")
            with patch("agari_research.training.validation_metrics", side_effect=scripted_validation([4.])):
                resumed = run_training(config, BehaviorDataset(), storage, run_name="selection-resume",
                                       resume=Path(interrupted["checkpoint"]))
            self.assertEqual(resumed["best_validation"]["bc"]["value"], 1.)
            self.assertEqual(file_sha256(alias), resumed["best_validation"]["bc"]["sha256"])
            best = torch.load(alias, weights_only=True)
            final = torch.load(storage.path("models/selection-resume/bc.pt"), weights_only=True)
            self.assertEqual((best["completed_updates"], final["completed_updates"]), (2, 6))
            self.assertEqual(best["validation_selection"], {"metric": "validation_loss", "value": 1., "update": 2})
            self.assertFalse(torch.equal(best["state_dict"]["a_head.weight"], final["state_dict"]["a_head.weight"]))
            self.assertTrue(storage.path("models/selection-resume/best-bc-agent.json").exists())
            left = torch.load(Path(complete["checkpoint"]), weights_only=True)
            right = torch.load(Path(resumed["checkpoint"]), weights_only=True)
            left.pop("run_name")
            right.pop("run_name")
            self.assertGreater(verify_resume.identical(left, right), 50)
            immutable = storage.path("models/selection-resume") / resumed["best_validation"]["bc"]["artifact"]
            immutable.write_bytes(b"changed committed validation model")
            with self.assertRaisesRegex(ValueError, "selected validation model"):
                run_training(config, BehaviorDataset(), storage, run_name="selection-resume", resume=Path(resumed["checkpoint"]))

    def test_cli_wall_budget_includes_dataset_preflight_and_rejects_invalid_limits(self):
        from agari_research.cli import main
        clock = [0.]
        def dataset(*args, **kwargs):
            clock[0] = 7.
            return BehaviorDataset()
        with tempfile.TemporaryDirectory() as directory:
            storage = Storage(Path(directory))
            path = storage.path("config.json")
            path.write_text(json.dumps(bc_config()))
            with patch("sys.argv", ["agari-train", "run", "--config", str(path), "--split", "split.json",
                                    "--run", "cli-budget", "--max-wall-seconds", "10"]), patch(
                    "agari_kit.storage.Storage.from_env", return_value=storage), patch(
                    "agari_research.cli.time", SimpleNamespace(monotonic=lambda: clock[0])), patch(
                    "agari_research.data.FrozenDataset", side_effect=dataset), patch(
                    "agari_research.training.run_training", return_value={"status": "complete"}) as train, contextlib.redirect_stdout(io.StringIO()):
                main()
            self.assertEqual(train.call_args.kwargs["max_wall_seconds"], 3.)
            for limit in (True, 0, -1, float("nan"), float("inf"), "10"):
                with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, "max_wall_seconds"):
                    run_training(bc_config(), BehaviorDataset(), storage, run_name="invalid-limit", max_wall_seconds=limit)

    def test_forced_accuracy_is_separate_from_choice_and_illegal_logits(self):
        rows = [(torch.tensor([9., 1., 999.]), torch.tensor(0), torch.tensor([True, True, False]), "DISCARD", False),
                (torch.tensor([9., 1., 999.]), torch.tensor(1), torch.tensor([True, True, False]), "PASS", False),
                (torch.tensor([9., 1., 999.]), torch.tensor(1), torch.tensor([False, True, False]), "PASS", True)]
        dataset = SimpleNamespace(policy_diagnostic_samples=lambda partition: iter(rows) if partition == "validation" else None)
        result = policy_diagnostics(torch.nn.Identity(), dataset, bc_config())
        self.assertEqual(result["choice"], {"samples": 2, "correct": 1, "accuracy": 0.5})
        self.assertEqual(result["forced"], {"samples": 1, "correct": 1, "accuracy": 1.})
        self.assertEqual(result["by_action_type"]["PASS"]["choice"]["accuracy"], 0.)
        self.assertEqual(result["by_action_type"]["PASS"]["forced"]["accuracy"], 1.)


class AgentIdentityTests(unittest.TestCase):
    def test_identity_ignores_locations_but_binds_configuration_weights_and_implementation(self):
        original = external_provenance()
        relocated = copy.deepcopy(original)
        relocated.update(name="@/new/config.json", descriptor_path="/new/config.json", descriptor_sha256="f" * 64)
        relocated["descriptor"]["artifacts"]["weights"]["path"] = "renamed.pt"
        relocated["artifacts"]["weights"].update(path="/new/renamed.pt", declared_path="renamed.pt")
        relocated["source"].update(module_path="/new/agent.py", roots=["/new"])
        self.assertEqual(agent_identity_sha256(original), agent_identity_sha256(relocated))
        for key in ("kwargs", "weights", "source"):
            changed = copy.deepcopy(original)
            if key == "kwargs":
                changed["descriptor"]["kwargs"]["temperature"] = 1
            elif key == "weights":
                changed["descriptor"]["artifacts"]["weights"]["sha256"] = "c" * 64
                changed["artifacts"]["weights"].update(sha256="c" * 64, declared_sha256="c" * 64)
            else:
                changed["source"]["source_sha256"] = "c" * 64
            self.assertNotEqual(agent_identity_sha256(original), agent_identity_sha256(changed))
        original["artifacts"]["weights"]["sha256"] = "c" * 64
        with self.assertRaises(DataError):
            agent_identity_sha256(original)

    def test_filter_rejects_empty_duplicate_or_unbound_seat_selection(self):
        for value in ({}, {"seats": [0]}, {"agent_identity_sha256": []},
                      {"agent_identity_sha256": ["a" * 64, "a" * 64]}, {"agent_identity_sha256": ["invalid"]}):
            with self.subTest(value=value), self.assertRaises(DataError):
                normalize_policy_filter(value)


class NativeIdentitySelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.storage = Storage(Path(cls.temporary.name))
        cls.records = [native_record(cls.storage, 610010, ["efficiency", "random", "efficiency", "random"]),
                       native_record(cls.storage, 610011, ["random", "efficiency", "random", "efficiency"])]

    def setUp(self):
        self.dataset = object.__new__(FrozenDataset)
        self.dataset.storage = self.storage
        self.dataset.sha256 = "a" * 64
        self.dataset.manifest = {"partitions": {"train": self.records, "validation": [], "test": []}}
        self.dataset.policy_filter = {"agent_identity_sha256": [agent_identity_sha256({"kind": "builtin", "name": "efficiency"})]}

    def test_rotating_teacher_seats_select_only_authenticated_native_actions(self):
        counts = self.dataset.sample_counts("train")
        grp = RankPredictor().eval()
        for index, record in enumerate(self.records):
            seats = [0, 2] if index == 0 else [1, 3]
            self.assertEqual(self.dataset.policy_players(record), seats)
            rounds = list(self.dataset.trace_game_rounds(record))
            expected = [decision for row in rounds for seat in seats for decision in row.players[seat]]
            bc = list(self.dataset.bc_game_samples(record))
            cql = list(self.dataset.cql_game_samples(record, grp, CONFIG))
            self.assertEqual(len(bc), len(expected))
            self.assertEqual(len(cql), len(bc))
            self.assertEqual(counts[index]["cql_samples"], len(bc))
            self.assertEqual(counts[index]["grp_samples"], len(rounds) * 4)
            for decision, a, b in zip(expected, bc, cql):
                self.assertTrue(torch.equal(a[0], policy_features(decision.observation)))
                self.assertEqual(a[1].item(), decision.action.encode())
                self.assertTrue(a[2][a[1]])
                self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b[:3])))
        plan = build_training_plan(bc_config() | {"policy_filter": self.dataset.policy_filter}, self.dataset)
        self.assertEqual(plan["policy_filter"], self.dataset.policy_filter)
        self.assertEqual(plan["stages"]["bc"]["target_samples"], 2 * sum(row["cql_samples"] for row in counts))

    def test_missing_identity_and_inconsistent_recorded_provenance_are_rejected(self):
        self.dataset.policy_filter = {"agent_identity_sha256": ["f" * 64]}
        with self.assertRaisesRegex(DataError, "No requested policy identity"):
            next(self.dataset.bc_game_samples(self.records[0]))
        self.setUp()
        original = self.records[0]
        with gzip.open(self.storage.path(original["trace_path"]), "rt") as stream:
            rows = [json.loads(line) for line in stream]
        rows[0]["agent_provenance"][0]["name"] = "random"
        changed = self.storage.path("changed-provenance.trace.jsonl")
        changed.write_text("".join(json.dumps(row) + "\n" for row in rows))
        receipt = json.loads(self.storage.path(original["validation_receipt"]).read_text())
        receipt["trace_sha256"] = file_sha256(changed)
        receipt_path = self.storage.path("changed-provenance.receipt.json")
        write_json(receipt_path, receipt)
        record = dict(original, trace_path=changed.name, validation_receipt=receipt_path.name,
                      validation_receipt_sha256=file_sha256(receipt_path))
        with self.assertRaisesRegex(DataError, "names disagree"):
            next(self.dataset.bc_game_samples(record))

    def test_validation_separates_choice_forced_and_native_action_types(self):
        self.dataset.policy_filter = {"agent_identity_sha256": [agent_identity_sha256({"kind": "builtin", "name": "random"})]}
        self.dataset.manifest["partitions"]["validation"] = self.records
        config = bc_config() | {"policy_filter": self.dataset.policy_filter}
        model = initialize_q(config, behavior_cloning=True).eval()
        result = validation_metrics("bc", self.dataset, {"bc": model}, config)
        diagnostics = result["validation_policy_diagnostics"]
        self.assertEqual(diagnostics["choice"]["samples"], result["validation_samples"])
        self.assertEqual(diagnostics["forced"]["accuracy"],
                         1. if diagnostics["forced"]["samples"] else None)
        self.assertGreater(diagnostics["by_action_type"]["CHI"]["choice"]["samples"], 0)
        self.assertGreater(diagnostics["by_action_type"]["PON"]["choice"]["samples"], 0)
        for group in ("choice", "forced"):
            self.assertEqual(sum(row[group]["samples"] for row in diagnostics["by_action_type"].values()),
                             diagnostics[group]["samples"])
        with self.assertRaises(DataError):
            next(self.dataset.policy_diagnostic_samples("test"))


if __name__ == "__main__":
    unittest.main()
