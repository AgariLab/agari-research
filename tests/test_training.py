import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from agari_kit.agent import Context
from agari_kit.engine import new_environment
from agari_kit.collection import file_sha256, verify_record
from agari_kit.dataset import make_split
from agari_kit.mjai import DataError, canonical_json, inspect_game
from agari_kit.storage import Storage
from agari_research.data import FrozenDataset
from agari_research.features import ENCODING_VERSION, UPSTREAM_COMMIT
from agari_research.losses import cql_loss, gae, ppo_loss
from agari_research.model_agent import ModelAgent
from agari_research.models.actor_critic import ActorCriticNetwork
from agari_research.models.q_network import QNetwork
from agari_research.runtime import require_matching_runtime, runtime_provenance
from agari_research.training import atomic_torch_save, initialize_policy, restore_rng, rng_state, run_training

CONFIG = json.loads((Path(__file__).parents[1] / "configs/smoke.json").read_text())


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_cql_penalizes_unobserved_legal_actions_and_ignores_illegal(self):
        q = torch.tensor([[1.0, 3.0, 999.0]], requires_grad=True)
        aux = torch.zeros(1, 4, requires_grad=True)
        loss, terms = cql_loss(q, aux, torch.tensor([0]), torch.tensor([[True, True, False]]),
                               torch.tensor([2.0]), torch.tensor([1]), alpha=5.0, aux_weight=0.2)
        expected = 1 + 5 * math.log1p(math.exp(2)) + 0.2 * math.log(4)
        self.assertAlmostEqual(loss.item(), expected, places=5)
        loss.backward()
        self.assertGreater(q.grad[0, 1].item(), 0)
        self.assertEqual(q.grad[0, 2].item(), 0)
        self.assertAlmostEqual(terms["td_loss"].item(), 1)

    def test_self_consistent_forged_receipt_is_replayed_and_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            storage = Storage(Path(directory))
            records = []
            for index in range(20):
                # Structural inspection accepts these shapes; 52 identical tiles
                # and an empty trace can never pass the actual game engine.
                events = [{"type": "start_game", "id": index},
                          {"type": "start_kyoku", "tehais": [["1m"] * 13 for _ in range(4)]},
                          {"type": "end_kyoku"}, {"type": "end_game", "scores": [25000] * 4}]
                mjai = storage.path(f"{index}.jsonl")
                trace = storage.path(f"{index}.trace.jsonl")
                receipt_path = storage.path(f"{index}.receipt.json")
                mjai.write_text("".join(canonical_json(event) + "\n" for event in events))
                trace.write_text('{"type":"agari_trace"}\n')
                receipt = dict(inspect_game(events), validation="engine-verified", engine_version="0.4.10",
                               mjai_file_sha256=file_sha256(mjai), trace_sha256=file_sha256(trace))
                receipt_path.write_text(canonical_json(receipt))
                record = {"game_id": str(index), "sha256": receipt["sha256"],
                          "played_at": f"2026-01-01T00:00:{index:02d}+00:00", "source": "forged-test",
                          "rules": "4p-red-half/tenhou", "usage_basis": "test", "validation": "engine-verified",
                          "mjai_path": mjai.name, "trace_path": trace.name,
                          "validation_receipt": receipt_path.name,
                          "validation_receipt_sha256": file_sha256(receipt_path)}
                # Every claimed hash matches, so integrity checks alone pass.
                verify_record(storage, record, replay=False)
                records.append(record)
            split = storage.path("split.json")
            split.write_text(canonical_json(make_split(records)))
            with self.assertRaises(DataError):
                FrozenDataset(storage, split)

    def test_cql_rejects_illegal_recorded_action(self):
        with self.assertRaises(ValueError):
            cql_loss(torch.zeros(1, 2), torch.zeros(1, 4), torch.tensor([1]),
                     torch.tensor([[True, False]]), torch.zeros(1), torch.zeros(1, dtype=torch.long))

    def test_gae_stops_at_each_complete_hand(self):
        advantages, returns = gae([0.2, 0.4], 1.0, 1.0, 0.95)
        self.assertAlmostEqual(advantages[1], 0.6)
        self.assertAlmostEqual(advantages[0], 0.77)
        self.assertAlmostEqual(returns[0], 0.97)
        # A new hand starts independently, even if the dealer is unchanged.
        next_advantages, next_returns = gae([0.4], -1.0, 1.0, 0.95)
        self.assertAlmostEqual(next_advantages[0], -1.4)
        self.assertAlmostEqual(next_returns[0], -1.0)

    def test_ppo_clipping_and_frozen_reference_kl(self):
        logits = torch.tensor([[0.0, 0.0, 1000.0]], requires_grad=True)
        reference = torch.tensor([[0.0, 0.0, -1000.0]])
        loss, terms = ppo_loss(logits, torch.tensor([0.0]), reference,
                               torch.tensor([[True, True, False]]), torch.tensor([0]),
                               torch.tensor([math.log(0.25)]), torch.tensor([1.0]), torch.tensor([0.0]),
                               clip=0.02, value_coef=0.5, entropy_coef=0.0, kl_coef=0.1)
        self.assertAlmostEqual(loss.item(), -1.02, places=6)
        self.assertAlmostEqual(terms["kl"].item(), 0)
        self.assertAlmostEqual(terms["entropy"].item(), math.log(2), places=6)
        loss.backward()
        self.assertEqual(logits.grad[0, 2].item(), 0)

    def test_q_to_actor_transfer_preserves_legal_action_order(self):
        q = QNetwork(**CONFIG["model"]).eval()
        policy = ActorCriticNetwork(**CONFIG["model"]).eval()
        reference = ActorCriticNetwork(**CONFIG["model"]).eval()
        initialize_policy(q, policy, reference)
        x = torch.randn(4, 74, 34)
        with torch.no_grad():
            self.assertTrue(torch.equal(q(x).argmax(1), policy(x)[0].argmax(1)))
            self.assertTrue(torch.equal(policy(x)[0], reference(x)[0]))
        self.assertFalse(any(parameter.requires_grad for parameter in reference.parameters()))

    def test_rng_checkpoint_is_safe_loadable_and_restores(self):
        import random
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rng.pt"
            atomic_torch_save(rng_state(), path)
            expected = (random.random(), np.random.random(), torch.rand(3))
            restored = torch.load(path, weights_only=True)
            restore_rng(restored)
            self.assertEqual(random.random(), expected[0])
            self.assertEqual(np.random.random(), expected[1])
            self.assertTrue(torch.equal(torch.rand(3), expected[2]))

    def test_resume_rejects_each_changed_runtime_component(self):
        current = runtime_provenance()
        paths = [("agari_kit", "source_sha256"), ("agari_kit", "version"),
                 ("engine", "binary_sha256"), ("engine", "version"),
                 ("python", "implementation"), ("python", "version"),
                 ("torch",), ("numpy",), ("lockfile_sha256",)]
        require_matching_runtime(copy.deepcopy(current), current)
        for path in paths:
            with self.subTest(component=path):
                changed = copy.deepcopy(current)
                target = changed
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = "changed-runtime-component"
                with self.assertRaisesRegex(ValueError, "runtime provenance mismatch"):
                    require_matching_runtime(changed, current)
        with self.assertRaisesRegex(ValueError, "runtime provenance mismatch"):
            require_matching_runtime(None, current)

    def test_training_rejects_runtime_before_loading_model_state(self):
        with tempfile.TemporaryDirectory() as directory:
            storage = Storage(Path(directory))
            changed = runtime_provenance()
            changed["torch"] = "a-different-runtime"
            checkpoint = storage.path("changed.pt")
            # Deliberately omit model/optimizer tensors: the runtime guard must
            # reject first, before trying to load or execute checkpoint models.
            atomic_torch_save({"runtime_provenance": changed}, checkpoint)
            dataset = SimpleNamespace(sha256="a" * 64)
            with self.assertRaisesRegex(ValueError, "runtime provenance mismatch: torch"):
                run_training(CONFIG, dataset, storage, run_name="runtime-rejection", resume=checkpoint)
            self.assertFalse(storage.path("checkpoints/runtime-rejection/latest.pt").exists())

    def test_exported_agent_masks_native_illegal_actions(self):
        model = QNetwork(**CONFIG["model"])
        payload = {"schema_version": 1, "encoding_version": ENCODING_VERSION,
                   "upstream_commit": UPSTREAM_COMMIT, "engine_version": "0.4.10", "kind": "cql",
                   "model_config": CONFIG["model"], "state_dict": model.state_dict()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            atomic_torch_save(payload, path)
            agent = ModelAgent(path)
            env = new_environment(2026)
            observations = env.reset()
            for _ in range(30):
                selected = {}
                for seat, observation in observations.items():
                    context = Context(seat, 2026)
                    agent.reset(context)
                    action = agent.act(observation, context)
                    self.assertIn(action.to_dict(), [legal.to_dict() for legal in observation.legal_actions()])
                    selected[seat] = action
                observations = env.step(selected)


if __name__ == "__main__":
    unittest.main()
