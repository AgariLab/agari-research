"""Real native games expose labels unavailable in executed MJAI events."""

import copy
import gzip
import json
import tempfile
import unittest
from pathlib import Path

import torch
from agari_kit.collection import file_sha256, record_game, verify_record, write_json
from agari_kit.dataset import make_split
from agari_kit.mjai import DataError
from agari_kit.storage import Storage
from agari_kit.validation import validate_mjai
from riichienv import ActionType, MjaiReplay

from agari_research.data import EpochBatchStream, FrozenDataset
from agari_research.features import policy_features, predicted_rewards, rank_of
from agari_research.models.grp_model import RankPredictor
from agari_research.planning import build_training_plan
from agari_research.trace_samples import SAMPLE_SEMANTICS_VERSION

CONFIG = json.loads((Path(__file__).parents[1] / "configs/smoke.json").read_text())


def native_record(storage, seed, agents):
    paths = {name: storage.path(f"{seed}.{name}.jsonl.gz") for name in ("raw", "mjai", "trace")}
    record_game(seed=seed, agents=agents, raw_path=paths["raw"], mjai_path=paths["mjai"], trace_path=paths["trace"])
    receipt = validate_mjai(paths["mjai"], trace_path=paths["trace"])
    receipt_path = storage.path(f"{seed}.verification.json")
    write_json(receipt_path, receipt)
    return {"game_id": "selfplay:" + receipt["sha256"], "sha256": receipt["sha256"],
            "source": "agarilab-selfplay", "played_at": f"2026-01-01T00:00:{seed % 60:02d}+00:00",
            "rules": "4p-red-half/tenhou", "usage_basis": "Generated native regression fixture",
            "validation": "engine-verified", "mjai_path": paths["mjai"].name,
            "trace_path": paths["trace"].name, "validation_receipt": receipt_path.name,
            "validation_receipt_sha256": file_sha256(receipt_path)}


class TraceLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.storage = Storage(Path(cls.temporary.name))
        cls.records = {}
        for seed in (610007, 610010):
            record = native_record(cls.storage, seed, ["efficiency", "random", "efficiency", "random"])
            verify_record(cls.storage, record, replay=True)
            cls.records[seed] = record
        # Two authenticated game fixtures exercise the reader without inventing
        # eighteen additional games to satisfy a 20-game chronological split.
        cls.dataset = object.__new__(FrozenDataset)
        cls.dataset.storage = cls.storage
        cls.dataset.sha256 = "a" * 64
        cls.dataset.manifest = {"partitions": {"train": list(cls.records.values()), "validation": [], "test": []}}
        cls.rounds = {seed: list(cls.dataset.trace_game_rounds(record)) for seed, record in cls.records.items()}
        cls.grp = RankPredictor().eval()

    def test_overridden_chi_stays_chi_with_identical_player_features(self):
        decision = next(d for d in self.rounds[610010][0].players[1] if d.step == 17)
        self.assertEqual(decision.action.action_type, ActionType.CHI)
        self.assertEqual(decision.action.encode(), 39)
        self.assertEqual(decision.action.to_dict()["consume_tiles"], [4, 14])
        winner = next(d for d in self.rounds[610010][0].players[3] if d.step == 17)
        self.assertEqual(winner.action.action_type, ActionType.PON)
        record = self.records[610010]
        old_round = next(iter(MjaiReplay.from_jsonl(str(self.storage.path(record["mjai_path"])), rule="tenhou").take_kyokus()))
        old = [(o, a) for o, a in old_round.steps(1) if o.encode() == decision.observation.encode()]
        self.assertEqual(len(old), 1)
        self.assertEqual(old[0][1].action_type, ActionType.PASS)
        flat = [d for r in self.rounds[610010] for ds in r.players for d in ds]
        index = next(i for i, d in enumerate(flat) if d.step == 17 and d.observation.player_id == 1)
        samples = list(self.dataset.cql_game_samples(record, self.grp, CONFIG))
        self.assertEqual(samples[index][1].item(), 39)
        self.assertTrue(torch.equal(samples[index][0], policy_features(decision.observation)))

    def test_both_submitted_ron_decisions_are_samples(self):
        round_samples = self.rounds[610007][0]
        winners = [(seat, d) for seat, ds in enumerate(round_samples.players) for d in ds if d.step == 71]
        self.assertEqual([seat for seat, _ in winners], [0, 2])
        for _, decision in winners:
            self.assertEqual(decision.action.action_type, ActionType.RON)
            self.assertEqual(decision.action.encode(), 79)
            self.assertGreater(len(decision.observation.legal_actions()), 1)
        old = next(iter(MjaiReplay.from_jsonl(str(self.storage.path(self.records[610007]["mjai_path"])), rule="tenhou").take_kyokus()))
        self.assertEqual(sum(a.action_type == ActionType.RON for _, a in old.steps(2)), 0)

    def test_counts_forced_filter_rounds_rewards_and_terminal_deposits(self):
        counts = self.dataset.sample_counts("train")
        config = copy.deepcopy(CONFIG)
        config["cql"]["gamma"] = 0.9
        for row, (seed, record) in zip(counts, self.records.items()):
            rounds = self.rounds[seed]
            samples = list(self.dataset.cql_game_samples(record, self.grp, config))
            self.assertEqual(row["grp_samples"], len(rounds) * 4)
            self.assertEqual(row["cql_samples"], len(samples))
            self.assertEqual(len(samples), sum(len(ds) for r in rounds for ds in r.players))
            self.assertEqual(len(samples), {610007: 177, 610010: 663}[seed])
            cursor = 0
            for r in rounds:
                rewards = predicted_rewards(self.grp, r.features, config["pts_weight"], config["device"])
                for seat, decisions in enumerate(r.players):
                    for offset, decision in enumerate(decisions):
                        sample = samples[cursor]
                        self.assertGreater(len(decision.observation.legal_actions()), 1)
                        self.assertEqual(sample[1].item(), decision.action.encode())
                        self.assertTrue(sample[2][sample[1]])
                        self.assertTrue(torch.isfinite(sample[0]).all())
                        self.assertAlmostEqual(sample[3].item(), rewards[seat] * 0.9 ** (len(decisions) - offset - 1), places=6)
                        self.assertEqual(sample[4].item(), rank_of(r.features["round_end_scores"], seat))
                        cursor += 1
        # End-game payout awards the two unclaimed deposits to the match leader;
        # GRP preserves the last round's pre-payout settlement, not final scores.
        final = self.rounds[610010][-1].features["round_end_scores"]
        receipt = json.loads(self.storage.path(self.records[610010]["validation_receipt"]).read_text())
        self.assertEqual(final, [35300, -700, 50600, 12800])
        self.assertEqual(receipt["scores"], [35300, -700, 52600, 12800])

    def test_native_single_sample_tail_and_epoch_resume_match(self):
        records = list(self.records.values())
        counts = {row["game_id"]: row["cql_samples"] for row in self.dataset.sample_counts("train")}
        factory = lambda record: self.dataset.cql_game_samples(record, self.grp, CONFIG)
        batch = sum(counts.values()) - 1
        complete = EpochBatchStream(records, counts, factory, batch, 2, seed=2026)
        expected = [complete.next("cpu") for _ in range(4)]
        self.assertEqual([len(row[0]) for row in expected], [839, 1, 839, 1])
        for update, seen in ((1, 839), (2, 840), (3, 1679)):
            resumed = EpochBatchStream(records, counts, factory, batch, 2, seen=seen, seed=2026)
            for target in expected[update:]:
                actual = resumed.next("cpu")
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, target)))
            with self.assertRaises(StopIteration):
                resumed.next("cpu")
        plan = build_training_plan(CONFIG, self.dataset)
        self.assertEqual(plan["sample_semantics_version"], SAMPLE_SEMANTICS_VERSION)

    def test_changed_trace_and_non_selfplay_source_are_rejected(self):
        record = dict(self.records[610010], source="external-mjai")
        with self.assertRaisesRegex(DataError, "locally generated"):
            next(self.dataset.trace_game_rounds(record))
        record = dict(self.records[610010], trace_path="changed.trace.jsonl.gz")
        self.storage.path(record["trace_path"]).write_bytes(self.storage.path(self.records[610010]["trace_path"]).read_bytes() + b"changed")
        with self.assertRaises(DataError):
            next(self.dataset.trace_game_rounds(record))
        with self.assertRaises(DataError):
            self.dataset.sample_counts("test")

    def test_foreign_producer_binary_needs_real_native_replay_compatibility(self):
        # Build a real 90/5/5 manifest: 19 fully generated and replayed games,
        # plus an unopened test record whose content deliberately does not exist.
        records = [dict(record) for record in self.records.values()]
        records.extend(native_record(self.storage, 710000 + index, ["tsumogiri"] * 4) for index in range(17))
        test = dict(records[-1], game_id="selfplay:" + "f" * 64, sha256="f" * 64,
                    mjai_path="unopened-test.jsonl", trace_path="unopened-test.trace.jsonl",
                    validation_receipt="unopened-test.verification.json")
        records.append(test)
        for index, record in enumerate(records):
            record["played_at"] = f"2026-01-01T00:00:{index:02d}+00:00"
        producer_receipt = json.loads(self.storage.path(records[0]["validation_receipt"]).read_text())
        producer_receipt["engine_binary_sha256"] = "a" * 64
        foreign = self.storage.path("foreign-producer.verification.json")
        write_json(foreign, producer_receipt)
        records[0].update(validation_receipt=foreign.name, validation_receipt_sha256=file_sha256(foreign))
        split = self.storage.path("migration-split.json")
        write_json(split, make_split(records))
        dataset = FrozenDataset(self.storage, split)  # No native validation mocks.
        self.assertIn({"version": "0.4.10", "binary_sha256": "a" * 64}, dataset.producer_engines)
        self.assertEqual(build_training_plan(CONFIG, dataset)["producer_engines"], dataset.producer_engines)
        self.assertGreater(len(list(dataset.trace_game_rounds(records[0]))), 0)

        # Matching file hashes cannot legitimize an Observation that differs
        # on the training platform: complete replay must reject the manifest.
        with gzip.open(self.storage.path(records[0]["trace_path"]), "rt") as stream:
            trace = [json.loads(line) for line in stream]
        seat = next(iter(trace[1]["observations_sha256"]))
        trace[1]["observations_sha256"][seat] = "0" * 64
        changed_trace = self.storage.path("incompatible.trace.jsonl")
        changed_trace.write_text("".join(json.dumps(row) + "\n" for row in trace))
        producer_receipt["trace_sha256"] = file_sha256(changed_trace)
        incompatible = self.storage.path("incompatible.verification.json")
        write_json(incompatible, producer_receipt)
        records[0].update(trace_path=changed_trace.name, validation_receipt=incompatible.name,
                          validation_receipt_sha256=file_sha256(incompatible))
        bad_split = self.storage.path("incompatible-split.json")
        write_json(bad_split, make_split(records))
        with self.assertRaisesRegex(DataError, "Observation hash mismatch"):
            FrozenDataset(self.storage, bad_split)


if __name__ == "__main__":
    unittest.main()
