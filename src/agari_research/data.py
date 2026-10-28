"""Verified whole-game manifests and bounded upstream-compatible sample streams."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from pathlib import Path

import torch
from agari_kit.collection import verify_record
from agari_kit.dataset import make_split
from agari_kit.engine import require_engine
from agari_kit.mjai import DataError, canonical_json, read_events
from agari_kit.storage import Storage
from riichienv import MjaiReplay
from agari_kit.validation import _trace_records

from .features import action_mask, grp_features, policy_features, predicted_rewards, rank_of
from .trace_samples import trace_rounds
from .selection import agent_identity_sha256, normalize_policy_filter


def require_trace_receipt(record: dict, receipt: dict) -> None:
    """Only content-bound, verified local traces define behavior action labels."""
    producer = receipt.get("producer_code")
    if (record.get("source") != "agarilab-selfplay" or not record.get("trace_path")
            or receipt.get("method") != "seeded-action-trace-v1"
            or not isinstance(producer, dict)):
        raise DataError("CQL requires locally generated self-play with a verified seeded trace")
    hashes = (receipt.get("engine_binary_sha256"), producer.get("source_sha256"))
    if (receipt.get("engine_version") != require_engine() or producer.get("package") != "agari-kit"
            or not isinstance(producer.get("version"), str) or not producer["version"]
            or any(not isinstance(value, str) or len(value) != 64
                   or any(c not in "0123456789abcdef" for c in value) for value in hashes)):
        raise DataError("Seeded training trace has invalid producer or engine identity")
    # Producer binaries differ across OS/architectures. Compatibility is proved
    # by full replay in FrozenDataset.__init__, not binary equality. The current
    # training runtime's binary remains strictly bound to checkpoint resumes.


class FrozenDataset:
    def __init__(self, storage: Storage, split_path: Path, *, policy_filter=None):
        self.storage = storage
        self.policy_filter = normalize_policy_filter(policy_filter)
        self.manifest = json.loads(split_path.read_text())
        body = {key: value for key, value in self.manifest.items() if key != "dataset_sha256"}
        digest = hashlib.sha256(canonical_json(body).encode()).hexdigest()
        if digest != self.manifest.get("dataset_sha256"):
            raise DataError("Frozen split digest does not match its contents")
        records = [record for part in ("train", "validation", "test")
                   for record in self.manifest["partitions"][part]]
        if make_split(records) != self.manifest:
            raise DataError("Split must use the verified chronological whole-game 90/5/5 policy")
        # All three partitions are checked for duplicate IDs and metadata above.
        # Held-out test content is never opened by training or model selection.
        producer_engines = set()
        for part in ("train", "validation"):
            for record in self.manifest["partitions"][part]:
                receipt = verify_record(storage, record, replay=True)
                require_trace_receipt(record, receipt)
                if self.policy_filter is not None:
                    self.policy_players(record, verified=True)
                producer_engines.add((receipt["engine_version"], receipt["engine_binary_sha256"]))
        self.producer_engines = [{"version": version, "binary_sha256": digest}
                                 for version, digest in sorted(producer_engines)]
        self.sha256 = digest

    def policy_players(self, record: dict, *, verified: bool = False) -> list[int]:
        """Select by this game's recorded content identity, never by seat number."""
        selection = getattr(self, "policy_filter", None)
        if selection is None:
            return list(range(4))
        if not verified:
            require_trace_receipt(record, verify_record(self.storage, record, replay=False))
        records = iter(_trace_records(self.storage.path(record["trace_path"])))
        try:
            _, header = next(records)
        except StopIteration as exc:
            raise DataError("Policy selection requires a seeded trace header") from exc
        finally:
            records.close()
        provenance, agents = header.get("agent_provenance"), header.get("agents")
        if (not isinstance(provenance, list) or len(provenance) != 4
                or not isinstance(agents, list) or len(agents) != 4):
            raise DataError("Policy selection requires provenance for all four trace players")
        selected = []
        for seat, item in enumerate(provenance):
            if not isinstance(item, dict) or item.get("name") != agents[seat]:
                raise DataError("Trace agent names disagree with their provenance")
            if agent_identity_sha256(item) in selection["agent_identity_sha256"]:
                selected.append(seat)
        if not selected:
            raise DataError(f"No requested policy identity occurs in game {record['game_id']}")
        return selected

    def records(self, partition: str, seed: int | None = None):
        if partition not in ("train", "validation"):
            raise DataError("The training reader cannot open the held-out test partition")
        records = list(self.manifest["partitions"][partition])
        if seed is not None:
            random.Random(seed).shuffle(records)
        return records

    def grp_samples(self, partition: str, seed: int | None = None):
        for record in self.records(partition, seed):
            yield from self.grp_game_samples(record)

    def grp_game_samples(self, record: dict):
        path = self.storage.path(record["mjai_path"])
        final_scores = None
        for event in read_events(path):
            if event["type"] == "end_game":
                final_scores = event.get("scores")
        if not isinstance(final_scores, list) or len(final_scores) != 4:
            raise DataError("GRP requires verified terminal four-player scores")
        replay = MjaiReplay.from_jsonl(str(path), rule="tenhou")
        for kyoku in replay.take_kyokus():
            features = kyoku.take_grp_features()
            for player in range(4):
                yield grp_features(features, player), torch.tensor(rank_of(final_scores, player))

    def cql_samples(self, partition: str, grp, config: dict, seed: int | None = None):
        grp.eval()
        for record in self.records(partition, seed):
            yield from self.cql_game_samples(record, grp, config)

    def cql_game_samples(self, record: dict, grp, config: dict):
        grp.eval()
        players = self.policy_players(record)
        for round_samples in self.trace_game_rounds(record):
            features = round_samples.features
            rewards = predicted_rewards(grp, features, config["pts_weight"], config["device"])
            for player in players:
                steps = round_samples.players[player]
                rank = rank_of(features["round_end_scores"], player)
                for index, decision in enumerate(steps):
                    observation, action = decision.observation, decision.action
                    mask = action_mask(observation)
                    action_id = action.encode()
                    if not 0 <= action_id < 82 or not mask[action_id]:
                        raise DataError("Replay action does not match the native legal mask")
                    target = rewards[player] * config["cql"]["gamma"] ** (len(steps) - index - 1)
                    yield (policy_features(observation), torch.tensor(action_id), mask,
                           torch.tensor(target, dtype=torch.float32), torch.tensor(rank))

    def bc_samples(self, partition: str, seed: int | None = None):
        for record in self.records(partition, seed):
            yield from self.bc_game_samples(record)

    def bc_game_samples(self, record: dict):
        players = self.policy_players(record)
        for round_samples in self.trace_game_rounds(record):
            for player in players:
                for decision in round_samples.players[player]:
                    mask = action_mask(decision.observation)
                    action_id = decision.action.encode()
                    if not 0 <= action_id < 82 or not mask[action_id]:
                        raise DataError("Replay action does not match the native legal mask")
                    yield policy_features(decision.observation), torch.tensor(action_id), mask

    def policy_diagnostic_samples(self, partition: str):
        """Validation-only labels retain forced actions as a separate group."""
        if partition != "validation":
            raise DataError("Policy diagnostics only read the validation partition")
        for record in self.records(partition):
            players = self.policy_players(record)
            for round_samples in self.trace_game_rounds(record, include_forced=True):
                for player in players:
                    for decision in round_samples.players[player]:
                        mask = action_mask(decision.observation)
                        action_id = decision.action.encode()
                        if not 0 <= action_id < 82 or not mask[action_id]:
                            raise DataError("Diagnostic action does not match the native legal mask")
                        yield (policy_features(decision.observation), torch.tensor(action_id), mask,
                               str(decision.action.action_type).split(".")[-1],
                               len(decision.observation.legal_actions()) == 1)

    def trace_game_rounds(self, record: dict, *, include_forced: bool = False):
        # Recheck the content-bound receipt on every pass, including resumes;
        # complete native/MJAI verification was performed at dataset opening.
        receipt = verify_record(self.storage, record, replay=False)
        require_trace_receipt(record, receipt)
        yield from trace_rounds(self.storage.path(record["mjai_path"]),
                                self.storage.path(record["trace_path"]),
                                producer=receipt["producer_code"], include_forced=include_forced)

    def sample_counts(self, partition: str) -> list[dict]:
        """Count the same native samples as training, never trace decision totals."""
        counts = []
        for record in self.records(partition):
            grp_count = cql_count = 0
            players = self.policy_players(record)
            for round_samples in self.trace_game_rounds(record):
                grp_count += 4
                cql_count += sum(len(round_samples.players[player]) for player in players)
            if not grp_count or not cql_count:
                raise DataError(f"Complete game produced no training samples: {record['game_id']}")
            counts.append({"game_id": record["game_id"], "sha256": record["sha256"],
                           "grp_samples": grp_count, "cql_samples": cql_count})
        return counts


class BatchStream:
    """Repeat deterministic epoch orders; saved sample cursor resumes exactly."""
    def __init__(self, factory, batch_size: int, seen: int = 0, seed: int = 0):
        self.factory, self.batch_size, self.seed = factory, batch_size, seed
        self.seen = seen
        self.iterator = self._samples()
        for _ in itertools.islice(self.iterator, seen):
            pass

    def _samples(self):
        for epoch in itertools.count():
            count = 0
            for sample in self.factory(self.seed + epoch):
                count += 1
                yield sample
            if not count:
                raise DataError("Training partition produced no usable samples")

    def next(self, device: str):
        samples = [next(self.iterator) for _ in range(self.batch_size)]
        self.seen += len(samples)
        return [torch.stack(values).to(device) for values in zip(*samples)]


class EpochBatchStream:
    """Finite complete epochs, exact per-game counts and no padding or dropped tail."""
    def __init__(self, records, counts, factory, batch_size: int, epochs: int,
                 seen: int = 0, seed: int = 0):
        if not records or batch_size < 1 or epochs < 1:
            raise DataError("Epoch streams require games, positive batch size and epochs")
        self.records, self.counts, self.factory = list(records), counts, factory
        if len({record["game_id"] for record in records}) != len(records):
            raise DataError("Duplicate games in epoch stream")
        if set(counts) != {record["game_id"] for record in records} or any(
                type(value) is not int or value < 1 for value in counts.values()):
            raise DataError("Epoch stream requires one positive sample count per game")
        self.batch_size, self.epochs, self.seed = batch_size, epochs, seed
        self.per_epoch = sum(counts.values())
        if not 0 <= seen <= self.per_epoch * epochs:
            raise DataError("Epoch sample cursor is out of range")
        self.seen = seen
        self.epoch, offset = divmod(seen, self.per_epoch)
        self.ordered = self._order(self.epoch)
        self.game_index = self.game_seen = 0
        self.iterator = None
        # Resume directly at the current game. Completed epochs and complete
        # games need no feature reconstruction or reward inference.
        while offset:
            count = self.counts[self.ordered[self.game_index]["game_id"]]
            if offset < count:
                self.game_seen = offset
                break
            offset -= count
            self.game_index += 1

    def _order(self, epoch):
        records = list(self.records)
        random.Random(self.seed + epoch).shuffle(records)
        return records

    def _sample(self):
        record = self.ordered[self.game_index]
        count = self.counts[record["game_id"]]
        if self.iterator is None:
            self.iterator = iter(self.factory(record))
            skipped = sum(1 for _ in itertools.islice(self.iterator, self.game_seen))
            if skipped != self.game_seen:
                raise DataError("Game ended before its saved sample cursor")
        marker = object()
        sample = next(self.iterator, marker)
        if sample is marker:
            raise DataError(f"Fewer samples than planned: {record['game_id']}")
        self.game_seen += 1
        self.seen += 1
        if self.game_seen == count:
            if next(self.iterator, marker) is not marker:
                raise DataError(f"More samples than planned: {record['game_id']}")
            self.game_index += 1
            self.game_seen, self.iterator = 0, None
            if self.game_index == len(self.ordered):
                self.epoch += 1
                self.game_index = 0
                self.ordered = self._order(self.epoch)
        return sample

    def next(self, device: str):
        if self.epoch >= self.epochs:
            raise StopIteration
        count = min(self.batch_size, self.per_epoch - self.seen % self.per_epoch)
        samples = [self._sample() for _ in range(count)]
        return [torch.stack(values).to(device) for values in zip(*samples)]
