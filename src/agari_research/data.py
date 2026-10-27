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
from agari_kit.mjai import DataError, canonical_json, read_events
from agari_kit.storage import Storage
from riichienv import MjaiReplay

from .features import action_mask, grp_features, policy_features, predicted_rewards, rank_of


class FrozenDataset:
    def __init__(self, storage: Storage, split_path: Path):
        self.storage = storage
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
        for part in ("train", "validation"):
            for record in self.manifest["partitions"][part]:
                receipt = verify_record(storage, record, replay=True)
                if receipt.get("engine_version") != "0.4.10":
                    raise DataError("Training receipt uses an unpinned engine version")
        self.sha256 = digest

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
        replay = MjaiReplay.from_jsonl(str(self.storage.path(record["mjai_path"])), rule="tenhou")
        for kyoku in replay.take_kyokus():
            features = kyoku.take_grp_features()
            rewards = predicted_rewards(grp, features, config["pts_weight"], config["device"])
            for player in range(4):
                steps = list(kyoku.steps(player))
                rank = rank_of(features["round_end_scores"], player)
                for index, (observation, action) in enumerate(steps):
                    mask = action_mask(observation)
                    action_id = action.encode()
                    if not 0 <= action_id < 82 or not mask[action_id]:
                        raise DataError("Replay action does not match the native legal mask")
                    target = rewards[player] * config["cql"]["gamma"] ** (len(steps) - index - 1)
                    yield (policy_features(observation), torch.tensor(action_id), mask,
                           torch.tensor(target, dtype=torch.float32), torch.tensor(rank))

    def sample_counts(self, partition: str) -> list[dict]:
        """Count the same native samples as training, never trace decision totals."""
        counts = []
        for record in self.records(partition):
            replay = MjaiReplay.from_jsonl(str(self.storage.path(record["mjai_path"])), rule="tenhou")
            grp_count = cql_count = 0
            for kyoku in replay.take_kyokus():
                grp_count += 4
                for player in range(4):
                    cql_count += sum(1 for _ in kyoku.steps(player))
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
