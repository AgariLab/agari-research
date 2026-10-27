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
