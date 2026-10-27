"""Evaluate every verified validation sample; train/test replay files stay closed."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from agari_kit.collection import verify_record, write_json
from agari_kit.dataset import make_split
from agari_kit.mjai import canonical_json
from agari_kit.storage import Storage
from agari_research.data import FrozenDataset
from agari_research.features import ENCODING_VERSION, UPSTREAM_COMMIT
from agari_research.losses import cql_loss
from agari_research.manifest import file_sha256
from agari_research.models.grp_model import RankPredictor
from agari_research.models.q_network import QNetwork
from agari_research.runtime import runtime_provenance
from agari_research.training import implementation_sha256


class ValidationOnlyDataset(FrozenDataset):
    def __init__(self, storage, split):
        self.storage = storage
        self.manifest = json.loads(split.read_text())
        body = {key: value for key, value in self.manifest.items() if key != "dataset_sha256"}
        self.sha256 = hashlib.sha256(canonical_json(body).encode()).hexdigest()
        records = [record for part in ("train", "validation", "test") for record in self.manifest["partitions"][part]]
        if self.sha256 != self.manifest.get("dataset_sha256") or make_split(records) != self.manifest:
            raise ValueError("Invalid frozen complete-game split")
        for record in self.manifest["partitions"]["validation"]:
            receipt = verify_record(storage, record, replay=True)
            if receipt.get("engine_version") != "0.4.10":
                raise ValueError("Unexpected validation engine version")

    def records(self, partition, seed=None):
        if partition != "validation" or seed is not None:
            raise ValueError("This reader only permits the fixed validation partition")
        return list(self.manifest["partitions"][partition])


def batches(samples, size):
    iterator = iter(samples)
    while rows := list(itertools.islice(iterator, size)):
        yield [torch.stack(values) for values in zip(*rows)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, help="Storage-relative frozen manifest")
    parser.add_argument("--models", required=True, help="Storage-relative exported model directory")
    parser.add_argument("--output", required=True, help="New storage-relative JSON report")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 1024:
        parser.error("batch-size must be 1..1024")
    started = time.monotonic()
    storage = Storage.from_env()
    storage.check(probe=True)
    output = storage.path(args.output)
    if output.exists():
        raise FileExistsError("Select a new report path")
    runtime, implementation = runtime_provenance(), implementation_sha256()
    dataset = ValidationOnlyDataset(storage, storage.path(args.split))
    model_dir = storage.path(args.models)
    paths = {name: model_dir / f"{name}.pt" for name in ("grp", "cql")}
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    payloads = {name: torch.load(path, map_location="cpu", weights_only=True) for name, path in paths.items()}
    for name, payload in payloads.items():
        if (payload.get("schema_version") != 1 or payload.get("kind") != name
                or payload.get("dataset_sha256") != dataset.sha256
                or payload.get("encoding_version") != ENCODING_VERSION
                or payload.get("upstream_commit") != UPSTREAM_COMMIT
                or payload.get("engine_version") != "0.4.10"):
            raise ValueError("Model does not belong to this frozen dataset/engine/encoding")
    if payloads["grp"]["config"] != payloads["cql"]["config"]:
        raise ValueError("GRP and CQL configurations differ")
    config = dict(payloads["cql"]["config"], device="cpu")
    torch.set_num_threads(config["torch_threads"])
    grp, q_network = RankPredictor(), QNetwork(**payloads["cql"]["model_config"])
    grp.load_state_dict(payloads["grp"]["state_dict"], strict=True)
    q_network.load_state_dict(payloads["cql"]["state_dict"], strict=True)
    grp.eval()
    q_network.eval()
    metrics = {}
    with torch.inference_mode():
        total = correct = count = 0
        for features, targets in batches(dataset.grp_samples("validation"), args.batch_size):
            logits = grp(features)
            total += F.cross_entropy(logits, targets, reduction="sum").item()
            correct += (logits.argmax(1) == targets).sum().item()
            count += len(targets)
        if not count:
            raise ValueError("No GRP validation samples")
        metrics["grp"] = {"samples": count, "loss": total / count, "accuracy": correct / count}
        sums, count = {}, 0
        for batch in batches(dataset.cql_samples("validation", grp, config), args.batch_size):
            values, auxiliary = q_network.forward_with_aux(batch[0])
            loss, terms = cql_loss(values, auxiliary, *batch[1:], alpha=config["cql"]["alpha"],
                                   aux_weight=config["cql"]["aux_weight"])
            size = len(batch[0])
            for key, value in dict(terms, loss=loss).items():
                if not torch.isfinite(value):
                    raise ValueError("Non-finite validation metric")
                sums[key] = sums.get(key, 0.0) + value.item() * size
            count += size
        if not count:
            raise ValueError("No CQL validation samples")
        metrics["cql"] = dict(samples=count, **{key: value / count for key, value in sums.items()})
    for record in dataset.manifest["partitions"]["validation"]:
        verify_record(storage, record, replay=False)
    if hashes != {name: file_sha256(path) for name, path in paths.items()}:
        raise ValueError("Model weights changed during validation")
    if runtime != runtime_provenance() or implementation != implementation_sha256():
        raise ValueError("Evaluation runtime or source changed during validation")
    report = {"status": "complete", "scope": "all samples in the full validation partition",
              "dataset_sha256": dataset.sha256, "validation_games": len(dataset.records("validation")),
              "train_content_read": False, "test_content_read": False, "batch_size": args.batch_size,
              "models": {name: {"path": str(path.relative_to(storage.root)), "sha256": hashes[name],
                                   "training_implementation_sha256": payloads[name]["implementation_sha256"],
                                   "training_runtime_provenance": payloads[name].get("runtime_provenance")}
                         for name, path in paths.items()},
              "evaluation_runtime_provenance": runtime, "evaluation_implementation_sha256": implementation,
              "metrics": metrics, "elapsed_seconds": time.monotonic() - started,
              "strength_evaluation": "not-performed"}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    print(json.dumps({key: report[key] for key in ("status", "validation_games", "metrics", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
