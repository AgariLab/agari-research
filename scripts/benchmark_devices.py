"""Bounded CPU/MPS CQL update microbenchmark, separate from strength evaluation."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path

import torch
from agari_kit.collection import write_json
from agari_kit.storage import Storage
from agari_research.losses import cql_loss
from agari_research.models.q_network import QNetwork
from agari_research.training import implementation_sha256


def benchmark(device, config, iterations=30):
    torch.manual_seed(config["seed"])
    model = QNetwork(**config["model"]).to(device).train()
    size = config["cql"]["batch_size"]
    # Fixed synthetic shapes isolate training compute from parsing / disk I/O.
    x = torch.randn(size, 74, 34).to(device)
    actions = torch.zeros(size, dtype=torch.long, device=device)
    mask = torch.ones(size, 82, dtype=torch.bool, device=device)
    targets, ranks = torch.zeros(size, device=device), torch.zeros(size, dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["cql"]["lr"])
    durations = []
    for index in range(iterations + 5):
        if device == "mps":
            torch.mps.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        q, aux = model.forward_with_aux(x)
        loss, _ = cql_loss(q, aux, actions, mask, targets, ranks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
        optimizer.step()
        if device == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - start
        if index >= 5:
            durations.append(elapsed)
    final_loss = loss.item()
    if not math.isfinite(final_loss):
        raise RuntimeError(f"Non-finite {device} benchmark")
    return {"iterations": iterations, "warmup": 5, "batch_size": size,
            "median_update_ms": statistics.median(durations) * 1000,
            "samples_per_second": size * len(durations) / sum(durations), "final_loss": final_loss}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/local-small.json"))
    parser.add_argument("--output", required=True, help="Storage-relative output JSON")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    torch.set_num_threads(config["torch_threads"])
    result = {"scope": "synthetic bounded CQL update throughput; excludes parsing, self-play and strength",
              "platform": platform.platform(), "torch": str(torch.__version__),
              "config": config, "implementation_sha256": implementation_sha256(),
              "cpu": benchmark("cpu", config), "mps_available": torch.backends.mps.is_available()}
    if result["mps_available"]:
        result["mps"] = benchmark("mps", config)
    storage = Storage.from_env()
    storage.check(probe=True)
    write_json(storage.path(args.output), result)
    print(json.dumps({key: value for key, value in result.items() if key != "config"}, indent=2))


if __name__ == "__main__":
    main()
