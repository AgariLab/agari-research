"""Local reproducible training entry point; no implicit network or dataset globs."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from agari_kit.storage import Storage


def main() -> None:
    started = time.monotonic()
    parser = argparse.ArgumentParser(prog="agari-train")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Train selected BC or GRP/CQL/PPO stages on a verified frozen split")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--split", required=True, help="Manifest path relative to AGARILAB_STORAGE_ROOT")
    run.add_argument("--run", required=True)
    run.add_argument("--resume", help="Checkpoint path relative to AGARILAB_STORAGE_ROOT")
    run.add_argument("--stop-after-updates", type=int)
    run.add_argument("--max-wall-seconds", type=float,
                     help="Save and stop at an update boundary after this invocation's time budget")
    plan = commands.add_parser("plan", help="Verify data and count native samples without training")
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--split", required=True, help="Manifest path relative to AGARILAB_STORAGE_ROOT")
    plan.add_argument("--output", required=True, help="Execution plan JSON relative to AGARILAB_STORAGE_ROOT")
    args = parser.parse_args()
    if args.command == "run" and args.max_wall_seconds is not None and (
            not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds <= 0):
        parser.error("max-wall-seconds must be finite and positive")
    from .data import FrozenDataset
    from .training import implementation_sha256, run_training, validate_config
    storage = Storage.from_env()
    storage.initialize()
    config = json.loads(args.config.read_text())
    validate_config(config)
    dataset = FrozenDataset(storage, storage.path(args.split), policy_filter=config.get("policy_filter"))
    if args.command == "plan":
        from agari_kit.collection import write_json
        from .planning import build_training_plan
        from .runtime import runtime_provenance
        result = {"plan": build_training_plan(config, dataset), "runtime_provenance": runtime_provenance(),
                  "implementation_sha256": implementation_sha256()}
        destination = storage.path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_json(destination, result)
        result = {"output": args.output, "sha256": result["plan"]["sha256"], "stages": result["plan"]["stages"]}
    else:
        remaining = (None if args.max_wall_seconds is None else
                     max(1e-9, args.max_wall_seconds - (time.monotonic() - started)))
        result = run_training(config, dataset, storage,
                              run_name=args.run, resume=storage.path(args.resume) if args.resume else None,
                              stop_after_updates=args.stop_after_updates, max_wall_seconds=remaining)
        result["command_elapsed_seconds"] = time.monotonic() - started
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
