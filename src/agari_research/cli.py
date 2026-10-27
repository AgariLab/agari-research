"""Local reproducible training entry point; no implicit network or dataset globs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agari_kit.storage import Storage


def main() -> None:
    parser = argparse.ArgumentParser(prog="agari-train")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Train GRP → CQL → PPO on a verified frozen split")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--split", required=True, help="Manifest path relative to AGARILAB_STORAGE_ROOT")
    run.add_argument("--run", required=True)
    run.add_argument("--resume", help="Checkpoint path relative to AGARILAB_STORAGE_ROOT")
    run.add_argument("--stop-after-updates", type=int)
    args = parser.parse_args()
    from .data import FrozenDataset
    from .training import run_training
    storage = Storage.from_env()
    storage.initialize()
    dataset = FrozenDataset(storage, storage.path(args.split))
    result = run_training(json.loads(args.config.read_text()), dataset, storage,
                          run_name=args.run, resume=storage.path(args.resume) if args.resume else None,
                          stop_after_updates=args.stop_after_updates)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
