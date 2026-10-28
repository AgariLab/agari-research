"""Collect only the registered 1,000-game teacher pilot after its independent gate.

Use PYTHONPATH pointing to both frozen worktree src directories. This launcher
does not change imports, collect 10k games, or start training.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new(path, value):
    """Never replace registration or an earlier invocation record."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def checkout_identity(path):
    def git(*args):
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True)
    head = git("rev-parse", "HEAD")
    top = git("rev-parse", "--show-toplevel")
    branch = git("symbolic-ref", "--quiet", "HEAD")
    dirty = git("status", "--porcelain", "--untracked-files=all")
    if (head.returncode or top.returncode or dirty.returncode or branch.returncode != 1
            or Path(top.stdout.strip()).resolve() != path or dirty.stdout.strip()):
        raise ValueError("Runtime requires clean, detached paired worktrees")
    return {"path": str(path), "commit": head.stdout.strip()}


def require_module_paths(root):
    roots = {}
    for name, checkout in (("agari_kit", "agari-kit"), ("agari_research", "agari-research")):
        expected = root / checkout / "src" / name
        module = importlib.import_module(name)
        if Path(module.__file__).resolve() != expected / "__init__.py":
            raise ValueError(f"Wrong {name} module source; set PYTHONPATH to the paired frozen src directories")
        roots[name] = str(expected)
        for key, loaded in tuple(sys.modules.items()):
            if key.startswith(name + ".") and getattr(loaded, "__file__", None):
                if not Path(loaded.__file__).resolve().is_relative_to(expected):
                    raise ValueError(f"Mixed {name} module source: {key}")
    return roots


def load_runtime(root):
    root = root.expanduser().resolve()
    checkouts = {name: checkout_identity(root / name) for name in ("agari-kit", "agari-research")}
    modules = require_module_paths(root)
    helper = root / "agari-research/scripts/phase2_contract.py"
    protocol = root / "agari-research/experiments/phase2-teacher-student-v1/protocol.json"
    spec = importlib.util.spec_from_file_location("pilot_frozen_phase2_contract", helper)
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    from agari_kit.agent import Rules
    from agari_kit.collection import collect_selfplay
    from agari_kit.engine import engine_identity
    from agari_kit.plugins import agent_provenance
    from agari_kit.provenance import source_fingerprint
    from agari_kit.storage import Storage
    require_module_paths(root)
    return SimpleNamespace(root=root, contract=contract, protocol_path=protocol,
        Storage=Storage, collect_selfplay=collect_selfplay, agent_provenance=agent_provenance,
        source_fingerprint=source_fingerprint, engine_identity=engine_identity, Rules=Rules,
        binding={"root": str(root), "checkouts": checkouts, "module_roots": modules,
                 "helper_sha256": digest(helper), "protocol_sha256": digest(protocol),
                 "entrypoint": str(Path(__file__).resolve()), "entrypoint_sha256": digest(__file__),
                 "python": {"executable": sys.executable, "version": sys.version,
                            "sha256": digest(Path(sys.executable).resolve())}})


# The subprocess only checks existing receipts and calls the frozen splitter.
# It imports no teacher class and never creates an Agent or starts collection.
SPLIT_CODE = """
import json, pathlib, sys
import agari_kit
from agari_kit.collection import verified_split
from agari_kit.dataset import Catalog
from agari_kit.storage import Storage
root, catalog_relative, split_relative, teacher_sha, expected_json = sys.argv[1:]
qualified = json.loads(expected_json)
expected = pathlib.Path(root) / 'agari-kit/src/agari_kit/__init__.py'
if pathlib.Path(agari_kit.__file__).resolve() != expected:
    raise ValueError('Wrong splitter module source')
storage = Storage.from_env()
catalog, output = storage.path(catalog_relative), storage.path(split_relative)
with Catalog(catalog) as source:
    rows = source.games()
if len(rows) != 1000:
    raise ValueError('Pilot split requires exactly 1000 complete games')
seen = set()
for row in rows:
    receipt = json.loads(storage.path(row['validation_receipt']).read_bytes())
    seed = receipt.get('seed')
    if (type(seed) is not int or not 2630000 <= seed < 2631000 or seed in seen
            or receipt.get('agent_provenance_sha256') != [teacher_sha] * 4
            or receipt.get('producer_code') != qualified['code']
            or receipt.get('rules') != qualified['rules']
            or receipt.get('engine_binary_sha256') != qualified['engine']['binary_sha256']
            or receipt.get('engine_version') != qualified['engine']['version']):
        raise ValueError('Pilot receipt seed, teacher or qualified collection identity mismatch')
    seen.add(seed)
print(json.dumps(verified_split(storage, catalog, output)), flush=True)
"""


def split_with_budget(runtime, storage, name, teacher_sha, qualified, invocation, remaining):
    environment = os.environ.copy()
    environment.update(PYTHONDONTWRITEBYTECODE="1", AGARILAB_STORAGE_ROOT=str(storage.root),
        PYTHONPATH=os.pathsep.join(str(runtime.root / repo / "src") for repo in ("agari-kit", "agari-research")))
    catalog = f"datasets/manifests/{name}/catalog.sqlite"
    target = f"datasets/manifests/{name}/split-v1.json"
    command = [sys.executable, "-B", "-c", SPLIT_CODE, str(runtime.root), catalog, target, teacher_sha,
               json.dumps(qualified, sort_keys=True)]
    stdout, stderr = invocation / "split.stdout.json", invocation / "split.stderr.log"
    with stdout.open("x") as out, stderr.open("x") as err:
        process = subprocess.Popen(command, env=environment, cwd=runtime.root, stdout=out, stderr=err)
        finished = False
        try:
            returncode = process.wait(timeout=remaining)
            finished = True
        except subprocess.TimeoutExpired:
            return None
        finally:
            # Also reap on Ctrl-C or any other parent-side exception, before
            # releasing the pilot lock. Preserve the original exception.
            original = sys.exception()
            try:
                if not finished:
                    try:
                        process.terminate()
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            except BaseException as cleanup:
                if original is not None:
                    original.add_note(f"Split process cleanup also failed: {cleanup}")
                else:
                    raise
    if returncode:
        raise RuntimeError(f"Frozen split verification failed ({returncode}); see {stderr}")
    result = json.loads(stdout.read_text())
    if result.get("games") != 1000 or Path(result.get("output", "")) != storage.path(target):
        raise ValueError("Unexpected frozen pilot split result")
    return {**result, "sha256": digest(storage.path(target)),
            "stdout_sha256": digest(stdout), "stderr_sha256": digest(stderr)}


def require_unchanged(runtime, current, teacher, provenance):
    require_module_paths(runtime.root)
    if (runtime.source_fingerprint() != current["code"] or runtime.engine_identity() != current["engine"]
            or runtime.agent_provenance(teacher) != provenance
            or digest(__file__) != runtime.binding["entrypoint_sha256"]
            or digest(runtime.protocol_path) != runtime.binding["protocol_sha256"]
            or digest(runtime.root / "agari-research/scripts/phase2_contract.py") != runtime.binding["helper_sha256"]):
        raise ValueError("Pilot implementation or qualified identity changed during invocation")


def run(args):
    started = time.monotonic()
    if not re.fullmatch(r"phase2-[A-Za-z0-9_.-]{1,72}", args.name):
        raise ValueError("Use a simple phase2-* pilot name")
    if (type(args.max_wall_seconds) not in (int, float) or not math.isfinite(args.max_wall_seconds)
            or not 0 < args.max_wall_seconds <= 172800):
        raise ValueError("Invocation budget must be positive and at most 172800 seconds")
    runtime = load_runtime(args.runtime_root)
    storage = runtime.Storage.from_env()
    storage.check(probe=True)
    descriptor = args.teacher_descriptor.expanduser().resolve()
    qualification_run = storage.path(args.qualification_run)
    protocol = runtime.contract.load_protocol(runtime.protocol_path)
    if protocol["seeds"]["teacher_1000"] != {"start": 2630000, "count": 1000, "rotations": 1}:
        raise ValueError("This entrypoint only supports the registered 1000-game pilot")
    qualification = runtime.contract.require_qualification(
        storage, qualification_run, runtime.protocol_path, descriptor)
    teacher = "@" + str(descriptor)
    provenance = runtime.agent_provenance(teacher)
    current = {"code": runtime.source_fingerprint(), "engine": runtime.engine_identity(),
               "rules": asdict(runtime.Rules())}
    if current != qualification["collection_identity"]:
        raise ValueError("Collection source, engine or rules differ from the qualified identity")
    require_module_paths(runtime.root)
    config = {"schema_version": 1, "source": "agarilab-selfplay", "games": 1000,
              "seed_start": 2630000, "agents": [teacher] * 4,
              "agent_provenance": [provenance] * 4, **current}
    registration = {"schema_version": 1, "kind": "qualified-teacher-1000-pilot-v1", "name": args.name,
                    "runtime": runtime.binding, "qualification": qualification, "collection": config,
                    "split": f"datasets/manifests/{args.name}/split-v1.json"}
    output = storage.path(f"runs/{args.name}-collection")
    if output.exists():
        if json.loads((output / "registration.json").read_bytes()) != registration:
            raise ValueError("Pilot registration identity changed; choose a new explicit run")
        if not (output / "invocations").is_dir():
            raise ValueError("Incomplete pilot registration; refusing recovery or overwrite")
    else:
        if any(storage.path(f"datasets/{kind}/{args.name}").exists()
               for kind in ("raw", "mjai", "manifests", "quarantine")):
            raise ValueError("Existing data has no matching pilot registration; refusing adoption")
        output.mkdir(parents=True, exist_ok=False)
        write_new(output / "registration.json", registration)
        (output / "invocations").mkdir()
    with (output / ".pilot.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another invocation owns this pilot") from None
        invocation = output / "invocations" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ-") + uuid.uuid4().hex)
        invocation.mkdir()
        registration_sha = digest(output / "registration.json")
        write_new(invocation / "started.json", {"schema_version": 1, "registration_sha256": registration_sha,
            "started_at": datetime.now(timezone.utc).isoformat(), "max_wall_seconds": args.max_wall_seconds})
        result = {"schema_version": 1, "registration_sha256": registration_sha,
                  "status": "failed", "invocation": str(invocation),
                  "preflight_seconds": time.monotonic() - started}
        try:
            acquisition_path = storage.path(f"datasets/manifests/{args.name}/acquisition.json")
            acquisition = json.loads(acquisition_path.read_bytes()) if acquisition_path.exists() else None
            if acquisition is not None and acquisition.get("config") != config:
                raise ValueError("Acquisition identity differs from the immutable pilot registration")
            remaining = args.max_wall_seconds - (time.monotonic() - started)
            if acquisition is not None and acquisition.get("status") == "complete":
                result["collection"] = {"status": "complete", "reentry": "verification-and-split-only"}
            elif remaining <= 0:
                result.update(status="paused", stage="preflight-budget")
                return result
            else:
                result["collector_budget_seconds"] = remaining
                result["collection"] = runtime.collect_selfplay(storage, name=args.name, games=1000,
                    seed_start=2630000, agents=[teacher] * 4, max_wall_seconds=remaining,
                    progress=lambda event: print(json.dumps(event), flush=True))
            if result["collection"]["status"] == "paused":
                result.update(status="paused", stage="collection")
                return result
            if result["collection"]["status"] != "complete":
                raise ValueError("Collector did not produce a complete or paused pilot")
            require_unchanged(runtime, current, teacher, provenance)
            remaining = args.max_wall_seconds - (time.monotonic() - started)
            result.update(status="complete-awaiting-split", stage="split")
            if remaining <= 0:
                return result
            teacher_sha = runtime.contract.content_sha256(provenance)
            split = split_with_budget(runtime, storage, args.name, teacher_sha, current, invocation, remaining)
            if split is not None:
                require_unchanged(runtime, current, teacher, provenance)
                result.update(status="ready", stage="complete", split=split)
            return result
        except BaseException as exc:
            result.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc)})
            raise
        finally:
            result.update(elapsed_seconds=time.monotonic() - started,
                          completed_at=datetime.now(timezone.utc).isoformat())
            write_new(invocation / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True, help="Root containing the two clean detached worktrees")
    parser.add_argument("--teacher-descriptor", type=Path, required=True)
    parser.add_argument("--qualification-run", required=True, help="Storage-relative completed qualification directory")
    parser.add_argument("--name", required=True, help="New phase2-* dataset name, reused only for identical resumes")
    parser.add_argument("--max-wall-seconds", type=float, default=172800,
                        help="Total invocation budget; preflight is deducted, collection pauses at whole-game boundaries")
    print(json.dumps(run(parser.parse_args())))


if __name__ == "__main__":
    main()
