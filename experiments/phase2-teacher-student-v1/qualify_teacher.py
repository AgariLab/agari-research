"""Register, run, and independently verify the phase-two teacher gate."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
from pathlib import Path

from agari_kit.collection import write_json
from agari_kit.evaluation_campaign import evaluate_campaign
from agari_kit.storage import Storage

_path = Path(__file__).resolve().parents[2] / "scripts/phase2_contract.py"
_spec = importlib.util.spec_from_file_location("phase2_contract", _path)
contract_tools = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(contract_tools)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("register", "baseline", "teacher", "report"))
    parser.add_argument("--teacher-descriptor", type=Path, required=True)
    parser.add_argument("--legality-report", required=True, help="Storage-relative bound native-probe qualification JSON")
    parser.add_argument("--run", default="phase2-teacher-qualification-v1")
    parser.add_argument("--max-wall-seconds", type=float, default=172800)
    args = parser.parse_args()
    if not re.fullmatch(r"phase2-[A-Za-z0-9_.-]{1,72}", args.run):
        raise ValueError("Qualification needs a new simple phase2-* run name")
    if not math.isfinite(args.max_wall_seconds) or not 0 < args.max_wall_seconds <= 172800:
        raise ValueError("Qualification invocation budget must be within 48 hours")
    storage = Storage.from_env()
    storage.check(probe=True)
    protocol_path = Path(__file__).with_name("protocol.json")
    registration, protocol, comparator = contract_tools.registration_for(
        storage, protocol_path, args.teacher_descriptor, args.legality_report)
    identities = registration["identities"]
    seeds = identities["teacher"]["seeds"]
    output = storage.path("runs/" + args.run)
    if args.stage == "register":
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "registration.json", registration)
        write_json(output / "comparison-contract.json", registration["contract"])
        print(json.dumps({"status": "registered-not-played", "output": str(output), "games_per_model": len(seeds) * 4}))
        return
    contract_tools.verify_registration(output, registration)
    if args.stage in identities:
        target = output / args.stage
        report = evaluate_campaign(seeds, identities[args.stage]["lineup"], target, resume=target.exists(),
                                   max_wall_seconds=args.max_wall_seconds)
        print(json.dumps({key: report[key] for key in ("status", "games", "candidate") if key in report}))
        return
    result = contract_tools.qualification_result(output, registration, comparator)
    destination = output / "qualification.json"
    if destination.exists():
        if json.loads(destination.read_bytes()) != result:
            raise ValueError("Qualification report differs from sealed source evidence")
    else:
        write_json(destination, result)
    comparison = result["comparison"]
    print(json.dumps({"status": result["status"], "games_per_model": comparison["games_per_model"],
                      "difference": comparison["difference"]["mean_rank"],
                      "ci95": comparison["ci95"]["mean_rank"], "output": str(destination)}))


if __name__ == "__main__":
    main()
