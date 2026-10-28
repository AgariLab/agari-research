"""Register, run, and independently verify the phase-two teacher gate.

Each side may run in its own process. Both use the same frozen code/environment.
The report step never starts games and verifies sealed campaign evidence first.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
from pathlib import Path

from agari_kit.collection import file_sha256, write_json
from agari_kit.evaluation_campaign import _identity, evaluate_campaign, verify_campaign
from agari_kit.storage import Storage


def comparison_module():
    path = Path(__file__).resolve().parents[2] / "scripts/compare_evaluations.py"
    spec = importlib.util.spec_from_file_location("phase2_comparison", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("register", "baseline", "teacher", "report"))
    parser.add_argument("--teacher-descriptor", type=Path, required=True)
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
    protocol = json.loads(protocol_path.read_text())
    base = "@" + str(storage.path(protocol["incumbent"]["descriptor"]))
    teacher = "@" + str(args.teacher_descriptor.resolve())
    lineups = {"baseline": [base] * 4, "teacher": [teacher, base, base, base]}
    seed_config = protocol["seeds"]["teacher_qualification"]
    seeds = list(range(seed_config["start"], seed_config["start"] + seed_config["count"]))
    identities = {name: dict(_identity(lineup), seeds=seeds) for name, lineup in lineups.items()}
    comparator = comparison_module()
    if comparator.comparison_conditions(identities["baseline"]) != comparator.comparison_conditions(identities["teacher"]):
        raise ValueError("Teacher qualification must share conditions and opponents")
    old_weights = identities["baseline"]["lineup_provenance"][0]["artifacts"]["weights_path"]["sha256"]
    if old_weights != protocol["incumbent"]["weights_sha256"]:
        raise ValueError("Incumbent weights differ from the pre-registered baseline")
    teacher_manifest_path = Path(identities["teacher"]["lineup_provenance"][0]["artifacts"]["manifest_path"]["path"])
    teacher_manifest = json.loads(teacher_manifest_path.read_bytes())
    published = protocol["teacher"]
    teacher_weight = teacher_manifest_path.parent / "source" / published["weights"]
    if (teacher_manifest["upstream"]["revision"] != published["revision"]
            or teacher_manifest["files"]["source/" + published["weights"]] != published["sha256"]
            or teacher_weight.stat().st_size != published["bytes"]
            or file_sha256(teacher_weight) != published["sha256"]):
        raise ValueError("Teacher differs from the pre-registered S42 publication")
    contract = {"schema_version": 1, "kind": "different-candidates",
                "hypothesis": "Frozen Mortal-S42 improves mean rank over the incumbent against three incumbent opponents",
                "primary_metric": "mean_rank",
                "baseline_provenance_sha256": comparator.identity_sha256(identities["baseline"]["lineup_provenance"][0]),
                "candidate_provenance_sha256": comparator.identity_sha256(identities["teacher"]["lineup_provenance"][0]),
                "conditions_sha256": comparator.identity_sha256(comparator.comparison_conditions(identities["baseline"]))}
    registration = {"schema_version": 1, "protocol_sha256": file_sha256(protocol_path),
                    "runner_sha256": file_sha256(Path(__file__)),
                    "comparator_sha256": file_sha256(Path(comparator.__file__)),
                    "identities": identities, "contract": contract}
    output = storage.path("runs/" + args.run)
    if args.stage == "register":
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "registration.json", registration)
        write_json(output / "comparison-contract.json", contract)
        print(json.dumps({"status": "registered-not-played", "output": str(output), "games_per_model": len(seeds) * 4}))
        return
    if json.loads((output / "registration.json").read_bytes()) != registration:
        raise ValueError("Registered qualification code/model/conditions changed; create a new experiment")
    if json.loads((output / "comparison-contract.json").read_bytes()) != contract:
        raise ValueError("Registered comparison contract changed")
    if args.stage in lineups:
        target = output / args.stage
        report = evaluate_campaign(seeds, lineups[args.stage], target, resume=target.exists(),
                                   max_wall_seconds=args.max_wall_seconds)
        print(json.dumps({key: report[key] for key in ("status", "games", "candidate") if key in report}))
        return
    # This stage inspects complete results once both independently sealed runs exist.
    reports = {name: verify_campaign(output / name) for name in lineups}
    comparison = comparator.compare(output / "baseline", output / "teacher", contract=contract)
    acceptable_attempts = all(report["attempts"]["failed"] == 0 and report["attempts"]["unsealed"] == 0
                              for report in reports.values())
    passed = (comparison["games_per_model"] == 1000 and comparison["ci95"]["mean_rank"][1] < 0
              and acceptable_attempts)
    result = {"schema_version": 1, "status": "passed" if passed else "not-qualified",
              "scope": "Teacher strength gate; targeted adapter legality evidence is required separately",
              "no_failed_or_unsealed_attempts": acceptable_attempts,
              "comparison": comparison, "registration_sha256": file_sha256(output / "registration.json")}
    destination = output / "qualification.json"
    if destination.exists():
        if json.loads(destination.read_bytes()) != result:
            raise ValueError("Qualification report differs from sealed source evidence")
    else:
        write_json(destination, result)
    print(json.dumps({"status": result["status"], "games_per_model": comparison["games_per_model"],
                      "difference": comparison["difference"]["mean_rank"],
                      "ci95": comparison["ci95"]["mean_rank"], "output": str(destination)}))


if __name__ == "__main__":
    main()
