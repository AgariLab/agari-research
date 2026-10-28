"""Compare complete fixed-opponent runs using paired whole-seed bootstrap."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

from agari_kit.collection import write_json
from agari_kit.storage import Storage


METRICS = ("mean_rank", "mean_point_delta", *(f"rank_{rank}_rate" for rank in range(1, 5)))


def read_run(directory: Path):
    paths = {name: directory / name for name in ("manifest.json", "games.jsonl")}
    contents = {name: path.read_bytes() for name, path in paths.items()}
    manifest = json.loads(contents["manifest.json"])
    if "campaign_plan_sha256" in manifest:
        from agari_kit.evaluation_campaign import verify_campaign
        verify_campaign(directory)
    rows = [json.loads(line) for line in contents["games.jsonl"].splitlines() if line.strip()]
    if manifest.get("status") != "complete":
        raise ValueError("Only completed evaluation runs can be compared")
    seeds, lineup = manifest["seeds"], manifest["lineup"]
    if (not seeds or any(type(seed) is not int or not 0 <= seed < 2**64 for seed in seeds)
            or len(set(seeds)) != len(seeds) or len(lineup) != 4
            or len(manifest["lineup_provenance"]) != 4):
        raise ValueError("Invalid evaluation fixture manifest")
    expected = len(seeds) * 4
    if len(rows) != expected or any(manifest.get(key) != expected
                                    for key in ("games_expected", "games_completed")):
        raise ValueError("Evaluation is missing complete seat rotations")
    indexed = {}
    for row in rows:
        seed, rotation = row["seed"], row["rotation"]
        if type(seed) is not int or type(rotation) is not int or seed not in seeds or rotation not in range(4):
            raise ValueError("Unexpected evaluation fixture")
        key = seed, rotation
        if key in indexed or row["seats"] != [lineup[(seat - rotation) % 4] for seat in range(4)]:
            raise ValueError("Duplicate or inconsistent seat rotation")
        scores, ranks = row["scores"], row["ranks"]
        if len(scores) != 4 or any(type(score) is not int for score in scores):
            raise ValueError("Invalid final scores")
        order = sorted(range(4), key=lambda seat: (-scores[seat], seat))
        if (any(type(rank) is not int for rank in ranks)
                or ranks != [order.index(seat) + 1 for seat in range(4)]):
            raise ValueError("Ranks do not match final scores and initial-seat tie order")
        if row.get("illegal_actions") != 0:
            raise ValueError("A run with illegal actions is not eligible for comparison")
        indexed[key] = row
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()}
    return manifest, indexed, hashes


def group_metrics(rows, seed):
    ranks = [rows[seed, seat]["ranks"][seat] for seat in range(4)]
    points = [rows[seed, seat]["scores"][seat] - 25000 for seat in range(4)]
    return (statistics.fmean(ranks), statistics.fmean(points),
            *(ranks.count(rank) / 4 for rank in range(1, 5)))


def inference_identity(provenance):
    if provenance.get("kind") == "builtin":
        return {"kind": "builtin", "name": provenance["name"]}
    if provenance.get("kind") != "external":
        raise ValueError("Candidate inference provenance is missing")
    source = provenance["source"]
    if not source.get("source_sha256") or not provenance.get("factory"):
        raise ValueError("Candidate needs a factory and source identity")
    return {"kind": "external", "factory": provenance["factory"],
            "source": {key: source[key] for key in ("package", "module", "source_sha256", "files")},
            "kwargs": provenance["descriptor"].get("kwargs", {}),
            "artifact_names": sorted(provenance["artifacts"])}


def identity_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def comparison_conditions(manifest):
    return {**{key: manifest[key] for key in ("rules", "engine", "code", "python")},
            "opponent_lineup": manifest["lineup"][1:],
            "opponent_provenance": manifest["lineup_provenance"][1:],
            "seeds": sorted(manifest["seeds"])}


def validate_contract(contract, old, new):
    """A different-candidates contract binds both complete candidate identities.

    SHA-256 values use canonical JSON (sorted keys, compact separators). Register
    the contract and hypothesis before evaluating/selecting the candidate. This
    explicit contract permits candidate factory/source/kwargs changes only; it
    never relaxes the shared rules, runtime, opponents or paired seed checks.
    """
    required = {"schema_version", "kind", "hypothesis", "primary_metric",
                "baseline_provenance_sha256", "candidate_provenance_sha256", "conditions_sha256"}
    if (not isinstance(contract, dict) or set(contract) != required
            or type(contract["schema_version"]) is not int or contract["schema_version"] != 1
            or contract["kind"] != "different-candidates" or contract["primary_metric"] != "mean_rank"
            or not isinstance(contract["hypothesis"], str) or not contract["hypothesis"].strip()):
        raise ValueError("Invalid explicit candidate comparison contract")
    expected = {"baseline_provenance_sha256": identity_sha256(old["lineup_provenance"][0]),
                "candidate_provenance_sha256": identity_sha256(new["lineup_provenance"][0]),
                "conditions_sha256": identity_sha256(comparison_conditions(old))}
    if any(contract[key] != value for key, value in expected.items()):
        raise ValueError("Candidate comparison contract identity mismatch")


def compare(baseline: Path, candidate: Path, *, bootstrap_samples: int = 2000, bootstrap_seed: int = 0,
            contract: dict | None = None):
    if bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    old, old_rows, old_hashes = read_run(baseline)
    new, new_rows, new_hashes = read_run(candidate)
    for key in ("rules", "engine", "code", "python"):
        if not old.get(key) or old[key] != new.get(key):
            raise ValueError(f"Evaluation conditions differ or are missing: {key}")
    if not old["engine"].get("binary_sha256"):
        raise ValueError("Comparison needs the actual engine binary identity")
    if (old["lineup"][1:] != new["lineup"][1:]
            or old["lineup_provenance"][1:] != new["lineup_provenance"][1:]):
        raise ValueError("Opponent configuration, artifacts or source differ")
    inference = inference_identity(old["lineup_provenance"][0])
    new_inference = inference_identity(new["lineup_provenance"][0])
    if contract is not None:
        validate_contract(contract, old, new)
    elif inference != new_inference:
        raise ValueError("Candidate inference factory, source or non-artifact arguments differ")
    if set(old_rows) != set(new_rows):
        raise ValueError("Runs must contain the same seeds and all four rotations")
    seeds = sorted(old["seeds"])
    if len(seeds) < 2:
        raise ValueError("At least two independent seed groups are required")
    old_groups = [group_metrics(old_rows, seed) for seed in seeds]
    new_groups = [group_metrics(new_rows, seed) for seed in seeds]
    differences = [tuple(after - before for before, after in zip(a, b))
                   for a, b in zip(old_groups, new_groups)]
    rng = random.Random(bootstrap_seed)
    distributions = [[] for _ in METRICS]
    # One resample selects complete paired groups. The four seats remain together.
    for _ in range(bootstrap_samples):
        sampled = rng.choices(differences, k=len(differences))
        for index, values in enumerate(zip(*sampled)):
            distributions[index].append(statistics.fmean(values))
    intervals = {}
    for key, values in zip(METRICS, distributions):
        values.sort()
        intervals[key] = [values[int(len(values) * .025)], values[min(len(values) - 1, int(len(values) * .975))]]
    means = lambda groups: dict(zip(METRICS, map(statistics.fmean, zip(*groups))))
    report = {
        "status": "complete", "games_per_model": len(old_rows), "seed_groups": len(seeds),
        "baseline": {"input_sha256": old_hashes, "candidate_provenance": old["lineup_provenance"][0],
                     "metrics": means(old_groups)},
        "candidate": {"input_sha256": new_hashes, "candidate_provenance": new["lineup_provenance"][0],
                      "metrics": means(new_groups)},
        "difference": means(differences), "difference_direction": "candidate minus baseline",
        "ci95": intervals, "ci_method": "paired-seed-group-percentile-bootstrap",
        "bootstrap_seed": bootstrap_seed, "bootstrap_samples": bootstrap_samples,
        "conditions": {key: old[key] for key in ("rules", "engine", "code", "python")},
        "candidate_inference": new_inference,
        "opponent_provenance": old["lineup_provenance"][1:], "seeds": seeds,
        "promotion_sample_size_met": len(old_rows) >= 10000,
        "strength_acceptance": "not-established-by-this-comparison",
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if contract is not None:
        report["baseline_inference"] = inference
        report["comparison_contract"] = contract
        report["comparison_contract_sha256"] = identity_sha256(contract)
    if all("candidate_diagnostics" in row for row in (*old_rows.values(), *new_rows.values())):
        from agari_kit.evaluation import DIAGNOSTIC_COUNTS, diagnostic_rates
        for label, rows in (("baseline", old_rows), ("candidate", new_rows)):
            counts = {key: sum(row["candidate_diagnostics"][key] for row in rows.values())
                      for key in DIAGNOSTIC_COUNTS}
            report[label]["diagnostics"] = diagnostic_rates(counts)
    for directory, hashes in ((baseline, old_hashes), (candidate, new_hashes)):
        if hashes != {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in hashes}:
            raise ValueError("Evaluation files changed during comparison")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="Storage-relative completed evaluation directory")
    parser.add_argument("--candidate", required=True, help="Storage-relative completed evaluation directory")
    parser.add_argument("--output", required=True, help="New storage-relative JSON report")
    parser.add_argument("--contract", type=Path, help="Pre-registered JSON contract allowing explicit candidate implementation differences")
    args = parser.parse_args()
    storage = Storage.from_env()
    storage.check(probe=True)
    output = storage.path(args.output)
    if output.exists():
        raise FileExistsError("Select a new comparison report path")
    contract = json.loads(args.contract.read_bytes()) if args.contract else None
    report = compare(storage.path(args.baseline), storage.path(args.candidate), contract=contract)
    report["inputs"] = {"baseline": args.baseline, "candidate": args.candidate}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    print(json.dumps({key: report[key] for key in ("games_per_model", "difference", "ci95")}, indent=2))


if __name__ == "__main__":
    main()
