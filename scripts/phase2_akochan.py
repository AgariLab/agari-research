"""Run a frozen candidate's 16+48 disjoint Akochan seed groups.

Only the run subcommand starts games. Registration requires the existing frozen
matrix's final-freeze and Akochan plan. Reports are exploratory diagnostics and
never select a candidate, alter the final comparison, or establish promotion.
The explicitly supplied tools/runtime worktrees and this helper must remain
unchanged across registration, continuation and reporting.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
import random
import re
import statistics
import sys
import time

PHASE_COUNTS = {"initial": 16, "extension": 48}
SIDES = ("incumbent", "candidate")
CONDITIONS = ("rules", "lineup", "lineup_provenance", "engine", "code", "python")
AKOCHAN_FACTORY = "agari_kit.opponents.akochan:AkochanAgent"


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Context:
    def __init__(self, tools_root, runtime_root):
        self.tools_root = Path(tools_root).expanduser().resolve()
        self.tools_checkout = self.tools_root
        script = self.tools_checkout / "scripts/phase2_evaluation_matrix.py"
        require(script.is_file() and script.resolve() == script,
                "tools-root must be the frozen research checkout containing scripts/phase2_evaluation_matrix.py")
        spec = importlib.util.spec_from_file_location("akochan_frozen_matrix", script)
        self.matrix = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.matrix)
        self.tools_identity = self.matrix.checkout_identity(self.tools_checkout)
        self.rt = self.matrix.Runtime(runtime_root)

    def binding(self):
        require(self.matrix.checkout_identity(self.tools_checkout) == self.tools_identity,
                "Frozen tools worktree changed")
        self.matrix.check_module_paths(self.rt.root)
        return {"tools_root": str(self.tools_root), "tools_checkout": self.tools_identity,
                "matrix_script_sha256": self.matrix.digest(self.matrix.__file__),
                "helper_path": str(Path(__file__).resolve()), "helper_sha256": self.matrix.digest(__file__),
                "runtime": self.rt.implementation_binding(),
                "python": {"executable": str(Path(sys.executable).resolve()), "sha256": self.matrix.digest(sys.executable)}}


def validate_plan(context, registered, frozen, development, plan):
    """Reconstruct every fixture; command strings in the old plan are never executed."""
    tools, rt = context.matrix, context.rt
    require(plan.get("schema_version") == 1 and plan.get("status") == "commands-only-not-played"
            and plan.get("winner") == frozen["winner"] == development["winner"]
            and plan.get("development_sha256") == tools.content_digest(development),
            "Akochan plan is not bound to the uniquely frozen final candidate")
    require(set(plan["phases"]) == set(PHASE_COUNTS) and set(plan["contracts"]) == set(PHASE_COUNTS),
            "Expected exactly initial and extension phases")
    settings = rt.protocol["seeds"]["akochan_exploratory"]
    require((settings["initial_count"], settings["count"], settings["rotations"]) == (16, 64, 4),
            "Unsupported registered Akochan seed budget")
    initial = plan["phases"]["initial"]
    require(set(initial) == set(SIDES), "Initial phase must contain both sides")
    opponent = initial["incumbent"]["lineup"][1]
    require(isinstance(opponent, str) and opponent.startswith("@"), "Akochan requires a frozen external descriptor")
    descriptor = Path(opponent[1:]).resolve()
    tools.require(rt.contract.storage_relative(rt.storage, descriptor), "Akochan descriptor must remain on artifact storage")
    require(rt.protocol["evaluation"]["akochan_opponents"] == ["akochan", "efficiency", "tsumogiri"],
            "Akochan opponent pool differs from registered protocol")
    offset = 0
    seen = set()
    for phase, count in PHASE_COUNTS.items():
        require(set(plan["phases"][phase]) == set(SIDES), "Each phase must contain incumbent and candidate")
        seeds = list(range(settings["start"] + offset, settings["start"] + offset + count))
        require(not seen & set(seeds), "Akochan phase seeds overlap")
        seen.update(seeds)
        for side in SIDES:
            candidate = frozen["fixtures"][side]["lineup"][0]
            expected = tools.identity(rt, [candidate, opponent, "efficiency", "tsumogiri"], seeds)
            require(expected == plan["phases"][phase][side], "Akochan fixture, descriptor, source or runtime differs")
            require(expected["lineup_provenance"][1].get("factory") == AKOCHAN_FACTORY,
                    "Opponent is not the exact Akochan adapter")
        sides = plan["phases"][phase]
        require(plan["contracts"][phase] == tools.explicit_contract(rt, sides["incumbent"], sides["candidate"]),
                "Akochan phase comparison contract changed")
        offset += count
    require(seen == rt.contract.seed_range(settings), "Akochan phases omit or duplicate registered seeds")
    # Constructor hashes the real binary/config/parameter inventory; it never
    # creates a process (the adapter starts one only on act).
    from agari_kit.agent import managed_agents
    with managed_agents([opponent]):
        pass
    return {"descriptor": rt.artifact(descriptor),
            "provenance": plan["phases"]["initial"]["incumbent"]["lineup_provenance"][1]}


def registration_for(context, matrix_relative, directory):
    tools, rt = context.matrix, context.rt
    registered = tools.load_matrix(rt, matrix_relative)
    frozen = tools.frozen_final(rt, registered)
    root = rt.path(matrix_relative)
    plan = tools.read(root / "akochan-plan.json")
    development = tools.read(root / "development-report.json")
    opponent = validate_plan(context, registered, frozen, development, plan)
    evidence = {"matrix": rt.artifact(root / "registration.json"),
                "plan": rt.artifact(root / "akochan-plan.json"),
                "final_freeze": rt.artifact(root / "final-freeze.json"),
                "final_reservation": rt.artifact(tools.final_reservation(rt)),
                "development": rt.artifact(root / "development-report.json")}
    return {"schema_version": 1, "kind": "phase2-akochan-exploration-v1", "matrix": matrix_relative, "directory": directory,
            "winner": frozen["winner"], "scale": frozen["scale"], "implementation": context.binding(),
            "evidence": evidence, "akochan": opponent, "phases": plan["phases"], "contracts": plan["contracts"],
            "scope": "Exploratory frozen-candidate diagnostics; no selection, data expansion or promotion decision",
            "extension_gate": "Both initial campaigns complete with zero failed or unsealed attempts; independent of scores",
            "bootstrap_samples": 2000, "bootstrap_seed": 0}


def register(context, matrix_relative, name):
    require(re.fullmatch(r"phase2-[A-Za-z0-9_.-]{1,72}", name), "Use a new simple phase2-* exploration name")
    tools, rt = context.matrix, context.rt
    output = rt.path("runs/" + name)
    require(not output.exists(), "Exploration registration is immutable; use a new name")
    registration = registration_for(context, matrix_relative, "runs/" + name)
    output.mkdir(parents=True, exist_ok=False)
    tools.publish(output / "registration.json", registration)
    return {"status": "registered-not-played", "run": "runs/" + name, "winner": registration["winner"],
            "initial_games_per_side": 64, "extension_games_per_side": 192, "total_games_per_side": 256}


def load_registration(context, relative):
    tools, rt = context.matrix, context.rt
    output = rt.path(relative)
    saved = tools.read(output / "registration.json")
    require(saved.get("directory") == relative, "Exploration registration belongs to another output directory")
    require(saved == registration_for(context, saved["matrix"], relative),
            "Exploration plan, frozen candidate, runtime or helper changed")
    return output, saved


def verify_initial(context, output, registration):
    return {side: context.matrix.verify_fixture(context.rt, output / "initial" / side,
                    registration["phases"]["initial"][side]) for side in SIDES}


def run_phase(context, relative, phase, side, *, resume, max_wall_seconds, invocation_started):
    require(phase in PHASE_COUNTS and side in SIDES, "Unknown exploration phase or side")
    require(type(max_wall_seconds) in (int, float) and math.isfinite(max_wall_seconds)
            and 0 < max_wall_seconds <= 172800, "Invocation budget must be finite, positive and within 48 hours")
    output, registration = load_registration(context, relative)
    if phase == "extension":
        verify_initial(context, output, registration)
    before = context.matrix.digest(output / "registration.json")
    report = context.matrix.execute(context.rt, registration["phases"][phase][side], output / phase / side,
                                    resume=resume, max_wall_seconds=max_wall_seconds, invocation_started=invocation_started)
    # Existing campaign evidence is retained on every error. Never rewrite a
    # failed attempt or continue automatically after a changed dependency.
    _, checked = load_registration(context, relative)
    require(checked == registration and context.matrix.digest(output / "registration.json") == before,
            "Exploration registration changed during execution")
    if phase == "extension":
        verify_initial(context, output, registration)
    return {key: report[key] for key in ("status", "reason", "games", "games_expected", "games_started") if key in report}


def combine_records(comparator, segments, phases, contracts, total_seeds):
    """Combine validated read_run outputs; preserve each seed's four rotations."""
    require(set(segments) == set(phases) == set(PHASE_COUNTS), "Both complete phases are required")
    require(len(total_seeds) == 64 and len(set(total_seeds)) == 64, "Exploration requires exactly 64 unique seeds")
    combined = {side: {} for side in SIDES}
    identities = {}
    seen = set()
    for phase, count in PHASE_COUNTS.items():
        require(set(segments[phase]) == set(phases[phase]) == set(SIDES), "Both paired sides are required in each phase")
        phase_seeds = phases[phase]["incumbent"]["seeds"]
        require(len(phase_seeds) == count and len(set(phase_seeds)) == count and not seen & set(phase_seeds),
                "Missing or duplicated seeds between Akochan phases")
        seen.update(phase_seeds)
        expected_keys = {(seed, rotation) for seed in phase_seeds for rotation in range(4)}
        for side in SIDES:
            manifest, rows, _ = segments[phase][side]
            expected = phases[phase][side]
            require(all(manifest.get(key) == expected[key] for key in (*CONDITIONS, "seeds")),
                    "Campaign differs from the registered Akochan fixture")
            require(expected["seeds"] == phase_seeds and set(rows) == expected_keys and len(rows) == count * 4,
                    "Missing, duplicate or unpaired complete seed rotations")
            current = {key: manifest[key] for key in CONDITIONS}
            require(side not in identities or current == identities[side], "Source, engine or lineup changed between phases")
            identities[side] = current
            require(not set(combined[side]) & set(rows), "Repeated games cannot count as new extension evidence")
            combined[side].update(rows)
        old, new = segments[phase]["incumbent"][0], segments[phase]["candidate"][0]
        require(comparator.comparison_conditions(old) == comparator.comparison_conditions(new),
                "Paired opponents, rules or runtime differ")
        comparator.validate_contract(contracts[phase], old, new)
    require(seen == set(total_seeds) and all(len(rows) == 256 for rows in combined.values()),
            "Merged exploration must cover the exact 64 registered groups and 256 games per side")
    return combined, identities


def paired_statistics(comparator, baseline, candidate, seeds, *, bootstrap_samples=2000, bootstrap_seed=0):
    """Same whole-seed resampling as the frozen comparator, applied after union."""
    require(bootstrap_samples >= 100, "Need at least 100 bootstrap samples")
    seeds = sorted(seeds)
    require(len(seeds) == len(set(seeds)) and len(seeds) >= 2, "Need unique paired seed groups")
    expected = {(seed, rotation) for seed in seeds for rotation in range(4)}
    require(set(baseline) == set(candidate) == expected, "Paired statistics require every rotation exactly once")
    old_groups = [comparator.group_metrics(baseline, seed) for seed in seeds]
    new_groups = [comparator.group_metrics(candidate, seed) for seed in seeds]
    differences = [tuple(after - before for before, after in zip(old, new)) for old, new in zip(old_groups, new_groups)]
    randomizer = random.Random(bootstrap_seed)
    distributions = [[] for _ in comparator.METRICS]
    for _ in range(bootstrap_samples):
        sampled = randomizer.choices(differences, k=len(differences))
        for index, values in enumerate(zip(*sampled)):
            distributions[index].append(statistics.fmean(values))
    intervals = {}
    for key, values in zip(comparator.METRICS, distributions):
        values.sort()
        intervals[key] = [values[int(len(values) * .025)], values[min(len(values)-1, int(len(values) * .975))]]
    means = lambda groups: dict(zip(comparator.METRICS, map(statistics.fmean, zip(*groups))))
    return {"games_per_model": len(baseline), "seed_groups": len(seeds), "seeds": seeds,
            "baseline": {"metrics": means(old_groups)}, "candidate": {"metrics": means(new_groups)},
            "difference": means(differences), "difference_direction": "candidate minus baseline", "ci95": intervals,
            "ci_method": "paired-seed-group-percentile-bootstrap", "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "paired_seed_groups": [{"seed": seed, "baseline": dict(zip(comparator.METRICS, old)),
                "candidate": dict(zip(comparator.METRICS, new)), "difference": dict(zip(comparator.METRICS, delta))}
                for seed, old, new, delta in zip(seeds, old_groups, new_groups, differences)]}


def campaign_evidence(context, path, fixture):
    evidence = context.matrix.verify_fixture(context.rt, path, fixture)
    return {**evidence, **{name: context.matrix.digest(path / name)
                           for name in ("report.json", "candidate-latencies.json")}}


def combined_cost(campaigns):
    """Sum sealed costs; compute latency percentiles from the concatenated samples."""
    elapsed = decisions = illegal = 0
    attempts = dict.fromkeys(("total", "failed", "unsealed"), 0)
    latencies = []
    for phase, (report, values) in campaigns.items():
        require(report["status"] == "complete" and report["games"] == PHASE_COUNTS[phase] * 4,
                "Campaign cost report has the wrong completion size")
        require(type(report["elapsed_seconds"]) in (int, float) and math.isfinite(report["elapsed_seconds"])
                and report["elapsed_seconds"] >= 0, "Invalid sealed game-loop cost")
        require(all(type(report[key]) is int and report[key] >= 0 for key in ("decisions", "illegal_actions", "candidate_decisions"))
                and report["illegal_actions"] == 0, "Invalid sealed decision/legality counts")
        require(all(type(report["attempts"][key]) is int and report["attempts"][key] >= 0 for key in attempts)
                and report["attempts"]["failed"] == report["attempts"]["unsealed"] == 0
                and report["attempts"]["total"] == PHASE_COUNTS[phase], "Failed, unsealed or duplicate attempts in cost evidence")
        require(isinstance(values, list) and len(values) == report["candidate_decisions"] and bool(values)
                and all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in values),
                "Candidate latency samples differ from the sealed decision count")
        elapsed += report["elapsed_seconds"]
        decisions += report["decisions"]
        illegal += report["illegal_actions"]
        for key in attempts:
            attempts[key] += report["attempts"][key]
        latencies.extend(values)
    require(set(campaigns) == set(PHASE_COUNTS), "Cost evidence must include both disjoint phases")
    latencies.sort()
    return {"elapsed_seconds": elapsed, "decisions": decisions, "illegal_actions": illegal, "attempts": attempts,
            "candidate_decisions": len(latencies), "candidate_latency_ms_mean": statistics.fmean(latencies),
            "candidate_latency_ms_p95": latencies[min(len(latencies)-1, int(len(latencies) * .95))],
            "candidate_latency_ms_max": latencies[-1],
            "elapsed_scope": "Sum of post-reset game-loop wall time; excludes construction/reset, reporting and validation",
            "decision_scope": "All four seats for decisions; candidate seat only for candidate_decisions and latency",
            "latency_scope": "Candidate agent.act wall time, including local IPC for external agents; excludes construction/reset and non-act engine work"}


def final_report(context, relative):
    output, registration = load_registration(context, relative)
    tools, rt = context.matrix, context.rt
    comparator = rt.comparator
    registration_sha = tools.digest(output / "registration.json")
    segments, evidence = {}, {}
    costs = {side: {} for side in SIDES}
    for phase in PHASE_COUNTS:
        segments[phase], evidence[phase] = {}, {}
        for side in SIDES:
            path = output / phase / side
            evidence[phase][side] = campaign_evidence(context, path, registration["phases"][phase][side])
            segments[phase][side] = comparator.read_run(path)
            costs[side][phase] = tools.read(path / "report.json"), tools.read(path / "candidate-latencies.json")
            require(all(segments[phase][side][2][name] == evidence[phase][side][name] for name in ("manifest.json", "games.jsonl")),
                    "Campaign changed between sealing and reading")
    total_seeds = sorted(rt.contract.seed_range(rt.protocol["seeds"]["akochan_exploratory"]))
    rows, identities = combine_records(comparator, segments, registration["phases"], registration["contracts"], total_seeds)
    report = paired_statistics(comparator, rows["incumbent"], rows["candidate"], total_seeds,
                                bootstrap_samples=registration["bootstrap_samples"], bootstrap_seed=registration["bootstrap_seed"])
    from agari_kit.evaluation import DIAGNOSTIC_COUNTS, diagnostic_rates
    for side, label in (("incumbent", "baseline"), ("candidate", "candidate")):
        require(all("candidate_diagnostics" in row for row in rows[side].values()), "Candidate diagnostics are missing")
        counts = {key: sum(row["candidate_diagnostics"][key] for row in rows[side].values()) for key in DIAGNOSTIC_COUNTS}
        report[label].update(provenance=identities[side]["lineup_provenance"][0], diagnostics=diagnostic_rates(counts),
                             cost=combined_cost(costs[side]))
    # Reverify sealed source bytes after all statistical work; no averaged phase CI.
    for phase in PHASE_COUNTS:
        for side in SIDES:
            require(campaign_evidence(context, output / phase / side, registration["phases"][phase][side]) == evidence[phase][side],
                    "Campaign evidence changed during aggregation")
    _, checked = load_registration(context, relative)
    require(checked == registration and tools.digest(output / "registration.json") == registration_sha,
            "Registered sources changed during aggregation")
    report.update(schema_version=1, kind="phase2-akochan-exploratory-comparison-v1", status="complete",
        registration_sha256=registration_sha, input_evidence=evidence,
        implementation=registration["implementation"], winner=registration["winner"], scale=registration["scale"],
        conditions={key: identities["incumbent"][key] for key in ("rules", "engine", "code", "python")},
        opponent_provenance=identities["incumbent"]["lineup_provenance"][1:],
        scope=registration["scope"], promotion_sample_size_met=False, strength_acceptance="exploratory-only",
        candidate_selection_changed=False, final_gate_changed=False)
    tools.publish(output / "report.json", report)
    return report


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools-root", type=Path, required=True, help="Clean detached tools research checkout, e.g. phase2-tools-20261029-v1/agari-research")
    parser.add_argument("--runtime-root", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    registration = commands.add_parser("register")
    registration.add_argument("--matrix", required=True, help="Storage-relative matrix with an existing final-freeze and Akochan plan")
    registration.add_argument("--run", required=True, help="New simple phase2-* exploration name")
    run = commands.add_parser("run")
    run.add_argument("--run", required=True, help="Storage-relative exploration directory")
    run.add_argument("--phase", choices=PHASE_COUNTS, required=True)
    run.add_argument("--side", choices=SIDES, required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--max-wall-seconds", type=float, default=172800)
    report = commands.add_parser("report")
    report.add_argument("--run", required=True)
    args = parser.parse_args()
    if args.command == "run":
        require(math.isfinite(args.max_wall_seconds) and 0 < args.max_wall_seconds <= 172800,
                "Invocation budget must be finite, positive and within 48 hours")
    context = Context(args.tools_root, args.runtime_root)
    if args.command == "register":
        result = register(context, args.matrix, args.run)
    elif args.command == "run":
        result = run_phase(context, args.run, args.phase, args.side, resume=args.resume,
                           max_wall_seconds=args.max_wall_seconds, invocation_started=started)
    else:
        full = final_report(context, args.run)
        result = {key: full[key] for key in ("status", "games_per_model", "seed_groups", "difference", "ci95", "scope")}
    print(context.matrix.canonical(result))


if __name__ == "__main__":
    main()
