"""Synthetic Akochan evidence only; never load real opponent parameters or play."""
from contextlib import nullcontext
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "scripts" / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


helper = module("akochan_helper_tests", "phase2_akochan.py")
matrix = module("akochan_matrix_tests", "phase2_evaluation_matrix.py")
comparator = module("akochan_comparator_tests", "compare_evaluations.py")


def fixture(side, seeds):
    return {"rules": {"preset": "tenhou"}, "engine": {"version": "0.4.10", "binary_sha256": "e" * 64},
            "code": {"package": "agari-kit", "source_sha256": "f" * 64}, "python": "fixed-python",
            "lineup": [side, "akochan", "efficiency", "tsumogiri"],
            "lineup_provenance": [{"kind": "builtin", "name": name} for name in (side, "akochan", "efficiency", "tsumogiri")],
            "seeds": list(seeds)}


def explicit(old, new):
    return {"schema_version": 1, "kind": "different-candidates", "hypothesis": "Synthetic candidate comparison",
            "primary_metric": "mean_rank", "baseline_provenance_sha256": comparator.identity_sha256(old["lineup_provenance"][0]),
            "candidate_provenance_sha256": comparator.identity_sha256(new["lineup_provenance"][0]),
            "conditions_sha256": comparator.identity_sha256(comparator.comparison_conditions(old))}


def records(manifest, candidate=False):
    from agari_kit.evaluation import DIAGNOSTIC_COUNTS
    result = []
    for seed in manifest["seeds"]:
        for rotation in range(4):
            rank = (seed + rotation) % 4
            if candidate:
                rank = max(0, rank-1) if seed % 3 else min(3, rank+1)
            values = [41000, 29000, 19000, 11000]
            scores = [0] * 4
            scores[rotation] = values.pop(rank)
            for seat in range(4):
                if seat != rotation:
                    scores[seat] = values.pop(0)
            order = sorted(range(4), key=lambda seat: (-scores[seat], seat))
            diagnostics = dict.fromkeys(DIAGNOSTIC_COUNTS, 0)
            diagnostics.update(hands=10, wins=seed % 3, deal_ins=seed % 2, riichi_hands=2, open_hands=4)
            result.append({"seed": seed, "rotation": rotation, "scores": scores, "candidate_diagnostics": diagnostics,
                "candidate_decisions": seed % 3 + 1, "decisions": 4 * (seed % 3 + 1), "elapsed_seconds": .1 * (seed % 3 + 1),
                "ranks": [order.index(seat) + 1 for seat in range(4)], "illegal_actions": 0,
                "seats": [manifest["lineup"][(seat-rotation) % 4] for seat in range(4)]})
    return result


def write_run(path, identity):
    path.mkdir(parents=True)
    rows = records(identity, identity["lineup"][0] == "candidate")
    manifest = dict(identity, status="complete", games_expected=len(rows), games_completed=len(rows))
    (path / "manifest.json").write_text(json.dumps(manifest))
    (path / "games.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    latencies = [float(row["seed"] % 100 + row["rotation"] + offset + .25)
                 for row in rows for offset in range(row["candidate_decisions"])]
    (path / "candidate-latencies.json").write_text(json.dumps(latencies))
    (path / "report.json").write_text(json.dumps({"status": "complete", "games": len(rows),
        "elapsed_seconds": sum(row["elapsed_seconds"] for row in rows), "decisions": sum(row["decisions"] for row in rows),
        "illegal_actions": 0, "candidate_decisions": len(latencies),
        "attempts": {"total": len(identity["seeds"]), "failed": 0, "unsealed": 0}}))
    return comparator.read_run(path)


class MergedStatisticsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.seeds = list(range(2670000, 2670064))
        self.phases, self.segments, self.contracts = {}, {}, {}
        for phase, seeds in (("initial", self.seeds[:16]), ("extension", self.seeds[16:])):
            self.phases[phase] = {side: fixture(side, seeds) for side in helper.SIDES}
            self.segments[phase] = {side: write_run(self.root / phase / side, self.phases[phase][side]) for side in helper.SIDES}
            self.contracts[phase] = explicit(*[self.phases[phase][side] for side in helper.SIDES])

    def combine(self):
        return helper.combine_records(comparator, self.segments, self.phases, self.contracts, self.seeds)

    def test_16_plus_48_reproduces_single_64_group_comparison_exactly(self):
        full = {side: fixture(side, self.seeds) for side in helper.SIDES}
        for side in helper.SIDES:
            write_run(self.root / "full" / side, full[side])
        reference = comparator.compare(self.root / "full/incumbent", self.root / "full/candidate",
            contract=explicit(full["incumbent"], full["candidate"]), bootstrap_samples=300, bootstrap_seed=17)
        rows, _ = self.combine()
        merged = helper.paired_statistics(comparator, rows["incumbent"], rows["candidate"], self.seeds,
                                           bootstrap_samples=300, bootstrap_seed=17)
        for key in ("games_per_model", "seed_groups", "seeds", "difference", "ci95", "ci_method", "bootstrap_samples", "bootstrap_seed"):
            self.assertEqual(merged[key], reference[key], key)
        self.assertEqual(merged["baseline"]["metrics"], reference["baseline"]["metrics"])
        self.assertEqual(merged["candidate"]["metrics"], reference["candidate"]["metrics"])
        self.assertEqual(len(merged["paired_seed_groups"]), 64)

    def test_repeated_initial_seed_cannot_count_as_extension_evidence(self):
        self.phases["extension"]["incumbent"]["seeds"][0] = self.seeds[0]
        with self.assertRaisesRegex(ValueError, "duplicated seeds"):
            self.combine()

    def test_missing_phase_or_rotation_is_rejected(self):
        saved = self.segments.pop("extension")
        with self.assertRaisesRegex(ValueError, "Both complete"):
            self.combine()
        self.segments["extension"] = saved
        self.segments["extension"]["candidate"][1].pop((self.seeds[-1], 3))
        with self.assertRaisesRegex(ValueError, "rotations"):
            self.combine()

    def test_wrong_engine_is_rejected_even_if_both_extension_sides_match(self):
        for side in helper.SIDES:
            self.phases["extension"][side]["engine"] = {"binary_sha256": "other"}
            self.segments["extension"][side][0]["engine"] = {"binary_sha256": "other"}
        with self.assertRaisesRegex(ValueError, "engine or lineup"):
            self.combine()

    def test_changed_candidate_or_opponent_does_not_merge(self):
        self.segments["extension"]["candidate"][0]["lineup_provenance"][0]["name"] = "substituted"
        with self.assertRaisesRegex(ValueError, "registered Akochan fixture"):
            self.combine()

    def test_duplicate_raw_game_is_rejected_before_union(self):
        path = self.root / "initial/incumbent/games.jsonl"
        lines = path.read_text().splitlines()
        lines[-1] = lines[0]
        path.write_text("\n".join(lines) + "\n")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            comparator.read_run(path.parent)

    def report_context(self):
        def verify(rt, path, expected):
            actual, _, hashes = comparator.read_run(path)
            helper.require(all(actual[key] == expected[key] for key in (*helper.CONDITIONS, "seeds")), "Synthetic fixture mismatch")
            return dict(hashes, **{"plan.json": "sealed-plan", "completion.json": "sealed-completion"})
        tools = SimpleNamespace(verify_fixture=Mock(side_effect=verify), digest=matrix.digest, publish=matrix.publish, read=matrix.read)
        rt = SimpleNamespace(comparator=comparator, protocol={"seeds": {"akochan_exploratory": {"start": self.seeds[0], "count": 64}}},
                             contract=SimpleNamespace(seed_range=lambda config: set(range(config["start"], config["start"] + config["count"]))))
        registration = {"phases": self.phases, "contracts": self.contracts, "bootstrap_samples": 2000, "bootstrap_seed": 0,
                        "implementation": {"helper_sha256": "synthetic"}, "winner": "B-2027", "scale": "large", "scope": "Exploratory only"}
        matrix.publish(self.root / "registration.json", registration)
        return SimpleNamespace(matrix=tools, rt=rt), registration

    def test_full_report_sums_diagnostics_and_only_publishes_exploration_report(self):
        context, registration = self.report_context()
        before = {path.relative_to(self.root): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        with patch.object(helper, "load_registration", return_value=(self.root, registration)):
            report = helper.final_report(context, "runs/synthetic")
        self.assertEqual((report["seed_groups"], report["games_per_model"]), (64, 256))
        self.assertEqual(report["baseline"]["diagnostics"]["hands"], 2560)
        self.assertEqual(report["baseline"]["cost"]["attempts"], {"total": 64, "failed": 0, "unsealed": 0})
        self.assertIn("local IPC", report["candidate"]["cost"]["latency_scope"])
        self.assertIn("report.json", report["input_evidence"]["initial"]["incumbent"])
        self.assertIn("candidate-latencies.json", report["input_evidence"]["extension"]["candidate"])
        self.assertEqual(report["strength_acceptance"], "exploratory-only")
        self.assertFalse(report["promotion_sample_size_met"])
        self.assertFalse(report["candidate_selection_changed"])
        self.assertFalse(report["final_gate_changed"])
        self.assertEqual(context.matrix.verify_fixture.call_count, 8)
        after = {path.relative_to(self.root): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(set(after) - set(before), {Path("report.json")})
        self.assertTrue(all(after[name] == value for name, value in before.items()))

    def test_unsealed_or_failed_campaign_never_publishes_report(self):
        context, registration = self.report_context()
        context.matrix.verify_fixture.side_effect = ValueError("failed or unsealed")
        with patch.object(helper, "load_registration", return_value=(self.root, registration)), self.assertRaisesRegex(ValueError, "unsealed"):
            helper.final_report(context, "runs/synthetic")
        self.assertFalse((self.root / "report.json").exists())


class CombinedCostTests(unittest.TestCase):
    def setUp(self):
        self.campaigns = {}
        for phase, groups, values in (("initial", 16, [1.] * 63 + [1000.]), ("extension", 48, [100.] * 192)):
            self.campaigns[phase] = ({"status": "complete", "games": groups * 4, "elapsed_seconds": float(groups),
                "decisions": groups * 16, "illegal_actions": 0, "candidate_decisions": len(values),
                "attempts": {"total": groups, "failed": 0, "unsealed": 0}}, values)

    def test_costs_sum_and_p95_uses_all_samples_instead_of_phase_percentiles(self):
        result = helper.combined_cost(self.campaigns)
        self.assertEqual(result["elapsed_seconds"], 64.)
        self.assertEqual(result["decisions"], 1024)
        self.assertEqual(result["candidate_decisions"], 256)
        self.assertEqual(result["candidate_latency_ms_p95"], 100.)
        self.assertEqual(result["candidate_latency_ms_max"], 1000.)
        self.assertNotEqual(result["candidate_latency_ms_p95"], (64 * 1 + 192 * 100) / 256)

    def test_missing_latency_sample_or_failed_attempt_cannot_be_reported(self):
        original = self.campaigns["extension"][1].pop()
        with self.assertRaisesRegex(ValueError, "latency samples"):
            helper.combined_cost(self.campaigns)
        self.campaigns["extension"][1].append(original)
        self.campaigns["extension"][0]["attempts"]["failed"] = 1
        with self.assertRaisesRegex(ValueError, "Failed"):
            helper.combined_cost(self.campaigns)


class ExecutionGateTests(unittest.TestCase):
    def setUp(self):
        self.output = Path("/synthetic/no-real-output")
        self.registration = {"matrix": "runs/matrix", "phases": {phase: {side: {"fixture": f"{phase}/{side}"}
                             for side in helper.SIDES} for phase in helper.PHASE_COUNTS}}
        self.context = SimpleNamespace(rt=None, matrix=SimpleNamespace(
            verify_fixture=Mock(return_value={"hash": "sealed"}), execute=Mock(return_value={"status": "complete", "games": 192}),
            digest=Mock(return_value="registration-hash")))

    def execute(self, phase="extension", resume=False):
        with patch.object(helper, "load_registration", return_value=(self.output, self.registration)):
            return helper.run_phase(self.context, "runs/exploration", phase, "candidate", resume=resume,
                                     max_wall_seconds=172800, invocation_started=123.)

    def test_extension_requires_both_initial_campaigns_and_zero_failures(self):
        self.context.matrix.verify_fixture.side_effect = [{"hash": "sealed"}, ValueError("failed or unsealed")]
        with self.assertRaisesRegex(ValueError, "failed or unsealed"):
            self.execute()
        self.context.matrix.execute.assert_not_called()
        self.assertEqual(self.context.matrix.verify_fixture.call_count, 2)

    def test_initial_missing_or_incomplete_never_starts_extension(self):
        self.context.matrix.verify_fixture.side_effect = FileNotFoundError("initial completion receipt missing")
        with self.assertRaises(FileNotFoundError):
            self.execute()
        self.context.matrix.execute.assert_not_called()

    def test_valid_extension_forwards_resume_and_total_invocation_budget(self):
        self.assertEqual(self.execute(resume=True)["status"], "complete")
        arguments = self.context.matrix.execute.call_args
        self.assertEqual(arguments.args[1], self.registration["phases"]["extension"]["candidate"])
        self.assertTrue(arguments.kwargs["resume"])
        self.assertEqual(arguments.kwargs["max_wall_seconds"], 172800)
        self.assertEqual(arguments.kwargs["invocation_started"], 123.)
        self.assertEqual(self.context.matrix.verify_fixture.call_count, 4)

    def test_initial_is_independent_of_unfinished_other_side(self):
        self.execute(phase="initial")
        self.context.matrix.verify_fixture.assert_not_called()
        self.context.matrix.execute.assert_called_once()

    def test_changed_registration_after_run_is_rejected_without_rewriting_evidence(self):
        changed = dict(self.registration, winner="another")
        with patch.object(helper, "load_registration", side_effect=[(self.output, self.registration), (self.output, changed)]):
            with self.assertRaisesRegex(ValueError, "changed during execution"):
                helper.run_phase(self.context, "runs/exploration", "initial", "candidate", resume=True,
                                 max_wall_seconds=172800, invocation_started=0.)
        self.context.matrix.execute.assert_called_once()

    def test_invalid_budget_rejected_before_loading_or_playing(self):
        with patch.object(helper, "load_registration") as load:
            for value in (True, 0, float("nan"), float("inf"), 172801):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    helper.run_phase(self.context, "runs/exploration", "initial", "candidate", resume=False,
                                     max_wall_seconds=value, invocation_started=0.)
            load.assert_not_called()
        self.context.matrix.execute.assert_not_called()

    def test_copied_registration_cannot_silently_redirect_the_campaign(self):
        context = SimpleNamespace(matrix=SimpleNamespace(read=Mock(return_value={"directory": "runs/registered"})),
                                  rt=SimpleNamespace(path=lambda path: Path("/synthetic") / path))
        with patch.object(helper, "registration_for") as rebuild, self.assertRaisesRegex(ValueError, "another output directory"):
            helper.load_registration(context, "runs/copied")
        rebuild.assert_not_called()


class FrozenPlanTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.descriptor = self.root / "akochan.json"
        self.descriptor.write_text("{}")
        self.settings = {"start": 2670000, "initial_count": 16, "count": 64, "rotations": 4}
        self.rt = SimpleNamespace(storage=SimpleNamespace(root=self.root), comparator=comparator,
            protocol={"seeds": {"akochan_exploratory": self.settings}, "evaluation": {"akochan_opponents": ["akochan", "efficiency", "tsumogiri"]}},
            artifact=lambda path: {"path": path.name, "sha256": matrix.digest(path)},
            contract=SimpleNamespace(storage_relative=lambda storage, path: path.relative_to(storage.root).as_posix(),
                seed_range=lambda settings: set(range(settings["start"], settings["start"] + settings["count"]))))
        def current_identity(rt, lineup, seeds):
            result = fixture(lineup[0], seeds)
            result["lineup"] = lineup
            result["lineup_provenance"][1] = {"kind": "external", "name": lineup[1], "factory": helper.AKOCHAN_FACTORY}
            return result
        self.tools = SimpleNamespace(require=matrix.require, identity=Mock(side_effect=current_identity), content_digest=matrix.content_digest,
                                     explicit_contract=lambda rt, old, new: explicit(old, new))
        self.context = SimpleNamespace(matrix=self.tools, rt=self.rt)
        self.frozen = {"winner": "B-2027", "fixtures": {side: {"lineup": [side]} for side in helper.SIDES}}
        self.development = {"winner": "B-2027", "frozen": "development"}
        self.plan = {"schema_version": 1, "status": "commands-only-not-played", "winner": "B-2027",
                     "development_sha256": matrix.content_digest(self.development), "phases": {}, "contracts": {}}
        offset = 0
        for phase, count in helper.PHASE_COUNTS.items():
            seeds = list(range(self.settings["start"] + offset, self.settings["start"] + offset + count))
            self.plan["phases"][phase] = {side: current_identity(self.rt, [side, "@" + str(self.descriptor), "efficiency", "tsumogiri"], seeds)
                                           for side in helper.SIDES}
            self.plan["contracts"][phase] = explicit(*[self.plan["phases"][phase][side] for side in helper.SIDES])
            offset += count

    def validate(self):
        with patch("agari_kit.agent.managed_agents", return_value=nullcontext()) as constructor:
            result = helper.validate_plan(self.context, {}, self.frozen, self.development, self.plan)
            constructor.assert_called_once_with(["@" + str(self.descriptor)])
            return result

    def test_exact_plan_matches_unique_final_candidate_and_hashes_real_descriptor(self):
        result = self.validate()
        self.assertEqual(result["descriptor"]["sha256"], matrix.digest(self.descriptor))
        self.assertEqual(self.tools.identity.call_count, 4)

    def test_pre_freeze_or_other_candidate_plan_cannot_start(self):
        self.plan["winner"] = "A-2026"
        with self.assertRaisesRegex(ValueError, "uniquely frozen"):
            self.validate()
        tools = SimpleNamespace(load_matrix=Mock(return_value={}), frozen_final=Mock(side_effect=ValueError("no valid final-freeze")))
        with self.assertRaisesRegex(ValueError, "final-freeze"):
            helper.registration_for(SimpleNamespace(matrix=tools, rt=None), "runs/matrix", "runs/exploration")
        tools.frozen_final.assert_called_once()

    def test_overlapping_extension_or_source_identity_change_is_rejected(self):
        saved = copy.deepcopy(self.plan)
        self.plan["phases"]["extension"]["candidate"]["seeds"][0] = self.settings["start"]
        with self.assertRaisesRegex(ValueError, "fixture"):
            self.validate()
        self.plan = saved
        self.plan["phases"]["initial"]["incumbent"]["lineup_provenance"][1]["factory"] = "another:Agent"
        with self.assertRaisesRegex(ValueError, "fixture"):
            self.validate()

    def test_missing_initial_side_is_rejected(self):
        self.plan["phases"]["initial"].pop("candidate")
        with self.assertRaisesRegex(ValueError, "both sides"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
