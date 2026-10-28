"""Pure synthetic attempt attribution and measured-cost accounting boundaries."""
import copy
import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).parents[1] / "scripts/report_teacher_student_results.py"
spec = importlib.util.spec_from_file_location("student_costs_test", SCRIPT)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)

TARGETS = {"grp": 10, "cql": 30}
CHECKPOINT = {"path": "checkpoints/phase2-C-2026/latest.pt", "sha256": "a" * 64}


def start(*, resume=False, limit=None):
    return {"resume": resume, "stop_after_updates": limit}


def finish(progress, *, status="complete", reason=None, runner_exit=0, child_exit=0, checkpoint=None):
    row = {"status": status, "stop_reason": reason, "exit_code": runner_exit,
           "process": {"exit_code": child_exit}, "cli_result": {"progress": progress}}
    if checkpoint is not None:
        row["latest_checkpoint"] = copy.deepcopy(checkpoint)
    return row


class AttemptAttributionTests(unittest.TestCase):
    def classify(self, started, finished, prior=()):
        return report.classify_attempt(started, finished, list(prior), "cql", TARGETS)

    def grp_boundary(self):
        return finish({"grp": 10, "cql": 0}, status="interrupted", reason="update_limit", runner_exit=2,
                      checkpoint=CHECKPOINT)

    def test_fresh_grp_and_hash_chained_resume_are_distinct_cost_stages(self):
        first = self.grp_boundary()
        grp = self.classify(start(limit=10), first)
        self.assertEqual(grp, {"stage": "grp", "start_progress": {"grp": 0, "cql": 0},
                               "end_progress": {"grp": 10, "cql": 0}, "planned_grp_boundary": True})
        second = finish(dict(TARGETS))
        second["resume_checkpoint"] = copy.deepcopy(CHECKPOINT)
        cql = self.classify(start(resume=True), second, [first])
        self.assertEqual(cql, {"stage": "cql", "start_progress": {"grp": 10, "cql": 0},
                               "end_progress": dict(TARGETS), "planned_grp_boundary": False})

    def test_single_invocation_crossing_stage_boundary_is_mixed_not_prorated(self):
        result = self.classify(start(), finish(dict(TARGETS)))
        self.assertEqual(result["stage"], "mixed-grp-cql")
        self.assertFalse(result["planned_grp_boundary"])
        self.assertNotIn("grp_wall_fraction", result)

    def test_resume_is_not_synonymous_with_cql_and_limit_is_additional(self):
        prior = finish({"grp": 3, "cql": 0}, status="paused", reason="wall_time_budget", runner_exit=2,
                       checkpoint=CHECKPOINT)
        current = self.grp_boundary()
        current["resume_checkpoint"] = copy.deepcopy(CHECKPOINT)
        result = self.classify(start(resume=True, limit=7), current, [prior])
        self.assertEqual(result["stage"], "grp")
        self.assertEqual(result["start_progress"], {"grp": 3, "cql": 0})
        self.assertFalse(result["planned_grp_boundary"])

    def test_missing_hash_path_or_conflicting_predecessor_leaves_cost_unattributed(self):
        current = finish(dict(TARGETS))
        current["resume_checkpoint"] = copy.deepcopy(CHECKPOINT)
        first = self.grp_boundary()
        scenarios = {
            "no predecessor": [],
            "different sha": [dict(first, latest_checkpoint=dict(CHECKPOINT, sha256="b" * 64))],
            "different path": [dict(first, latest_checkpoint=dict(CHECKPOINT, path="checkpoints/other/latest.pt"))],
            "conflicting progress": [first, dict(first, cli_result={"progress": {"grp": 9, "cql": 0}})],
            "unsealed progress": [dict(first, cli_result={})],
        }
        for name, prior in scenarios.items():
            with self.subTest(name=name):
                result = self.classify(start(resume=True), current, prior)
                self.assertEqual(result["stage"], "unattributed")
                self.assertIsNone(result["start_progress"])
                self.assertEqual(result["end_progress"], TARGETS)
                self.assertFalse(result["planned_grp_boundary"])

    def test_repeated_same_checkpoint_with_identical_progress_is_unambiguous(self):
        first = self.grp_boundary()
        second = finish(dict(TARGETS))
        second["resume_checkpoint"] = copy.deepcopy(CHECKPOINT)
        result = self.classify(start(resume=True), second, [first, copy.deepcopy(first)])
        self.assertEqual(result["stage"], "cql")

    def test_budget_stop_failure_and_wrong_limit_never_become_planned_grp_pause(self):
        variants = {
            "budget boundary": dict(self.grp_boundary(), status="paused", stop_reason="wall_time_budget"),
            "failed wrapper": dict(self.grp_boundary(), status="failed", exit_code=1),
            "failed child": dict(self.grp_boundary(), process={"exit_code": 1}),
            "wrong exit code": dict(self.grp_boundary(), exit_code=0),
        }
        for name, finished in variants.items():
            with self.subTest(name=name):
                self.assertFalse(self.classify(start(limit=10), finished)["planned_grp_boundary"])
        for limit in (None, 9, 11):
            with self.subTest(limit=limit):
                self.assertFalse(self.classify(start(limit=limit), self.grp_boundary())["planned_grp_boundary"])

    def test_zero_invalid_or_out_of_order_progress_is_not_guessed(self):
        for progress in ({"grp": 0, "cql": 0}, {"grp": 11, "cql": 0}, {"grp": True, "cql": 0},
                         {"grp": 1.5, "cql": 0}, {"grp": 0, "cql": 3}, {"grp": 10}):
            with self.subTest(progress=progress):
                result = self.classify(start(), finish(progress))
                self.assertEqual(result["stage"], "unattributed")
                self.assertFalse(result["planned_grp_boundary"])
        current = finish({"grp": 9, "cql": 0})
        current["resume_checkpoint"] = copy.deepcopy(CHECKPOINT)
        self.assertEqual(self.classify(start(resume=True), current, [self.grp_boundary()])["stage"], "unattributed")

    def test_missing_process_or_unsealed_attempt_has_no_inferred_stage(self):
        for started, finished in ((None, None), (start(), None),
                                  (start(), {"status": "failed"}),
                                  (start(resume=True), {"status": "already-complete"})):
            with self.subTest(finished=finished):
                result = self.classify(started, finished)
                self.assertEqual(result, {"stage": "unattributed", "start_progress": None,
                                          "end_progress": None, "planned_grp_boundary": False})

    def test_bc_method_is_known_even_if_failed_child_has_no_progress_snapshot(self):
        result = report.classify_attempt(start(), {"status": "failed", "process": {"exit_code": 9}}, [], "bc", {"bc": 30})
        self.assertEqual(result["stage"], "bc")
        self.assertIsNone(result["end_progress"])


class CostAggregationTests(unittest.TestCase):
    def row(self, *, status="complete", **values):
        return {"status": status, "costs": dict.fromkeys(report.COSTS) | values}

    def test_failed_measured_attempt_is_kept_and_missing_costs_stay_unknown(self):
        rows = [self.row(status="failed", invocation_wall_seconds=12, child_wall_seconds=10,
                         user_seconds=15, system_seconds=2, max_rss_bytes=1024),
                self.row(status="failed", invocation_wall_seconds=3), self.row(status="unsealed")]
        result = report.summarize_costs(rows)
        self.assertEqual(result["invocation_wall_seconds"], {"observed_sum": 15, "known_attempts": 2, "unknown_attempts": 1})
        self.assertEqual(result["child_wall_seconds"], {"observed_sum": 10, "known_attempts": 1, "unknown_attempts": 2})
        self.assertEqual(result["user_seconds"]["observed_sum"], 15)  # Multi-thread CPU may exceed wall time.
        self.assertEqual(result["max_rss_bytes"], {"observed_max": 1024, "known_attempts": 1, "unknown_attempts": 2})

    def test_overlapping_invocations_sum_work_without_inventing_batch_wall(self):
        first = self.row(invocation_wall_seconds=10, child_wall_seconds=8, max_rss_bytes=2048)
        second = self.row(invocation_wall_seconds=10, child_wall_seconds=9, max_rss_bytes=1024)
        first.update(started_unix_seconds=100, finished_unix_seconds=110)
        second.update(started_unix_seconds=101, finished_unix_seconds=111)
        result = report.summarize_costs([first, second])
        self.assertEqual(result["invocation_wall_seconds"]["observed_sum"], 20)
        self.assertEqual(result["child_wall_seconds"]["observed_sum"], 17)
        self.assertEqual(result["max_rss_bytes"]["observed_max"], 2048)
        self.assertNotIn("batch_wall_seconds", result)

    def test_no_observations_are_null_but_observed_zero_is_known(self):
        for rows, unknown in (([], 0), ([self.row(), self.row(status="unsealed")], 2)):
            with self.subTest(unknown=unknown):
                for value in report.summarize_costs(rows).values():
                    self.assertEqual(value["known_attempts"], 0)
                    self.assertEqual(value["unknown_attempts"], unknown)
                    self.assertIsNone(value.get("observed_sum", value.get("observed_max")))
        zero = report.summarize_costs([self.row(system_seconds=0)])
        self.assertEqual(zero["system_seconds"], {"observed_sum": 0, "known_attempts": 1, "unknown_attempts": 0})

    def test_invalid_cost_values_are_rejected_instead_of_poisoning_totals(self):
        for field in report.COSTS:
            for value in (-1, float("inf"), float("nan"), True, "10"):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, "Invalid recorded attempt cost"):
                        report.summarize_costs([self.row(**{field: value})])


if __name__ == "__main__":
    unittest.main()
