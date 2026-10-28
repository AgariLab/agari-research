"""Synthetic continuation-release gates; no real release/audit is written."""
from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True

from agari_kit.storage import Storage

spec = importlib.util.spec_from_file_location("continued_release", Path(__file__).with_name("build_model_release_v2_continued.py"))
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def fixtures():
    identity = {"weights": "a"*64, "descriptor": "b"*64, "kit": "c"*64, "research": "d"*64,
                "engine": "e"*64, "launcher": release.LAUNCHER_SHA256, "kit_files": 19, "research_files": 17}
    prior = {"weights": {"sha256": identity["weights"]},
        "agent": {"descriptor_artifact": {"sha256": identity["descriptor"]}},
        "online_validation": {"recorded_runtime": {"agari_kit": {"source_sha256": identity["kit"]},
            "engine": {"binary_sha256": identity["engine"]}}, "recorded_source": {"source_sha256": identity["research"]}}}
    failure = {"global_attempt": 23, "campaign_game": 22, "failure": "connection_closed",
        "start_game_observed": False, "events_count": 0, "requests": 0, "seat": None, "scores": None,
        "server_outcome": "unknown", "rating_impact": "unknown", "source_stage": "stopped_batch",
        "integrity_verified": True}
    fresh = {"status": "ready_with_interruption", "final_acceptance": False,
        "original_zero_failure_criterion_met": False, "target_completed_games": 100, "expected_attempts": 101,
        "integrity_errors": [], "pending_reasons": [], "attempt_failures": [failure],
        "attempts": [{"global_attempt": i, "status": "interrupted" if i == 23 else "complete"}
                     for i in range(1, 102)],
        "raw_files": [{"path": f"synthetic-only/{i}", "bytes": 1, "sha256": "f"*64} for i in range(410)],
        "totals": {"attempts": 101, "games_completed": 100, "healthy_completed": 100, "failed_attempts": 1,
                   "invalid_attempts": 0, "pending_attempts": 0},
        "stage_totals": {"first": {"attempts": 1, "games_completed": 1, "failed_attempts": 0},
            "stopped_batch": {"attempts": 22, "games_completed": 21, "failed_attempts": 1},
            "continuation": {"attempts": 78, "games_completed": 78, "failed_attempts": 0}},
        "identity_checks": {"actual": identity, "expected": identity, "matches": True}}
    return prior, fresh


def write_published(directory, fresh, *, changes=None, attempts=None):
    report = dict(copy.deepcopy(fresh), status="completed_with_interruption", final_acceptance=True,
                  published_at="2026-10-28T00:00:00+00:00")
    report.update(changes or {})
    docs = {"report.json": report, "attempts.json": fresh["attempts"] if attempts is None else attempts,
            "raw-sha256.json": fresh["raw_files"]}
    for name, value in docs.items():
        (directory/name).write_text(json.dumps(value)+"\n")
    (directory/"SHA256SUMS").write_text("".join(f"{release.sha256(directory/name)}  {name}\n" for name in sorted(docs)))


class ContinuationGateTests(unittest.TestCase):
    def setUp(self):
        self.prior, self.fresh = fixtures()

    def test_expected_100_completed_and_one_preserved_unknown_outcome_pass(self):
        release.require_complete_continuation(self.fresh, self.prior)

    def test_uninterrupted_success_or_missing_interruption_is_rejected(self):
        for mutate in (lambda f: f.update(original_zero_failure_criterion_met=True),
                       lambda f: f.update(status="ready"), lambda f: f.update(attempt_failures=[])):
            fresh = copy.deepcopy(self.fresh)
            mutate(fresh)
            with self.assertRaises(ValueError):
                release.require_complete_continuation(fresh, self.prior)

    def test_second_failure_and_incomplete_continuation_are_rejected(self):
        for key, value in (("failed_attempts", 2), ("healthy_completed", 99), ("games_completed", 99),
                           ("invalid_attempts", 1), ("pending_attempts", 1)):
            fresh = copy.deepcopy(self.fresh)
            fresh["totals"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                release.require_complete_continuation(fresh, self.prior)

    def test_invented_server_outcome_or_rating_impact_is_rejected(self):
        for key, value in (("server_outcome", "no-game"), ("rating_impact", "none"), ("scores", [25000]*4),
                           ("start_game_observed", True), ("global_attempt", 24), ("events_count", False)):
            fresh = copy.deepcopy(self.fresh)
            fresh["attempt_failures"][0][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "interruption"):
                release.require_complete_continuation(fresh, self.prior)

    def test_99_attempts_as_completed_or_wrong_stage_split_is_rejected(self):
        fresh = copy.deepcopy(self.fresh)
        fresh["stage_totals"]["stopped_batch"]["games_completed"] = 22
        fresh["stage_totals"]["continuation"]["games_completed"] = 77
        with self.assertRaisesRegex(ValueError, "split"):
            release.require_complete_continuation(fresh, self.prior)

    def test_raw_evidence_cannot_be_omitted_or_duplicated(self):
        for duplicate in (False, True):
            fresh = copy.deepcopy(self.fresh)
            if duplicate:
                fresh["raw_files"][-1] = fresh["raw_files"][0]
            else:
                fresh["raw_files"].pop()
            with self.subTest(duplicate=duplicate), self.assertRaisesRegex(ValueError, "410"):
                release.require_complete_continuation(fresh, self.prior)

    def test_source_identity_must_match_v1(self):
        fresh = copy.deepcopy(self.fresh)
        fresh["identity_checks"]["actual"]["research"] = "0"*64
        with self.assertRaisesRegex(ValueError, "immutable v1"):
            release.require_complete_continuation(fresh, self.prior)


class PublishedAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="agari-continued-release-test-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.patcher = patch.object(release, "FINAL_AUDIT", self.directory)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        _, self.fresh = fixtures()

    def test_published_core_matches_independent_continued_audit(self):
        write_published(self.directory, self.fresh)
        published = release.verify_published_audit(self.fresh)
        self.assertEqual(published["status"], "completed_with_interruption")
        self.assertFalse(published["original_zero_failure_criterion_met"])

    def test_resigned_false_success_or_wrong_status_is_rejected(self):
        for change in ({"original_zero_failure_criterion_met": True}, {"status": "passed"}):
            write_published(self.directory, self.fresh, changes=change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                release.verify_published_audit(self.fresh)

    def test_success_only_100_row_inventory_cannot_hide_attempt23(self):
        write_published(self.directory, self.fresh,
                        attempts=[row for row in self.fresh["attempts"] if row["global_attempt"] != 23])
        with self.assertRaisesRegex(ValueError, "attempts"):
            release.verify_published_audit(self.fresh)

    def test_changed_bytes_and_duplicate_checksums_are_rejected(self):
        write_published(self.directory, self.fresh)
        with (self.directory/"attempts.json").open("a") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(ValueError, "hash differs"):
            release.verify_published_audit(self.fresh)
        write_published(self.directory, self.fresh)
        path = self.directory/"SHA256SUMS"
        with path.open("a") as stream:
            stream.write(path.read_text().splitlines()[0]+"\n")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            release.verify_published_audit(self.fresh)


class PublicationTests(unittest.TestCase):
    def test_changed_legacy_helper_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory(prefix="agari-continued-release-test-") as tmp:
            helper = Path(tmp)/"old.py"
            helper.write_text("raise RuntimeError('must not run')")
            with patch.object(release, "BASE_HELPER", helper), patch.object(release, "load_module") as load:
                with self.assertRaisesRegex(ValueError, "helper changed"):
                    release.load_base_helper()
                load.assert_not_called()

    def test_pending_write_does_not_create_release_or_call_publisher(self):
        with tempfile.TemporaryDirectory(prefix="agari-continued-release-test-") as tmp:
            root = Path(tmp)
            with patch.object(release.Storage, "from_env", return_value=Storage(root)), \
                 patch.object(release, "EXPERIMENT", root), \
                 patch.object(release, "prepare", return_value=({"status": "pending"}, None)), \
                 patch.object(release, "load_base_helper") as load, \
                 patch.object(sys, "argv", ["continued.py", "--write"]), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(release.main(), 2)
                load.assert_not_called()
            self.assertFalse(json.loads(output.getvalue())["release_written"])
            self.assertEqual(list(root.iterdir()), [])

    def test_existing_release_is_not_overwritten(self):
        helper = release.load_base_helper()
        with tempfile.TemporaryDirectory(prefix="agari-continued-release-test-") as tmp:
            target = Path(tmp)/"synthetic.json"
            helper.publish_new(target, b"original")
            with self.assertRaises(FileExistsError):
                helper.publish_new(target, b"replacement")
            self.assertEqual(target.read_bytes(), b"original")

    def test_second_copy_failure_reports_partial_and_preserves_first(self):
        publish = release.load_base_helper().publish_new
        with tempfile.TemporaryDirectory(prefix="agari-continued-release-test-") as tmp:
            root = Path(tmp)
            model = root/"models"/release.RUN
            model.mkdir(parents=True)
            public, external = root/"model-release-v2.json", model/"release-v2.json"

            def fail_second(path, content):
                if path == public:
                    raise OSError("synthetic second-copy failure")
                publish(path, content)

            with patch.object(release.Storage, "from_env", return_value=Storage(root)), \
                 patch.object(release, "EXPERIMENT", root), \
                 patch.object(release, "prepare", return_value=({"status": "ready_with_interruption"},
                     {"release_id": "synthetic-test-only", "status": "completed_with_interruption"})), \
                 patch.object(release, "load_base_helper", return_value=SimpleNamespace(publish_new=fail_second)), \
                 patch.object(sys, "argv", ["continued.py", "--write"]), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(release.main(), 1)
            result = json.loads(output.getvalue())
            self.assertEqual(result["status"], "partial")
            self.assertTrue(result["release_written"])
            self.assertEqual(result["created_paths"], [str(external)])
            self.assertTrue(external.exists())
            self.assertFalse(public.exists())


class DirectSourceSplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="agari-continued-source-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old = self.root/"runs"/release.STOPPED
        self.new = self.root/"runs"/release.CONTINUATION
        self.failed = self.old/"game-000022"
        self.failed.mkdir(parents=True)
        self.new.mkdir(parents=True)
        self.write(self.old/"manifest.json", {"status": "stopped", "games_requested": 99,
                                             "games_recorded": 22, "error": "game_incomplete"})
        self.write(self.old/"summary.json", {"status": "stopped", "games_requested": 99,
            "games_attempted": 22, "games_completed": 21, "healthy_games": 21, "error": "game_incomplete"})
        self.write(self.new/"manifest.json", {"status": "complete", "games_requested": 78,
                                             "games_recorded": 78, "error": None})
        self.write(self.new/"summary.json", {"status": "complete", "games_requested": 78,
            "games_attempted": 78, "games_completed": 78, "healthy_games": 78, "error": None})
        (self.failed/"events.jsonl").write_bytes(b"")
        self.write(self.failed/"manifest.json", {"status": "incomplete", "games_recorded": 1})
        self.write(self.failed/"report.json", {"games_requested": 1, "games_completed": 0,
            "validation_passed": None, "degraded_games": 0, "counts": {}})
        self.write(self.failed/"games.jsonl", {"mode": "ranked", "status": "interrupted", "seat": None,
            "scores": None, "validation_passed": None, "degraded": False, "counts": {},
            "failure": "connection_closed", "game_index": 0})

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value)+"\n")

    def test_expected_split_retains_the_original_stopped_batch(self):
        result = release.verify_source_split(Storage(self.root))
        self.assertEqual(result["completed_before_continuation"], 22)
        self.assertEqual(result["completed_in_continuation"], 78)
        self.assertEqual(result["stages"][1]["status"], "stopped")
        self.assertEqual(result["failure_outcome"]["server_outcome"], "unknown")
        self.assertEqual(result["failure_raw_files"]["events.jsonl"]["bytes"], 0)

    def test_failure_event_rewrite_is_rejected_independently_of_summary(self):
        (self.failed/"events.jsonl").write_text('{"event":"start_game"}\n')
        with self.assertRaisesRegex(ValueError, "failure event stream changed"):
            release.verify_source_split(Storage(self.root))

    def test_old_batch_cannot_be_relabeled_complete(self):
        summary = json.loads((self.old/"summary.json").read_text())
        summary["status"] = "complete"
        self.write(self.old/"summary.json", summary)
        with self.assertRaisesRegex(ValueError, "must remain recorded as stopped"):
            release.verify_source_split(Storage(self.root))


if __name__ == "__main__":
    unittest.main()
