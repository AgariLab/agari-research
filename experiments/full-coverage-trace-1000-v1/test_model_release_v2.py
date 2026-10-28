"""Synthetic publication-gate tests; never create a real release or ranked audit."""

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

SPEC = importlib.util.spec_from_file_location("model_release_v2", Path(__file__).with_name("build_model_release_v2.py"))
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def fresh_fixture():
    return {"status": "ready", "final_acceptance": False, "errors": [],
            "auditor_sha256": "a" * 64, "games": [{"game_number": 1, "status": "complete"}],
            "raw_files": [{"path": "synthetic-only", "bytes": 1, "sha256": "b" * 64}],
            "totals": {"games_verified_complete": 1}}


def write_fixture(directory, fresh, *, report_change=None, games=None, raw_files=None):
    report = dict(copy.deepcopy(fresh), status="passed", final_acceptance=True,
                  published_at="2026-10-28T00:00:00+00:00")
    if report_change:
        report.update(report_change)
    documents = {"report.json": report, "games.json": fresh["games"] if games is None else games,
                 "raw-sha256.json": fresh["raw_files"] if raw_files is None else raw_files}
    for name, value in documents.items():
        (directory / name).write_text(json.dumps(value) + "\n")
    (directory / "SHA256SUMS").write_text("".join(
        f"{release.sha256(directory/name)}  {name}\n" for name in sorted(documents)))


class FinalAuditBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="agari-release-v2-test-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.patcher = patch.object(release, "FINAL_AUDIT", self.directory)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.fresh = fresh_fixture()

    def test_only_publication_fields_may_differ_from_fresh_independent_audit(self):
        write_fixture(self.directory, self.fresh)
        report = release.verify_published_audit(self.fresh)
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["final_acceptance"])

    def test_passed_label_with_self_consistent_hashes_cannot_replace_fresh_evidence(self):
        write_fixture(self.directory, self.fresh, report_change={"totals": {"games_verified_complete": 100}})
        with self.assertRaisesRegex(ValueError, "fresh independent"):
            release.verify_published_audit(self.fresh)

    def test_separate_game_and_raw_inventories_must_match_fresh_result(self):
        for change in ({"games": []}, {"raw_files": []}):
            with self.subTest(change=change):
                write_fixture(self.directory, self.fresh, **change)
                with self.assertRaisesRegex(ValueError, "differ"):
                    release.verify_published_audit(self.fresh)

    def test_changed_bytes_and_duplicate_checksums_are_rejected(self):
        write_fixture(self.directory, self.fresh)
        with (self.directory/"games.json").open("a") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(ValueError, "hash differs"):
            release.verify_published_audit(self.fresh)
        write_fixture(self.directory, self.fresh)
        sums = self.directory/"SHA256SUMS"
        with sums.open("a") as stream:
            stream.write(sums.read_text().splitlines()[0] + "\n")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            release.verify_published_audit(self.fresh)

    def test_unaccepted_report_and_symlink_are_rejected(self):
        write_fixture(self.directory, self.fresh, report_change={"final_acceptance": False})
        with self.assertRaisesRegex(ValueError, "separately accepted"):
            release.verify_published_audit(self.fresh)
        target = self.directory/"games.json"
        target.unlink()
        target.symlink_to(self.directory/"raw-sha256.json")
        with self.assertRaisesRegex(ValueError, "symlink"):
            release.verify_published_audit(self.fresh)


class ReleaseGateTests(unittest.TestCase):
    def test_second_copy_failure_reports_the_already_created_first_copy(self):
        with tempfile.TemporaryDirectory(prefix="agari-release-v2-test-") as tmp:
            root = Path(tmp)
            model = root/"models"/release.RUN
            model.mkdir(parents=True)
            external = model/"release-v2.json"
            public = root/"model-release-v2.json"
            original_publish = release.publish_new

            def publish(path, content):
                if path == public:
                    raise OSError("synthetic second-copy failure")
                original_publish(path, content)

            with patch.object(release.Storage, "from_env", return_value=Storage(root)), \
                 patch.object(release, "EXPERIMENT", root), \
                 patch.object(release, "prepare", return_value=({"status": "ready"}, {"release_id": "synthetic-test-only"})), \
                 patch.object(release, "publish_new", side_effect=publish), \
                 patch.object(sys, "argv", ["build_model_release_v2.py", "--write"]), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(release.main(), 1)
            report = json.loads(output.getvalue())
            self.assertEqual(report["status"], "partial")
            self.assertTrue(report["release_written"])
            self.assertEqual(report["created_paths"], [str(external)])
            self.assertTrue(external.is_file())
            self.assertFalse(public.exists())
            self.assertEqual(json.loads(external.read_text())["release_id"], "synthetic-test-only")

    def test_ready_label_cannot_bypass_the_100_game_requirement(self):
        with tempfile.TemporaryDirectory(prefix="agari-release-v2-test-") as tmp:
            root = Path(tmp)
            script = root/"synthetic-auditor.py"
            script.write_text("# Synthetic identity only; never executed.\n")
            fresh = {"status": "ready", "final_acceptance": False,
                     "auditor_sha256": release.sha256(script), "target_games": 100,
                     "games": [{}] * 99, "errors": [], "pending_reasons": [],
                     "totals": {"games_verified_complete": 99, "invalid_games": 0, "pending_games": 0}}
            with patch.object(release, "AUDITOR", script), \
                 patch.object(release, "verify_v1", return_value={}), \
                 patch.object(release, "load_module", return_value=SimpleNamespace(audit_ranked=lambda *_: fresh)):
                with self.assertRaisesRegex(ValueError, "exactly 100"):
                    release.prepare(Storage(root))
            self.assertEqual(list(root.iterdir()), [script])

    def test_pending_write_refuses_without_creating_any_release(self):
        with tempfile.TemporaryDirectory(prefix="agari-release-v2-test-") as tmp:
            root = Path(tmp)
            script = root/"synthetic-auditor.py"
            script.write_text("# Synthetic identity only; never executed.\n")
            fresh = {"status": "pending", "auditor_sha256": release.sha256(script),
                     "totals": {"games_verified_complete": 9}, "errors": [],
                     "pending_reasons": ["campaign_running"]}
            calls = []

            def audit(*args, **kwargs):
                calls.append((args, kwargs))
                return fresh

            with patch.object(release.Storage, "from_env", return_value=Storage(root)), \
                 patch.object(release, "EXPERIMENT", root), \
                 patch.object(release, "AUDITOR", script), \
                 patch.object(release, "verify_v1", return_value={}), \
                 patch.object(release, "load_module", return_value=SimpleNamespace(audit_ranked=audit)), \
                 patch.object(sys, "argv", ["build_model_release_v2.py", "--write"]), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(release.main(), 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "pending")
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][1], {})  # Production path never overrides identities.
            self.assertEqual(list(root.iterdir()), [script])

    def test_immutable_v1_hash_is_checked_before_importing_its_verifier(self):
        with tempfile.TemporaryDirectory(prefix="agari-release-v2-test-") as tmp:
            root = Path(tmp)
            model = root/"models"/release.RUN
            model.mkdir(parents=True)
            (model/"release-v1.json").write_text("{}")
            (root/"model-release-v1.json").write_text("{}")
            (root/"build_model_release.py").write_text("raise RuntimeError('must not execute')")
            with patch.object(release, "EXPERIMENT", root), patch.object(release, "load_module") as load:
                with self.assertRaisesRegex(ValueError, "Immutable v1 hash"):
                    release.verify_v1(Storage(root))
                load.assert_not_called()

    def test_exclusive_publication_preserves_existing_bytes(self):
        with tempfile.TemporaryDirectory(prefix="agari-release-v2-test-") as tmp:
            target = Path(tmp)/"synthetic-release.json"
            release.publish_new(target, b"original")
            with self.assertRaises(FileExistsError):
                release.publish_new(target, b"replacement")
            self.assertEqual(target.read_bytes(), b"original")


if __name__ == "__main__":
    unittest.main()
