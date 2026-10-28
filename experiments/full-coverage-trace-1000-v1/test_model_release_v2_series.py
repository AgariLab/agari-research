"""Temporary synthetic release gates; never write a real audit or release."""

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
from unittest.mock import Mock, patch

sys.dont_write_bytecode = True

from agari_kit.storage import Storage

spec = importlib.util.spec_from_file_location("series_release", Path(__file__).with_name("build_model_release_v2_series.py"))
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def fixtures(extra_interruption=False):
    identity = {"weights": "a" * 64, "descriptor": "b" * 64, "kit": "c" * 64,
                "research": "d" * 64, "engine": "e" * 64, "launcher": release.LAUNCHER_SHA256,
                "kit_files": 19, "research_files": 17}
    prior = {"created_at": "2026-10-01T00:00:00+00:00", "release_id": "synthetic/release-v1",
             "scope": "synthetic immutable history", "weights": {"sha256": identity["weights"]},
             "agent": {"descriptor_artifact": {"sha256": identity["descriptor"]}},
             "training": {"source": {"commit": "historical-training-commit"}},
             "evaluation": {"rules": {"game_mode": "4p-red-half"}, "source": "historical-evaluation"},
             "online_validation": {"recorded_runtime": {"agari_kit": {"source_sha256": identity["kit"]},
                 "engine": {"binary_sha256": identity["engine"]}},
                 "recorded_source": {"source_sha256": identity["research"]},
                 "source_commit_matched_by_bytes": "historical-validation-byte-match"},
             "verification": {"scope": "immutable original verifier"},
             "release_policy": {"immutable": True}, "limitations": ["historical limitation"],
             "unanticipated_v1_field": {"keep": [True, 3, "all original information"]}}
    interrupted_ids = {23, 30, 40} if extra_interruption else {23, 30}
    attempts = [{"global_attempt": number, "batch_id": "fixture", "local_attempt": number,
                 "runtime_variant": "uninstrumented" if number <= 30 else "parent_transport_observer",
                 "status": "interrupted" if number in interrupted_ids else "complete",
                 "healthy": number not in interrupted_ids}
                for number in range(1, 101 + len(interrupted_ids))]
    failures = [{"global_attempt": number, "batch_id": "fixture", "local_attempt": number,
                 "runtime_variant": "uninstrumented" if number <= 30 else "parent_transport_observer",
                 "failure": "connection_closed", "start_game_observed": number != 23,
                 "end_game_observed": False, "scores": None, "requests": 0 if number == 23 else 115,
                 "server_outcome": "unknown", "rating_impact": "unknown"}
                for number in sorted(interrupted_ids)]
    fresh = {"status": "ready", "final_acceptance": False, "target_complete_games": 100,
             "completion_target_met": True, "integrity_status": "verified", "integrity_errors": [],
             "pending_reasons": [], "original_zero_failure_criterion_met": False,
             "outcome": "completed_with_interruptions", "attempts": attempts, "failures": failures,
             "totals": {"attempts": len(attempts), "games_completed": 100, "healthy_completed": 100,
                 "interrupted_attempts": len(failures), "invalid_attempts": 0, "pending_attempts": 0},
             "runtime_variants": {
                 "uninstrumented": {"attempts": 30, "games_completed": 28,
                     "interrupted_attempts": 2, "wrapper_sha256": None},
                 "parent_transport_observer": {"attempts": len(attempts) - 30, "games_completed": 72,
                     "interrupted_attempts": len(failures) - 2, "wrapper_sha256": release.WRAPPER_SHA256}},
             "registry": {"path": "fixture", "sha256": "f" * 64, "revision": 1,
                          "root_sha256": release.REGISTRY_ROOT_SHA256},
             "identity_checks": {"actual": identity, "expected": copy.deepcopy(identity), "matches": True},
             "batches": [{"id": "fixture", "status": "stopped"}],
             "source_files": [], "raw_files": [], "auditor_sha256": "f" * 64,
             "legacy_auditor_sha256": release.LEGACY_AUDITOR_SHA256,
             "future_audit_field": {"must_not_be_projected_away": True}}
    fresh["totals"].update(completed_counts={"requests": 10000},
                           interrupted_counts={"requests": 115 * (len(failures) - 1)},
                           all_observed_counts={"requests": 10000 + 115 * (len(failures) - 1)})
    for name, group in fresh["runtime_variants"].items():
        original = name == "uninstrumented"
        complete_requests = 4367 if original else 5633
        interrupted_requests = 115 if original else 115 * (len(failures) - 2)
        group.update(same_runtime_as_uninstrumented=original,
                     completed_counts={"requests": complete_requests},
                     interrupted_counts={"requests": interrupted_requests},
                     all_observed_counts={"requests": complete_requests + interrupted_requests})
    return prior, fresh


def write_published(directory, fresh, *, changes=None, attempts=None, sources=None):
    directory.mkdir(exist_ok=True)
    report = dict(copy.deepcopy(fresh), status="published", final_acceptance=True,
                  published_at="2026-10-28T00:00:00+00:00")
    report.update(changes or {})
    docs = {"report.json": report, "attempts.json": fresh["attempts"] if attempts is None else attempts,
            "source-sha256.json": fresh["source_files"] if sources is None else sources}
    for name, value in docs.items():
        (directory / name).write_text(json.dumps(value) + "\n")
    (directory / "SHA256SUMS").write_text("".join(
        release.sha256(directory / name) + "  " + name + "\n" for name in sorted(docs)))


class GateTests(unittest.TestCase):
    def test_hundred_complete_preserves_two_or_more_interruptions(self):
        for extra in (False, True):
            prior, fresh = fixtures(extra)
            release.require_complete_series(fresh, prior)

    def test_incomplete_excess_pending_or_unhealthy_games_are_rejected(self):
        for key, value in (("games_completed", 99), ("games_completed", 101), ("healthy_completed", 99),
                           ("pending_attempts", 1), ("invalid_attempts", 1), ("interrupted_attempts", True)):
            prior, fresh = fixtures()
            fresh["totals"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                release.require_complete_series(fresh, prior)

    def test_uninterrupted_or_provisional_acceptance_is_rejected(self):
        for key, value in (("original_zero_failure_criterion_met", True), ("integrity_status", "provisional"),
                           ("final_acceptance", True), ("status", "published"), ("completion_target_met", False)):
            prior, fresh = fixtures()
            fresh[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                release.require_complete_series(fresh, prior)

    def test_missing_duplicate_or_reordered_attempts_are_rejected(self):
        for mutation in (lambda rows: rows.pop(), lambda rows: rows.__setitem__(22, rows[21]),
                         lambda rows: rows.reverse()):
            prior, fresh = fixtures()
            mutation(fresh["attempts"])
            with self.assertRaises(ValueError):
                release.require_complete_series(fresh, prior)

    def test_historical_failure_or_unknown_outcome_cannot_be_changed(self):
        for key, value in (("server_outcome", "did-not-count"), ("rating_impact", "none"),
                           ("scores", [25000] * 4), ("requests", 0), ("end_game_observed", True)):
            prior, fresh = fixtures()
            fresh["failures"][1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                release.require_complete_series(fresh, prior)

    def test_runtime_groups_and_v1_identity_cannot_be_collapsed_or_rewritten(self):
        for mutate in (
            lambda f: f["runtime_variants"].pop("parent_transport_observer"),
            lambda f: f["runtime_variants"]["parent_transport_observer"].update(wrapper_sha256="0" * 64),
            lambda f: f["attempts"][31].update(runtime_variant="uninstrumented"),
            lambda f: f["identity_checks"]["actual"].update(research="0" * 64),
            lambda f: f["registry"].update(root_sha256="0" * 64),
        ):
            prior, fresh = fixtures()
            mutate(fresh)
            with self.assertRaises(ValueError):
                release.require_complete_series(fresh, prior)


class PublishedAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="agari-series-published-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.patch = patch.object(release, "FINAL_AUDIT", self.directory)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        _, self.fresh = fixtures()

    def test_every_published_field_matches_fresh(self):
        write_published(self.directory, self.fresh)
        self.assertEqual(release.verify_published_audit(self.fresh)["status"], "published")

    def test_resigned_changes_in_arbitrary_fields_or_json_types_fail(self):
        for changes in ({"future_audit_field": {"must_not_be_projected_away": 1}},
                        {"original_zero_failure_criterion_met": True},
                        {"batches": []}, {"runtime_variants": {}}, {"extra_field": "unverified"}):
            write_published(self.directory, self.fresh, changes=changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                release.verify_published_audit(self.fresh)

    def test_success_only_attempts_and_omitted_sources_fail(self):
        write_published(self.directory, self.fresh,
                        attempts=[row for row in self.fresh["attempts"] if row["status"] == "complete"])
        with self.assertRaisesRegex(ValueError, "attempts"):
            release.verify_published_audit(self.fresh)
        self.fresh["source_files"] = [{"path": "synthetic", "bytes": 0, "sha256": "e" * 64}]
        write_published(self.directory, self.fresh, sources=[])
        with self.assertRaisesRegex(ValueError, "source inventory"):
            release.verify_published_audit(self.fresh)

    def test_corrupt_duplicate_checksum_or_extra_file_fails(self):
        write_published(self.directory, self.fresh)
        path = self.directory / "attempts.json"
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "hash differs"):
            release.verify_published_audit(self.fresh)
        write_published(self.directory, self.fresh)
        sums = self.directory / "SHA256SUMS"
        sums.write_text(sums.read_text() + sums.read_text().splitlines()[0] + "\n")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            release.verify_published_audit(self.fresh)
        write_published(self.directory, self.fresh)
        (self.directory / "extra.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            release.verify_published_audit(self.fresh)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="agari-series-preparation-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.repo = self.root / "repo"
        self.experiment = self.repo / "experiments/model"
        self.ranked = self.repo / "experiments/ranked"
        self.experiment.mkdir(parents=True)
        self.ranked.mkdir()
        self.storage = Storage(self.root / "storage")
        self.model = self.storage.root / "models" / release.RUN
        self.model.mkdir(parents=True)
        self.prior, self.fresh = fixtures()
        self.paths = {"REPO": self.repo, "EXPERIMENT": self.experiment,
                      "AUDITOR": self.ranked / "audit.py", "REGISTRY": self.ranked / "registry.json",
                      "FINAL_AUDIT": self.ranked / "published", "SCRIPT": self.experiment / "generator.py"}
        for name in ("BASE_HELPER", "LEGACY_AUDITOR", "LAUNCHER", "WRAPPER"):
            original = getattr(release, name)
            destination = self.experiment / original.name
            destination.write_bytes(original.read_bytes())
            self.paths[name] = destination
        for name in ("AUDITOR", "SCRIPT"):
            self.paths[name].write_text("# synthetic verifier fixture\n")
        self.plan = self.ranked / "plan.json"
        self.plan.write_text('{"synthetic_plan":true}')
        self.plan_entry = {"path": str(self.plan), "bytes": self.plan.stat().st_size,
                           "sha256": release.sha256(self.plan)}
        self.paths["REGISTRY"].write_text(json.dumps({"plans": [self.plan_entry]}))
        self.raw = self.storage.root / "runs/riichi-ranked-first-v1/manifest.json"
        self.raw.parent.mkdir(parents=True)
        self.raw.write_text(json.dumps({"rules_expected": self.prior["evaluation"]["rules"]}))
        for path in (self.model / "release-v1.json", self.experiment / "model-release-v1.json"):
            path.write_text(json.dumps(self.prior))
        self.fresh["auditor_sha256"] = release.sha256(self.paths["AUDITOR"])
        self.root_sha = release.sha256(self.paths["REGISTRY"])
        self.fresh["registry"].update(path=str(self.paths["REGISTRY"]), sha256=self.root_sha, root_sha256=self.root_sha)
        self.fresh["registry_chain"] = [{"path": str(self.paths["REGISTRY"]), "sha256": self.root_sha, "revision": 1}]
        self.fresh["source_files"] = [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": release.sha256(path)}
            for path in [self.raw, self.plan] + [self.paths[name] for name in ("AUDITOR", "REGISTRY", "LEGACY_AUDITOR", "LAUNCHER", "WRAPPER")]]
        self.fresh["raw_files"] = [copy.deepcopy(self.fresh["source_files"][0])]
        self.audit = Mock(side_effect=lambda **kwargs: copy.deepcopy(self.fresh))
        self.verify_v1 = Mock(side_effect=lambda storage: copy.deepcopy(self.prior))
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        for name, path in self.paths.items():
            self.stack.enter_context(patch.object(release, name, path))
        self.stack.enter_context(patch.object(release, "REGISTRY_ROOT_SHA256", self.root_sha))
        self.stack.enter_context(patch.object(release, "load_base_helper", return_value=SimpleNamespace(verify_v1=self.verify_v1)))
        self.stack.enter_context(patch.object(release, "load_module", return_value=SimpleNamespace(audit_ranked_series=self.audit)))
        write_published(self.paths["FINAL_AUDIT"], self.fresh)

    def test_prepare_preserves_all_v1_information_and_entire_fresh_audit(self):
        status, result = release.prepare(self.storage)
        self.assertEqual(status["status"], "ready_with_interruptions")
        self.assertEqual(result["ranked"]["series_audit"], self.fresh)
        for key, value in self.prior.items():
            self.assertTrue(result[key] == value or result["inherited_v1_fields"].get(key) == value)
        self.assertEqual(self.audit.call_count, 2)
        self.assertEqual(self.verify_v1.call_count, 2)
        self.assertFalse((self.model / "release-v2.json").exists())
        self.assertFalse((self.experiment / "model-release-v2.json").exists())

    def test_pending_and_unpublished_audit_create_no_release(self):
        self.fresh.update(status="pending", completion_target_met=False, pending_reasons=["still_running"])
        status, result = release.prepare(self.storage)
        self.assertEqual(status["status"], "pending")
        self.assertIsNone(result)
        self.fresh.update(status="ready", completion_target_met=True, pending_reasons=[])
        with patch.object(release, "FINAL_AUDIT", self.ranked / "not-published"):
            status, result = release.prepare(self.storage)
        self.assertEqual(status["reason"], "independent-series-audit-not-published")
        self.assertIsNone(result)

    def test_explicit_registry_head_is_bound_and_root_inventory_is_retained(self):
        head = self.ranked / "registry-v2.json"
        head.write_text(json.dumps({"plans": [self.plan_entry], "synthetic_revision": 2}))
        self.fresh["registry"].update(path=str(head), sha256=release.sha256(head), revision=2)
        self.fresh["registry_chain"].append({"path": str(head), "sha256": release.sha256(head), "revision": 2})
        self.fresh["source_files"].append({"path": str(head), "bytes": head.stat().st_size,
                                           "sha256": release.sha256(head)})
        write_published(self.paths["FINAL_AUDIT"], self.fresh)
        _, result = release.prepare(self.storage, registry_path=head)
        self.assertEqual(result["ranked"]["registry"]["sha256"], release.sha256(head))
        self.assertTrue(all(call.kwargs == {"registry_path": head} for call in self.audit.call_args_list))
        self.fresh["source_files"] = [row for row in self.fresh["source_files"]
                                       if row["path"] != str(self.paths["REGISTRY"])]
        with self.assertRaisesRegex(ValueError, "omitted"):
            release.verify_source_inventory(self.fresh, head)

    def test_omitted_old_plan_is_rejected_even_when_registry_is_present(self):
        self.fresh["source_files"] = [row for row in self.fresh["source_files"] if row["path"] != str(self.plan)]
        with self.assertRaisesRegex(ValueError, "historical execution plan"):
            release.verify_source_inventory(self.fresh)

    def test_intermediate_registry_ancestor_cannot_be_omitted(self):
        for revision in (2, 3):
            head = self.ranked / f"registry-v{revision}.json"
            head.write_text(json.dumps({"plans": [self.plan_entry], "synthetic_revision": revision}))
            self.fresh["registry_chain"].append({"path": str(head), "sha256": release.sha256(head),
                                                 "revision": revision})
            self.fresh["source_files"].append({"path": str(head), "sha256": release.sha256(head),
                                               "bytes": head.stat().st_size})
        self.fresh["registry"].update(path=str(head), sha256=release.sha256(head), revision=3)
        middle = str(self.ranked / "registry-v2.json")
        self.fresh["source_files"] = [row for row in self.fresh["source_files"] if row["path"] != middle]
        with self.assertRaisesRegex(ValueError, "registry ancestor"):
            release.verify_source_inventory(self.fresh, head)

    def test_changed_raw_source_rejected_before_release(self):
        self.raw.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, "source bytes changed"):
            release.prepare(self.storage)

    def test_duplicate_or_omitted_source_inventory_is_rejected(self):
        self.fresh["source_files"].append(self.fresh["source_files"][0])
        with self.assertRaisesRegex(ValueError, "duplicate source"):
            release.verify_source_inventory(self.fresh)
        self.fresh["source_files"].pop()
        self.fresh["source_files"].pop()
        with self.assertRaisesRegex(ValueError, "omitted"):
            release.verify_source_inventory(self.fresh)

    def test_second_audit_or_v1_drift_is_rejected(self):
        changed = copy.deepcopy(self.fresh)
        changed["future_audit_field"]["must_not_be_projected_away"] = False
        self.audit.side_effect = [copy.deepcopy(self.fresh), changed]
        with self.assertRaisesRegex(ValueError, "Series evidence changed"):
            release.prepare(self.storage)
        self.audit.side_effect = lambda **kwargs: copy.deepcopy(self.fresh)
        changed_prior = copy.deepcopy(self.prior)
        changed_prior["training"]["source"]["commit"] = "invented"
        self.verify_v1.side_effect = [copy.deepcopy(self.prior), changed_prior]
        with self.assertRaisesRegex(ValueError, "Historical v1 evidence changed"):
            release.prepare(self.storage)

    def test_invalid_or_inconsistent_pending_cannot_be_published(self):
        for status in ("invalid", "pending"):
            self.fresh.update(status=status, integrity_errors=["synthetic_source_mismatch"])
            with self.subTest(status=status), self.assertRaises(ValueError):
                release.prepare(self.storage)


class PublicationTests(unittest.TestCase):
    def test_changed_helper_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.py"
            path.write_text("raise RuntimeError('must not execute')")
            with patch.object(release, "BASE_HELPER", path), patch.object(release, "load_module") as load:
                with self.assertRaisesRegex(ValueError, "helper changed"):
                    release.load_base_helper()
                load.assert_not_called()

    def test_pending_default_write_and_verify_have_correct_exit_without_writes(self):
        for argv, code in (([], 0), (["--write"], 2), (["--verify-release"], 2)):
            with self.subTest(argv=argv), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with (patch.object(release.Storage, "from_env", return_value=Storage(root)),
                      patch.object(release, "EXPERIMENT", root),
                      patch.object(release, "prepare", return_value=({"status": "pending"}, None)),
                      patch.object(release, "load_base_helper") as helper,
                      contextlib.redirect_stdout(io.StringIO()) as output):
                    self.assertEqual(release.main(argv), code)
                helper.assert_not_called()
                self.assertFalse(json.loads(output.getvalue())["release_written"])
                self.assertEqual(list(root.iterdir()), [])

    def test_existing_file_or_dangling_symlink_is_never_overwritten(self):
        for symlink in (False, True):
            with self.subTest(symlink=symlink), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                target = root / "model-release-v2.json"
                if symlink:
                    target.symlink_to(root / "missing-target")
                else:
                    target.write_bytes(b"original")
                with (patch.object(release.Storage, "from_env", return_value=Storage(root)),
                      patch.object(release, "EXPERIMENT", root), patch.object(release, "prepare") as prepare,
                      contextlib.redirect_stdout(io.StringIO())):
                    self.assertEqual(release.main(["--write"]), 1)
                prepare.assert_not_called()
                self.assertTrue(target.is_symlink() if symlink else target.read_bytes() == b"original")

    def test_exclusive_publish_and_partial_second_copy_preserve_first(self):
        publish = release.load_base_helper().publish_new
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "models" / release.RUN
            model.mkdir(parents=True)
            external, public = model / "release-v2.json", root / "model-release-v2.json"

            def fail_second(path, content):
                if path == public:
                    raise OSError("synthetic second-copy failure")
                publish(path, content)

            with (patch.object(release.Storage, "from_env", return_value=Storage(root)),
                  patch.object(release, "EXPERIMENT", root),
                  patch.object(release, "prepare", return_value=({"status": "ready_with_interruptions"},
                      {"release_id": "synthetic-only", "status": "completed_with_interruptions"})),
                  patch.object(release, "load_base_helper", return_value=SimpleNamespace(publish_new=fail_second)),
                  contextlib.redirect_stdout(io.StringIO()) as output):
                self.assertEqual(release.main(["--write"]), 1)
            value = json.loads(output.getvalue())
            self.assertEqual(value["status"], "partial")
            self.assertEqual(value["created_paths"], [str(external)])
            self.assertTrue(value["release_written"])
            self.assertFalse(public.exists())
            original = external.read_bytes()
            with self.assertRaises(FileExistsError):
                publish(external, b"replacement")
            self.assertEqual(external.read_bytes(), original)

    def test_two_identical_copies_verify_but_resigned_changed_content_does_not(self):
        helper = release.load_base_helper()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "models" / release.RUN
            model.mkdir(parents=True)
            external, public = model / "release-v2.json", root / "model-release-v2.json"
            prepared = {"release_id": "synthetic-only", "status": "completed_with_interruptions",
                        "created_at": "2026-10-28T00:00:00+00:00", "original_zero_failure_criterion_met": False}

            def prepare(*args, **kwargs):
                return {"status": "ready_with_interruptions"}, copy.deepcopy(prepared)

            with (patch.object(release.Storage, "from_env", return_value=Storage(root)),
                  patch.object(release, "EXPERIMENT", root), patch.object(release, "prepare", side_effect=prepare),
                  patch.object(release, "load_base_helper", return_value=helper),
                  contextlib.redirect_stdout(io.StringIO()) as output):
                self.assertEqual(release.main(["--write"]), 0)
                self.assertEqual(external.read_bytes(), public.read_bytes())
                prepared["created_at"] = "2026-10-29T00:00:00+00:00"
                self.assertEqual(release.main(["--verify-release"]), 0)
                changed = json.loads(external.read_bytes())
                changed["original_zero_failure_criterion_met"] = True
                for path in (external, public):
                    path.write_text(json.dumps(changed))
                self.assertEqual(release.main(["--verify-release"]), 1)
            rows = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual([row["status"] for row in rows], ["published", "verified-existing", "invalid"])
            self.assertTrue(json.loads(external.read_bytes())["original_zero_failure_criterion_met"])


if __name__ == "__main__":
    unittest.main()
