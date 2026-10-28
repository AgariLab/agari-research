"""Temporary-only fixtures test resumed completion without erasing interruption."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


legacy_tests = module('_continued_fixture_helpers', 'test_audit_ranked.py')
audit = module('_continued_audit_under_test', 'audit_ranked_continuation.py')
save, lines, load = legacy_tests.save, legacy_tests.lines, legacy_tests.load


class ContinuedFixture:
    def __init__(self, root):
        old = legacy_tests.Fixture(root)
        self.root, self.first, self.stopped = root, old.first, old.campaign
        self.paths, self.expected = old.paths, old.expected
        self.continuation = root / 'continuation'
        self.continuation.mkdir()
        self.old_records = copy.deepcopy(old.records[:21])
        self.new_records = copy.deepcopy(old.records[21:])
        for number, record in enumerate(self.new_records, 1):
            source = self.stopped / record['directory']
            destination = self.continuation / f'game-{number:06d}'
            source.rename(destination)
            record.update(game_number=number, directory=destination.name)
        failure = self.stopped / 'game-000022'
        failure.mkdir()
        template = load(self.stopped / 'game-000021/manifest.json')
        template['status'] = 'incomplete'
        save(failure / 'manifest.json', template)
        lines(failure / 'games.jsonl', [{'mode': 'ranked', 'status': 'interrupted',
            'seat': None, 'scores': None, 'validation_passed': None, 'degraded': False,
            'counts': {}, 'failure': 'connection_closed', 'game_index': 0}])
        (failure / 'events.jsonl').write_bytes(b'')
        save(failure / 'report.json', {'games_requested': 1, 'games_completed': 0,
            'validation_passed': None, 'degraded_games': 0, 'counts': {}})
        self.old_records.append({'game_number': 22, 'directory': 'game-000022',
            'status': 'interrupted', 'healthy': False, 'errors': list(audit.FAILURE_ERRORS),
            'seat': None, 'scores': None, 'degraded': False,
            'counts': dict.fromkeys(audit.base.COUNT_FIELDS, 0),
            'latency': {'count': 0, 'mean_ms': None, 'p50_ms': None, 'p95_ms': None, 'max_ms': None},
            'elapsed_seconds': 33.53039512503892})
        old_manifest = load(self.stopped / 'manifest.json')
        new_manifest = dict(old_manifest, games_requested=78, status='complete',
            games_recorded=78, current_game=None, error=None, started_at=audit.CONTINUATION_STARTED_AT)
        old_manifest.update(status='stopped', games_recorded=22, current_game=22,
            error='game_incomplete', started_at='2026-10-27T23:42:13.741091+00:00',
            updated_at='2026-10-28T01:37:48.081576+00:00')
        save(self.stopped / 'manifest.json', old_manifest)
        save(self.continuation / 'manifest.json', new_manifest)
        lines(self.stopped / 'games.jsonl', self.old_records)
        lines(self.continuation / 'games.jsonl', self.new_records)
        self.rollup(self.stopped, self.old_records, requested=99, status='stopped', current=22, error='game_incomplete')
        self.rollup(self.continuation, self.new_records, requested=78, status='complete', current=None, error=None)
        self.pins = self.compute_pins()

    def rollup(self, root, records, *, requested, status, current, error):
        complete = [row for row in records if row['status'] == 'complete']
        count = sum(row['latency']['count'] for row in complete)
        save(root / 'summary.json', {'schema_version': 1, 'games_requested': requested,
            'games_attempted': len(records), 'games_completed': len(complete), 'healthy_games': len(complete),
            'status': status, 'current_game': current, 'error': error,
            'counts': {key: sum(row['counts'][key] for row in records) for key in audit.base.COUNT_FIELDS},
            'latency': {'count': count,
                'mean_ms': sum(row['latency']['count'] * row['latency']['mean_ms'] for row in complete) / count,
                'max_ms': max(row['latency']['max_ms'] for row in complete)}, 'elapsed_seconds': 500.0})

    def compute_pins(self):
        pins = {}
        for label in audit.HISTORY_PINS:
            stage, relative = label.split('/', 1)
            path = (self.first if stage == 'first' else self.stopped) / relative
            data = path.read_bytes()
            pins[label] = (len(data), audit.base.digest(data))
        return pins

    def run(self):
        return audit.audit_ranked_continuation(self.first, self.stopped, self.continuation,
            paths=self.paths, expected=self.expected, history_pins=self.pins)

    def publish(self, result, directory):
        return audit.write_continued_audit(result, directory,
            paths=self.paths, expected=self.expected, history_pins=self.pins)

    def mark_running(self, completed=77):
        for name in ('manifest.json', 'summary.json'):
            path = self.continuation / name
            value = load(path); value.update(status='running', current_game=completed + 1)
            save(path, value)
        lines(self.continuation / 'games.jsonl', self.new_records[:completed])
        path = self.continuation / f'game-{completed + 1:06d}' / 'manifest.json'
        value = load(path); value['status'] = 'running'; save(path, value)


class ContinuedAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='continued-ranked-audit-test-')
        self.addCleanup(self.tmp.cleanup)
        self.fixture = ContinuedFixture(Path(self.tmp.name))

    def reject(self, code):
        result = self.fixture.run()
        self.assertEqual(result['status'], 'invalid', result['integrity_errors'])
        self.assertTrue(any(code in item for item in result['integrity_errors']), result['integrity_errors'])
        with self.assertRaises(audit.base.AuditError):
            self.fixture.publish(result, self.fixture.root / 'not-created')
        self.assertFalse((self.fixture.root / 'not-created').exists())
        return result

    def test_100_complete_101_attempts_is_not_zero_failure_success(self):
        result = self.fixture.run()
        self.assertEqual(result['status'], 'ready_with_interruption', result['integrity_errors'])
        self.assertFalse(result['final_acceptance'])
        self.assertFalse(result['original_zero_failure_criterion_met'])
        self.assertEqual(len(result['attempts']), 101)
        self.assertEqual(len(result['raw_files']), 410)
        self.assertEqual(result['totals']['games_completed'], 100)
        self.assertEqual(result['totals']['failed_attempts'], 1)
        self.assertEqual(result['stage_totals']['stopped_batch'],
                         {'attempts': 22, 'games_completed': 21, 'failed_attempts': 1})
        self.assertEqual(result['stage_totals']['continuation'],
                         {'attempts': 78, 'games_completed': 78, 'failed_attempts': 0})
        self.assertEqual(result['attempts'][22]['global_attempt'], 23)
        self.assertIsNone(result['attempts'][22]['completed_game'])
        self.assertEqual(result['attempts'][23]['completed_game'], 23)
        self.assertEqual(result['attempts'][-1]['global_attempt'], 101)
        self.assertEqual(result['attempts'][-1]['completed_game'], 100)
        failure = result['attempt_failures'][0]
        self.assertEqual(failure['failure'], 'connection_closed')
        self.assertFalse(failure['start_game_observed'])
        self.assertEqual(failure['server_outcome'], 'unknown')
        self.assertEqual(failure['rating_impact'], 'unknown')
        self.assertIsNone(failure['scores'])
        self.assertAlmostEqual(result['totals']['local_decision_ms']['p95_ms'], 7.08405)
        self.assertAlmostEqual(result['totals']['server_ack_elapsed_ms']['p95_ms'], 584.05)
        self.assertEqual(result, self.fixture.run())

    def test_publication_normalizes_only_three_fields_and_is_exclusive(self):
        result = self.fixture.run()
        output = self.fixture.root / 'continued-audit'
        report = self.fixture.publish(result, output)
        self.assertEqual(report['status'], 'completed_with_interruption')
        self.assertFalse(report['original_zero_failure_criterion_met'])
        normalized = dict(report, status='ready_with_interruption', final_acceptance=False)
        normalized.pop('published_at')
        self.assertEqual(normalized, result)
        self.assertEqual(load(output / 'attempts.json'), result['attempts'])
        self.assertEqual(load(output / 'raw-sha256.json'), result['raw_files'])
        for line in (output / 'SHA256SUMS').read_text().splitlines():
            checksum, name = line.split('  ')
            self.assertEqual(checksum, audit.base.digest((output / name).read_bytes()))
        before = {path.name: path.read_bytes() for path in output.iterdir()}
        with self.assertRaises(FileExistsError):
            self.fixture.publish(result, output)
        self.assertEqual(before, {path.name: path.read_bytes() for path in output.iterdir()})

    def test_any_historical_pinned_byte_change_rejects(self):
        for relative in ('manifest.json', 'summary.json', 'games.jsonl', 'game-000022/events.jsonl'):
            with self.subTest(relative=relative):
                path = self.fixture.stopped / relative
                original = path.read_bytes(); path.write_bytes(original + b' ')
                self.reject('historical_evidence_changed')
                path.write_bytes(original)

    def test_repinning_cannot_hide_changed_failure_semantics(self):
        path = self.fixture.stopped / 'game-000022/games.jsonl'
        game = load(path); game['failure'] = 'different_failure'; save(path, game)
        self.fixture.pins = self.fixture.compute_pins()
        self.reject('historical_failure_record_mismatch')

    def test_repinning_cannot_invent_no_game_or_rating_result(self):
        path = self.fixture.stopped / 'game-000022/events.jsonl'
        lines(path, [{'event': 'start_game', 'seat': 0, 'game_index': 0}])
        self.fixture.pins = self.fixture.compute_pins()
        self.reject('historical_failure_events_not_empty')

    def test_historical_failure_cannot_be_deleted_or_renumbered(self):
        lines(self.fixture.stopped / 'games.jsonl', self.fixture.old_records[:21])
        self.fixture.pins = self.fixture.compute_pins()
        self.reject('historical_attempt_count_changed')

    def test_second_failure_is_retained_and_rejects(self):
        path = self.fixture.continuation / 'game-000005/games.jsonl'
        game = load(path); game['failure'] = 'connection_closed'; save(path, game)
        result = self.reject('new_continuation_failure')
        self.assertEqual(len(result['attempts']), 101)
        self.assertEqual(result['totals']['failed_attempts'], 2)
        self.assertEqual(result['attempts'][27]['status'], 'invalid')
        self.assertEqual(len(result['attempt_failures']), 1)  # Only the fixed interruption is authorized.

    def test_missing_new_game_is_not_silently_filtered(self):
        (self.fixture.continuation / 'game-000005').rename(self.fixture.root / 'missing-new-game')
        result = self.reject('batch_missing_or_extra_attempt_directory')
        self.assertEqual(len(result['attempts']), 101)
        self.assertEqual(result['attempts'][27]['status'], 'invalid')

    def test_77_new_completions_cannot_publish(self):
        lines(self.fixture.continuation / 'games.jsonl', self.fixture.new_records[:77])
        self.reject('continuation_not_78_complete_attempts')

    def test_running_probe_never_reads_current_events_or_publishes(self):
        self.fixture.mark_running()
        original = audit.base.Reader.read
        forbidden = self.fixture.continuation / 'game-000078/events.jsonl'
        def guarded(reader, path, **kwargs):
            self.assertNotEqual(path, forbidden)
            return original(reader, path, **kwargs)
        with patch.object(audit.base.Reader, 'read', guarded):
            result = self.fixture.run()
        self.assertEqual(result['status'], 'pending', result['integrity_errors'])
        self.assertEqual(result['totals']['games_completed'], 99)
        with self.assertRaises(audit.base.AuditError):
            self.fixture.publish(result, self.fixture.root / 'not-created')
        self.assertFalse((self.fixture.root / 'not-created').exists())

    def test_concurrent_append_and_snapshot_changes_stay_pending(self):
        self.fixture.mark_running()
        with (self.fixture.continuation / 'games.jsonl').open('ab') as stream:
            stream.write(b'{"game_number":')
        with patch.object(audit.base.Reader, 'unchanged', return_value=False):
            result = self.fixture.run()
        self.assertEqual(result['status'], 'pending', result['integrity_errors'])
        self.assertIn('continuation_index_inflight', result['pending_reasons'])
        self.assertIn('source_changed_during_probe', result['pending_reasons'])

    def test_parent_rollup_or_start_time_tampering_rejects(self):
        for name, mutate, code in (
            ('summary.json', lambda x: x['counts'].update(requests=999), 'batch_summary_counts_mismatch'),
            ('summary.json', lambda x: x.update(games_completed=77), 'batch_summary_totals_mismatch'),
            ('manifest.json', lambda x: x.update(started_at='2026-10-28T01:00:00+00:00'), 'continuation_start_identity_mismatch'),
        ):
            with self.subTest(code=code):
                path = self.fixture.continuation / name
                original = path.read_bytes(); value = load(path); mutate(value); save(path, value)
                self.reject(code); path.write_bytes(original)

    def test_actual_identity_change_and_publication_race_reject(self):
        result = self.fixture.run()
        (self.fixture.paths.model_dir / 'ppo.pt').write_bytes(b'changed weights')
        self.reject('actual_identity_differs_from_pinned_release')
        with self.assertRaisesRegex(audit.base.AuditError, 'independent_reaudit_changed'):
            self.fixture.publish(result, self.fixture.root / 'not-created')
        self.assertFalse((self.fixture.root / 'not-created').exists())

    def test_original_auditor_remains_invalid_for_this_campaign(self):
        old = audit.base.audit_ranked(self.fixture.first, self.fixture.stopped,
                                      paths=self.fixture.paths, expected=self.fixture.expected)
        self.assertEqual(old['status'], 'invalid')
        self.assertFalse(old['final_acceptance'])
        self.assertEqual(audit.base.digest(audit.LEGACY_PATH.read_bytes()), audit.LEGACY_SHA256)

    def test_new_defaulted_ack_and_score_mismatch_are_rejected(self):
        path = self.fixture.continuation / 'game-000001/events.jsonl'
        original = path.read_bytes()
        for mutation, code in (
            (lambda events: events[2].update(status='defaulted'), 'nonaccepted_ack'),
            (lambda events: events[-1]['scores'].__setitem__(0, 12345), 'end_game_missing_or_mismatch'),
        ):
            with self.subTest(code=code):
                events = [json.loads(line) for line in original.splitlines()]
                mutation(events); lines(path, events); self.reject(code); path.write_bytes(original)


if __name__ == '__main__':
    unittest.main()
