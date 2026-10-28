"""Synthetic-only series, append-chain, concurrency and diagnostic regression tests."""
import copy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import shutil
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


helpers = module('_series_fixture_helpers', 'test_audit_ranked_continuation.py')
audit = module('_series_audit_under_test', 'audit_ranked_series.py')
save, lines, load = helpers.save, helpers.lines, helpers.load


class SeriesFixture:
    def __init__(self, root):
        root = root.resolve()
        old = helpers.ContinuedFixture(root)
        self.root, self.paths, self.expected = root, old.paths, old.expected
        self.runs = root / 'runs'; self.runs.mkdir()
        self.first = self.runs / 'riichi-ranked-first-v1'; old.first.rename(self.first)
        self.stopped = self.runs / 'riichi-ranked-campaign-remaining99-v1'; old.stopped.rename(self.stopped)
        self.continued = self.runs / 'riichi-ranked-continuation78-v1'; old.continuation.rename(self.continued)
        self.wrapper = root / 'diagnostic_wrapper.py'; self.wrapper.write_text('# synthetic wrapper; never run\n')
        self.wrapper_sha = audit.base.digest(self.wrapper.read_bytes())
        self.pool = root / 'pool'; self.pool.mkdir()
        self.template_manifest = load(self.continued / 'manifest.json')
        self.diag = self.runs / 'riichi-ranked-diagnostic-v1'
        self.make_single(self.diag, self.continued / 'game-000007', old.new_records[6],
                         '2026-10-28T02:53:41+00:00', '2026-10-28T02:57:18+00:00')
        self.names = [f'riichi-ranked-observed-{number:06d}-v1' for number in range(1, 72)]
        start = datetime(2026, 10, 28, 3, 4, tzinfo=timezone.utc)
        for number, name in enumerate(self.names):
            source = self.continued / f'game-{number + 8:06d}'
            saved = self.pool / source.name; source.rename(saved)
            begin = start + timedelta(minutes=number * 2)
            self.make_single(self.runs / name, saved, old.new_records[number + 7],
                             begin.isoformat(), (begin + timedelta(minutes=1)).isoformat())
        self.make_interrupted_child(self.continued / 'game-000007', no_start=False)
        continued_records = copy.deepcopy(old.new_records[:7])
        continued_records[-1] = self.interrupted_index(self.continued / 'game-000007', 7)
        lines(self.continued / 'games.jsonl', continued_records)
        manifest = load(self.continued / 'manifest.json')
        manifest.update(status='stopped', games_recorded=7, current_game=7, error='game_incomplete',
                        updated_at='2026-10-28T02:25:27+00:00')
        save(self.continued / 'manifest.json', manifest)
        self.rollup(self.continued, continued_records)
        self.plan = root / 'plan-v1.json'
        save(self.plan, self.plan_value(self.names, 29, 2, '2026-10-28T03:02:48+00:00'))
        histories = []
        for batch_id, path, kind, variant, requested, status in (
            ('first', self.first, 'standalone', 'uninstrumented', 1, 'complete'),
            ('remaining99', self.stopped, 'campaign', 'uninstrumented', 99, 'stopped'),
            ('continuation78', self.continued, 'campaign', 'uninstrumented', 78, 'stopped'),
            ('diagnostic1', self.diag, 'campaign', 'parent_transport_observer', 1, 'complete')):
            pins = {p.relative_to(path).as_posix(): {'bytes': p.stat().st_size, 'sha256': audit.base.digest(p.read_bytes())}
                    for p in sorted(path.rglob('*')) if p.is_file()}
            histories.append({'id': batch_id, 'run': path.name, 'kind': kind, 'runtime_variant': variant,
                              'requested_games': requested, 'terminal_status': status, 'pins': pins})
        registry = {'schema_version': 1, 'series_id': 'synthetic-series', 'revision': 1,
            'target_complete_games': 100, 'runs_root': str(self.runs), 'predecessor': None,
            'legacy_auditor_sha256': audit.LEGACY_SHA256, 'expected_identity': self.expected,
            'diagnostic_wrapper': {'path': str(self.wrapper), 'sha256': self.wrapper_sha},
            'history_batches': histories, 'plans': [self.reference(self.plan)]}
        self.registry = root / 'registry-v1.json'; save(self.registry, registry)
        self.anchor = audit.base.digest(self.registry.read_bytes())

    def reference(self, path):
        return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': audit.base.digest(path.read_bytes())}

    def plan_value(self, names, completed, failures, created):
        return {'schema_version': 1, 'created_at': created, 'target_complete_games': 100,
            'complete_games_before_execution': completed, 'prior_interrupted_attempts': failures,
            'maximum_new_attempts': len(names), 'mode': 'ranked', 'runtime_variant': 'parent_transport_observer',
            'wrapper_sha256': self.wrapper_sha, 'supervisor_sha256': self.expected['launcher'],
            'kit_source_sha256': self.expected['kit'], 'research_source_sha256': self.expected['research'],
            'weights_sha256': self.expected['weights'], 'descriptor_sha256': self.expected['descriptor'],
            'engine_version': '0.4.10', 'proxy_policy': 'environment', 'ping_interval_seconds': 20,
            'ping_timeout_seconds': 20, 'games_per_run': 1, 'concurrent_connections': 1,
            'stop_on_first_unsuccessful_run': True, 'retry_failed_run': False,
            'validation_repeated': False, 'original_zero_failure_criterion_met': False, 'planned_runs': names}

    def make_single(self, destination, child, record, started, ended):
        destination.mkdir()
        shutil.copytree(child, destination / 'game-000001')
        record = copy.deepcopy(record); record.update(game_number=1, directory='game-000001')
        lines(destination / 'games.jsonl', [record])
        manifest = dict(self.template_manifest, games_requested=1, games_recorded=1,
                        started_at=started, updated_at=ended, status='complete', current_game=None, error=None)
        save(destination / 'manifest.json', manifest)
        self.rollup(destination, [record])
        self.diagnostic(destination, 'complete')

    def rollup(self, root, records):
        manifest = load(root / 'manifest.json')
        count = sum(row['latency']['count'] for row in records)
        save(root / 'summary.json', {'schema_version': 1, 'games_requested': manifest['games_requested'],
            'games_attempted': len(records), 'games_completed': sum(row['status'] == 'complete' for row in records),
            'healthy_games': sum(row['healthy'] for row in records), 'status': manifest['status'],
            'current_game': manifest['current_game'], 'error': manifest['error'],
            'counts': {key: sum(row['counts'].get(key, 0) for row in records) for key in audit.base.COUNT_FIELDS},
            'latency': {'count': count,
                'mean_ms': sum(row['latency']['count'] * row['latency']['mean_ms'] for row in records if row['latency']['count']) / count if count else None,
                'max_ms': max((row['latency']['max_ms'] for row in records if row['latency']['count']), default=None)},
            'elapsed_seconds': 500.0})

    def diagnostic(self, root, status, created=1):
        directory = root / 'diagnostic'; directory.mkdir(exist_ok=True)
        save(directory / 'manifest.json', {'schema_version': 1, 'mode': 'ranked', 'games_requested': 1,
            'wrapper_sha256': self.wrapper_sha, 'supervisor_sha256': self.expected['launcher'],
            'runtime_variant': 'parent_transport_observer', 'same_runtime_as_uninstrumented': False,
            'automatic_reconnect': False, 'transport_config': audit.CONFIG, 'websockets_version': '15.0.1',
            'python_version': [3, 12, 16], 'proxy_policy': 'environment', 'network_path': 'proxy'})
        events = [{'event': 'diagnostic_ready', 'elapsed_seconds': 0},
                  {'event': 'connection_attempt', 'elapsed_seconds': 1, 'attempt': 1}]
        if created:
            events.extend([{'event': 'connection_created', 'elapsed_seconds': 2, 'connection_id': 1},
                {'event': 'handshake_complete', 'elapsed_seconds': 3, 'connection_id': 1},
                {'event': 'recv_cancelled', 'elapsed_seconds': 4, 'connection_id': 1},
                {'event': 'connection_lost', 'elapsed_seconds': 5, 'connection_id': 1, 'abnormal_no_close_frames': True},
                {'event': 'context_exit', 'elapsed_seconds': 6, 'connection_id': 1, 'abnormal_no_close_frames': True}])
        else:
            events.append({'event': 'handshake_failed', 'elapsed_seconds': 2, 'exception_category': 'connection_refused'})
        lines(directory / 'events.jsonl', events)
        save(directory / 'result.json', {'schema_version': 1, 'status': 'finalized', 'error': None,
            'campaign_status': status, 'connection_attempts': 1, 'connections_created': created,
            'events_recorded': len(events), 'elapsed_seconds': 10})

    def make_interrupted_child(self, child, no_start):
        manifest = load(child / 'manifest.json'); manifest['status'] = 'incomplete'; save(child / 'manifest.json', manifest)
        game = load(child / 'games.jsonl')
        game.update(status='interrupted', scores=None, failure='connection_closed')
        events = [json.loads(line) for line in (child / 'events.jsonl').read_text().splitlines()]
        if no_start:
            events = []; game.update(seat=None, counts={})
        else:
            events = events[:-1]
        lines(child / 'events.jsonl', events); save(child / 'games.jsonl', game)
        save(child / 'report.json', {'games_requested': 1, 'games_completed': 0,
            'validation_passed': None, 'degraded_games': 0, 'counts': game['counts']})

    def interrupted_index(self, child, number):
        game = load(child / 'games.jsonl')
        local = [json.loads(line)['latency_ms'] for line in (child / 'events.jsonl').read_text().splitlines()
                 if json.loads(line)['event'] == 'decision']
        errors = ['game_incomplete', 'game_failed']
        if not game['counts'].get('requests', 0):
            errors.append('action_receipts_mismatch')
        errors.append('transport_incomplete')
        return {'game_number': number, 'directory': child.name, 'status': 'interrupted', 'healthy': False,
            'errors': errors, 'seat': game['seat'], 'scores': None, 'degraded': False,
            'counts': dict.fromkeys(audit.base.COUNT_FIELDS, 0) | game['counts'],
            'latency': audit._latency_index(local), 'elapsed_seconds': 12.0}

    def fail_first_planned(self, no_start=True):
        root = self.runs / self.names[0]
        self.make_interrupted_child(root / 'game-000001', no_start)
        record = self.interrupted_index(root / 'game-000001', 1)
        lines(root / 'games.jsonl', [record])
        manifest = load(root / 'manifest.json'); manifest.update(status='stopped', current_game=1, error='game_incomplete')
        save(root / 'manifest.json', manifest); self.rollup(root, [record]); self.diagnostic(root, 'stopped', created=0 if no_start else 1)
        for name in self.names[1:]:
            shutil.rmtree(self.runs / name)

    def running_first(self):
        root = self.runs / self.names[0]
        for name in ('manifest.json', 'summary.json'):
            path = root / name; value = load(path); value.update(status='running', current_game=1)
            save(path, value)
        (root / 'games.jsonl').write_bytes(b'')
        path = root / 'game-000001/manifest.json'; value = load(path); value['status'] = 'running'; save(path, value)

    def options(self):
        return {'paths': self.paths, 'expected': self.expected, 'wrapper_path': self.wrapper,
                'root_registry_sha256': self.anchor}

    def run(self, registry=None):
        return audit.audit_ranked_series(registry or self.registry, **self.options())

    def forge_head(self, plan_path):
        value = load(self.registry)
        value.update(revision=2, predecessor={'path': str(self.registry), 'sha256': self.anchor},
                     plans=value['plans'] + [self.reference(plan_path)])
        path = self.root / 'registry-v2.json'; save(path, value); return path


class SeriesAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='ranked-series-test-'); self.addCleanup(self.tmp.cleanup)
        self.fixture = SeriesFixture(Path(self.tmp.name))

    def test_complete_keeps_two_failures_and_distinct_runtime(self):
        result = self.fixture.run()
        self.assertEqual(result['status'], 'ready', result['integrity_errors'])
        self.assertEqual(result['integrity_status'], 'verified')
        self.assertFalse(result['final_acceptance']); self.assertFalse(result['original_zero_failure_criterion_met'])
        self.assertEqual(result['totals']['attempts'], 102); self.assertEqual(result['totals']['games_completed'], 100)
        self.assertEqual(result['totals']['completed_counts']['requests'], 300)
        self.assertEqual(result['totals']['interrupted_counts']['requests'], 3)
        self.assertEqual([row['global_attempt'] for row in result['failures']], [23, 30])
        self.assertEqual(result['runtime_variants']['uninstrumented']['games_completed'], 28)
        self.assertEqual(result['runtime_variants']['parent_transport_observer']['games_completed'], 72)
        self.assertEqual(len(result['raw_files']), 846)
        self.assertAlmostEqual(result['totals']['completed_local_decision_ms']['p95_ms'], 7.08405)
        self.assertEqual(result, self.fixture.run())
        self.assertTrue(all(row['server_outcome'] == row['rating_impact'] == 'unknown' for row in result['failures']))
        self.assertEqual(result['batches'][3]['diagnostic']['abnormal_no_close_frames_observations'], 2)
        self.assertTrue(result['batches'][3]['successful'])

    def test_publish_reaudits_is_exclusive_and_only_changes_publication_fields(self):
        result = self.fixture.run(); output = self.fixture.root / 'output'
        report = audit.write_series_audit(result, output, **self.fixture.options())
        normalized = dict(report, status='ready', final_acceptance=False); normalized.pop('published_at')
        self.assertEqual(normalized, result)
        self.assertEqual(load(output / 'attempts.json'), result['attempts'])
        self.assertEqual(load(output / 'source-sha256.json'), result['source_files'])
        with self.assertRaises(FileExistsError): audit.write_series_audit(result, output, **self.fixture.options())
        for line in (output / 'SHA256SUMS').read_text().splitlines():
            checksum, name = line.split('  '); self.assertEqual(checksum, audit.base.digest((output / name).read_bytes()))

    def test_active_events_are_not_read_and_unexecuted_items_are_planned(self):
        self.fixture.running_first()
        for name in self.fixture.names[1:]: shutil.rmtree(self.fixture.runs / name)
        forbidden = self.fixture.runs / self.fixture.names[0] / 'game-000001/events.jsonl'
        original = audit.base.Reader.read
        def guarded(reader, path, **kwargs):
            self.assertNotEqual(path, forbidden); return original(reader, path, **kwargs)
        with patch.object(audit.base.Reader, 'read', guarded): result = self.fixture.run()
        self.assertEqual(result['status'], 'pending', result['integrity_errors'])
        self.assertEqual(result['totals']['games_completed'], 29)
        self.assertEqual(result['totals']['pending_attempts'], 1); self.assertEqual(result['totals']['planned_batches'], 70)
        with self.assertRaises(audit.base.AuditError): audit.write_series_audit(result, self.fixture.root / 'blocked', **self.fixture.options())
        self.assertFalse((self.fixture.root / 'blocked').exists())

    def test_handshake_before_connection_created_is_retained_not_invalid(self):
        self.fixture.fail_first_planned(no_start=True)
        result = self.fixture.run()
        self.assertEqual(result['status'], 'pending', result['integrity_errors'])
        self.assertEqual(result['integrity_status'], 'verified')
        self.assertEqual(result['totals']['interrupted_attempts'], 3)
        self.assertEqual(result['failures'][-1]['requests'], 0)
        self.assertFalse(result['failures'][-1]['start_game_observed'])
        self.assertEqual(result['plans'][0]['status'], 'stopped')

    def test_failure_does_not_get_replaced_by_later_planned_run(self):
        keep = self.fixture.root / 'later'; shutil.copytree(self.fixture.runs / self.fixture.names[1], keep)
        self.fixture.fail_first_planned(); keep.rename(self.fixture.runs / self.fixture.names[1])
        result = self.fixture.run()
        self.assertEqual(result['status'], 'invalid')
        self.assertIn('plan_started_after_gap_active_or_failed_run', result['integrity_errors'])

    def test_exact_child_diagnostic_and_empty_directory_inventories(self):
        root = self.fixture.runs / self.fixture.names[0]
        for path, is_dir in ((root / 'game-000001/extra.json', False),
                             (root / 'diagnostic/extra.json', False), (root / 'diagnostic/empty', True)):
            with self.subTest(path=str(path)):
                path.mkdir() if is_dir else path.write_text('{}')
                result = self.fixture.run(); self.assertEqual(result['status'], 'invalid')
                path.rmdir() if is_dir else path.unlink()

    def test_unregistered_run_or_unindexed_attempt_is_rejected(self):
        extra = self.fixture.runs / 'riichi-ranked-unregistered-v1'; extra.mkdir()
        self.assertIn('unregistered_ranked_run', self.fixture.run()['integrity_errors']); extra.rmdir()
        child = self.fixture.runs / self.fixture.names[0] / 'game-000002'; child.mkdir()
        result = self.fixture.run(); self.assertEqual(result['status'], 'invalid')
        self.assertTrue(any('unindexed_or_missing_attempt_directory' in code for code in result['integrity_errors']))

    def test_root_file_or_symlink_is_invalid_not_running(self):
        root = self.fixture.runs / self.fixture.names[0]; held = self.fixture.root / 'held'; root.rename(held)
        for symlink in (False, True):
            with self.subTest(symlink=symlink):
                root.symlink_to(held, target_is_directory=True) if symlink else root.write_text('not a directory')
                result = self.fixture.run(); self.assertEqual(result['status'], 'invalid')
                root.unlink()
        held.rename(root)

    def test_dangling_run_symlink_is_invalid(self):
        root = self.fixture.runs / self.fixture.names[0]
        shutil.rmtree(root); root.symlink_to(self.fixture.root / 'absent', target_is_directory=True)
        result = self.fixture.run()
        self.assertEqual(result['status'], 'invalid')
        self.assertTrue(any('run_directory_missing_or_symlink' in code for code in result['integrity_errors']))

    def test_terminal_timestamps_are_required_aware_and_ordered(self):
        path = self.fixture.runs / self.fixture.names[0] / 'manifest.json'
        original = path.read_bytes()
        for mutation in (lambda x: x.pop('started_at'), lambda x: x.pop('updated_at'),
                         lambda x: x.update(started_at='2026-10-28T03:04:00'),
                         lambda x: x.update(updated_at='2026-10-28T01:00:00+00:00')):
            value = load(path); mutation(value); save(path, value)
            self.assertEqual(self.fixture.run()['status'], 'invalid')
            path.write_bytes(original)

    def test_handshake_failure_cannot_accompany_observed_start(self):
        root = self.fixture.runs / self.fixture.names[0]
        self.fixture.diagnostic(root, 'complete', created=0)
        result = self.fixture.run()
        self.assertEqual(result['status'], 'invalid')
        self.assertTrue(any('diagnostic_connection_accounting' in code for code in result['integrity_errors']))

    def test_running_to_next_run_snapshot_race_is_pending(self):
        self.fixture.running_first()
        original = audit.base.Reader.unchanged
        def changed_raw_only(reader):
            return False if any(path.startswith(str(self.fixture.runs) + '/') for path in reader.files) else original(reader)
        with patch.object(audit.base.Reader, 'unchanged', changed_raw_only): result = self.fixture.run()
        self.assertEqual(result['status'], 'pending', result['integrity_errors'])
        self.assertIn('plan_advanced_during_probe', result['pending_reasons'])
        stable = self.fixture.run(); self.assertEqual(stable['status'], 'invalid')

    def test_completed_event_second_read_cannot_mix_snapshots(self):
        path = self.fixture.runs / self.fixture.names[0] / 'game-000001/events.jsonl'
        original = audit.base.Reader.read; reads = 0
        def changing(reader, selected, **kwargs):
            nonlocal reads
            if selected == path:
                reads += 1
                if reads == 2:
                    events = [json.loads(line) for line in path.read_text().splitlines()]
                    events[1]['latency_ms'] += 1; lines(path, events)
            return original(reader, selected, **kwargs)
        with patch.object(audit.base.Reader, 'read', changing): result = self.fixture.run()
        self.assertIn('terminal_events_changed_during_audit', result['integrity_errors'])
        self.assertEqual(result['status'], 'invalid')

    def test_history_registry_plan_and_wrapper_tampering_reject(self):
        for path in (self.fixture.registry, self.fixture.plan, self.fixture.wrapper,
                     self.fixture.stopped / 'game-000022/events.jsonl'):
            with self.subTest(path=str(path)):
                before = path.read_bytes(); path.write_bytes(before + b' ')
                self.assertEqual(self.fixture.run()['status'], 'invalid'); path.write_bytes(before)

    def test_append_plan_requires_prior_terminal_and_preserves_chain(self):
        self.fixture.fail_first_planned()
        plan2 = self.fixture.root / 'plan-v2.json'
        save(plan2, self.fixture.plan_value(['riichi-ranked-extra-000001-v1'], 29, 3, '2026-10-28T10:00:00+00:00'))
        target = self.fixture.root / 'registry-v2.json'
        audit.append_plan(self.fixture.registry, plan2, target, **self.fixture.options())
        result = self.fixture.run(target)
        self.assertEqual(result['status'], 'pending', result['integrity_errors'])
        self.assertEqual([row['revision'] for row in result['registry_chain']], [1, 2])
        self.assertEqual([row['status'] for row in result['plans']], ['stopped', 'planned'])
        sources = {row['path'] for row in result['source_files']}
        self.assertTrue({str(self.fixture.registry), str(target), str(self.fixture.plan), str(plan2)} <= sources)
        with self.assertRaises(FileExistsError): audit.append_plan(self.fixture.registry, plan2, target, **self.fixture.options())

    def test_append_while_active_and_cross_plan_overlap_reject(self):
        self.fixture.running_first()
        for name in self.fixture.names[1:]: shutil.rmtree(self.fixture.runs / name)
        plan2 = self.fixture.root / 'plan-v2.json'; new_name = 'riichi-ranked-extra-000001-v1'
        save(plan2, self.fixture.plan_value([new_name], 29, 2, '2026-10-28T10:00:00+00:00'))
        with self.assertRaisesRegex(audit.base.AuditError, 'append_requires_prior_plan_terminal'):
            audit.append_plan(self.fixture.registry, plan2, self.fixture.root / 'not-created', **self.fixture.options())
        self.assertFalse((self.fixture.root / 'not-created').exists())
        head = self.fixture.forge_head(plan2)
        shutil.copytree(self.fixture.diag, self.fixture.runs / new_name)
        result = self.fixture.run(head)
        self.assertIn('new_plan_started_before_prior_plan_terminal', result['integrity_errors'])
        self.assertEqual(result['status'], 'invalid')

    def test_publication_rechecks_every_dependency(self):
        result = self.fixture.run(); self.fixture.wrapper.write_text('# changed after audit')
        with self.assertRaisesRegex(audit.base.AuditError, 'independent_reaudit_changed'):
            audit.write_series_audit(result, self.fixture.root / 'not-created', **self.fixture.options())
        self.assertFalse((self.fixture.root / 'not-created').exists())


if __name__ == '__main__': unittest.main()
