"""Temporary synthetic campaigns exercise publication gates; no network/token use."""
import copy
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import tempfile
import unittest
from unittest.mock import patch

MODULE_PATH = Path(__file__).with_name('audit_ranked.py')
SPEC = importlib.util.spec_from_file_location('ranked_audit_under_test', MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def save(path, value):
    path.write_text(json.dumps(value) + '\n')


def lines(path, values):
    path.write_text(''.join(json.dumps(value) + '\n' for value in values))


def load(path):
    return json.loads(path.read_text())


class Fixture:
    def __init__(self, root):
        self.root, self.first, self.campaign = root, root / 'first', root / 'campaign'
        model, kit, research = root / 'model', root / 'kit', root / 'research'
        for path in (self.first, self.campaign, model, kit, research):
            path.mkdir()
        (model / 'ppo.pt').write_bytes(b'synthetic weights, never loaded')
        (kit / '__init__.py').write_text('__version__ = "0.1.0"\n')
        (research / 'model_agent.py').write_text('# synthetic source, never imported\n')
        engine, launcher = root / 'engine.so', root / 'launcher.py'
        engine.write_bytes(b'synthetic native identity, never loaded')
        launcher.write_text('# synthetic supervisor, never executed\n')
        weights_sha = audit.digest((model / 'ppo.pt').read_bytes())
        descriptor = {'artifacts': {'weights_path': {'path': 'ppo.pt', 'sha256': weights_sha}},
                      'factory': 'agari_research.model_agent:ModelAgent',
                      'kwargs': {'device': 'cpu', 'torch_threads': 1}}
        save(model / 'ppo-agent.json', descriptor)
        self.paths = audit.IdentityPaths(model, kit, research, engine, launcher)
        self.expected = {'weights': weights_sha,
            'descriptor': audit.digest((model / 'ppo-agent.json').read_bytes()),
            'kit': audit.source_digest(kit)[0], 'research': audit.source_digest(research, '0/')[0],
            'engine': audit.digest(engine.read_bytes()), 'launcher': audit.digest(launcher.read_bytes()),
            'kit_files': 1, 'research_files': 1}
        _, identity = audit.identities(self.paths, self.expected)
        self.records = []
        all_latency = []
        for index in range(100):
            directory = self.first if index == 0 else self.campaign / f'game-{index:06d}'
            directory.mkdir(exist_ok=True)
            manifest = dict(identity, schema_version=1, mode='ranked',
                endpoint='wss://game.riichi.dev/ws/ranked', agent=identity['agent_provenance']['name'],
                rules_expected=copy.deepcopy(audit.RULES), games_requested=1, games_recorded=1,
                reserve_ms=100, status='complete')
            save(directory / 'manifest.json', manifest)
            seat, scores = index % 4, [20000 + index, 25000, 27000, 28000 - index]
            counts = dict.fromkeys(('requests', 'replies_prepared', 'replies_sent', 'ack_accepted'), 3)
            game = dict(mode='ranked', status='complete', seat=seat, scores=scores,
                        validation_passed=None, degraded=False, counts=counts, game_index=0)
            lines(directory / 'games.jsonl', [game])
            save(directory / 'report.json', dict(games_requested=1, games_completed=1,
                 validation_passed=None, degraded_games=0, counts=counts))
            latencies = [1 + index / 1000, 3 + index / 1000, 7 + index / 1000]
            events = [{'event': 'start_game', 'seat': seat, 'game_index': 0}]
            for request_id, latency, elapsed in zip((2, 7, 11), latencies, (300 + index, 400 + index, 500 + index)):
                events.append(dict(event='decision', request_id=request_id, latency_ms=latency,
                                   action_type='dahai', game_index=0))
                events.append(dict(event='action_ack', request_id=request_id, status='accepted',
                                   elapsed_ms=elapsed, bank_consumed_ms=0, bank_ms=15000, game_index=0))
            events.append(dict(event='end_game', scores=scores, game_index=0))
            lines(directory / 'events.jsonl', events)
            if index:
                all_latency.extend(latencies)
                self.records.append(dict(game_number=index, directory=directory.name,
                    status='complete', healthy=True, errors=[], seat=seat, scores=scores,
                    degraded=False, counts=dict.fromkeys(audit.COUNT_FIELDS, 0) | counts,
                    latency=dict(count=3, mean_ms=statistics.fmean(latencies),
                        p50_ms=latencies[1], p95_ms=latencies[-1], max_ms=latencies[-1]), elapsed_seconds=5.0))
        lines(self.campaign / 'games.jsonl', self.records)
        save(self.campaign / 'manifest.json', dict(identity, schema_version=1, mode='ranked',
            agent=identity['agent_provenance']['name'], games_requested=99, reserve_ms=100,
            idle_timeout=1800, status='complete', current_game=None, error=None,
            games_recorded=99, launcher_sha256=self.expected['launcher']))
        save(self.campaign / 'summary.json', dict(schema_version=1, games_requested=99,
            games_attempted=99, games_completed=99, healthy_games=99, status='complete',
            current_game=None, error=None,
            counts=dict.fromkeys(audit.COUNT_FIELDS, 0) | dict.fromkeys(audit.GOOD_COUNTS, 297),
            latency=dict(count=297, mean_ms=statistics.fmean(all_latency), max_ms=max(all_latency)),
            elapsed_seconds=500.0))

    def run(self):
        return audit.audit_ranked(self.first, self.campaign, paths=self.paths, expected=self.expected)

    def publish(self, result, output):
        return audit.write_final_audit(result, output, paths=self.paths, expected=self.expected)

    def directory(self, number=1):
        return self.campaign / f'game-{number:06d}'


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='ranked-audit-test-')
        self.addCleanup(self.tmp.cleanup)
        self.fixture = Fixture(Path(self.tmp.name))

    def reject(self, code=None):
        result = self.fixture.run()
        self.assertEqual(result['status'], 'invalid', result['errors'])
        if code:
            self.assertTrue(any(code in error for error in result['errors']), result['errors'])
        output = self.fixture.root / 'rejected-output'
        with self.assertRaises(audit.AuditError):
            self.fixture.publish(result, output)
        self.assertFalse(output.exists())
        return result

    def test_complete_is_read_only_ready_and_pools_linear_quantiles(self):
        result = self.fixture.run()
        self.assertEqual(result['status'], 'ready', result['errors'])
        self.assertFalse(result['final_acceptance'])
        self.assertEqual(len(result['games']), 100)
        self.assertEqual(len(result['raw_files']), 403)
        self.assertEqual(result['totals']['local_decision_ms']['count'], 300)
        self.assertAlmostEqual(result['totals']['local_decision_ms']['p95_ms'], 7.08405)
        self.assertAlmostEqual(result['totals']['server_ack_elapsed_ms']['p95_ms'], 584.05)
        self.assertAlmostEqual(result['totals']['local_decision_ms']['mean_ms'], 11 / 3 + .0495)
        self.assertEqual(result, self.fixture.run())
        self.assertFalse((self.fixture.root / 'final-audit').exists())

    def test_publication_is_exclusive_and_hashes_bind_documents(self):
        result = self.fixture.run()
        output = self.fixture.root / 'final-audit'
        report = self.fixture.publish(result, output)
        self.assertEqual(report['status'], 'passed')
        self.assertTrue(report['final_acceptance'])
        normalized = dict(report, status='ready', final_acceptance=False)
        normalized.pop('published_at')
        self.assertEqual(normalized, result)
        self.assertEqual(load(output / 'games.json'), result['games'])
        self.assertEqual(load(output / 'raw-sha256.json'), result['raw_files'])
        for line in (output / 'SHA256SUMS').read_text().splitlines():
            checksum, name = line.split('  ')
            self.assertEqual(checksum, audit.digest((output / name).read_bytes()))
        before = {p.name: p.read_bytes() for p in output.iterdir()}
        with self.assertRaises(FileExistsError):
            self.fixture.publish(result, output)
        self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_99_games_cannot_publish(self):
        lines(self.fixture.campaign / 'games.jsonl', self.fixture.records[:-1])
        self.reject('missing_or_extra_campaign_game')

    def test_running_child_events_are_never_read(self):
        fixture = self.fixture
        for name in ('manifest.json', 'summary.json'):
            value = load(fixture.campaign / name)
            value.update(status='running', current_game=99)
            save(fixture.campaign / name, value)
        lines(fixture.campaign / 'games.jsonl', fixture.records[:-1])
        child = fixture.directory(99)
        value = load(child / 'manifest.json'); value['status'] = 'running'
        save(child / 'manifest.json', value)
        original = audit.Reader.read
        accessed = []
        def record_read(reader, path, **kwargs):
            accessed.append(path)
            self.assertNotEqual(path, child / 'events.jsonl')
            return original(reader, path, **kwargs)
        with patch.object(audit.Reader, 'read', record_read):
            result = fixture.run()
        self.assertEqual(result['status'], 'pending')
        self.assertEqual(result['errors'], [])
        self.assertEqual(result['totals']['games_verified_complete'], 99)
        with self.assertRaises(audit.AuditError):
            fixture.publish(result, fixture.root / 'not-created')
        self.assertFalse((fixture.root / 'not-created').exists())

    def test_running_partial_jsonl_is_pending(self):
        for name in ('manifest.json', 'summary.json'):
            path = self.fixture.campaign / name
            value = load(path); value['status'] = 'running'; save(path, value)
        with (self.fixture.campaign / 'games.jsonl').open('ab') as stream:
            stream.write(b'{"game_number":')
        result = self.fixture.run()
        self.assertEqual(result['status'], 'pending', result['errors'])
        self.assertIn('campaign_index_inflight', result['pending_reasons'])

    def test_running_snapshot_change_remains_pending(self):
        for name in ('manifest.json', 'summary.json'):
            path = self.fixture.campaign / name
            value = load(path); value['status'] = 'running'; save(path, value)
        with patch.object(audit.Reader, 'unchanged', return_value=False):
            result = self.fixture.run()
        self.assertEqual(result['status'], 'pending')
        self.assertIn('source_changed_during_probe', result['pending_reasons'])

    def test_failed_game_is_retained_and_rejects(self):
        directory = self.fixture.directory(3)
        value = load(directory / 'games.jsonl'); value['failure'] = 'connection_closed'
        save(directory / 'games.jsonl', value)
        result = self.reject('game_incomplete_or_failed')
        self.assertEqual(len(result['games']), 100)
        self.assertEqual(result['games'][3]['status'], 'invalid')

    def test_missing_or_reordered_campaign_record_rejects(self):
        records = copy.deepcopy(self.fixture.records)
        records[3], records[4] = records[4], records[3]
        lines(self.fixture.campaign / 'games.jsonl', records)
        self.reject('campaign_game_number_mismatch')

    def test_end_game_is_mandatory_and_matches_scores(self):
        path = self.fixture.directory() / 'events.jsonl'
        events = [json.loads(line) for line in path.read_text().splitlines()]
        events[-1]['scores'][0] += 100
        lines(path, events)
        self.reject('end_game_missing_or_mismatch')

    def test_duplicate_or_unknown_receipt_rejects(self):
        path = self.fixture.directory() / 'events.jsonl'
        events = [json.loads(line) for line in path.read_text().splitlines()]
        events[4]['request_id'] = events[2]['request_id']
        lines(path, events)
        self.reject('duplicate_or_unmatched_ack')

    def test_defaulted_ack_rejects_even_with_forged_clean_summary(self):
        path = self.fixture.directory() / 'events.jsonl'
        events = [json.loads(line) for line in path.read_text().splitlines()]
        events[2]['status'] = 'defaulted'
        lines(path, events)
        self.reject('nonaccepted_ack')

    def test_fatal_counter_cannot_be_hidden_in_rollups(self):
        path = self.fixture.directory() / 'games.jsonl'
        game = load(path); game['counts']['local_timeouts'] = 1; save(path, game)
        self.reject('nonzero_fatal_count')

    def test_nonfatal_diagnostics_match_supervisor_policy(self):
        fixture = self.fixture
        for name in ('games.jsonl', 'report.json'):
            path = fixture.directory() / name
            value = load(path)
            value['counts'].update(duplicate_start=1, duplicate_requests=2,
                                   stale_requests=3, duplicate_acks=4, unknown_messages=5)
            save(path, value)
        records = copy.deepcopy(fixture.records)
        records[0]['counts'].update(duplicate_start=1, duplicate_requests=2,
                                    stale_requests=3, duplicate_acks=4, unknown_messages=5)
        lines(fixture.campaign / 'games.jsonl', records)
        summary = load(fixture.campaign / 'summary.json')
        summary['counts'].update(duplicate_start=1, duplicate_requests=2,
                                 stale_requests=3, duplicate_acks=4, unknown_messages=5)
        save(fixture.campaign / 'summary.json', summary)
        result = fixture.run()
        self.assertEqual(result['status'], 'ready', result['errors'])
        self.assertEqual(result['totals']['diagnostic_counts']['duplicate_acks'], 4)

    def test_source_and_weight_mutations_reject(self):
        for path in (self.fixture.paths.kit_src / '__init__.py', self.fixture.paths.model_dir / 'ppo.pt'):
            with self.subTest(path=path.name):
                before = path.read_bytes(); path.write_bytes(before + b'changed')
                self.reject('actual_identity_differs_from_pinned_release')
                path.write_bytes(before)

    def test_child_identity_and_parent_summary_tampering_reject(self):
        for path, change, code in (
            (self.fixture.directory() / 'manifest.json',
             lambda x: x['engine'].update(binary_sha256='0' * 64), 'game_identity_mismatch'),
            (self.fixture.campaign / 'summary.json',
             lambda x: x['counts'].update(requests=298), 'campaign_summary_counts_mismatch'),
            (self.fixture.campaign / 'summary.json',
             lambda x: x['latency'].update(mean_ms=999), 'campaign_summary_latency_mismatch'),
        ):
            with self.subTest(code=code):
                before = path.read_bytes(); value = load(path); change(value); save(path, value)
                self.reject(code); path.write_bytes(before)

    def test_unindexed_extra_game_rejects(self):
        (self.fixture.campaign / 'game-000100').mkdir()
        self.reject('missing_or_extra_campaign_game')

    def test_write_reaudits_inputs_and_identity(self):
        result = self.fixture.run()
        (self.fixture.paths.model_dir / 'ppo.pt').write_bytes(b'changed after audit')
        with self.assertRaisesRegex(audit.AuditError, 'independent_reaudit_changed'):
            self.fixture.publish(result, self.fixture.root / 'not-created')
        self.assertFalse((self.fixture.root / 'not-created').exists())

    def test_unfinished_manifest_never_yields_pass_from_end_game_only(self):
        path = self.fixture.first / 'manifest.json'
        value = load(path); value['status'] = 'interrupted'; save(path, value)
        self.reject('transport_not_complete')

    def test_duplicate_keys_and_nonfinite_latency_are_rejected(self):
        path = self.fixture.directory() / 'events.jsonl'
        before = path.read_bytes()
        path.write_bytes(before.replace(b'"latency_ms": 1.001', b'"latency_ms": NaN', 1))
        self.reject('nonfinite_json')
        path.write_bytes(before.replace(b'"seat": 1', b'"seat": 1, "seat": 1', 1))
        self.reject('duplicate_json_key')


if __name__ == '__main__':
    unittest.main()
