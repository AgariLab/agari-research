#!/usr/bin/env python3
"""Audit 100 completed games across 101 attempts, retaining one fixed interruption.

The original zero-failure audit is immutable and does not pass this campaign.
No network, credentials, environment lookup, model load, or toolkit imports.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
LEGACY_PATH = HERE / 'audit_ranked.py'
LEGACY_SHA256 = '191c396e01574782942059c902ff6ae7ad915c7a68e000cf663f9c43077bbc40'
CONTINUATION_STARTED_AT = '2026-10-28T01:42:37.069470+00:00'
DEFAULT_CONTINUATION = Path('/Volumes/新加卷/AgariLab/runs/riichi-ranked-continuation78-v1')
HISTORY_PINS = {
    'first/manifest.json': (2238, '664c0c9e62b43b8588f9999987d432ca79eb32fd6d1848e79f54aca23b386a28'),
    'first/games.jsonl': (250, '95dfed903415964073d319f592e7e530d2e1607db0693a907952e7a6369381ea'),
    'first/events.jsonl': (32696, '16f1aa4b1b26d635105b1f419ebab928820d0fb71de8e13892896b688b5dfef3'),
    'first/report.json': (221, '426ad886fce666644ea0684fad02ed341309e9932f655e15a891252ffb8b374c'),
    'stopped/manifest.json': (2311, '159603577b0d996fb39ff6ce5fc4b1f67687a6904a2c5e33252c89494b088e44'),
    'stopped/summary.json': (804, 'd7ceeee5b93c6456e87926b3dcfbc67dd75d2a607d62eaa6664bbe55e7ef7dd5'),
    'stopped/games.jsonl': (16164, '572339e4d7500c42d4a9819a8184473671428046bb28bc88a615a9a51101a479'),
    'stopped/game-000022/manifest.json': (2240, '902d713bf4f24d13e0f7a7b41c8c7bfdf55bef475bd8aad7c2e7f870cefafcc9'),
    'stopped/game-000022/games.jsonl': (183, '76ca552c90e7bb828467926ec9fc51ad695aa6b965a45af80b5d6fd34e4c5fca'),
    'stopped/game-000022/events.jsonl': (0, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'),
    'stopped/game-000022/report.json': (119, '2defeeefb69d8ba515baa263c1f8c921e2004521b8e8c18e1c6f996235c991bf'),
}
FAILURE_ERRORS = ['game_incomplete', 'game_failed', 'action_receipts_mismatch', 'transport_incomplete']


def _load_legacy():
    if LEGACY_PATH.is_symlink() or hashlib.sha256(LEGACY_PATH.read_bytes()).hexdigest() != LEGACY_SHA256:
        raise ValueError('legacy_auditor_identity_changed')
    name = '_fixed_ranked_auditor_for_continuation'
    spec = importlib.util.spec_from_file_location(name, LEGACY_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = _load_legacy()


def _verify_history(reader, first, stopped, pins):
    base.require(set(pins) == set(HISTORY_PINS), 'historical_pin_set_changed')
    checked = {}
    for label, (size, sha256) in sorted(pins.items()):
        stage, relative = label.split('/', 1)
        root = first if stage == 'first' else stopped
        data = reader.read(root / relative)
        base.require(len(data) == size and base.digest(data) == sha256, 'historical_evidence_changed:' + label)
        checked[label] = {'bytes': size, 'sha256': sha256}
    return checked


def _batch_header(root, requested, identity, launcher, reader):
    manifest = reader.object(root / 'manifest.json')
    summary = reader.object(root / 'summary.json')
    base.require(base.exact_int(manifest.get('schema_version'), 1)
                 and base.exact_int(summary.get('schema_version'), 1)
                 and manifest.get('mode') == 'ranked'
                 and base.exact_int(manifest.get('games_requested'), requested)
                 and base.exact_int(summary.get('games_requested'), requested)
                 and base.exact_int(manifest.get('reserve_ms'), 100)
                 and manifest.get('idle_timeout') == 1800, 'batch_configuration_mismatch')
    base.require(all(manifest.get(key) == value for key, value in identity.items())
                 and manifest.get('agent') == identity['agent_provenance']['name']
                 and manifest.get('launcher_sha256') == launcher, 'batch_identity_mismatch')
    return manifest, summary


def _complete_attempt(root, local_number, global_attempt, completed_game, stage, identity, reader, record=None):
    directory = root if stage == 'first' else root / f'game-{local_number:06d}'
    if not directory.is_dir() or directory.is_symlink():
        return {'directory': str(directory), 'status': 'invalid',
                'global_attempt': global_attempt, 'campaign_game': local_number,
                'completed_game': None, 'source_stage': stage,
                'integrity_errors': ['missing_or_symlink_game_directory']}, [], []
    # The historical helper checks the supervisor's LOCAL game_number. Global
    # attempt and completed-game numbering are separate and never hide a loss.
    row, local, server = base.audit_game(directory, 1 if stage == 'first' else local_number + 1,
                                         identity, reader, record)
    row.pop('game_number')
    row['integrity_errors'] = row.pop('errors')
    row.update(global_attempt=global_attempt, campaign_game=local_number,
               completed_game=completed_game if row['status'] == 'complete' else None, source_stage=stage)
    if record is not None and row['status'] != 'complete':
        row['integrity_errors'].append('recorded_continuation_or_historical_game_not_complete')
        row['status'] = 'invalid'
    return row, local, server


def _historical_failure(stopped, record, identity, reader):
    directory = stopped / 'game-000022'
    base.require(directory.is_dir() and not directory.is_symlink(), 'historical_failure_directory_invalid')
    manifest = reader.object(directory / 'manifest.json')
    records = base.jsonl(reader.read(directory / 'games.jsonl'))
    report = reader.object(directory / 'report.json')
    events = reader.read(directory / 'events.jsonl')
    base.require(manifest.get('status') == 'incomplete' and manifest.get('mode') == 'ranked'
                 and base.exact_int(manifest.get('schema_version'), 1)
                 and base.exact_int(manifest.get('games_requested'), 1)
                 and base.exact_int(manifest.get('games_recorded'), 1)
                 and base.exact_int(manifest.get('reserve_ms'), 100)
                 and manifest.get('endpoint') == 'wss://game.riichi.dev/ws/ranked'
                 and manifest.get('rules_expected') == base.RULES
                 and manifest.get('agent') == identity['agent_provenance']['name']
                 and all(manifest.get(key) == value for key, value in identity.items()),
                 'historical_failure_manifest_mismatch')
    expected_game = {'mode': 'ranked', 'status': 'interrupted', 'seat': None, 'scores': None,
                     'validation_passed': None, 'degraded': False, 'counts': {},
                     'failure': 'connection_closed', 'game_index': 0}
    base.require(records == [expected_game] and type(records[0]['game_index']) is int
                 and records[0]['degraded'] is False, 'historical_failure_record_mismatch')
    base.require(report == {'games_requested': 1, 'games_completed': 0, 'validation_passed': None,
                            'degraded_games': 0, 'counts': {}}
                 and all(type(report[key]) is int for key in ('games_requested', 'games_completed', 'degraded_games')),
                 'historical_failure_report_mismatch')
    base.require(events == b'', 'historical_failure_events_not_empty')
    zero_counts = dict.fromkeys(base.COUNT_FIELDS, 0)
    expected_record = {'game_number': 22, 'directory': 'game-000022', 'status': 'interrupted',
        'healthy': False, 'errors': FAILURE_ERRORS, 'seat': None, 'scores': None, 'degraded': False,
        'counts': zero_counts,
        'latency': {'count': 0, 'mean_ms': None, 'p50_ms': None, 'p95_ms': None, 'max_ms': None},
        'elapsed_seconds': 33.53039512503892}
    base.require(record == expected_record and record['healthy'] is False and record['degraded'] is False,
                 'historical_failure_parent_record_mismatch')
    failure = {'global_attempt': 23, 'campaign_game': 22, 'failure': 'connection_closed',
        'source_stage': 'stopped_batch', 'source_directory': str(directory),
        'start_game_observed': False, 'events_count': 0, 'requests': 0,
        'seat': None, 'scores': None, 'server_outcome': 'unknown', 'rating_impact': 'unknown',
        'integrity_verified': True}
    attempt = dict(failure, directory=str(directory), status='interrupted', completed_game=None,
        counts=zero_counts, local_decision_ms=base.stats([]), server_ack_elapsed_ms=base.stats([]),
        observed_errors=list(FAILURE_ERRORS), integrity_errors=[],
        elapsed_seconds=record['elapsed_seconds'], events_sha256=base.digest(events))
    return attempt, failure


def _check_directory_index(root, count):
    actual = sorted(path.name for path in root.iterdir() if path.is_dir() or path.is_symlink())
    base.require(actual == [f'game-{number:06d}' for number in range(1, count + 1)],
                 'batch_missing_or_extra_attempt_directory')


def _check_rollup(manifest, summary, rows, *, attempted, completed, status, current, error):
    base.require(manifest.get('status') == summary.get('status') == status
                 and base.exact_int(manifest.get('games_recorded'), attempted)
                 and manifest.get('current_game') == summary.get('current_game') == current
                 and manifest.get('error') == summary.get('error') == error,
                 'batch_terminal_manifest_mismatch')
    base.require(base.exact_int(summary.get('games_attempted'), attempted)
                 and base.exact_int(summary.get('games_completed'), completed)
                 and base.exact_int(summary.get('healthy_games'), completed), 'batch_summary_totals_mismatch')
    base.require(len(rows) == attempted and sum(row['status'] == 'complete' for row in rows) == completed,
                 'batch_attempt_reconciliation_failed')
    counts = {key: sum(row['counts'][key] for row in rows) for key in base.COUNT_FIELDS}
    base.require(base.normalized_counts(summary.get('counts')) == counts, 'batch_summary_counts_mismatch')
    healthy = [row for row in rows if row['status'] == 'complete']
    n = sum(row['local_decision_ms']['count'] for row in healthy)
    base.same_numbers(summary.get('latency'), {
        'count': n,
        'mean_ms': sum(row['local_decision_ms']['count'] * row['local_decision_ms']['mean_ms'] for row in healthy) / n,
        'max_ms': max(row['local_decision_ms']['max_ms'] for row in healthy)}, 'batch_summary_latency_mismatch')
    base.require(base.finite(summary.get('elapsed_seconds')), 'batch_summary_elapsed_invalid')


def _totals(attempts, local, server):
    completed = [row for row in attempts if row['status'] == 'complete']
    ranks = Counter(row['rank_min'] for row in completed if row['rank_min'] == row['rank_max'])
    return {
        'attempts': len(attempts), 'games_completed': len(completed),
        'healthy_completed': len(completed),
        'failed_attempts': sum(row['status'] in ('interrupted', 'invalid') for row in attempts),
        'invalid_attempts': sum(row['status'] == 'invalid' for row in attempts),
        'pending_attempts': sum(row['status'] == 'pending' for row in attempts),
        'score_sum': sum(row['score'] for row in completed),
        'mean_score': statistics.fmean(row['score'] for row in completed) if completed else None,
        'mean_point_delta': statistics.fmean(row['point_delta'] for row in completed) if completed else None,
        'mean_rank_min': statistics.fmean(row['rank_min'] for row in completed) if completed else None,
        'mean_rank_max': statistics.fmean(row['rank_max'] for row in completed) if completed else None,
        'untied_rank_counts': {str(rank): ranks[rank] for rank in range(1, 5)},
        'tied_games': sum(row['rank_min'] != row['rank_max'] for row in completed),
        'counts': {key: sum(row['counts'][key] for row in completed) for key in base.COUNT_FIELDS},
        'diagnostic_counts': {key: sum(row['counts'][key] for row in completed) for key in base.DIAGNOSTIC_COUNTS},
        'local_decision_ms': base.stats(local), 'server_ack_elapsed_ms': base.stats(server),
        'bank_consumed_ms': sum(row['bank_consumed_ms'] for row in completed),
        'statistics_denominator': 'Verified completed games; every failed attempt is separately retained, with unknown outcome and no invented score/rank.',
    }


def audit_ranked_continuation(first_game=base.DEFAULT_FIRST, stopped_campaign=base.DEFAULT_CAMPAIGN,
                              continuation=DEFAULT_CONTINUATION, *, paths=None, expected=None, history_pins=None):
    """Read-only audit; keyword identity/pin overrides are for synthetic tests only."""
    first, stopped, resumed = map(lambda path: Path(path).resolve(), (first_game, stopped_campaign, continuation))
    reader, errors, pending, attempts, failures = base.Reader(), [], [], [], []
    local_all, server_all = [], []
    result = {'schema_version': 1, 'status': 'pending', 'final_acceptance': False,
        'acceptance_profile': '100_completed_with_one_fixed_historical_interruption',
        'final_acceptance_scope': 'Evidence reconciles resumed completion; original zero-failure reliability criterion remains unmet.',
        'original_zero_failure_criterion_met': False, 'target_completed_games': 100, 'expected_attempts': 101,
        'auditor_sha256': base.digest(Path(__file__).read_bytes()), 'legacy_auditor_sha256': LEGACY_SHA256,
        'inputs': {'first_game': str(first), 'stopped_campaign': str(stopped), 'continuation': str(resumed)},
        'integrity_errors': errors, 'pending_reasons': pending, 'attempts': attempts, 'attempt_failures': failures,
        'identity_checks': {}, 'historical_pins': {}, 'raw_files': [], 'totals': {}, 'stage_totals': {},
        'recovery_evidence': {},
        'statistics_method': {
            'quantiles': 'Pooled individual actions; linear interpolation at sorted position (n - 1) * p.',
            'supervisor_p95': 'Historical nearest-rank p95 is recomputed only to check each index, not used for pooled final p95.',
            'rank': 'rank_min/rank_max from scores; retain intervals for ties without inventing server tie-break.',
            'point_delta': 'Final score minus 25000; no rating, uma or oka.',
            'local_decision_ms': 'Client socket.recv timestamp to response preparation: queueing, parsing, IPC/inference and legality checks; excludes loading, socket.send and network round trip.',
            'server_ack_elapsed_ms': 'Server-reported elapsed time includes request/response network round trip; not pure inference or client-measured ack RTT.'},
        'limitations': [
            'Original zero-failure campaign stopped. This resumed-completion audit does not turn it into 100 consecutive healthy attempts.',
            'No start_game was observed for historical attempt 23. Empty local events do not establish that the server never started a game or that rating was unaffected.',
            'Compact local transport records are not full MJAI replays or independently signed server records.',
            'Nonfatal filtered-message counts are reconciled across recorded counters, not reconstructed from omitted messages.',
            'Online matchmaking results are descriptive and do not replace fixed-opponent playing-strength evaluation.']}
    running = False
    try:
        base.require(base.digest(LEGACY_PATH.read_bytes()) == LEGACY_SHA256, 'legacy_auditor_identity_changed')
        paths, expected = paths or base.IdentityPaths.defaults(), base.EXPECTED if expected is None else expected
        pins = HISTORY_PINS if history_pins is None else history_pins
        actual, identity = base.identities(paths, expected)
        result['identity_checks'] = {'actual': actual, 'expected': dict(expected), 'matches': True}
        result['historical_pins'] = _verify_history(reader, first, stopped, pins)
        row, local, server = _complete_attempt(first, 1, 1, 1, 'first', identity, reader)
        attempts.append(row)
        if row['status'] == 'complete':
            local_all.extend(local); server_all.extend(server)
        old_manifest, old_summary = _batch_header(stopped, 99, identity, actual['launcher'], reader)
        old_records = base.jsonl(reader.read(stopped / 'games.jsonl'))
        base.require(len(old_records) == 22, 'historical_attempt_count_changed')
        for number, record in enumerate(old_records[:21], 1):
            row, local, server = _complete_attempt(stopped, number, number + 1, number + 1,
                                                  'stopped_batch', identity, reader, record)
            attempts.append(row)
            if row['status'] == 'complete':
                local_all.extend(local); server_all.extend(server)
        failure_row, failure = _historical_failure(stopped, old_records[21], identity, reader)
        attempts.append(failure_row); failures.append(failure)
        _check_directory_index(stopped, 22)
        base.require(all(row['status'] == 'complete' for row in attempts[:22]), 'historical_complete_game_invalid')
        _check_rollup(old_manifest, old_summary, attempts[1:23], attempted=22, completed=21,
                      status='stopped', current=22, error='game_incomplete')
        manifest, summary = _batch_header(resumed, 78, identity, actual['launcher'], reader)
        base.require(manifest.get('started_at') == CONTINUATION_STARTED_AT, 'continuation_start_identity_mismatch')
        base.require(datetime.fromisoformat(manifest['started_at']) > datetime.fromisoformat(old_manifest['updated_at']),
                     'continuation_precedes_historical_stop')
        result['recovery_evidence'] = {
            'original_status': old_manifest['status'], 'original_error': old_manifest['error'],
            'original_requested_games': 99, 'original_attempts': 22, 'original_completed_games': 21,
            'original_updated_at': old_manifest['updated_at'], 'continuation_started_at': manifest['started_at'],
            'continuation_requested_games': 78, 'same_model_source_engine_launcher': True,
            'server_outcome_of_interruption': 'unknown', 'rating_impact_of_interruption': 'unknown'}
        terminal = manifest.get('status') == summary.get('status') == 'complete'
        running = manifest.get('status') == 'running' or summary.get('status') == 'running'
        if running:
            pending.append('continuation_running')
        elif not terminal:
            errors.append('continuation_stopped_or_failed')
        try:
            records = base.jsonl(reader.read(resumed / 'games.jsonl'))
        except base.AuditError:
            if not running:
                raise
            records = []
            pending.append('continuation_index_inflight')
        base.require(len(records) <= 78, 'too_many_continuation_attempts')
        for number, record in enumerate(records, 1):
            row, local, server = _complete_attempt(resumed, number, number + 23, number + 22,
                                                  'continuation', identity, reader, record)
            attempts.append(row)
            if row['status'] == 'complete':
                local_all.extend(local); server_all.extend(server)
        if terminal:
            base.require(len(records) == 78, 'continuation_not_78_complete_attempts')
            _check_directory_index(resumed, 78)
            base.require(all(row['status'] == 'complete' for row in attempts[23:]), 'new_continuation_failure')
            _check_rollup(manifest, summary, attempts[23:], attempted=78, completed=78,
                          status='complete', current=None, error=None)
        else:
            pending.append('full_100_completed_games_not_final')
        _verify_history(reader, first, stopped, pins)
        base.require(base.identities(paths, expected)[0] == actual, 'identity_changed_during_audit')
        base.require(base.digest(LEGACY_PATH.read_bytes()) == LEGACY_SHA256, 'legacy_auditor_identity_changed')
        if not reader.unchanged():
            if running:
                pending.append('source_changed_during_probe')
            else:
                errors.append('source_changed_during_final_audit')
    except (base.AuditError, OSError, KeyError, TypeError, IndexError, ValueError) as exc:
        errors.append(str(exc) if isinstance(exc, base.AuditError) else 'unreadable_or_malformed_continuation_evidence')
    for row in attempts:
        errors.extend(f'attempt_{row["global_attempt"]}:{code}' for code in row['integrity_errors'])
        if row['status'] == 'pending':
            pending.append(f'attempt_{row["global_attempt"]}_pending')
    complete = [row for row in attempts if row['status'] == 'complete']
    if len({row['events_sha256'] for row in complete}) != len(complete):
        errors.append('duplicate_complete_game_event_evidence')
    result['totals'] = _totals(attempts, local_all, server_all)
    for stage in ('first', 'stopped_batch', 'continuation'):
        rows = [row for row in attempts if row['source_stage'] == stage]
        result['stage_totals'][stage] = {'attempts': len(rows),
            'games_completed': sum(row['status'] == 'complete' for row in rows),
            'failed_attempts': sum(row['status'] in ('interrupted', 'invalid') for row in rows)}
    result['raw_files'] = sorted(reader.files.values(), key=lambda item: item['path'])
    if errors:
        result['status'] = 'invalid'
    elif pending or len(complete) != 100:
        result['status'] = 'pending'
    elif len(attempts) != 101 or len(failures) != 1 or len(reader.files) != 410:
        errors.append('final_101_attempt_100_complete_410_file_reconciliation_failed')
        result['status'] = 'invalid'
    else:
        result['status'] = 'ready_with_interruption'
    return result


def write_continued_audit(result, output, *, paths=None, expected=None, history_pins=None):
    base.require(result.get('status') == 'ready_with_interruption' and result.get('final_acceptance') is False
                 and result.get('original_zero_failure_criterion_met') is False
                 and not result.get('integrity_errors') and not result.get('pending_reasons')
                 and result.get('totals', {}).get('attempts') == 101
                 and result.get('totals', {}).get('games_completed') == 100
                 and result.get('totals', {}).get('healthy_completed') == 100
                 and result.get('totals', {}).get('failed_attempts') == 1
                 and len(result.get('attempts', [])) == 101 and len(result.get('raw_files', [])) == 410,
                 'continued_audit_not_ready')
    inputs = result.get('inputs', {})
    base.require(set(inputs) == {'first_game', 'stopped_campaign', 'continuation'}, 'continued_inputs_missing')
    fresh = audit_ranked_continuation(inputs['first_game'], inputs['stopped_campaign'], inputs['continuation'],
                                      paths=paths, expected=expected, history_pins=history_pins)
    base.require(fresh == result and fresh['status'] == 'ready_with_interruption', 'independent_reaudit_changed')
    for item in result['raw_files']:
        path = Path(item['path'])
        base.require(not path.is_symlink() and path.is_file() and path.stat().st_size == item['bytes']
                     and base.digest(path.read_bytes()) == item['sha256'], 'source_changed_before_publication')
    base.require(base.digest(Path(__file__).read_bytes()) == result['auditor_sha256'], 'auditor_changed_before_publication')
    output = Path(output)
    output.mkdir(exist_ok=False)
    report = dict(result, status='completed_with_interruption', final_acceptance=True,
                  published_at=datetime.now(timezone.utc).isoformat())
    documents = {'report.json': report, 'attempts.json': result['attempts'], 'raw-sha256.json': result['raw_files']}
    for name, value in documents.items():
        with (output / name).open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    with (output / 'SHA256SUMS').open('x', encoding='utf-8') as stream:
        stream.write(''.join(f'{base.digest((output / name).read_bytes())}  {name}\n' for name in sorted(documents)))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first-game', type=Path, default=base.DEFAULT_FIRST)
    parser.add_argument('--stopped-campaign', type=Path, default=base.DEFAULT_CAMPAIGN)
    parser.add_argument('--continuation', type=Path, default=DEFAULT_CONTINUATION)
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--report-dir', type=Path, default=HERE / 'continued-audit')
    args = parser.parse_args(argv)
    result = audit_ranked_continuation(args.first_game, args.stopped_campaign, args.continuation)
    if args.write:
        try:
            result = write_continued_audit(result, args.report_dir)
        except (base.AuditError, OSError) as exc:
            print(json.dumps({'status': result['status'], 'final_acceptance': False,
                'write_error': str(exc) if isinstance(exc, base.AuditError) else 'report_directory_exists_or_unwritable',
                'integrity_errors': result['integrity_errors'], 'pending_reasons': result['pending_reasons']}))
            return 2
    print(json.dumps({key: result[key] for key in ('status', 'final_acceptance',
        'original_zero_failure_criterion_met', 'integrity_errors', 'pending_reasons',
        'stage_totals', 'totals', 'attempt_failures')}, ensure_ascii=False, indent=2))
    return 0 if result['status'] in ('ready_with_interruption', 'completed_with_interruption') else 1


if __name__ == '__main__':
    raise SystemExit(main())
