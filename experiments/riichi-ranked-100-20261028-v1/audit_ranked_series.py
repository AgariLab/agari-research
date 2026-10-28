#!/usr/bin/env python3
"""Read-only, append-registry audit of ranked attempts; never runs a bot."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import sys

HERE = Path(__file__).resolve().parent
DEFAULT_REGISTRY = HERE / 'series-registry-v1.json'
ROOT_REGISTRY_SHA256 = '564d5f67baaa9dd0c901313ae3c5879b4e81ffb437c1d7e38c5f760a4fa5cc2f'
LEGACY_SHA256 = '191c396e01574782942059c902ff6ae7ad915c7a68e000cf663f9c43077bbc40'
WRAPPER_SHA256 = '9c8dd327662276771082a5dd543b7bd4b00d8fa419ce3de64962a808cd10e294'
LEGACY_PATH = HERE / 'audit_ranked.py'
if hashlib.sha256(LEGACY_PATH.read_bytes()).hexdigest() != LEGACY_SHA256:
    raise ValueError('legacy_auditor_identity_changed')
_spec = importlib.util.spec_from_file_location('_series_fixed_legacy', LEGACY_PATH)
base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = base
_spec.loader.exec_module(base)
CONFIG = {'ping_interval': 20, 'ping_timeout': 20, 'open_timeout': 15, 'close_timeout': 2,
          'max_size': 2097152, 'max_queue': 16, 'write_limit': 32768, 'compression': 'deflate'}
FAILURE_CODES = {'connection_closed', 'receive_timeout', 'protocol_error', 'client_error',
                 'connection_or_worker_failed'}
DIAGNOSTIC_EVENTS = {'diagnostic_ready', 'connection_attempt', 'connection_created',
    'handshake_complete', 'handshake_failed', 'recv_cancelled', 'recv_failed', 'send_failed',
    'connection_lost', 'context_exit'}
DIAGNOSTIC_KEYS = {'event', 'elapsed_seconds', 'attempt', 'connection_id', 'recv_count',
    'seconds_since_last_recv', 'library_latency_seconds', 'received_close_code', 'sent_close_code',
    'received_close_first', 'local_keepalive_timeout_observed', 'abnormal_no_close_frames',
    'exception_category', 'cause_category', 'receive_exception_category', 'http_status'}
SAFE_CATEGORIES = {None, 'cancelled', 'connection_closed_ok', 'connection_closed_error',
    'connection_closed', 'proxy_http_status', 'proxy_invalid_response', 'proxy_error',
    'invalid_proxy', 'handshake_http_status', 'handshake_invalid_response', 'handshake_security_error',
    'handshake_error', 'invalid_uri', 'payload_too_big', 'websocket_protocol_error',
    'tls_certificate_error', 'tls_error', 'dns_error', 'timeout', 'connection_reset',
    'connection_refused', 'connection_aborted', 'broken_pipe', 'eof', 'os_error',
    'diagnostic_error', 'other_error'}


def _timestamp(value):
    base.require(type(value) is str, 'missing_or_invalid_timestamp')
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise base.AuditError('missing_or_invalid_timestamp') from None
    base.require(parsed.tzinfo is not None and parsed.utcoffset() is not None, 'timestamp_requires_timezone')
    return parsed


def _read_registry(path, reader, anchor, seen=None, chain=None):
    seen = set() if seen is None else seen
    path = Path(path).resolve()
    base.require(path not in seen and len(seen) < 64, 'registry_chain_cycle_or_too_long')
    seen.add(path)
    raw = reader.read(path)
    value = base.decode(raw)
    base.require(type(value) is dict and value.get('schema_version') == 1
                 and value.get('target_complete_games') == 100, 'registry_schema_or_target')
    prior = value.get('predecessor')
    if prior is None:
        base.require(value.get('revision') == 1 and base.digest(raw) == anchor, 'root_registry_identity_changed')
    else:
        base.require(type(prior) is dict and set(prior) == {'path', 'sha256'}, 'registry_predecessor_schema')
        parent_path = Path(prior['path']).resolve()
        parent = _read_registry(parent_path, reader, anchor, seen, chain)
        base.require(reader.files[str(parent_path)]['sha256'] == prior['sha256'], 'registry_predecessor_changed')
        base.require(value['revision'] == parent['revision'] + 1
                     and len(value['plans']) == len(parent['plans']) + 1
                     and value['plans'][:-1] == parent['plans'], 'registry_not_append_only')
        ignored = {'revision', 'predecessor', 'plans'}
        base.require({k: v for k, v in value.items() if k not in ignored}
                     == {k: v for k, v in parent.items() if k not in ignored}, 'registry_history_or_policy_changed')
    if chain is not None:
        chain.append({'path': str(path), 'sha256': base.digest(raw), 'revision': value['revision']})
    return value


def _plan(entry, reader, expected, wrapper_sha):
    path = Path(entry['path']).resolve()
    data = reader.read(path)
    base.require(len(data) == entry['bytes'] and base.digest(data) == entry['sha256'], 'plan_identity_changed')
    value = base.decode(data)
    fixed = {'schema_version': 1, 'target_complete_games': 100, 'mode': 'ranked',
        'runtime_variant': 'parent_transport_observer', 'wrapper_sha256': wrapper_sha,
        'supervisor_sha256': expected['launcher'], 'kit_source_sha256': expected['kit'],
        'research_source_sha256': expected['research'], 'weights_sha256': expected['weights'],
        'descriptor_sha256': expected['descriptor'], 'engine_version': '0.4.10',
        'proxy_policy': 'environment', 'ping_interval_seconds': 20, 'ping_timeout_seconds': 20,
        'games_per_run': 1, 'concurrent_connections': 1, 'stop_on_first_unsuccessful_run': True,
        'retry_failed_run': False, 'validation_repeated': False, 'original_zero_failure_criterion_met': False}
    base.require(type(value) is dict and all(type(value.get(k)) is type(v) and value[k] == v for k, v in fixed.items()),
                 'plan_policy_or_identity_mismatch')
    names = value.get('planned_runs')
    base.require(type(names) is list and names and len(names) == value.get('maximum_new_attempts')
                 and len(set(names)) == len(names)
                 and all(type(name) is str and re.fullmatch(r'riichi-ranked-[a-z0-9-]+', name) for name in names),
                 'plan_run_names_invalid')
    base.require(base.integer(value.get('complete_games_before_execution'))
                 and base.integer(value.get('prior_interrupted_attempts')), 'plan_baseline_invalid')
    _timestamp(value['created_at'])
    return value


def _inventory(root):
    base.require(root.is_dir() and not root.is_symlink(), 'run_directory_missing_or_symlink')
    files, directories = {}, []
    for path in root.rglob('*'):
        base.require(not path.is_symlink(), 'run_contains_symlink')
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path
        elif path.is_dir():
            directories.append(path.relative_to(root).as_posix())
        else:
            raise base.AuditError('run_contains_nonregular_entry')
    base.require(all(any(name.startswith(directory + '/') for name in files) for directory in directories),
                 'run_contains_unregistered_empty_directory')
    return files


def _pins(root, pins, reader):
    actual = _inventory(root)
    base.require(set(actual) == set(pins), 'sealed_batch_inventory_changed')
    for relative, expected in pins.items():
        data = reader.read(actual[relative])
        base.require(len(data) == expected['bytes'] and base.digest(data) == expected['sha256'],
                     'sealed_batch_bytes_changed')


def _manifest_identity(manifest, identity):
    base.require(manifest.get('mode') == 'ranked'
                 and manifest.get('agent') == identity['agent_provenance']['name']
                 and all(manifest.get(k) == v for k, v in identity.items()), 'recorded_identity_mismatch')


def _latency_index(local):
    ordered = sorted(local)
    return {'count': len(local), 'mean_ms': statistics.fmean(local) if local else None,
        'p50_ms': statistics.median(local) if local else None,
        'p95_ms': ordered[math.ceil(len(local) * .95) - 1] if local else None,
        'max_ms': max(local) if local else None}


def _interrupted(directory, local_number, identity, reader, record):
    manifest = reader.object(directory / 'manifest.json')
    _manifest_identity(manifest, identity)
    base.require(manifest.get('status') in ('incomplete', 'interrupted')
                 and manifest.get('rules_expected') == base.RULES
                 and base.exact_int(manifest.get('games_requested'), 1)
                 and base.exact_int(manifest.get('games_recorded'), 1)
                 and manifest.get('endpoint') == 'wss://game.riichi.dev/ws/ranked'
                 and base.exact_int(manifest.get('reserve_ms'), 100), 'interrupted_manifest_schema')
    records = base.jsonl(reader.read(directory / 'games.jsonl'))
    base.require(len(records) == 1 and type(records[0]) is dict, 'interrupted_game_record_count')
    game = records[0]
    base.require(game.get('mode') == 'ranked' and game.get('status') == 'interrupted'
                 and base.exact_int(game.get('game_index'), 0)
                 and game.get('failure') in FAILURE_CODES and type(game.get('degraded')) is bool
                 and game.get('validation_passed') is None and game.get('scores') is None,
                 'interrupted_game_schema')
    counts = base.normalized_counts(game.get('counts'))
    report = reader.object(directory / 'report.json')
    base.require(base.exact_int(report.get('games_requested'), 1)
                 and base.exact_int(report.get('games_completed'), 0)
                 and base.exact_int(report.get('degraded_games'), int(game['degraded']))
                 and report.get('validation_passed') is None
                 and base.normalized_counts(report.get('counts')) == counts, 'interrupted_report_mismatch')
    raw = reader.read(directory / 'events.jsonl')
    events = base.jsonl(raw)
    local, server, decisions, ack_keys, timeouts = [], [], {}, set(), set()
    ack_counts = Counter()
    started, seat, bank = False, None, 0
    for index, event in enumerate(events):
        base.require(type(event) is dict and base.exact_int(event.get('game_index'), 0), 'interrupted_event_schema')
        kind = event.get('event')
        if kind == 'start_game':
            base.require(index == 0 and not started and set(event) == {'event', 'seat', 'game_index'}
                         and type(event['seat']) is int and event['seat'] in range(4), 'interrupted_start_mismatch')
            started, seat = True, event['seat']
            continue
        base.require(started and base.integer(event.get('request_id')), 'interrupted_event_outside_game')
        request = event['request_id']
        if kind == 'decision':
            base.require(set(event) == {'event', 'request_id', 'latency_ms', 'action_type', 'game_index'}
                         and request not in decisions and (not decisions or request > max(decisions))
                         and base.finite(event.get('latency_ms')) and event.get('action_type') in base.ACTION_TYPES,
                         'interrupted_decision_invalid')
            decisions[request] = event
            local.append(event['latency_ms'])
        elif kind == 'action_ack':
            status = event.get('status')
            base.require(set(event) == {'event', 'request_id', 'status', 'elapsed_ms', 'bank_consumed_ms', 'bank_ms', 'game_index'}
                         and status in ('accepted', 'rejected', 'unparseable', 'defaulted', 'stale')
                         and (request, status) not in ack_keys
                         and (status == 'defaulted' or request in decisions)
                         and all(base.finite(event.get(k)) for k in ('elapsed_ms', 'bank_consumed_ms', 'bank_ms')),
                         'interrupted_ack_invalid')
            ack_keys.add((request, status)); ack_counts['ack_' + status] += 1
            server.append(event['elapsed_ms']); bank += event['bank_consumed_ms']
        elif kind == 'inference_timeout':
            base.require(set(event) == {'event', 'request_id', 'game_index'} and request not in timeouts,
                         'interrupted_timeout_invalid')
            timeouts.add(request)
        else:
            raise base.AuditError('interrupted_unexpected_event_or_end_game')
    base.require(game.get('seat') == seat and (seat is None or type(game['seat']) is int), 'interrupted_seat_mismatch')
    base.require(counts['replies_prepared'] == len(decisions) and counts['local_timeouts'] == len(timeouts)
                 and counts['requests'] >= len(set(decisions) | timeouts | {request for request, _ in ack_keys})
                 and counts['requests'] >= counts['replies_prepared'] >= counts['replies_sent'] >= counts['ack_accepted']
                 and all(counts.get(k, 0) == ack_counts[k] for k in ('ack_accepted', 'ack_rejected', 'ack_unparseable', 'ack_defaulted', 'ack_stale')),
                 'interrupted_observed_count_mismatch')
    expected_errors = ['game_incomplete', 'game_failed']
    if game['degraded']:
        expected_errors.append('game_degraded')
    expected_errors.extend(k for k in base.BAD_COUNTS if counts[k])
    if counts['requests'] == 0 or counts['requests'] != counts['replies_sent'] or counts['requests'] != counts['ack_accepted']:
        expected_errors.append('action_receipts_mismatch')
    expected_errors.append('transport_incomplete')
    if record is not None:
        base.require(record.get('game_number') == local_number and record.get('directory') == directory.name
                     and record.get('status') == 'interrupted' and record.get('healthy') is False
                     and record.get('errors') == expected_errors and record.get('seat') == seat
                     and record.get('scores') is None and record.get('degraded') is game['degraded']
                     and base.normalized_counts(record.get('counts')) == counts
                     and base.finite(record.get('elapsed_seconds')), 'interrupted_parent_record_mismatch')
        base.same_numbers(record.get('latency'), _latency_index(local), 'interrupted_parent_latency_mismatch')
    return {'status': 'interrupted', 'healthy': False, 'failure': game['failure'], 'seat': seat,
        'scores': None, 'requests': counts['requests'], 'counts': counts,
        'degraded': game['degraded'], 'start_game_observed': started, 'end_game_observed': False,
        'event_lines': len(events), 'events_sha256': base.digest(raw),
        'server_outcome': 'unknown', 'rating_impact': 'unknown', 'observed_errors': expected_errors,
        'local_decision_ms': base.stats(local), 'server_ack_elapsed_ms': base.stats(server),
        'bank_consumed_ms': bank, 'integrity_errors': []}, local, server


def _attempt(directory, number, identity, reader, record=None):
    row = {'directory': str(directory), 'local_attempt': number, 'healthy': False,
           'status': 'invalid', 'integrity_errors': []}
    try:
        base.require(directory.is_dir() and not directory.is_symlink(), 'attempt_directory_invalid')
        manifest = reader.object(directory / 'manifest.json')
        if manifest.get('status') == 'running':
            row['status'] = 'pending'
            return row, [], []
        base.require(set(_inventory(directory)) == {'manifest.json', 'games.jsonl', 'events.jsonl', 'report.json'},
                     'unexpected_attempt_file_inventory')
        if manifest.get('status') == 'complete':
            checked, local, server = base.audit_game(directory, 1 if record is None else number + 1,
                                                    identity, reader, record)
            checked.pop('game_number')
            checked['integrity_errors'] = checked.pop('errors')
            row.update(checked)
            row['healthy'] = row['status'] == 'complete'
            row['requests'] = row.get('counts', {}).get('requests', 0)
            row.update(start_game_observed=row['healthy'], end_game_observed=row['healthy'])
            return row, local, server
        checked, local, server = _interrupted(directory, number, identity, reader, record)
        row.update(checked)
        return row, local, server
    except (base.AuditError, OSError, KeyError, TypeError, IndexError) as exc:
        row['integrity_errors'].append(str(exc) if isinstance(exc, base.AuditError) else 'malformed_attempt')
        return row, [], []


def _diagnostic(root, campaign_status, reader, wrapper_sha, launcher_sha, start_observed):
    directory = root / 'diagnostic'
    base.require(directory.is_dir() and not directory.is_symlink(), 'diagnostic_directory_missing')
    base.require(set(_inventory(directory)) == {'manifest.json', 'events.jsonl', 'result.json'},
                 'unexpected_diagnostic_file_inventory')
    manifest = reader.object(directory / 'manifest.json')
    base.require(manifest.get('schema_version') == 1 and manifest.get('mode') == 'ranked'
                 and manifest.get('games_requested') == 1 and manifest.get('wrapper_sha256') == wrapper_sha
                 and manifest.get('supervisor_sha256') == launcher_sha
                 and manifest.get('runtime_variant') == 'parent_transport_observer'
                 and manifest.get('same_runtime_as_uninstrumented') is False
                 and manifest.get('automatic_reconnect') is False and manifest.get('transport_config') == CONFIG
                 and manifest.get('websockets_version') == '15.0.1'
                 and manifest.get('python_version') == [3, 12, 16]
                 and manifest.get('proxy_policy') == 'environment' and manifest.get('network_path') == 'proxy',
                 'diagnostic_runtime_identity_mismatch')
    result = reader.object(directory / 'result.json')
    events = base.jsonl(reader.read(directory / 'events.jsonl'))
    base.require(result.get('status') == 'finalized' and result.get('error') is None
                 and result.get('campaign_status') == campaign_status
                 and result.get('events_recorded') == len(events)
                 and base.exact_int(result.get('connection_attempts'), 1)
                 and type(result.get('connections_created')) is int and result['connections_created'] in (0, 1),
                 'diagnostic_result_mismatch')
    prior = -1
    for event in events:
        base.require(type(event) is dict and not set(event) - DIAGNOSTIC_KEYS
                     and event.get('event') in DIAGNOSTIC_EVENTS and base.finite(event.get('elapsed_seconds'))
                     and event['elapsed_seconds'] >= prior, 'diagnostic_event_schema')
        prior = event['elapsed_seconds']
        for key in ('exception_category', 'cause_category', 'receive_exception_category'):
            base.require(event.get(key) in SAFE_CATEGORIES, 'diagnostic_exception_schema')
        for key in ('connection_id', 'attempt'):
            base.require(key not in event or base.exact_int(event[key], 1), 'diagnostic_connection_id_invalid')
        base.require('recv_count' not in event or base.integer(event['recv_count']), 'diagnostic_recv_count_invalid')
        for key in ('local_keepalive_timeout_observed', 'abnormal_no_close_frames'):
            base.require(key not in event or type(event[key]) is bool, 'diagnostic_boolean_invalid')
    counts = Counter(event['event'] for event in events)
    base.require(counts['diagnostic_ready'] == counts['connection_attempt'] == 1
                 and counts['connection_created'] == result['connections_created']
                 and counts['handshake_complete'] + counts['handshake_failed'] == 1
                 and (counts['handshake_complete'] == 0 or result['connections_created'] == 1)
                 and (not start_observed or counts['handshake_complete'] == 1),
                 'diagnostic_connection_accounting')
    return {'runtime_variant': 'parent_transport_observer', 'wrapper_sha256': wrapper_sha,
        'same_runtime_as_uninstrumented': False, 'event_counts': dict(sorted(counts.items())),
        'abnormal_no_close_frames_observations': sum(event.get('abnormal_no_close_frames') is True for event in events),
        'local_keepalive_timeout_observations': sum(event.get('local_keepalive_timeout_observed') is True for event in events),
        'closure_flags_change_completed_game_status': False,
        'closure_interpretation': 'Transport shutdown observations are retained separately; verified end_game and game records determine completion.'}


def _rollup(manifest, summary, records, rows):
    base.require(len(records) == len(rows) and base.exact_int(manifest.get('games_recorded'), len(records))
                 and base.exact_int(summary.get('games_attempted'), len(records)), 'batch_attempt_count_mismatch')
    complete = sum(row['status'] == 'complete' for row in rows)
    base.require(base.exact_int(summary.get('games_completed'), complete)
                 and base.exact_int(summary.get('healthy_games'), complete), 'batch_completion_count_mismatch')
    counts = {key: sum(row['counts'][key] for row in rows) for key in base.COUNT_FIELDS}
    base.require(base.normalized_counts(summary.get('counts')) == counts, 'batch_counts_mismatch')
    count = sum(row['local_decision_ms']['count'] for row in rows)
    base.same_numbers(summary.get('latency'), {'count': count,
        'mean_ms': sum(row['local_decision_ms']['count'] * row['local_decision_ms']['mean_ms'] for row in rows if row['local_decision_ms']['count']) / count if count else None,
        'max_ms': max((row['local_decision_ms']['max_ms'] for row in rows if row['local_decision_ms']['count']), default=None)},
        'batch_latency_mismatch')
    base.require(base.finite(summary.get('elapsed_seconds')), 'batch_elapsed_invalid')
    status = manifest.get('status')
    base.require(status == summary.get('status'), 'batch_terminal_status_mismatch')
    if status == 'complete':
        base.require(complete == len(rows) == manifest['games_requested']
                     and manifest.get('current_game') is summary.get('current_game') is None
                     and manifest.get('error') is summary.get('error') is None, 'batch_complete_state_mismatch')
    else:
        base.require(status == 'stopped' and rows and rows[-1]['status'] == 'interrupted'
                     and all(row['status'] == 'complete' for row in rows[:-1])
                     and manifest.get('current_game') == summary.get('current_game') == len(rows)
                     and manifest.get('error') == summary.get('error') == rows[-1]['observed_errors'][0],
                     'batch_stopped_state_mismatch')


def _batch(entry, runs_root, identity, reader, wrapper_sha, launcher_sha):
    root = runs_root / entry['run']
    result = {'batch_id': entry['id'], 'run': entry['run'], 'runtime_variant': entry['runtime_variant'],
              'status': 'planned', 'requested_games': entry['requested_games'], 'integrity_errors': [],
              'pending_reasons': [], 'successful': False, 'attempts': 0,
              'games_completed': 0, 'interrupted_attempts': 0}
    rows, local, server = [], [], []
    if not root.exists() and not root.is_symlink():
        if 'pins' in entry:
            result['integrity_errors'].append('sealed_batch_missing')
            result['status'] = 'invalid'
        return result, rows, local, server
    try:
        base.require(root.is_dir() and not root.is_symlink(), 'run_directory_missing_or_symlink')
        if 'pins' in entry:
            _pins(root, entry['pins'], reader)
        if entry['kind'] == 'standalone':
            row, loc, srv = _attempt(root, 1, identity, reader)
            rows.append(row); local.extend(loc); server.extend(srv)
            result.update(status=row['status'], successful=row['status'] == 'complete')
        else:
            # A live writer can have claimed the directory before its first
            # atomic metadata replacement. Never read unfinished child events.
            if not all((root / name).is_file() for name in ('manifest.json', 'summary.json', 'games.jsonl')):
                base.require('pins' not in entry, 'sealed_parent_file_missing')
                result.update(status='running', pending_reasons=['parent_initialization_pending'])
                return result, rows, local, server
            manifest, summary = reader.object(root / 'manifest.json'), reader.object(root / 'summary.json')
            _manifest_identity(manifest, identity)
            base.require(manifest.get('games_requested') == summary.get('games_requested') == entry['requested_games']
                         and manifest.get('launcher_sha256') == launcher_sha
                         and manifest.get('reserve_ms') == 100 and manifest.get('idle_timeout') == 1800,
                         'batch_configuration_mismatch')
            running = 'running' in (manifest.get('status'), summary.get('status'))
            started = _timestamp(manifest.get('started_at'))
            if not running:
                base.require(_timestamp(manifest.get('updated_at')) >= started, 'batch_end_precedes_start')
            try:
                records = base.jsonl(reader.read(root / 'games.jsonl'))
            except base.AuditError:
                if not running:
                    raise
                records = []
                result['pending_reasons'].append('parent_index_inflight')
            base.require(len(records) <= entry['requested_games'], 'too_many_attempts_in_batch')
            for number, record in enumerate(records, 1):
                row, loc, srv = _attempt(root / f'game-{number:06d}', number, identity, reader, record)
                rows.append(row); local.extend(loc); server.extend(srv)
            game_dirs = sorted(path.name for path in root.iterdir() if path.name.startswith('game-'))
            expected_dirs = [f'game-{number:06d}' for number in range(1, len(records) + 1)]
            if running:
                result.update(status='running')
                result['pending_reasons'].append('batch_running')
                if len(game_dirs) > len(records) or summary.get('games_attempted', 0) > len(records):
                    rows.append({'directory': str(root / f'game-{len(records) + 1:06d}'),
                        'local_attempt': len(records) + 1, 'status': 'pending', 'healthy': False,
                        'integrity_errors': [], 'pending_reason': 'awaiting_terminal_parent_index'})
            else:
                base.require(game_dirs == expected_dirs, 'unindexed_or_missing_attempt_directory')
                base.require(all(row['status'] in ('complete', 'interrupted') for row in rows), 'batch_contains_unverifiable_attempt')
                _rollup(manifest, summary, records, rows)
                result['status'] = manifest['status']
                result['successful'] = manifest['status'] == 'complete'
                if entry['runtime_variant'] == 'parent_transport_observer':
                    if not (root / 'diagnostic/result.json').is_file():
                        base.require('pins' not in entry, 'sealed_diagnostic_result_missing')
                        result.update(status='running', successful=False)
                        result['pending_reasons'].append('diagnostic_finalization_pending')
                    else:
                        try:
                            result['diagnostic'] = _diagnostic(root, manifest['status'], reader, wrapper_sha,
                                launcher_sha, any(row.get('start_game_observed') for row in rows))
                        except base.AuditError as exc:
                            if 'pins' not in entry and str(exc) in ('invalid_json', 'unterminated_jsonl'):
                                result.update(status='running', successful=False)
                                result['pending_reasons'].append('diagnostic_finalization_inflight')
                            else:
                                raise
                allowed_dirs = set(expected_dirs) | ({'diagnostic'} if entry['runtime_variant'] == 'parent_transport_observer' else set())
                base.require({path.name for path in root.iterdir() if path.is_dir() or path.is_symlink()} == allowed_dirs,
                             'unregistered_batch_subdirectory')
                allowed_files = {'manifest.json', 'summary.json', 'games.jsonl'}
                base.require({path.name for path in root.iterdir() if path.is_file()} == allowed_files, 'unregistered_parent_file')
            result['started_at'] = manifest.get('started_at')
            result['updated_at'] = manifest.get('updated_at')
        if 'pins' in entry:
            base.require(result['status'] == entry['terminal_status'], 'sealed_terminal_status_changed')
            _pins(root, entry['pins'], reader)
    except (base.AuditError, OSError, KeyError, TypeError, IndexError) as exc:
        result['status'] = 'invalid'
        result['integrity_errors'].append(str(exc) if isinstance(exc, base.AuditError) else 'malformed_batch')
    for row in rows:
        row.update(batch_id=entry['id'], runtime_variant=entry['runtime_variant'])
    result.update(attempts=len(rows), games_completed=sum(row['status'] == 'complete' for row in rows),
                  interrupted_attempts=sum(row['status'] == 'interrupted' for row in rows))
    return result, rows, local, server


def _group(rows):
    complete = [row for row in rows if row['status'] == 'complete']
    interrupted = [row for row in rows if row['status'] == 'interrupted']
    def counts(selected):
        return {key: sum(row['counts'][key] for row in selected) for key in base.COUNT_FIELDS}
    return {'attempts': len(rows), 'games_completed': len(complete), 'healthy_completed': len(complete),
        'interrupted_attempts': len(interrupted), 'invalid_attempts': sum(row['status'] == 'invalid' for row in rows),
        'pending_attempts': sum(row['status'] == 'pending' for row in rows),
        'completed_counts': counts(complete), 'interrupted_counts': counts(interrupted),
        'all_observed_counts': counts(complete + interrupted)}


def _identity_sources(paths, wrapper, inputs):
    sources = [Path(__file__).resolve(), LEGACY_PATH, wrapper, paths.launcher, paths.engine_binary,
               paths.model_dir / 'ppo-agent.json', paths.model_dir / 'ppo.pt']
    sources += sorted(paths.kit_src.rglob('*.py')) + sorted(paths.research_src.rglob('*.py'))
    for path in sources:
        base.require(path.is_file() and not path.is_symlink(), 'identity_dependency_missing_or_symlink')
        data = path.read_bytes()
        inputs.files[str(path.resolve())] = {'path': str(path.resolve()), 'bytes': len(data), 'sha256': base.digest(data)}


def audit_ranked_series(registry_path=DEFAULT_REGISTRY, *, paths=None, expected=None,
                         wrapper_path=None, root_registry_sha256=None):
    """Audit all registered sources. Keyword overrides are synthetic-test only."""
    registry_path = Path(registry_path).resolve()
    raw, inputs = base.Reader(), base.Reader()
    errors, pending, batches, attempts = [], [], [], []
    completed_local, completed_server, failed_local, failed_server = [], [], [], []
    anchor = ROOT_REGISTRY_SHA256 if root_registry_sha256 is None else root_registry_sha256
    result = {'schema_version': 1, 'status': 'pending', 'final_acceptance': False,
        'target_complete_games': 100, 'completion_target_met': False,
        'integrity_status': 'provisional', 'integrity_errors': errors, 'pending_reasons': pending,
        'original_zero_failure_criterion_met': False, 'outcome': 'in_progress',
        'auditor_sha256': base.digest(Path(__file__).read_bytes()), 'legacy_auditor_sha256': LEGACY_SHA256,
        'registry': {}, 'registry_chain': [], 'plans': [], 'identity_checks': {}, 'batches': batches, 'attempts': attempts, 'failures': [],
        'totals': {}, 'runtime_variants': {}, 'raw_files': [], 'source_files': [],
        'statistics_method': {'quantiles': 'Pooled actions with linear interpolation at (n - 1) * p.',
            'completed_statistics': 'Only independently verified complete, healthy games; interruptions never receive invented scores or ranks.',
            'all_observed_statistics': 'Includes interrupted-attempt actions and is reported separately.',
            'local_decision_ms': 'Client receipt to response preparation; excludes model loading, send and network round trip.',
            'server_ack_elapsed_ms': 'Server-reported request elapsed including network round trip; not pure inference or client ack RTT.',
            'rank': 'Score-based rank_min/rank_max; retain tie intervals.', 'point_delta': 'Final score minus 25000, excluding rating/uma/oka.'},
        'limitations': ['Integrity, completion target and zero-failure reliability are separate conclusions.',
            'Local interruptions retain unknown final server outcome/rating even when observed requests were accepted.',
            'Filtered-message diagnostic counters are reconciled from persisted counts, not omitted raw messages.',
            'Observer transport is a distinct runtime variant despite identical model and core package sources.',
            'Engine identity refers to the local native module; these records do not establish the server engine release timeline.',
            'Shutdown close-frame diagnostics do not override a verified end_game.',
            'Compact local logs are not full MJAI replays or signed server results.']}
    running_seen = False
    try:
        registry = _read_registry(registry_path, inputs, anchor, chain=result['registry_chain'])
        result['registry'] = {'path': str(registry_path), 'sha256': inputs.files[str(registry_path)]['sha256'],
                              'revision': registry['revision'], 'root_sha256': anchor}
        paths = paths or base.IdentityPaths.defaults()
        expected = base.EXPECTED if expected is None else expected
        base.require(registry['expected_identity'] == expected and registry['legacy_auditor_sha256'] == LEGACY_SHA256,
                     'registry_core_identity_mismatch')
        actual, identity = base.identities(paths, expected)
        result['identity_checks'] = {'actual': actual, 'expected': dict(expected), 'matches': True}
        wrapper = Path(wrapper_path or registry['diagnostic_wrapper']['path']).resolve()
        wrapper_sha = base.digest(wrapper.read_bytes())
        base.require(wrapper_sha == registry['diagnostic_wrapper']['sha256']
                     and (root_registry_sha256 is not None or wrapper_sha == WRAPPER_SHA256), 'wrapper_identity_mismatch')
        runs_root = Path(registry['runs_root']).resolve()
        entries = list(registry['history_batches'])
        plans = [_plan(entry, inputs, expected, wrapper_sha) for entry in registry['plans']]
        registered_names = {entry['run'] for entry in entries}
        base.require(len(registered_names) == len(entries), 'duplicate_historical_run')
        plan_groups = []
        for number, plan in enumerate(plans, 1):
            names = plan['planned_runs']
            base.require(not registered_names.intersection(names), 'run_registered_twice')
            registered_names.update(names)
            planned = [{'id': f'plan-{number:03d}-{index:06d}', 'run': name, 'kind': 'campaign',
                        'runtime_variant': 'parent_transport_observer', 'requested_games': 1}
                       for index, name in enumerate(names, 1)]
            plan_groups.append((plan, planned)); entries.extend(planned)
        actual_names = {path.name for path in runs_root.glob('riichi-ranked-*') if path.is_dir() or path.is_symlink()}
        base.require(actual_names <= registered_names, 'unregistered_ranked_run')
        row_number = 0
        def collect(entry):
            nonlocal row_number, running_seen
            batch, rows, _, _ = _batch(entry, runs_root, identity, raw, wrapper_sha, actual['launcher'])
            batches.append(batch)
            running_seen |= batch['status'] == 'running'
            errors.extend(f'{entry["id"]}:{code}' for code in batch['integrity_errors'])
            pending.extend(f'{entry["id"]}:{code}' for code in batch['pending_reasons'])
            for row in rows:
                row_number += 1
                row['global_attempt'] = row_number
                attempts.append(row)
                errors.extend(f'attempt_{row_number}:{code}' for code in row['integrity_errors'])
                if row['status'] in ('complete', 'interrupted'):
                    # Reread only terminal attempt records already verified by
                    # the helper; never read an active child's events for stats.
                    event_bytes = raw.read(Path(row['directory']) / 'events.jsonl')
                    base.require(base.digest(event_bytes) == row['events_sha256'], 'terminal_events_changed_during_audit')
                    events = base.jsonl(event_bytes)
                    local = [e['latency_ms'] for e in events if e['event'] == 'decision']
                    server = [e['elapsed_ms'] for e in events if e['event'] == 'action_ack']
                    (completed_local if row['status'] == 'complete' else failed_local).extend(local)
                    (completed_server if row['status'] == 'complete' else failed_server).extend(server)
            return batch
        for entry in registry['history_batches']:
            collect(entry)
        previous_plan_state, previous_plan_end = None, None
        for plan_number, (plan, group) in enumerate(plan_groups, 1):
            any_started = any((runs_root / entry['run']).exists() for entry in group)
            if previous_plan_state is not None and previous_plan_state not in ('complete', 'stopped'):
                if any_started:
                    if running_seen and not raw.unchanged():
                        pending.append('cross_plan_advanced_during_probe')
                        break
                    raise base.AuditError('new_plan_started_before_prior_plan_terminal')
                pending.append('new_plan_waiting_for_prior_plan_terminal')
            before = _group(attempts)
            base.require(before['games_completed'] == plan['complete_games_before_execution']
                         and before['interrupted_attempts'] == plan['prior_interrupted_attempts'], 'plan_baseline_mismatch')
            blocked, blocked_status, blocked_root = False, None, None
            previous_end = previous_plan_end
            group_batches = []
            for entry in group:
                exists = (runs_root / entry['run']).exists()
                if blocked and exists:
                    if (blocked_status == 'running' and not raw.unchanged()) or (blocked_status == 'planned' and blocked_root.exists()):
                        pending.append('plan_advanced_during_probe')
                        break
                    raise base.AuditError('plan_started_after_gap_active_or_failed_run')
                batch = collect(entry)
                group_batches.append(batch)
                if exists and batch.get('started_at'):
                    started = _timestamp(batch['started_at'])
                    base.require(started >= _timestamp(plan['created_at']), 'run_precedes_plan')
                    if previous_end is not None:
                        base.require(started >= previous_end, 'plan_runs_overlap')
                if batch.get('updated_at') and batch['successful']:
                    previous_end = _timestamp(batch['updated_at'])
                if not blocked and not batch['successful']:
                    blocked, blocked_status, blocked_root = True, batch['status'], runs_root / entry['run']
            created = [batch for batch in group_batches if batch['status'] != 'planned']
            if any(batch['status'] == 'running' for batch in created):
                state = 'running'
            elif any(batch['status'] == 'invalid' for batch in created):
                state = 'invalid'
            elif created and created[-1]['status'] == 'stopped':
                state = 'stopped'
            elif len(created) == len(group) and all(batch['successful'] for batch in created):
                state = 'complete'
            else:
                state = 'pending' if created else 'planned'
            terminal_at = created[-1].get('updated_at') if created and state in ('complete', 'stopped') else None
            result['plans'].append({'plan_number': plan_number, 'status': state, 'planned_runs': len(group),
                'created_runs': len(created), 'terminal_at': terminal_at})
            previous_plan_state = state
            previous_plan_end = _timestamp(terminal_at) if terminal_at else None
        for entry in registry['history_batches']:
            _pins(runs_root / entry['run'], entry['pins'], raw)
        base.require(base.identities(paths, expected)[0] == actual, 'identity_changed_during_audit')
        _identity_sources(paths, wrapper, inputs)
        base.require(inputs.files[str(LEGACY_PATH)]['sha256'] == LEGACY_SHA256
                     and inputs.files[str(wrapper)]['sha256'] == wrapper_sha, 'audit_dependency_changed')
        if not raw.unchanged():
            if running_seen:
                pending.append('source_changed_during_probe')
            else:
                errors.append('source_changed_during_terminal_audit')
        base.require(inputs.unchanged(), 'registry_plan_or_identity_source_changed')
    except (base.AuditError, OSError, KeyError, TypeError, IndexError, ValueError) as exc:
        errors.append(str(exc) if isinstance(exc, base.AuditError) else 'unreadable_or_malformed_series')
    complete = [row for row in attempts if row['status'] == 'complete']
    interrupted = [row for row in attempts if row['status'] == 'interrupted']
    if len({row['events_sha256'] for row in complete}) != len(complete):
        errors.append('duplicate_complete_game_evidence')
    totals = _group(attempts)
    totals['planned_batches'] = sum(batch['status'] == 'planned' for batch in batches)
    totals['all_observed_counts_scope'] = 'Verified terminal attempts only; active event streams are deliberately not consumed.'
    ranks = Counter(row['rank_min'] for row in complete if row['rank_min'] == row['rank_max'])
    totals.update(score_sum=sum(row['score'] for row in complete),
        mean_score=statistics.fmean(row['score'] for row in complete) if complete else None,
        mean_point_delta=statistics.fmean(row['point_delta'] for row in complete) if complete else None,
        mean_rank_min=statistics.fmean(row['rank_min'] for row in complete) if complete else None,
        mean_rank_max=statistics.fmean(row['rank_max'] for row in complete) if complete else None,
        untied_rank_counts={str(rank): ranks[rank] for rank in range(1, 5)},
        tied_games=sum(row['rank_min'] != row['rank_max'] for row in complete),
        completed_local_decision_ms=base.stats(completed_local), completed_server_ack_elapsed_ms=base.stats(completed_server),
        interrupted_local_decision_ms=base.stats(failed_local), interrupted_server_ack_elapsed_ms=base.stats(failed_server),
        all_observed_local_decision_ms=base.stats(completed_local + failed_local),
        all_observed_server_ack_elapsed_ms=base.stats(completed_server + failed_server))
    result['totals'] = totals
    result['failures'] = [{key: row[key] for key in ('global_attempt', 'batch_id', 'local_attempt',
        'runtime_variant', 'failure', 'start_game_observed', 'end_game_observed', 'seat', 'scores',
        'requests', 'server_outcome', 'rating_impact')} for row in interrupted]
    for variant in ('uninstrumented', 'parent_transport_observer'):
        result['runtime_variants'][variant] = dict(_group([row for row in attempts if row['runtime_variant'] == variant]),
            wrapper_sha256=None if variant == 'uninstrumented' else locals().get('wrapper_sha'),
            same_runtime_as_uninstrumented=variant == 'uninstrumented')
    result['raw_files'] = sorted(raw.files.values(), key=lambda item: item['path'])
    result['source_files'] = sorted((inputs.files | raw.files).values(), key=lambda item: item['path'])
    result['completion_target_met'] = totals['games_completed'] >= 100
    if totals['games_completed'] > 100:
        errors.append('target_overrun_all_records_retained')
    result['original_zero_failure_criterion_met'] = not interrupted and not errors and totals['games_completed'] == 100
    if errors:
        result.update(status='invalid', integrity_status='invalid')
    elif pending or totals['pending_attempts']:
        result.update(status='pending', integrity_status='provisional')
    elif totals['games_completed'] != 100:
        result.update(status='pending', integrity_status='verified')
        pending.append('complete_game_target_not_met')
    else:
        result.update(status='ready', integrity_status='verified', outcome='completed_with_interruptions' if interrupted else 'completed_without_interruptions')
    return result


def write_series_audit(result, output, **test_options):
    base.require(result.get('status') == 'ready' and result.get('integrity_status') == 'verified'
                 and result.get('final_acceptance') is False and result.get('completion_target_met') is True
                 and not result.get('integrity_errors') and not result.get('pending_reasons')
                 and result.get('totals', {}).get('games_completed') == 100, 'series_not_ready')
    fresh = audit_ranked_series(result['registry']['path'], **test_options)
    base.require(fresh == result, 'independent_reaudit_changed')
    for item in result['source_files']:
        path = Path(item['path'])
        base.require(path.is_file() and not path.is_symlink() and path.stat().st_size == item['bytes']
                     and base.digest(path.read_bytes()) == item['sha256'], 'source_changed_before_publication')
    output = Path(output)
    output.mkdir(exist_ok=False)
    report = dict(result, status='published', final_acceptance=True,
                  published_at=datetime.now(timezone.utc).isoformat())
    docs = {'report.json': report, 'attempts.json': result['attempts'], 'source-sha256.json': result['source_files']}
    for name, value in docs.items():
        with (output / name).open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    with (output / 'SHA256SUMS').open('x') as stream:
        stream.write(''.join(f'{base.digest((output / name).read_bytes())}  {name}\n' for name in sorted(docs)))
    return report


def append_plan(registry_path, plan_path, output, *, root_registry_sha256=None, **test_options):
    """Only append a prewritten plan; never launch or modify a run."""
    reader = base.Reader()
    registry_path, plan_path, output = map(lambda p: Path(p).resolve(), (registry_path, plan_path, output))
    anchor = root_registry_sha256 or ROOT_REGISTRY_SHA256
    prior = _read_registry(registry_path, reader, anchor)
    data = reader.read(plan_path)
    entry = {'path': str(plan_path), 'bytes': len(data), 'sha256': base.digest(data)}
    plan = _plan(entry, reader, prior['expected_identity'], prior['diagnostic_wrapper']['sha256'])
    audit = audit_ranked_series(registry_path, root_registry_sha256=root_registry_sha256, **test_options)
    base.require(audit['integrity_status'] == 'verified' and not audit['integrity_errors']
                 and audit['totals']['pending_attempts'] == 0 and not audit['completion_target_met']
                 and audit['plans'] and audit['plans'][-1]['status'] in ('complete', 'stopped'),
                 'append_requires_prior_plan_terminal')
    base.require(plan['complete_games_before_execution'] == audit['totals']['games_completed']
                 and plan['prior_interrupted_attempts'] == audit['totals']['interrupted_attempts']
                 and _timestamp(plan['created_at']) >= _timestamp(audit['plans'][-1]['terminal_at']),
                 'append_plan_baseline_or_time_mismatch')
    old_names = {batch['run'] for batch in prior['history_batches']}
    for existing in prior['plans']:
        old_names.update(_plan(existing, reader, prior['expected_identity'], prior['diagnostic_wrapper']['sha256'])['planned_runs'])
    base.require(not old_names.intersection(plan['planned_runs']), 'append_reuses_run_name')
    runs_root = Path(prior['runs_root'])
    base.require(not any((runs_root / name).exists() for name in plan['planned_runs']), 'append_requires_unstarted_runs')
    next_registry = dict(prior, revision=prior['revision'] + 1,
        predecessor={'path': str(registry_path), 'sha256': base.digest(registry_path.read_bytes())},
        plans=prior['plans'] + [entry])
    base.require(reader.unchanged(), 'registry_or_plan_changed_before_append')
    with output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(next_registry, ensure_ascii=False, indent=2) + '\n')
    return {'path': str(output), 'sha256': base.digest(output.read_bytes()), 'revision': next_registry['revision']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--report-dir', type=Path, default=HERE / 'series-audit')
    parser.add_argument('--append-plan', type=Path)
    parser.add_argument('--output-registry', type=Path)
    args = parser.parse_args(argv)
    if args.append_plan:
        base.require(not args.write and args.output_registry is not None, 'append_requires_output_without_write')
        print(json.dumps(append_plan(args.registry, args.append_plan, args.output_registry), ensure_ascii=False))
        return 0
    result = audit_ranked_series(args.registry)
    if args.write:
        try:
            result = write_series_audit(result, args.report_dir)
        except (base.AuditError, OSError) as exc:
            print(json.dumps({'status': result['status'], 'final_acceptance': False,
                'write_error': str(exc) if isinstance(exc, base.AuditError) else 'output_exists_or_unwritable',
                'integrity_errors': result['integrity_errors'], 'pending_reasons': result['pending_reasons']}))
            return 2
    print(json.dumps({key: result[key] for key in ('status', 'final_acceptance', 'completion_target_met',
        'integrity_status', 'integrity_errors', 'pending_reasons', 'original_zero_failure_criterion_met',
        'totals', 'failures', 'runtime_variants')}, ensure_ascii=False, indent=2))
    return 0 if result['status'] in ('ready', 'published') else 1


if __name__ == '__main__':
    raise SystemExit(main())
