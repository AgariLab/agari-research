#!/usr/bin/env python3
"""Independently audit the fixed first-game + remaining-99 ranked campaign.

No network, credentials, environment lookup, model loading, or toolkit imports.
Default: read-only pending/invalid/ready result. --write requires 100 healthy
games and exclusively creates final-audit; it never replaces prior evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
STORAGE = Path('/Volumes/新加卷/AgariLab')
DEFAULT_FIRST = STORAGE / 'runs/riichi-ranked-first-v1'
DEFAULT_CAMPAIGN = STORAGE / 'runs/riichi-ranked-campaign-remaining99-v1'
EXPECTED = {
    'weights': '613d6a20cf1057a9c6556153a5506fdc11d115f62aaf5de83e39143572bd9096',
    'descriptor': '51495822b3af9c6aff6f11a59aa3db0c1380ed4238088b22d4bc2741e052a381',
    'kit': 'e1991979d79a373ca54050786616f57ded571a8d20de326a6e66982244dd1979',
    'research': '43eabb39e1518ec550f8692bce79224060f07aa280f12a254b990ccc2c3e7e21',
    'engine': '169c8c2c53bdace01e4ca319f54050565fe728a5b1b2fc6b7f3046738887071d',
    'launcher': 'cd570e8abc1b5b028661f6dfa7881ebdea7c650db11b7d5a0c7fd53adbe4aede',
    'kit_files': 19, 'research_files': 17,
}
GOOD_COUNTS = ('requests', 'replies_prepared', 'replies_sent', 'ack_accepted')
BAD_COUNTS = (
    'local_timeouts', 'send_timeouts', 'ack_rejected', 'ack_unparseable',
    'ack_defaulted', 'ack_stale', 'unknown_ack_status', 'suppressed_after_timeout',
)
DIAGNOSTIC_COUNTS = (
    'duplicate_start', 'duplicate_requests', 'stale_requests', 'duplicate_acks',
    'unknown_messages',
)
COUNT_FIELDS = BAD_COUNTS + DIAGNOSTIC_COUNTS + GOOD_COUNTS
RULES = {'game_mode': '4p-red-half', 'preset': 'tenhou', 'engine_version': '0.4.10'}
ACTION_TYPES = {'dahai', 'none', 'chi', 'pon', 'daiminkan', 'ankan', 'kakan',
                'reach', 'hora', 'ryukyoku'}
SECRET_PATTERNS = (
    re.compile(rb'eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+'),
    re.compile(rb'-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----'),
)


class AuditError(ValueError):
    """Messages are fixed diagnostic codes, never source payloads."""


@dataclass(frozen=True)
class IdentityPaths:
    model_dir: Path
    kit_src: Path
    research_src: Path
    engine_binary: Path
    launcher: Path

    @classmethod
    def defaults(cls):
        candidates = list((WORKSPACE / 'agari-research/.venv/lib').glob(
            'python*/site-packages/riichienv/_riichienv*.so'))
        require(len(candidates) == 1, 'engine_binary_not_unique')
        return cls(STORAGE / 'models/full-coverage-trace-1000-v1',
                   WORKSPACE / 'agari-kit/src/agari_kit',
                   WORKSPACE / 'agari-research/src/agari_research', candidates[0],
                   WORKSPACE / 'agari-kit/scripts/run_online_campaign.py')


def require(condition, code):
    if not condition:
        raise AuditError(code)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def exact_int(value, expected):
    return type(value) is int and value == expected


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, 'duplicate_json_key')
        result[key] = value
    return result


def decode(data):
    try:
        return json.loads(data, object_pairs_hook=_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(AuditError('nonfinite_json')))
    except (UnicodeError, json.JSONDecodeError):
        raise AuditError('invalid_json') from None


def jsonl(data):
    require(not data or data.endswith(b'\n'), 'unterminated_jsonl')
    return [decode(line) for line in data.splitlines()]


def source_digest(root, prefix=''):
    hasher = hashlib.sha256()
    files = sorted(root.rglob('*.py'))
    require(bool(files), 'empty_source_tree')
    for path in files:
        require(not path.is_symlink() and path.is_file(), 'source_symlink_or_nonfile')
        hasher.update((prefix + path.relative_to(root).as_posix()).encode())
        hasher.update(b'\0'); hasher.update(path.read_bytes()); hasher.update(b'\0')
    return hasher.hexdigest(), len(files)


class Reader:
    def __init__(self):
        self.files = {}

    def read(self, path, *, limit=16 * 1024 * 1024):
        require(not path.is_symlink() and path.is_file(), 'missing_or_symlink_file')
        require(path.stat().st_size <= limit, 'oversize_record')
        data = path.read_bytes()
        require(len(data) <= limit, 'oversize_record')
        require(not any(pattern.search(data) for pattern in SECRET_PATTERNS), 'secret_pattern_detected')
        self.files[str(path)] = {'path': str(path), 'bytes': len(data), 'sha256': digest(data)}
        return data

    def object(self, path):
        value = decode(self.read(path))
        require(type(value) is dict, 'expected_json_object')
        return value

    def unchanged(self):
        return all(not Path(path).is_symlink() and Path(path).is_file()
                   and digest(Path(path).read_bytes()) == info['sha256']
                   for path, info in self.files.items())


def identities(paths, expected):
    descriptor_path = paths.model_dir / 'ppo-agent.json'
    weights_path = paths.model_dir / 'ppo.pt'
    for path in (descriptor_path, weights_path, paths.engine_binary, paths.launcher):
        require(path.is_file() and not path.is_symlink(), 'identity_file_missing_or_symlink')
    descriptor = decode(descriptor_path.read_bytes())
    kit, kit_files = source_digest(paths.kit_src)
    research, research_files = source_digest(paths.research_src, '0/')
    actual = {'weights': digest(weights_path.read_bytes()),
              'descriptor': digest(descriptor_path.read_bytes()), 'kit': kit, 'research': research,
              'engine': digest(paths.engine_binary.read_bytes()),
              'launcher': digest(paths.launcher.read_bytes()),
              'kit_files': kit_files, 'research_files': research_files}
    require(actual == expected, 'actual_identity_differs_from_pinned_release')
    require(descriptor == {
        'artifacts': {'weights_path': {'path': 'ppo.pt', 'sha256': expected['weights']}},
        'factory': 'agari_research.model_agent:ModelAgent',
        'kwargs': {'device': 'cpu', 'torch_threads': 1}}, 'unexpected_descriptor')
    name = '@' + str(descriptor_path)
    provenance = {
        'kind': 'external', 'name': name, 'descriptor_path': str(descriptor_path),
        'descriptor_sha256': actual['descriptor'], 'descriptor': descriptor,
        'factory': 'agari_research.model_agent:ModelAgent',
        'artifacts': {'weights_path': {'path': str(weights_path), 'declared_path': 'ppo.pt',
            'declared_sha256': actual['weights'], 'sha256': actual['weights']}},
        'source': {'package': 'agari_research', 'module': 'agari_research.model_agent',
            'module_path': str(paths.research_src / 'model_agent.py'),
            'roots': [str(paths.research_src)], 'source_sha256': research, 'files': research_files}}
    return actual, {'agent_provenance': provenance,
        'code': {'package': 'agari-kit', 'version': '0.1.0', 'source_sha256': kit},
        'engine': {'version': '0.4.10', 'binary_sha256': actual['engine']}}


def stats(values):
    if not values:
        return {'count': 0, 'min_ms': None, 'mean_ms': None, 'p50_ms': None,
                'p95_ms': None, 'p99_ms': None, 'max_ms': None}
    ordered = sorted(values)
    def quantile(p):
        index = (len(ordered) - 1) * p
        low, high = math.floor(index), math.ceil(index)
        return ordered[low] + (ordered[high] - ordered[low]) * (index - low)
    return {'count': len(values), 'min_ms': ordered[0], 'mean_ms': statistics.fmean(values),
            'p50_ms': quantile(.5), 'p95_ms': quantile(.95), 'p99_ms': quantile(.99),
            'max_ms': ordered[-1]}


def normalized_counts(value):
    require(type(value) is dict and not set(value) - set(COUNT_FIELDS), 'unknown_count_field')
    require(all(integer(v) for v in value.values()), 'invalid_count_value')
    return {key: value.get(key, 0) for key in COUNT_FIELDS}


def same_numbers(actual, expected, code):
    require(type(actual) is dict and set(actual) == set(expected), code)
    for key, value in expected.items():
        other = actual[key]
        if value is None:
            require(other is None, code)
        elif type(value) is int:
            require(type(other) is int and other == value, code)
        else:
            require(finite(other) and math.isclose(other, value, rel_tol=1e-12, abs_tol=1e-9), code)


def audit_game(directory, number, identity, reader, expected_record=None):
    """Never read events for a manifest that is not complete."""
    row = {'game_number': number, 'directory': str(directory), 'status': 'invalid', 'errors': []}
    if type(expected_record) is dict:
        # Keep a safe projection of every indexed attempt, including failures;
        # never copy arbitrary server/exception text into a derived report.
        row['campaign_record'] = {
            'recorded_complete': expected_record.get('status') == 'complete',
            'recorded_healthy': expected_record.get('healthy') is True,
            'recorded_error_count': len(expected_record['errors']) if type(expected_record.get('errors')) is list else None,
        }
    local, server = [], []
    try:
        manifest = reader.object(directory / 'manifest.json')
        if manifest.get('status') == 'running':
            row['status'] = 'pending'
            return row, local, server
        require(manifest.get('status') == 'complete', 'transport_not_complete')
        require(exact_int(manifest.get('schema_version'), 1) and manifest.get('mode') == 'ranked'
                and manifest.get('endpoint') == 'wss://game.riichi.dev/ws/ranked', 'transport_mode')
        require(exact_int(manifest.get('games_requested'), 1) and exact_int(manifest.get('games_recorded'), 1)
                and exact_int(manifest.get('reserve_ms'), 100) and manifest.get('rules_expected') == RULES,
                'transport_configuration')
        require(all(manifest.get(key) == value for key, value in identity.items())
                and manifest.get('agent') == identity['agent_provenance']['name'], 'game_identity_mismatch')
        records = jsonl(reader.read(directory / 'games.jsonl'))
        require(len(records) == 1 and type(records[0]) is dict, 'single_game_record_required')
        game = records[0]
        report = reader.object(directory / 'report.json')
        row['recorded_failure'] = bool(game.get('failure'))
        if type(game.get('seat')) is int and game['seat'] in range(4):
            row['recorded_seat'] = game['seat']
        if type(game.get('scores')) is list and len(game['scores']) == 4 and all(type(s) is int for s in game['scores']):
            row['recorded_scores'] = game['scores']
        require(game.get('status') == 'complete' and game.get('mode') == 'ranked'
                and exact_int(game.get('game_index'), 0) and not game.get('failure'), 'game_incomplete_or_failed')
        require(game.get('degraded') is False and 'validation_passed' in game
                and game['validation_passed'] is None, 'game_degraded_or_validation')
        seat, scores = game.get('seat'), game.get('scores')
        require(type(seat) is int and seat in range(4), 'invalid_seat')
        require(type(scores) is list and len(scores) == 4 and all(type(s) is int for s in scores), 'invalid_scores')
        counts = normalized_counts(game.get('counts'))
        require(all(counts[key] == 0 for key in BAD_COUNTS), 'nonzero_fatal_count')
        require(counts['requests'] > 0 and len({counts[key] for key in GOOD_COUNTS}) == 1, 'receipt_count_mismatch')
        require(exact_int(report.get('games_requested'), 1) and exact_int(report.get('games_completed'), 1)
                and exact_int(report.get('degraded_games'), 0) and 'validation_passed' in report
                and report['validation_passed'] is None,
                'game_report_result_mismatch')
        require(normalized_counts(report.get('counts')) == counts, 'game_report_counts_mismatch')
        event_bytes = reader.read(directory / 'events.jsonl')
        events = jsonl(event_bytes)
        require(len(events) == 2 * counts['requests'] + 2, 'event_count_mismatch')
        require(events[0] == {'event': 'start_game', 'seat': seat, 'game_index': 0}, 'start_game_mismatch')
        require(events[-1] == {'event': 'end_game', 'scores': scores, 'game_index': 0}, 'end_game_missing_or_mismatch')
        require(exact_int(events[0]['seat'], seat) and exact_int(events[0]['game_index'], 0)
                and exact_int(events[-1]['game_index'], 0)
                and all(type(s) is int for s in events[-1]['scores']), 'invalid_boundary_event_types')
        decisions, acknowledgements, actions = {}, set(), Counter()
        bank_consumed = 0
        for event in events[1:-1]:
            require(type(event) is dict and exact_int(event.get('game_index'), 0), 'invalid_event')
            request_id = event.get('request_id')
            require(integer(request_id), 'invalid_request_id')
            if event.get('event') == 'decision':
                require(set(event) == {'event', 'request_id', 'latency_ms', 'action_type', 'game_index'}, 'unexpected_decision_fields')
                require(request_id not in decisions and (not decisions or request_id > max(decisions)), 'duplicate_or_stale_decision')
                require(finite(event.get('latency_ms')) and event.get('action_type') in ACTION_TYPES, 'invalid_decision')
                decisions[request_id] = event
                actions[event['action_type']] += 1
                local.append(event['latency_ms'])
            elif event.get('event') == 'action_ack':
                require(set(event) == {'event', 'request_id', 'status', 'elapsed_ms', 'bank_consumed_ms', 'bank_ms', 'game_index'}, 'unexpected_ack_fields')
                require(request_id in decisions and request_id not in acknowledgements, 'duplicate_or_unmatched_ack')
                require(event.get('status') == 'accepted', 'nonaccepted_ack')
                require(all(finite(event.get(key)) for key in ('elapsed_ms', 'bank_consumed_ms', 'bank_ms')), 'invalid_ack_timing')
                acknowledgements.add(request_id)
                server.append(event['elapsed_ms'])
                bank_consumed += event['bank_consumed_ms']
            else:
                raise AuditError('unexpected_event_type')
        require(set(decisions) == acknowledgements and len(decisions) == counts['requests'], 'event_receipts_mismatch')
        row.update(status='complete', seat=seat, scores=scores, score=scores[seat],
            point_delta=scores[seat] - 25000,
            rank_min=1 + sum(s > scores[seat] for s in scores),
            rank_max=sum(s >= scores[seat] for s in scores),
            degraded=False, counts=counts,
            diagnostic_counts={key: counts[key] for key in DIAGNOSTIC_COUNTS},
            event_lines=len(events), events_sha256=digest(event_bytes),
            action_counts=dict(sorted(actions.items())), local_decision_ms=stats(local),
            server_ack_elapsed_ms=stats(server), bank_consumed_ms=bank_consumed)
        # Campaign supervisor uses nearest-rank p95. Reproduce that only to
        # check its index; our final statistics always use linear interpolation.
        if expected_record is not None:
            require(type(expected_record) is dict, 'invalid_campaign_game_record')
            require(exact_int(expected_record.get('game_number'), number - 1)
                    and expected_record.get('directory') == directory.name, 'campaign_game_number_mismatch')
            for key, value in (('status', 'complete'), ('healthy', True), ('errors', []),
                               ('seat', seat), ('scores', scores), ('degraded', False)):
                require(type(expected_record.get(key)) is type(value) and expected_record[key] == value,
                        'campaign_game_result_mismatch')
            require(normalized_counts(expected_record.get('counts')) == counts, 'campaign_game_counts_mismatch')
            require(finite(expected_record.get('elapsed_seconds')), 'invalid_campaign_game_elapsed')
            ordered = sorted(local)
            same_numbers(expected_record.get('latency'), {'count': len(local),
                'mean_ms': statistics.fmean(local), 'p50_ms': statistics.median(local),
                'p95_ms': ordered[math.ceil(len(local) * .95) - 1], 'max_ms': max(local)},
                'campaign_game_latency_mismatch')
    except (AuditError, OSError, KeyError, TypeError, IndexError) as exc:
        row['status'] = 'invalid'
        row['errors'].append(str(exc) if isinstance(exc, AuditError) else 'unreadable_or_malformed_game')
    return row, local, server


def audit_ranked(first_game=DEFAULT_FIRST, campaign=DEFAULT_CAMPAIGN, *, paths=None, expected=None):
    """Pure read-only audit. Overrides exist for synthetic tests, not the CLI."""
    first_game, campaign = Path(first_game).resolve(), Path(campaign).resolve()
    reader, errors, pending, rows = Reader(), [], [], []
    local_all, server_all = [], []
    result = {'schema_version': 1, 'status': 'pending', 'final_acceptance': False,
              'auditor_sha256': digest(Path(__file__).read_bytes()), 'target_games': 100,
              'inputs': {'first_game': str(first_game), 'campaign': str(campaign)},
              'errors': errors, 'pending_reasons': pending, 'games': rows,
              'identity_checks': {}, 'raw_files': [], 'totals': {},
              'statistics_method': {
                'quantiles': 'linear interpolation at sorted position (n - 1) * p; pooled individual actions',
                'rank': 'rank_min/rank_max from final scores; ties remain intervals without assuming server tie-break',
                'point_delta': 'final seat score minus 25000; excludes rating/uma/oka',
                'local_decision_ms': 'Client socket.recv receipt timestamp to response preparation: queueing, parsing, IPC/inference and legality checks; excludes loading, socket.send and network round trip.',
                'server_ack_elapsed_ms': 'Server-reported request elapsed time, including request/response network round trip; not pure inference or client-measured ack RTT.',
                'supervisor_p95': 'Nearest-rank only for checking the historical campaign index; never used as final pooled p95.'},
              'limitations': ['Compact local transport records, not full MJAI replays or independently signed server records.',
                              'Duplicate/stale request notifications and unknown messages are filtered before event emission; their nonfatal counters are reconciled across game/report/campaign records and summed, not reconstructed from omitted messages.',
                              'Online matchmaking sample; descriptive performance is not a fixed-opponent playing-strength acceptance.']}
    try:
        paths = paths or IdentityPaths.defaults()
        expected = EXPECTED if expected is None else expected
        actual, identity = identities(paths, expected)
        result['identity_checks'] = {'actual': actual, 'expected': dict(expected), 'matches': True}
        row, local, server = audit_game(first_game, 1, identity, reader)
        rows.append(row)
        if row['status'] == 'complete':
            local_all.extend(local); server_all.extend(server)
        top = reader.object(campaign / 'manifest.json')
        summary = reader.object(campaign / 'summary.json')
        terminal = top.get('status') == summary.get('status') == 'complete'
        running = top.get('status') == 'running' or summary.get('status') == 'running'
        require(exact_int(top.get('schema_version'), 1) and exact_int(summary.get('schema_version'), 1)
                and top.get('mode') == 'ranked' and top.get('games_requested') == summary.get('games_requested') == 99,
                'campaign_configuration')
        require(top.get('reserve_ms') == 100 and top.get('idle_timeout') == 1800
                and top.get('agent') == identity['agent_provenance']['name'], 'campaign_configuration')
        require(all(top.get(key) == value for key, value in identity.items())
                and top.get('launcher_sha256') == actual['launcher'], 'campaign_identity_mismatch')
        if running:
            pending.append('campaign_running')
        elif not terminal:
            errors.append('campaign_not_complete')
        try:
            records = jsonl(reader.read(campaign / 'games.jsonl'))
        except AuditError:
            if running:
                pending.append('campaign_index_inflight')
                records = []
            else:
                raise
        require(len(records) <= 99, 'too_many_campaign_records')
        for number, record in enumerate(records, 1):
            # The path comes from its required sequence position, never from
            # untrusted JSON. No failed attempt can be filtered or renumbered.
            directory = campaign / f'game-{number:06d}'
            row, local, server = audit_game(directory, number + 1, identity, reader, record)
            rows.append(row)
            if row['status'] == 'complete':
                local_all.extend(local); server_all.extend(server)
        directories = sorted(p.name for p in campaign.iterdir() if p.is_dir() or p.is_symlink())
        expected_dirs = [f'game-{number:06d}' for number in range(1, len(records) + 1)]
        if terminal:
            require(len(records) == 99 and directories == expected_dirs, 'missing_or_extra_campaign_game')
            require(top.get('games_recorded') == 99 and top.get('current_game') is None and top.get('error') is None,
                    'campaign_manifest_totals')
            require(summary.get('games_attempted') == summary.get('games_completed') == summary.get('healthy_games') == 99
                    and summary.get('current_game') is None and summary.get('error') is None,
                    'campaign_summary_totals')
            completed_rows = rows[1:]
            require(all(row['status'] == 'complete' for row in completed_rows), 'campaign_contains_bad_game')
            count_totals = {key: sum(row['counts'][key] for row in completed_rows) for key in COUNT_FIELDS}
            require(normalized_counts(summary.get('counts')) == count_totals, 'campaign_summary_counts_mismatch')
            # Use per-game independent means with the supervisor's exact
            # weighting order to verify its summary, without using it as data.
            n = sum(row['local_decision_ms']['count'] for row in completed_rows)
            same_numbers(summary.get('latency'), {'count': n,
                'mean_ms': sum(row['local_decision_ms']['count'] * row['local_decision_ms']['mean_ms'] for row in completed_rows) / n,
                'max_ms': max(row['local_decision_ms']['max_ms'] for row in completed_rows)}, 'campaign_summary_latency_mismatch')
            require(finite(summary.get('elapsed_seconds')), 'invalid_campaign_elapsed')
        else:
            # A running child is deliberately not opened beyond its manifest.
            allowed = expected_dirs + [f'game-{len(records) + 1:06d}']
            if not all(name in allowed for name in directories):
                # Parent JSON files are atomically replaced, but the JSONL
                # append and creation of the next child occur between them.
                # Running layouts cannot justify final acceptance or a claim
                # of corruption; terminal layouts must match exactly above.
                pending.append('campaign_layout_requires_terminal_snapshot')
            pending.append('full_100_not_final')
        require(identities(paths, expected)[0] == actual, 'identity_changed_during_audit')
        if not reader.unchanged():
            if running:
                pending.append('source_changed_during_probe')
            else:
                errors.append('source_changed_during_final_audit')
    except (AuditError, OSError, KeyError, TypeError, IndexError) as exc:
        errors.append(str(exc) if isinstance(exc, AuditError) else 'unreadable_or_malformed_campaign')
    for row in rows:
        errors.extend(f'game_{row["game_number"]}:{code}' for code in row['errors'])
        if row['status'] == 'pending':
            pending.append(f'game_{row["game_number"]}_pending')
    complete = [row for row in rows if row['status'] == 'complete']
    if len({row['events_sha256'] for row in complete}) != len(complete):
        errors.append('duplicate_game_event_evidence')
    ranks = Counter(row['rank_min'] for row in complete if row['rank_min'] == row['rank_max'])
    result['totals'] = {
        'games_audited': len(rows), 'games_verified_complete': len(complete),
        'invalid_games': sum(row['status'] == 'invalid' for row in rows),
        'pending_games': sum(row['status'] == 'pending' for row in rows),
        'score_sum': sum(row['score'] for row in complete),
        'mean_score': statistics.fmean(row['score'] for row in complete) if complete else None,
        'mean_point_delta': statistics.fmean(row['point_delta'] for row in complete) if complete else None,
        'mean_rank_min': statistics.fmean(row['rank_min'] for row in complete) if complete else None,
        'mean_rank_max': statistics.fmean(row['rank_max'] for row in complete) if complete else None,
        'untied_rank_counts': {str(rank): ranks[rank] for rank in range(1, 5)},
        'tied_games': sum(row['rank_min'] != row['rank_max'] for row in complete),
        'counts': {key: sum(row['counts'][key] for row in complete) for key in COUNT_FIELDS},
        'diagnostic_counts': {key: sum(row['counts'][key] for row in complete) for key in DIAGNOSTIC_COUNTS},
        'local_decision_ms': stats(local_all), 'server_ack_elapsed_ms': stats(server_all),
        'bank_consumed_ms': sum(row['bank_consumed_ms'] for row in complete),
        'descriptive_statistics_scope': 'Verified complete games only; all invalid/pending attempts remain in games and prohibit final acceptance.',
    }
    result['raw_files'] = sorted(reader.files.values(), key=lambda item: item['path'])
    if errors:
        result['status'] = 'invalid'
    elif pending or len(complete) != 100:
        result['status'] = 'pending'
    else:
        result['status'] = 'ready'
    return result


def write_final_audit(result, output, *, paths=None, expected=None):
    """Publish only a ready audit, and never overwrite an existing directory."""
    require(result.get('status') == 'ready' and result.get('final_acceptance') is False
            and not result.get('errors') and not result.get('pending_reasons')
            and result.get('totals', {}).get('games_verified_complete') == 100
            and len(result.get('games', [])) == 100, 'final_audit_not_ready')
    inputs = result.get('inputs', {})
    require(set(inputs) == {'first_game', 'campaign'}, 'audit_inputs_missing')
    fresh = audit_ranked(inputs['first_game'], inputs['campaign'], paths=paths, expected=expected)
    require(fresh == result and fresh['status'] == 'ready', 'independent_reaudit_changed')
    # Recheck the exact bytes after any delay between auditing and publication.
    for item in result['raw_files']:
        path = Path(item['path'])
        require(not path.is_symlink() and path.is_file()
                and path.stat().st_size == item['bytes']
                and digest(path.read_bytes()) == item['sha256'], 'source_changed_before_publication')
    output = Path(output)
    output.mkdir(exist_ok=False)
    report = dict(result, status='passed', final_acceptance=True,
                  published_at=datetime.now(timezone.utc).isoformat())
    documents = {'report.json': report, 'games.json': result['games'], 'raw-sha256.json': result['raw_files']}
    for name, value in documents.items():
        with (output / name).open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    with (output / 'SHA256SUMS').open('x', encoding='utf-8') as stream:
        stream.write(''.join(f'{digest((output / name).read_bytes())}  {name}\n' for name in sorted(documents)))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first-game', type=Path, default=DEFAULT_FIRST)
    parser.add_argument('--campaign', type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--report-dir', type=Path, default=HERE / 'final-audit')
    args = parser.parse_args(argv)
    result = audit_ranked(args.first_game, args.campaign)
    if args.write:
        try:
            result = write_final_audit(result, args.report_dir)
        except (AuditError, OSError) as exc:
            print(json.dumps({'status': result['status'], 'final_acceptance': False,
                'write_error': str(exc) if isinstance(exc, AuditError) else 'report_directory_exists_or_unwritable',
                'errors': result['errors'], 'pending_reasons': result['pending_reasons']}))
            return 2
    print(json.dumps({key: result[key] for key in
        ('status', 'final_acceptance', 'errors', 'pending_reasons', 'totals')}, ensure_ascii=False, indent=2))
    return 0 if result['status'] in ('ready', 'passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
