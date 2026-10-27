"""Independent selected-action census by seeded native engine execution.

No MjaiReplay, no training data reader, no trace_samples import. Test metadata
is used only to validate the frozen split; test artifact contents are unopened.
"""
import gzip
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

from riichienv import GameRule,RiichiEnv
from agari_kit.collection import file_sha256,verify_record,write_json
from agari_kit.dataset import make_split
from agari_kit.engine import engine_identity
from agari_kit.mjai import canonical_json,read_events
from agari_kit.provenance import source_fingerprint
from agari_kit.storage import Storage

storage=Storage.from_env()
base=storage.path('runs/trace-selected-actions-audit-v1')
engine=engine_identity()
assert engine=={'version':'0.4.10','binary_sha256':'169c8c2c53bdace01e4ca319f54050565fe728a5b1b2fc6b7f3046738887071d'}
kit=source_fingerprint()
started=time.perf_counter()
metadata={'started_at':datetime.now(timezone.utc).isoformat(),'engine':engine,'code':kit,'python':sys.version,
          'audit_script_sha256':file_sha256(Path(__file__)),'selection':'len(native_observation.legal_actions()) > 1',
          'action_match':'Full native Action.to_dict equals recorded submitted action before engine resolves simultaneous claims',
          'grp_count':'Native executed start_kyoku count times four players','test_content_read':False,
          'forbidden_readers_imported':False}
write_json(base/'started.json',metadata)


def audit_game(record):
    game_started=time.perf_counter()
    receipt=verify_record(storage,record,replay=False)
    trace=storage.path(record['trace_path']); mjai=storage.path(record['mjai_path']); receipt_path=storage.path(record['validation_receipt'])
    hashes={'trace_sha256':file_sha256(trace),'mjai_file_sha256':file_sha256(mjai),'receipt_sha256':file_sha256(receipt_path)}
    action_types=Counter(); all_types=Counter(); forced_types=Counter(); per_seat=Counter()
    multiple_ron=[]; step=0; footer=None
    opener=gzip.open if trace.suffix=='.gz' else open
    with opener(trace,'rt',encoding='utf-8') as stream:
        header=json.loads(next(stream))
        assert header['type']=='agari_trace' and header['schema_version']==1
        assert header['rules']=={'engine_version':'0.4.10','game_mode':'4p-red-half','preset':'tenhou'}
        seed=header['seed']; assert type(seed) is int and 0<=seed<2**64
        env=RiichiEnv(seed=seed,game_mode='4p-red-half',rule=GameRule.default_tenhou())
        observations=env.reset()
        for line in stream:
            entry=json.loads(line)
            if entry.get('type')=='trace_end':
                footer=entry
                assert not stream.read().strip(), 'Records after trace_end'
                break
            assert not env.done() and observations and step<100000
            assert type(entry['step']) is int and entry['step']==step
            seats={str(seat) for seat in observations}
            assert set(entry['actions'])==seats==set(entry['observations_sha256'])
            selected={}; ron_players=[]
            for seat,obs in observations.items():
                view_hash=hashlib.sha256(obs.serialize_to_base64().encode()).hexdigest()
                assert view_hash==entry['observations_sha256'][str(seat)], 'Observation mismatch'
                legal=list(obs.legal_actions()); assert legal
                recorded_key=canonical_json(entry['actions'][str(seat)])
                matching=[action for action in legal if canonical_json(action.to_dict())==recorded_key]
                assert len(matching)==1, 'Submitted action is not one unique native legal Action.to_dict'
                action=matching[0]; selected[seat]=action
                action_type=str(action.action_type)
                all_types[action_type]+=1
                if len(legal)>1:
                    action_types[action_type]+=1
                    per_seat[str(seat)]+=1
                else:
                    forced_types[action_type]+=1
                if action_type=='ActionType.RON': ron_players.append(seat)
            if len(ron_players)>1: multiple_ron.append({'step':step,'players':sorted(ron_players)})
            observations=env.step(selected)
            step+=1
    assert footer is not None and env.done()
    scores=list(env.scores())
    assert footer['steps']==step and footer['scores']==scores
    generated=env.mjai_log
    assert generated and generated[-1]['type']=='end_game'
    generated[-1]=dict(generated[-1],scores=scores)
    archived=list(read_events(mjai))
    assert len(generated)==len(archived)
    canonical_digest=hashlib.sha256()
    for actual,expected in zip(archived,generated):
        text=canonical_json(actual)
        assert text==canonical_json(expected), 'MJAI differs from native seeded execution'
        canonical_digest.update((text+'\n').encode())
    assert canonical_digest.hexdigest()==record['sha256']==receipt['sha256']
    rounds=sum(event['type']=='start_kyoku' for event in generated)
    assert rounds==sum(event['type']=='end_kyoku' for event in generated)
    assert hashes=={'trace_sha256':file_sha256(trace),'mjai_file_sha256':file_sha256(mjai),'receipt_sha256':file_sha256(receipt_path)}
    assert sum(action_types.values())+sum(forced_types.values())==sum(all_types.values())
    return {'game_id':record['game_id'],'canonical_mjai_sha256':record['sha256'],'seed':seed,'rounds':rounds,'grp_samples':rounds*4,
            'policy_samples':sum(action_types.values()),'action_types':dict(sorted(action_types.items())),
            'all_selected_action_types':dict(sorted(all_types.items())),'forced_action_types':dict(sorted(forced_types.items())),
            'per_player_policy_samples':dict(sorted(per_seat.items())),'steps':step,'all_selected_actions':sum(all_types.values()),
            'simultaneous_multi_ron_steps':multiple_ron,'scores':scores,'input_hashes':hashes,
            'input_paths':{key:record[key] for key in ('trace_path','mjai_path','validation_receipt')},
            'producer_code':header['code'],'producer_agents':header['agents'],'elapsed_seconds':time.perf_counter()-game_started}


def audit_dataset(name,old_relative):
    start=time.perf_counter()
    split_path=storage.path(f'datasets/manifests/{name}/split-v1.json')
    split_hash=file_sha256(split_path)
    split=json.loads(split_path.read_text())
    manifest_records=[record for part in ('train','validation','test') for record in split['partitions'][part]]
    assert make_split(manifest_records)==split
    old_path=storage.path(old_relative); old_hash=file_sha256(old_path); old=json.loads(old_path.read_text())
    assert old['dataset_sha256']==split['dataset_sha256']
    result={'dataset':name,'dataset_sha256':split['dataset_sha256'],'split_file_sha256':split_hash,
            'split_path':str(split_path.relative_to(storage.root)),'engine':engine,'audit_script_sha256':metadata['audit_script_sha256'],
            'test_content_read':False,'old_native_report':old_relative,'old_native_report_sha256':old_hash,'partitions':{}}
    for partition in ('train','validation'):
        partition_started=time.perf_counter(); rows=[]; actions=Counter(); all_actions=Counter(); forced=Counter()
        old_part=old['partitions'][partition]
        old_rows={row['game_id']:row for row in old_part['per_game']}
        records=split['partitions'][partition]
        assert set(old_rows)=={r['game_id'] for r in records}
        deltas=[]
        with (base/f'{name}-{partition}-games.jsonl').open('x') as stream:
            for index,record in enumerate(records,1):
                row=audit_game(record); rows.append(row)
                actions.update(row['action_types']); all_actions.update(row['all_selected_action_types']); forced.update(row['forced_action_types'])
                previous=old_rows[row['game_id']]
                old_policy=previous.get('cql_samples',previous.get('policy_samples'))
                assert type(old_policy) is int
                old_rounds=previous['rounds']
                difference={'game_id':row['game_id'],'old_native_policy_samples':old_policy,'selected_policy_samples':row['policy_samples'],
                            'policy_sample_delta':row['policy_samples']-old_policy,'grp_sample_delta':row['grp_samples']-old_rounds*4}
                if 'actions' in previous:
                    keys=set(previous['actions'])|set(row['action_types'])
                    difference['action_type_deltas']={key:row['action_types'].get(key,0)-previous['actions'].get(key,0)
                                                      for key in sorted(keys) if row['action_types'].get(key,0)!=previous['actions'].get(key,0)}
                deltas.append(difference)
                stream.write(canonical_json(row)+'\n'); stream.flush()
                if index%25==0 or index==len(records):
                    progress={'dataset':name,'partition':partition,'games':index,'expected':len(records),'policy_samples':sum(actions.values()),
                              'elapsed_seconds':time.perf_counter()-partition_started}
                    write_json(base/'progress.json',progress); print(json.dumps(progress),flush=True)
        old_policy=sum(row.get('cql_samples',row.get('policy_samples')) for row in old_rows.values())
        old_rounds=sum(row['rounds'] for row in old_rows.values())
        totals={'games':len(rows),'rounds':sum(row['rounds'] for row in rows),'grp_samples':sum(row['grp_samples'] for row in rows),
                'policy_samples':sum(actions.values()),'action_types':dict(sorted(actions.items())),
                'all_selected_action_types':dict(sorted(all_actions.items())),'forced_action_types':dict(sorted(forced.items())),
                'all_selected_actions':sum(all_actions.values()),'trace_steps':sum(row['steps'] for row in rows),
                'simultaneous_multi_ron_steps':sum(len(row['simultaneous_multi_ron_steps']) for row in rows),
                'per_game_file':f'{name}-{partition}-games.jsonl','per_game_file_sha256':file_sha256(base/f'{name}-{partition}-games.jsonl'),
                'elapsed_seconds':time.perf_counter()-partition_started,'difference_from_old_native':{
                    'old_policy_samples':old_policy,'policy_sample_delta':sum(actions.values())-old_policy,
                    'old_grp_samples':old_rounds*4,'grp_sample_delta':sum(row['grp_samples'] for row in rows)-old_rounds*4,
                    'games_with_sample_count_difference':sum(row['policy_sample_delta']!=0 for row in deltas),
                    'per_game':deltas}}
        if 'action_types' in old_part:
            keys=set(old_part['action_types'])|set(actions)
            totals['difference_from_old_native']['action_type_deltas']={key:actions[key]-old_part['action_types'].get(key,0)
                for key in sorted(keys) if actions[key]!=old_part['action_types'].get(key,0)}
            totals['difference_from_old_native']['games_with_action_label_difference']=sum(bool(row['action_type_deltas']) for row in deltas)
        result['partitions'][partition]=totals
    assert file_sha256(split_path)==split_hash and file_sha256(old_path)==old_hash
    result['elapsed_seconds']=time.perf_counter()-start
    write_json(base/f'{name}.json',result)
    print(json.dumps({'dataset_complete':name,'summary':{part:{key:value for key,value in data.items()
                     if key in ('games','rounds','grp_samples','policy_samples','action_types','simultaneous_multi_ron_steps')}
                     for part,data in result['partitions'].items()}}),flush=True)
    return result

try:
    reports=[]
    for name,old in [('meld-coverage-v1','runs/meld-coverage-audit-v1/report.json'),
                     ('selfplay-v1','runs/baseline-coverage-audit-v1/native-sample-counts.json')]:
        reports.append(audit_dataset(name,old))
    assert engine_identity()==engine and source_fingerprint()==kit
    forbidden=[name for name in sys.modules if name=='agari_research.data' or 'trace_samples' in name]
    assert not forbidden
    result=dict(metadata,status='complete',elapsed_seconds=time.perf_counter()-started,
                finished_at=datetime.now(timezone.utc).isoformat(),dataset_reports={row['dataset']:f'{row["dataset"]}.json' for row in reports},
                dataset_report_sha256={row['dataset']:file_sha256(base/f'{row["dataset"]}.json') for row in reports},forbidden_imports=forbidden)
    write_json(base/'report.json',result)
    print(json.dumps(result),flush=True)
except Exception as error:
    write_json(base/'failure.json',dict(metadata,status='failed',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.perf_counter()-started))
    raise
