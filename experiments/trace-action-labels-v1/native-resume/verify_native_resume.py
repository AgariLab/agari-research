"""Bounded real seeded-trace epoch resume check. Artifacts stay on project storage."""
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from agari_kit.collection import write_json
from agari_kit.storage import Storage
from agari_research.data import FrozenDataset
from agari_research.manifest import file_sha256
from agari_research.planning import build_training_plan
from agari_research.runtime import runtime_provenance
from agari_research.training import implementation_sha256, validate_config

storage=Storage.from_env()
base=storage.path('runs/trace-native-resume-v1')
repo=Path('/Users/zhou/MyCodes/AgariLab/agari-research')
split='datasets/manifests/smoke-v1/split-v1.json'
continuous='trace-native-continuous-v1'
interrupted='trace-native-interrupted-v1'
for name in (continuous,interrupted):
    for kind in ('runs','models','checkpoints'):
        if storage.path(f'{kind}/{name}').exists():
            raise RuntimeError(f'Refusing existing experiment directory: {kind}/{name}')
started=time.perf_counter()
implementation=implementation_sha256()
assert implementation == "2f6e3618a1fa7812837b042e257d654c6e4d94f95764904002dbfef0bb9afedf"
runtime=runtime_provenance()
config=json.loads((base/'config.json').read_text())
validate_config(config)
dataset=FrozenDataset(storage,storage.path(split))
plan=build_training_plan(config,dataset)
assert plan['sample_semantics_version'] == 'seeded-trace-player-decisions-v1'
write_json(base/'execution-plan.json',{'plan':plan,'implementation_sha256':implementation,'runtime_provenance':runtime})
for stage in ('grp','cql'):
    assert 0 < plan['stages'][stage]['last_batch_samples'] < config[stage]['batch_size']
grp_epoch=plan['stages']['grp']['updates_per_epoch']
grp_target=plan['stages']['grp']['target_updates']
cql_epoch=plan['stages']['cql']['updates_per_epoch']
first_stop=grp_epoch+max(1,grp_epoch//2)
assert grp_epoch < first_stop < grp_target
second_stop=grp_target+cql_epoch-first_stop
print(json.dumps({'plan':plan['stages'],'first_stop_total_updates':first_stop,'second_invocation_updates':second_stop}),flush=True)
checks=[]

def invoke(label,name,stop=None,resume=False):
    command=[sys.executable,'-m','agari_research.cli','run','--config',str(base/'config.json'),'--split',split,'--run',name]
    if stop is not None:
        command+=['--stop-after-updates',str(stop)]
    if resume:
        command+=['--resume',f'checkpoints/{name}/latest.pt']
    start=time.perf_counter()
    result=subprocess.run(command,cwd=repo,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    elapsed=time.perf_counter()-start
    (base/f'{label}.log').write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f'{label} failed; see {label}.log')
    state=torch.load(storage.path(f'checkpoints/{name}/latest.pt'),map_location='cpu',weights_only=True)
    record={'label':label,'seconds':elapsed,'command':command,'progress':state['progress'],
            'samples_seen':state['samples_seen'],'schedulers':{stage:{'T_max':value['T_max'],'last_epoch':value['last_epoch']}
            for stage,value in state['schedulers'].items()}}
    checks.append(record)
    write_json(base/f'{label}-checkpoint.json',record)
    print(json.dumps(record),flush=True)
    return state

full=invoke('continuous',continuous)
first=invoke('interrupted-grp',interrupted,first_stop)
assert first['progress']=={'grp':first_stop,'cql':0,'ppo':0}
assert first['samples_seen']['grp'] > plan['stages']['grp']['samples_per_epoch']
second=invoke('interrupted-cql',interrupted,second_stop,True)
assert second['progress']=={'grp':grp_target,'cql':cql_epoch,'ppo':0}
assert second['samples_seen']['cql']==plan['stages']['cql']['samples_per_epoch']
resumed=invoke('resumed-complete',interrupted,resume=True)
assert implementation_sha256()==implementation
assert runtime_provenance()==runtime
spec=importlib.util.spec_from_file_location('verify_resume',repo/'scripts/verify_resume.py')
verification=importlib.util.module_from_spec(spec)
spec.loader.exec_module(verification)
full.pop('run_name')
resumed.pop('run_name')
for stage in ('grp','cql','ppo'):
    assert full['progress'][stage]==plan['stages'][stage]['target_updates']
tensors=verification.identical(full,resumed)
coverage={}
for name in (continuous,interrupted):
    ledger=json.loads(storage.path(f'runs/{name}/coverage.json').read_text())
    for stage in ('grp','cql'):
        row=ledger['stages'][stage]
        assert row['complete_epochs']==2
        assert row['unique_games_seen']==len(dataset.records('train'))
        assert row['samples_seen']==plan['stages'][stage]['target_samples']
        assert all(game['samples_seen']==2*game['samples_per_epoch'] for game in row['games'])
    coverage[name]=ledger
    write_json(base/f'{name}-coverage.json',ledger)
report={'status':'exact-match','dataset_sha256':dataset.sha256,'split':split,'split_file_sha256':file_sha256(storage.path(split)),
        'partitions':{key:len(dataset.manifest['partitions'][key]) for key in ('train','validation','test')},
        'training_plan_sha256':plan['sha256'],'sample_semantics_version':plan['sample_semantics_version'],'producer_engines':plan['producer_engines'],'implementation_sha256':implementation,'runtime_provenance':runtime,
        'research_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        'config_sha256':file_sha256(base/'config.json'),'script_sha256':file_sha256(Path(__file__)),
        'comparison_script_sha256':file_sha256(repo/'scripts/verify_resume.py'),'tensors_compared':tensors,
        'checked':'All final checkpoint fields except distinct run_name, including native data cursors, model, optimizer, scheduler and RNG',
        'invocations':checks,'total_seconds':time.perf_counter()-started,'test_content_read':False,
        'checkpoint_sha256':{name:file_sha256(storage.path(f'checkpoints/{name}/latest.pt')) for name in (continuous,interrupted)},
        'coverage':{name:{stage:{key:value for key,value in ledger['stages'][stage].items() if key!='games'}
                          for stage in ('grp','cql')} for name,ledger in coverage.items()},
        'models':{name:f'models/{name}' for name in (continuous,interrupted)}}
write_json(base/'resume-equivalence.json',report)
print(json.dumps({'status':report['status'],'tensors':tensors,'total_seconds':report['total_seconds'],
                  'report':str(base/'resume-equivalence.json')}),flush=True)

# Verify the actual small tail batches and held-out validation scope in both runs.
batch_evidence={}
for name in (continuous, interrupted):
    rows=[json.loads(line) for line in storage.path(f'runs/{name}/metrics.jsonl').read_text().splitlines()]
    stages={}
    for stage in ('grp','cql'):
        metrics=[row for row in rows if row['stage']==stage]
        tails=[{'update':row['update'],'batch_samples':row['batch_samples']} for row in metrics if row['batch_samples'] != config[stage]['batch_size']]
        expected=[{'update':plan['stages'][stage]['updates_per_epoch'] * epoch,'batch_samples':plan['stages'][stage]['last_batch_samples']} for epoch in (1,2)]
        assert tails == expected
        validation={key:value for key,value in metrics[-1].items() if key.startswith('validation_')}
        assert validation['validation_scope']=='full' and validation['validation_sample_limit'] is None
        stages[stage]={'tail_batches':tails,'validation':validation}
    batch_evidence[name]=stages
write_json(base/'batch-validation-evidence.json',batch_evidence)
print(json.dumps({'tail_batches_verified':True,'batch_evidence':str(base/'batch-validation-evidence.json')}),flush=True)
