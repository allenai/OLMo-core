"""Idempotent controller for the explicitly approved 3-lineage/stage matrix."""

import argparse
import copy
import fcntl
import json
import os
import subprocess
import sys
import time

from olmoe3_qkgain_plan import *
from olmoe3_lr_sweep_watch import Controller,atomic_json,log,replace_env,status

TRAIN_TEMPLATE='01M2V7H5S2SWN7WWW981VG50GX'
CPU_TEMPLATE='01M2VHM12SRGJJ6KGRN6P5D22D'
UPLOADER_REF='50069318bd7b6bcfed655a8a01d2892e56b7abff'


def add_mounts(t):
    for volume in ('olmo-3p5-checkpoints','dolma-3p5','oe-training-default','oe-adapt-default'):
        path='/weka/'+volume
        if not any(x['mountPath']==path for x in t['datasets']):
            t['datasets'].append(dict(mountPath=path,source=dict(weka=volume)))


def training_spec(template,r,commit,hosts):
    s=copy.deepcopy(template);t=s['tasks'][0];s['tasks']=[t]
    t.update(name='train',replicas=r.nodes,leaderSelection=True,timeout='720h')
    t['resources']['gpuCount']=r.gpus//r.nodes
    t['arguments']=['python','src/examples/olmo_ddp/olmoe3_qkgain_node.py',r.run_id]
    t['context'].update(priority='urgent',minRuntime='6h',autoResume=True)
    t['constraints']={'cluster':['ai2/holmes'],'hostname':hosts}
    t['result']={'path':'/noop-results'}
    replace_env(t,dict(GIT_REF=commit,GIT_BRANCH=BRANCH,NUM_NODES=r.nodes,
        GANTRY_TASK_NAME='train',GANTRY_INSTALL_CMD='true',
        GANTRY_POST_SETUP_CMD='bash src/examples/olmo_ddp/olmoe3_hero_decay_setup.sh',
        OLMO35_DECAY_CPU_VALIDATE=None,WANDB_RUN_ID=None,WANDB_RESUME=None,
        OLMO_PROFILE_LB_COUNT_BATCHED='0',
        CACHED_PATH_CACHE_ROOT=str(ROOT/'data-work/cached-path-metadata'),
        OLMOE3_BEAKER_WORKSPACE=WORKSPACE,RESULTS_DIR='/noop-results'))
    # No inherited start/load/sweep flag may redirect this independent campaign.
    t['envVars']=[e for e in t['envVars'] if not e['name'].startswith(('HYBRID_','QKGAIN_','OLMO35_MT_LOAD','OLMO35_LC_LOAD','OLMO35_DECAY_LOAD'))]
    add_mounts(t)
    s['retry']={'allowedTaskRetries':0};s['description']=json.dumps(r.as_dict())
    return s


def cpu_spec(template,commit,mode,train_template=None):
    if mode=='config':
        assert train_template is not None
        template=train_template
    s=copy.deepcopy(template);t=s['tasks'][0];s['tasks']=[t]
    for key in ('resources','replicas','leaderSelection','synchronizedStartTimeout','hostNetworking','propagateFailure','propagatePreemption'):
        t.pop(key,None)
    t.update(name=mode,timeout='720h' if mode=='watch' else '12h')
    t['constraints']={'cluster':['ai2/phobos' if mode=='watch' else 'ai2/rhea']}
    t['context']=dict(priority='urgent',minRuntime='0s',autoResume=mode=='watch')
    t['result']={'path':'/noop-results'}
    add_mounts(t)
    if mode=='watch':
        t['arguments']=['bash','-euc',"gh auth setup-git && exec uv run --no-project --with 'beaker-py==2.7.2' --with 'olmo-checkpoint-uploader @ git+https://github.com/jacob-morrison/olmo-checkpoint-uploader.git@"+UPLOADER_REF+"' python -u src/examples/olmo_ddp/olmoe3_qkgain_control.py watch"]
        setup='true'
    elif mode=='download':
        t['arguments']=['bash','-euc',"exec uv run --no-project --with 'huggingface-hub==1.29.0' python -u src/examples/olmo_ddp/olmoe3_qkgain_control.py download"]
        setup='true'
    else:
        t['arguments']=['python','src/examples/olmo_ddp/olmoe3_qkgain_train.py','--validate']
        setup='bash src/examples/olmo_ddp/olmoe3_hero_decay_setup.sh'
    replace_env(t,dict(GIT_REF=commit,GIT_BRANCH=BRANCH,GANTRY_TASK_NAME=mode,
        GANTRY_INSTALL_CMD='true',GANTRY_POST_SETUP_CMD=setup,NUM_NODES='1',
        OLMO35_DECAY_CPU_VALIDATE='1',RESULTS_DIR='/noop-results',
        CACHED_PATH_CACHE_ROOT=str(ROOT/'data-work/cached-path-metadata'),
        HF_XET_HIGH_PERFORMANCE='1',WANDB_MODE='disabled'))
    if not any(e['name']=='HF_TOKEN' for e in t['envVars']):
        t['envVars'].append(dict(name='HF_TOKEN',secret='jacobm_HF_TOKEN'))
    s['retry']={'allowedTaskRetries':0};s['description']=f'{CAMPAIGN}: {mode}; no Beaker result payloads'
    assert not t.get('resources')
    return s


def control(b,commit,path=AUTOMATION,workspace=WORKSPACE):
    c=object.__new__(Controller)
    c.beaker,c.commit,c.workspace=b,commit,b.workspace.get(workspace)
    c.automation,c.last_status=path,{}
    return c


def download():
    from huggingface_hub import HfApi
    import olmoe3_hero_bucket_download as d
    assert MOUNT.is_mount()
    d.SCRATCH=DOWNLOAD_ROOT
    d.prepare_scratch()
    p=d.download(HfApi(),'emo',6000)
    assert p==REFERENCE
    validate_checkpoint(p,6000,16777216,64)
    atomic_json(AUTOMATION/'download-success.json',dict(passed=True,path=str(p)))


def ensure_saved(c,name,make):
    """Reuse immutable submitted specs, even if host availability changes later."""
    path=c.automation/'specs'/f'{name}.json'
    spec=json.loads(path.read_text()) if path.exists() else make()
    w=c.ensure(name,spec)
    return w,c.report(w)


def watch():
    from beaker import Beaker,BeakerExperimentSpec
    from olmo_checkpoint_uploader.models import Registration
    from olmo_checkpoint_uploader.state import StateStore
    from olmo_checkpoint_uploader.backend import HuggingFaceBucketBackend
    from olmoe3_qkgain_eval import eval_specs,validate_result
    assert MOUNT.is_mount()
    AUTOMATION.mkdir(parents=True,exist_ok=True)
    commit=os.environ['GIT_REF']
    with (AUTOMATION/'LOCK').open('a') as lock, Beaker.from_env(check_for_upgrades=False) as b:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        c=control(b,commit);ec=control(b,commit,AUTOMATION/'evals',EVAL_WORKSPACE)
        templates=b.experiment.get_spec(b.workload.get(TRAIN_TEMPLATE)).to_json()
        cpu=b.experiment.get_spec(b.workload.get(CPU_TEMPLATE)).to_json()
        # Only remove hosts from the qualified allowlist; never widen node eligibility.
        hosts_path=AUTOMATION/'hosts.json'
        if not hosts_path.exists():
            registered={n.hostname for n in b.node.list()}
            prior=templates['tasks'][0]['constraints']['hostname']
            hosts=[h for h in prior if h in registered]
            assert len(hosts)>=8
            atomic_json(hosts_path,hosts)
        hosts=json.loads(hosts_path.read_text())
        backend=HuggingFaceBucketBackend();backend.assert_private(BUCKET)
        store=StateStore(CONTROL,STATE)
        for r in runs()+runs(True):
            r.root.mkdir(parents=True,exist_ok=True)
            store.register(Registration(run_id=r.run_id,lineage_id=r.run_id,checkpoint_root=str(r.root),
                bucket_id=BUCKET,remote_prefix=r.prefix,deletion_mode='apply',min_local_checkpoints=2,delete_grace_seconds=3600))
        plan=dict(commit=commit,runs=[r.as_dict() for r in runs()],
                  numerical_parity='standing user waiver; structural and runtime checks retained')
        path=AUTOMATION/'plan.json'
        if path.exists():assert json.loads(path.read_text())==plan
        else:atomic_json(path,plan)
        log('QKGAIN_CAMPAIGN_ARMED',**plan)
        previous=None
        while True:
            rows={}
            gate,gs=ensure_saved(c,CAMPAIGN+'-config',lambda:cpu_spec(cpu,commit,'config',templates))
            dl,ds=ensure_saved(c,CAMPAIGN+'-download',lambda:cpu_spec(cpu,commit,'download'))
            rows['config']=dict(status=gs,id=gate.experiment.id if gate else None)
            rows['download']=dict(status=ds,id=dl.experiment.id if dl else None)
            fs=os.statvfs(MOUNT);admit=fs.f_bavail*fs.f_frsize>=12_000_000_000_000 and status(b.workload.get(UPLOADER))=='STATUS_RUNNING'
            for arm in ARMS:
                for stage in STAGES:
                    r=Run(arm,stage)
                    try:
                        if gs!='STATUS_SUCCEEDED':
                            rows[r.run_id]=dict(waiting='config gate');break
                        if r.source:
                            if stage=='pt':
                                if ds!='STATUS_SUCCEEDED':
                                    rows[r.run_id]=dict(waiting='archive restore');break
                            else:
                                parent=Run(arm,STAGES[STAGES.index(stage)-1])
                                receipt=AUTOMATION/'submissions'/f'{parent.run_id}-train.json'
                                if not receipt.exists():break
                                pw=b.workload.get(json.loads(receipt.read_text())['experiment_id'])
                                if status(pw)!='STATUS_SUCCEEDED':break
                                assert json.loads((parent.root/'audit/success.json').read_text())['passed']
                            validate_checkpoint(r.source)
                        if not admit:
                            rows[r.run_id]=dict(waiting='uploader/storage admission');break
                        sm=Run(arm,stage,True)
                        sw,ss=ensure_saved(c,sm.run_id,lambda:training_spec(templates,sm,commit,hosts))
                        if ss!='STATUS_SUCCEEDED':
                            rows[r.run_id]=dict(smoke_status=ss,smoke=sw.experiment.id if sw else None);break
                        assert json.loads((sm.root/'audit/success.json').read_text())['passed']
                        tw,ts=ensure_saved(c,r.run_id+'-train',lambda:training_spec(templates,r,commit,hosts))
                        row=dict(status=ts,id=tw.experiment.id if tw else None)
                        rows[r.run_id]=row
                        if ts!='STATUS_SUCCEEDED':break
                        assert json.loads((r.root/'audit/success.json').read_text())['passed']
                        validate_checkpoint(r.root/f'step{r.end}',r.end,r.batch,r.gpus)
                        specs=eval_specs(b,r,commit)
                        estate={}
                        for kind,spec in specs.items():
                            BeakerExperimentSpec.from_json(copy.deepcopy(spec))
                            ew,es=ensure_saved(ec,r.run_id+'-'+kind,lambda spec=spec:spec)
                            estate[kind]=dict(status=es,id=ew.experiment.id if ew else None)
                            if es=='STATUS_SUCCEEDED':validate_result(r,kind)
                            if kind=='convert' and es!='STATUS_SUCCEEDED':break
                        row['evals']=estate
                        archived=store.load_checkpoint(r.run_id,r.end)
                        row['uploaded']=bool(archived and archived.remote_verified)
                    except Exception as e:
                        rows[r.run_id]=dict(state='needs_attention',error=f'{type(e).__name__}: {e}')
                        break
            atomic_json(AUTOMATION/'status.json',dict(updated_at=time.time(),runs=rows))
            if rows!=previous:log('QKGAIN_STATUS',runs=rows);previous=rows
            if all(rows.get(r.run_id,{}).get('uploaded') and len(rows[r.run_id].get('evals',{}))==5
                   and all(e['status']=='STATUS_SUCCEEDED' for e in rows[r.run_id]['evals'].values()) for r in runs()):
                log('QKGAIN_ALL_STAGES_UPLOADED_AND_EVALUATED');return
            time.sleep(60)


if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='download':download()
    elif mode=='watch':watch()
    else:raise ValueError(mode)
