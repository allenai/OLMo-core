"""Current-replica rendezvous and bounded save/resume smoke on each stage's topology."""

import hashlib
import json
import os
import re
import subprocess
import sys
import time

from olmoe3_qkgain_plan import *
from olmoe3_lr_sweep_watch import atomic_json
from olmoe3_profile_node import resolve_ready_leader
from olmoe3_hero_decay_runtime import verify_runtime


def main():
    from beaker import Beaker
    verify_runtime()
    r=find_run(sys.argv[1]); rank=int(os.environ.get('BEAKER_REPLICA_RANK','0'))
    local=r.gpus//r.nodes
    assert int(os.environ.get('BEAKER_REPLICA_COUNT','1'))==r.nodes
    assert int(os.environ['BEAKER_ASSIGNED_GPU_COUNT'])==local
    exp,job=os.environ['BEAKER_EXPERIMENT_ID'],os.environ['BEAKER_JOB_ID']
    # nvidia-smi may show all eight physical GPUs even for a six-GPU allocation.
    from olmoe3_profile_topology import validate_topology
    topo=subprocess.check_output(['nvidia-smi','topo','-m'],text=True,timeout=30)
    n=sum(bool(re.match(r'^GPU\d+\s',line)) for line in topo.splitlines())
    assert n in (local,8)
    atomic_json(ROOT/'topology'/exp/f'{job}.json',validate_topology(topo,n))
    ready=ROOT/'rendezvous'/exp
    atomic_json(ready/f'{job}.json',dict(job=job,rank=rank))
    host='127.0.0.1'
    if r.nodes>1:
        with Beaker.from_env(check_for_upgrades=False) as b:
            deadline=time.monotonic()+900
            while time.monotonic()<deadline:
                leader=resolve_ready_leader(b,b.workload.get(exp),ready,r.nodes)
                if leader:break
                print('WAIT_CURRENT_REPLICAS',rank,flush=True);time.sleep(10)
            else:raise TimeoutError('Current replica rendezvous')
        _,host=leader
    existing=sorted((int(p.name[4:]) for p in r.root.glob('step*')
                     if re.fullmatch(r'step\d+',p.name) and (p/'.metadata.json').is_file()))
    start=existing[-1] if existing else r.start
    source=r.root/f'step{start}' if existing else r.source
    target=r.start+4 if r.smoke else r.end
    if start==target:
        validate_checkpoint(r.root/f'step{target}',target,r.batch,r.gpus)
        if rank==0:
            if r.smoke or r.stage in ('mt','lc','sft'):
                for k in range(r.gpus):
                    proof=json.loads((r.root/'audit'/f'restore-{r.start+2}-rank{k}.json').read_text())
                    assert proof['passed'] and not proof['fresh_stage']
            atomic_json(r.root/'audit/success.json',dict(passed=True,step=target,gpus=r.gpus,smoke=r.smoke,
                checkpoint_metadata_sha256=hashlib.sha256((r.root/f'step{target}/.metadata.json').read_bytes()).hexdigest()))
        return
    # New MT branches validate save/resume within their own allocation, then
    # continue automatically. No second GPU queue or discarded smoke training.
    stops=([r.start+2,target] if r.smoke else ([2,4,target] if r.stage in ('mt','lc','sft') else [target]))
    port=29000+int(hashlib.sha256(exp.encode()).hexdigest()[:8],16)%1000
    for i,stop in enumerate(stops):
        if start>=stop:continue
        env=dict(os.environ,QKGAIN_RUN=r.run_id,QKGAIN_START=str(start),QKGAIN_STOP=str(stop))
        if source:env['QKGAIN_LOAD']=str(source)
        else:env.pop('QKGAIN_LOAD',None)
        print('QKGAIN_AGENT',json.dumps(dict(run=r.run_id,start=start,stop=stop,rank=rank)),flush=True)
        subprocess.run([sys.executable,'-m','torch.distributed.run',f'--nnodes={r.nodes}',
            f'--nproc-per-node={local}',f'--node-rank={rank}','--rdzv-backend=static',
            f'--rdzv-endpoint={host}:{port+i}',f'--rdzv-id={exp}-{i}',
            '--rdzv-conf=read_timeout=900','--max-restarts=0',
            'src/examples/olmo_ddp/olmoe3_qkgain_train.py','train',r.run_id,'ai2/holmes'],env=env,check=True)
        source=r.root/f'step{stop}';start=stop
        validate_checkpoint(source,stop,r.batch,r.gpus)
    if rank==0:
        if r.smoke or r.stage in ('mt','lc','sft'):
            for k in range(r.gpus):
                proof=json.loads((r.root/'audit'/f'restore-{r.start+2}-rank{k}.json').read_text())
                assert proof['passed'] and not proof['fresh_stage']
        atomic_json(r.root/'audit/success.json',dict(passed=True,step=target,gpus=r.gpus,smoke=r.smoke,
            checkpoint_metadata_sha256=hashlib.sha256((r.root/f'step{target}/.metadata.json').read_bytes()).hexdigest()))


if __name__=='__main__':main()
