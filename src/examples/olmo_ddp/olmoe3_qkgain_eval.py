"""Scoped adapters to the frozen conversion, ordinary RULER, and chat eval recipes."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from olmoe3_qkgain_plan import *
from olmoe3_lr_sweep_watch import atomic_json


def eval_specs(b,r,commit):
    import olmoe3_hero_decay_eval as d
    import olmoe3_hero_sft_eval_control as s
    oldroot,oldend=d.EVAL_ROOT,d.END
    d.EVAL_ROOT,d.END=EVAL_ROOT/r.run_id,r.end
    proxy=SimpleNamespace(run_id=r.run_id,arm='emo')
    result={}
    try:
        kinds=('convert','math500','ifbench','humaneval','alpaca') if r.stage=='sft' else ('convert','gen_mc','math','code','ruler')
        for kind in kinds:
            base='gen_mc' if kind in ('ruler',*s.BUNDLES) else kind
            template=b.experiment.get_spec(b.workload.get(d.TEMPLATES[base])).to_json()
            if kind in s.BUNDLES:
                s.model_path=lambda run:r.hf
                spec=s.spec_for(template,kind,proxy,commit)
                old=f'olmoe3_hero_sft_eval.py --run {r.run_id} --bundle {kind}'
                new=f'olmoe3_qkgain_eval.py {kind} --run {r.run_id} --source /tmp/hero-ladder'
            else:
                spec=d.build_spec(template,base,proxy,commit)
                old=f'olmoe3_hero_decay_eval.py {base} --arm emo'
                new=f'olmoe3_qkgain_eval.py {kind} --run {r.run_id}'
            t=spec['tasks'][0]
            assert t['arguments'][0].count(old)==1
            t['arguments'][0]=t['arguments'][0].replace(old,new)
            if not any(x['mountPath']=='/weka/oe-adapt-default' for x in t['datasets']):
                t['datasets'].append(dict(mountPath='/weka/oe-adapt-default',source=dict(weka='oe-adapt-default')))
            spec['description']=json.dumps(dict(run=r.as_dict(),stage=kind,model=str(r.hf),numerical_parity=False))
            result[kind]=spec
    finally:
        d.EVAL_ROOT,d.END=oldroot,oldend
    return result


def validate_result(r,kind):
    from olmoe3_hero_4t_eval_policy import validate_export
    validate_export(r.hf)
    if kind=='convert':return
    if kind=='ruler':
        import olmoe3_hero_lc_ruler as rr
        rr.CAMPAIGN=CAMPAIGN+'-ruler'
        rr.verify_success(r.hf)
        return
    if r.stage=='sft':
        path=r.hf.parent/'posttrain-evals-r1'/kind/'success.json'
        row=json.loads(path.read_text())
        assert row['passed'] and row['bundle']==kind and row['model']==str(r.hf)
        assert row['metrics_sha256']==hashlib.sha256((path.parent/'metrics.json').read_bytes()).hexdigest()
    else:
        paths=list(r.hf.parent.glob(f'pilot-olmobase-{kind}-*/full-eval-pilot-success.json'))
        assert paths
        row=json.loads(paths[-1].read_text())
        assert row['passed'] and row['bundle']==kind
        assert row['metrics_sha256']==hashlib.sha256(Path(row['metrics']).read_bytes()).hexdigest()


def convert_checkpoint(r,source):
    import olmoe3_hero_decay_plan as cp
    from olmoe3_hero_4t_eval_policy import install_conversion,validate_export
    root=r.hf.parent
    if r.hf.exists():
        validate_export(r.hf,hash_weights=True)
        return
    cp.validate_checkpoint=validate_checkpoint
    receipt=cp.verified_copy(r.root/f'step{r.end}',root/'olmo-core',r.end)
    atomic_json(root/'download-success.json',dict(passed=True,source_kind='local_copy',source=receipt['source'],
        arm='emo',step=r.end,tokens=r.end*r.batch,raw_path=str(root/'olmo-core'),all_file_hashes_verified=True,bytes=receipt['total_bytes']))
    atomic_json(root/'source-receipt.json',receipt)
    sys.path.insert(0,str(source/'src/examples/olmo_ddp'))
    import hero_hf_convert as convert
    import olmo_core.nn.hf.convert_checkpoint as exporter
    original=exporter.convert_checkpoint_to_hf
    def export(*args,**kwargs):
        kwargs['max_sequence_length']=r.sequence
        if r.stage=='sft':kwargs['tokenizer_id']=str(SFT_DATA/'train/tokenizer')
        result=original(*args,**kwargs)
        if r.stage=='sft':
            from olmoe3_hero_sft_metadata import install_metadata
            install_metadata(args[1],SFT_DATA/'train/tokenizer')
        return result
    exporter.convert_checkpoint_to_hf=export
    convert.SCRATCH=EVAL_ROOT/r.run_id
    convert.BATCH=r.batch
    convert.TARGETS={0:r.end}
    convert.prepare_scratch=lambda:None
    install_conversion(convert)
    sys.argv=[convert.__file__,'--arm','emo','--step',str(r.end),'--full','--portable-reference','--precise']
    convert.main()
    validate_export(r.hf,hash_weights=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('kind',choices=('convert','gen_mc','math','code','ruler','math500','ifbench','humaneval','alpaca'))
    p.add_argument('--run',required=True)
    p.add_argument('--source',required=True,type=Path)
    args=p.parse_args();r=find_run(args.run)
    assert not r.smoke and MOUNT.is_mount()
    from olmoe3_hero_decay_eval import CORE_REF,HELPER_REF
    expected=CORE_REF if args.kind=='convert' else HELPER_REF
    assert subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()==expected
    assert not subprocess.check_output(['git','-C',str(args.source),'status','--porcelain'],text=True).strip()
    owner=EVAL_ROOT/r.run_id/'_OWNER.json'
    payload=dict(campaign=CAMPAIGN,run=r.run_id)
    if owner.exists():assert json.loads(owner.read_text())==payload
    else:atomic_json(owner,payload)
    if args.kind=='convert':return convert_checkpoint(r,args.source)
    from olmoe3_hero_4t_eval_policy import validate_export,install_runtime
    validate_export(r.hf,hash_weights=True)
    if r.stage=='sft':
        import olmoe3_hero_sft_plan as sp
        import olmoe3_hero_sft_convert as sc
        import olmoe3_hero_sft_eval as se
        sp.DATA=SFT_DATA
        sp.find_run=lambda name:SimpleNamespace(run_id=r.run_id,arm='emo',smoke=False)
        sc.export_root=lambda run:EVAL_ROOT/r.run_id
        os.environ['QKGAIN_SFT_STEP']=str(r.end)
        sys.argv=[se.__file__,'--run',r.run_id,'--bundle',args.kind]
        se.main()
    elif args.kind=='ruler':
        import olmoe3_hero_lc_ruler as rr
        rr.CAMPAIGN=CAMPAIGN+'-ruler'
        rr.model_path=lambda milestone,arm:r.hf
        sys.argv=[rr.__file__,'--milestone','lc100b','--arm','emo','--helper',str(args.source),'--instances','4']
        rr.main()
    else:
        sys.path.insert(0,str(args.source/'ladders/olmoe3/workloads'))
        import hero_full_eval as runtime
        install_runtime(runtime)
        runtime.FAST_MODELS={r.hf}
        sys.argv=[runtime.__file__,args.kind,str(r.hf),'--instances','4' if args.kind=='gen_mc' else '8','--fast-pilot']
        runtime.main()
    validate_result(r,args.kind)


if __name__=='__main__':main()
