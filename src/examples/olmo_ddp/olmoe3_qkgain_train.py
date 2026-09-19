"""One config adapter for the approved model/stage matrix; qualified kernels unchanged."""

import copy
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import ClassVar

import olmoe3_small_hero as hero
import torch
from olmoe3_lr_sweep_watch import atomic_json
from olmoe3_qkgain_plan import *
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.internal.experiment import CliContext, SubCmd, build_config, main
from olmo_core.optim.scheduler import WSD, LinearWithWarmup
from olmo_core.train import Duration
from olmo_core.train.callbacks import Callback
from olmo_core.train.callbacks.checkpoint_ready_notifier import CheckpointReadyNotifierCallback
from olmo_core.train.common import LoadStrategy
from olmo_core.train.utils import EnvRngStates


def current():
    return find_run(os.environ["QKGAIN_RUN"])


def scheduler(r):
    return WSD(warmup=2000, decay=667, decay_fraction=None) if r.stage == "pt" else (
        LinearWithWarmup(warmup_fraction=.03, alpha_f=0) if r.stage == "sft" else
        LinearWithWarmup(warmup=2000, alpha_f=0))


def source_for(r):
    return Path(os.environ["QKGAIN_LOAD"]) if os.environ.get("QKGAIN_LOAD") else r.source


def common_components(ctx, **kwargs):
    r = find_run(ctx.run_name)
    common = hero.qualified.common_components(ctx, **kwargs)
    common.save_folder = str(r.root)
    if r.stage != "pt":
        common.work_dir = str(ROOT / "data-work" / r.stage)
    return common


def model_config(common):
    r = find_run(common.run_name)
    model = hero.qualified.model_config(common)
    if r.arm.startswith("3to1"):
        for layer in (3, 11):
            model.block_overrides[layer] = copy.deepcopy(model.block_overrides[7])
    for block in [model.block, *model.block_overrides.values()]:
        mixer = block.sequence_mixer
        if hasattr(mixer, "qk_norm_per_head_gains"):
            mixer.qk_norm_per_head_gains = r.split
        router = getattr(block, "routed_experts_router", None)
        if router is not None and not r.emo:
            router.emo = None
        if r.stage == "sft" and hasattr(mixer, "use_cute_kernel"):
            mixer.use_cute_kernel = False
    if r.stage in ("lc", "sft"):
        model.recompute_each_block = True
        model.recompute_all_blocks_by_chunk = False
    assert not model.two_batch_overlap
    model.validate()
    return model


def sft_adapter():
    import olmoe3_hero_sft as sft
    sft.DATA, sft.CACHE = SFT_DATA, SFT_CACHE
    sft.ROOT_WORK = ROOT / "data-work/sft"
    sft.find_run, sft.GPUS = find_run, 8
    sft.data_plan = lambda: json.loads(SFT_DATA_PLAN.read_text())
    return sft


def data_components(common):
    r = find_run(common.run_name)
    if r.stage == "pt":
        return hero.data_components(common)
    if r.stage == "mt":
        import olmoe3_hero_mt as mt
        mt.BATCH = r.batch
        mt.REQUESTED_TOKENS = 35641421562
        return mt.data_components(common)
    if r.stage == "lc":
        import olmoe3_hero_lc as lc
        lc.BATCH = r.batch
        lc.DATA_WORK = ROOT / "data-work/lc"
        return lc.data_components(common)
    return sft_adapter().data_components(common)


def train_module_config(common):
    r = find_run(common.run_name)
    tm = hero.qualified.train_module_config(common)
    tm.rank_microbatch_size = r.microbatch
    tm.optim.lr = r.lr
    tm.scheduler = scheduler(r)
    tm.reset_optimizer_states_on_load = r.stage != "pt" and source_for(r) == r.source
    if r.stage == "sft":
        tm.optim.weight_decay = 0.0
        tm.z_loss_multiplier = None
        tm.compile_model = False
    return tm


def equal(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b,(list,tuple)) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    if torch.is_tensor(a):
        return torch.equal(a.cpu(),b.cpu())
    if hasattr(a,"shape"):
        return bool((a==b).all())
    return a==b


@dataclass
class Audit(Callback):
    """Cross-topology model transfer and same-topology exact resume; reject nonfinite training."""
    priority: ClassVar[int] = 10
    run_id: str = ""

    def post_checkpoint_loaded(self, path):
        r=find_run(self.run_id); actual=hero.state_sample(self.trainer)
        fresh = r.stage != "pt" and Path(path)==r.source
        if fresh:
            saved=json.loads((Path(path)/"resume_audit/rank0.json").read_text())
            keys={k for k in actual["tensors"] if k.startswith("model_param/")}
            keys.update(self.trainer.train_module._persistent_model_buffer_state_dict())
            assert keys and all(actual["tensors"][k]==saved["tensors"][k] for k in keys)
            assert self.step==self.trainer.global_train_tokens_seen==self.trainer.data_loader.tokens_processed==0
            optim=self.trainer.train_module.optim
            assert not optim._losses and not optim._grad_norms
            checked=0
            for name,t in optim.states.items():
                if name.endswith((".exp_avg",".exp_avg_sq",".step")):
                    t=t.to_local() if hasattr(t,"to_local") else t
                    assert not torch.count_nonzero(t).item(),name
                    checked+=1
            assert checked
        else:
            saved=json.loads((Path(path)/"resume_audit"/f"rank{get_rank()}.json").read_text())
            assert equal(saved,actual),"Full-state resume samples changed"
            state=torch.load(Path(path)/"train"/f"rank{get_rank()}.pt",map_location="cpu",weights_only=False)
            assert equal(state["rng"],EnvRngStates.current_state().as_dict())
            assert equal(state["data_loader"],self.trainer.data_loader.state_dict())
        atomic_json(r.root/"audit"/f"restore-{self.step}-rank{get_rank()}.json",
                    dict(passed=True,fresh_stage=fresh,source=str(path)))

    def pre_train(self):
        r=find_run(self.run_id)
        assert MOUNT.is_mount() and get_world_size()==r.gpus
        assert self.trainer.global_train_tokens_seen==self.trainer.data_loader.tokens_processed==self.step*r.batch
        assert self.step==int(os.environ["QKGAIN_START"])
        reg=json.loads((CONTROL/"registrations"/f"{r.run_id}.json").read_text())
        assert reg['enabled'] and reg['checkpoint_root']==str(r.root) and reg['remote_prefix']==r.prefix
        assert reg['bucket_id']==BUCKET and reg['min_local_checkpoints']==2
        assert reg['deletion_mode']=='apply' and reg['delete_grace_seconds']>=3600
        assert not (r.root/"STORAGE_PAUSED.json").exists()
        (r.root/"audit").mkdir(parents=True,exist_ok=True)
        self.first=True
        tm=self.trainer.train_module; original=tm.save_state_dict_direct
        def save(directory,**kw):
            before=hero.state_sample(self.trainer)
            original(directory,**kw)
            after=hero.state_sample(self.trainer)
            assert equal(before,after),"Synchronous checkpoint mutated live state"
            atomic_json(Path(directory).parent/"resume_audit"/f"rank{get_rank()}.json",after)
        tm.save_state_dict_direct=save
        atomic_json(r.root/"audit"/f"session-{self.step}-rank{get_rank()}.json",
                    dict(passed=True,run=r.as_dict(),source_commit=os.environ.get('GIT_REF')))
        if self.step==0:
            atomic_json(r.root/"audit"/f"initial-rank{get_rank()}.json",hero.state_sample(self.trainer))

    def pre_step(self,batch):
        if not self.first:return
        r=find_run(self.run_id)
        row=dict(step=self.step,input_sha256=hashlib.sha256(batch['input_ids'].cpu().numpy().tobytes()).hexdigest())
        if r.stage=='sft':
            plan=sft_adapter().data_plan()
            assert plan['passed'] and plan['steps_per_epoch']==1680 and plan['total_steps']==3360
            assert self.trainer.data_loader.total_batches==1680
            mask=batch['label_mask']; ids=batch['input_ids']
            assert 'doc_lens' in batch and mask.dtype==torch.bool and mask.any() and (~mask).any()
            assert not mask[ids==100277].any() and not mask[:,0].any()
            row['supervised_tokens']=int(mask.sum())
        atomic_json(r.root/'audit'/f'batch-{self.step}-rank{get_rank()}.json',row)
        self.first=False

    def log_metrics(self,step,metrics):
        r=find_run(self.run_id)
        for k,v in metrics.items():
            if k in ('train/CE loss','optim/total grad norm'):
                assert math.isfinite(float(v)),(step,k,v)
            if k.startswith('optim/LR ('):
                assert math.isclose(float(v),scheduler(r).get_lr(r.lr,step,r.end),rel_tol=1e-6,abs_tol=1e-10)
        if get_rank()==0:
            with (r.root/'audit/metrics.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,**metrics))+'\n')


@dataclass
class Finish(Callback):
    priority: ClassVar[int] = -20
    run_id: str = ''
    def post_train(self):
        r=find_run(self.run_id)
        assert self.step==int(os.environ['QKGAIN_STOP']), 'Stopped early; no downstream release'
        if get_rank()==0:
            atomic_json(r.root/'audit'/f'complete-{self.step}.json',dict(passed=True,step=self.step,tokens=self.step*r.batch))


def trainer_config(common):
    r=find_run(common.run_name)
    cfg=hero.qualified.trainer_config(common)
    cfg.callbacks.pop('integration_audit',None)
    cfg.max_duration=Duration.steps(r.end)
    cfg.hard_stop=Duration.steps(int(os.environ.get('QKGAIN_STOP',str(r.end))))
    load=source_for(r)
    cfg.load_path=str(load) if load else None
    cfg.load_strategy=LoadStrategy.always if load else LoadStrategy.never
    fresh=r.stage!='pt' and load==r.source
    cfg.load_optim_state=cfg.load_trainer_state=not fresh
    cp=cfg.callbacks['checkpointer']
    cp.save_interval=None; cp.fixed_steps=r.saves; cp.ephemeral_save_interval=None
    cp.pre_train_checkpoint=r.start==0 and load!=r.root/'step0'
    cp.save_async=False; cp.max_checkpoints=None
    cfg.callbacks['qkgain_audit']=Audit(run_id=r.run_id)
    # The existing guard only needs the campaign run resolver.
    hero.find_run=find_run
    cfg.callbacks['storage']=hero.StorageGuard(run_id=r.run_id)
    cfg.callbacks['finish']=Finish(run_id=r.run_id)
    cfg.callbacks['checkpoint_ready']=CheckpointReadyNotifierCallback(inbox_dir=str(CONTROL/'inbox'),run_id=r.run_id,lineage_id=r.run_id)
    if r.stage=='sft':
        cfg.callbacks.pop('lm_evaluator',None)
        cfg.callbacks['sft_validation']=sft_adapter().SFTValidation(run_id=r.run_id)
    if r.smoke:
        cfg.no_evals=True; cfg.metrics_collect_interval=1
    wb=cfg.callbacks['wandb']; wb.project='olmo3p5-hero'; wb.group=CAMPAIGN
    wb.tags=[r.arm,r.stage,'emo-on' if r.emo else 'emo-off','smoke' if r.smoke else 'production']
    wb.notes=json.dumps(r.as_dict())
    return cfg


def builder(r):
    return partial(build_config,global_batch_size=r.batch,max_sequence_length=r.sequence,
        num_nodes=r.nodes,common_config_builder=common_components,data_config_builder=data_components,
        model_config_builder=model_config,train_module_config_builder=train_module_config,
        trainer_config_builder=trainer_config,beaker_image=hero.qualified.base.BEAKER_IMAGE,
        beaker_workspace=WORKSPACE,include_default_evals=False,num_execution_units=1)


def validate():
    self_test()
    for r in runs()+runs(True):
        os.environ.update(QKGAIN_RUN=r.run_id,QKGAIN_START=str(r.start),QKGAIN_STOP=str(r.end))
        c=builder(r)(CliContext(__file__,SubCmd.dry_run,r.run_id,'ai2/holmes',[]))
        c.as_dict(json_safe=True)
        expected=794233472 if r.arm=='7to1-split' else 787364992-(0 if r.split else 5120)
        assert c.model.num_active_params==expected,(r.run_id,c.model.num_active_params)
        assert c.data_loader.global_batch_size==r.batch and c.train_module.rank_microbatch_size==r.microbatch
        assert c.trainer.max_duration.value==r.end and not c.trainer.callbacks['checkpointer'].save_async
        assert c.train_module.scheduler.get_lr(r.lr,r.end,r.end)==0
        assert c.train_module.ep_config is None and c.train_module.pp_config is None
        for block in [c.model.block,*c.model.block_overrides.values()]:
            router=getattr(block,'routed_experts_router',None)
            assert router is None or (router.emo is not None)==r.emo
        print('QKGAIN_CONFIG_OK',json.dumps(r.as_dict()),flush=True)


if __name__=='__main__':
    hero.qualified.apply_policy()
    if sys.argv[1:]==['--validate']:
        validate()
    else:
        main(config_builder=builder(current()))
