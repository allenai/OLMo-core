"""Native-checkpoint LC -> SFT pipeline, independent of slow HF export/evaluation."""

import copy
import fcntl
import json
import os
import time

from olmoe3_lr_sweep_watch import atomic_json, log, replace_env, status
from olmoe3_qkgain_control import (
    CPU_TEMPLATE,
    TRAIN_TEMPLATE,
    control,
    cpu_spec,
    ensure_saved,
    training_spec,
    verify_native,
)
from olmoe3_qkgain_eval import eval_specs, validate_result
from olmoe3_qkgain_plan import *


def items():
    """Only the six authorized descendants; never restart any PT or MT work."""
    return [Run(arm, stage) for stage in ('lc', 'sft') for arm in ARMS]


def watcher_spec(template, commit, gate):
    """A resource-free CPU watcher; the existing PT/MT controller remains untouched."""
    spec = cpu_spec(template, commit, 'watch')
    task = spec['tasks'][0]
    task['arguments'][-1] = task['arguments'][-1].replace(
        'olmoe3_qkgain_control.py watch', 'olmoe3_qkgain_posttrain_control.py watch'
    )
    replace_env(task, {'POSTTRAIN_CONFIG_GATE': gate})
    spec['description'] = 'Three LC4Mi/warmup400 -> SFT2Mi/2epoch chains; native dependencies; parallel exports/evals'
    assert task['constraints']['cluster'] == ['ai2/phobos'] and 'resources' not in task
    return spec


def validate_data():
    """Verify the actual cached packed data and batch-aware loaders before GPU admission."""
    from olmo_core.data import TokenizerConfig
    from olmoe3_qkgain_train import sft_adapter, sft_data_plan, validate

    validate()
    assert MOUNT.is_mount()
    for r in items():
        if r.stage != 'sft':
            continue
        plan = sft_data_plan(r)
        assert plan['steps_per_epoch'] == 420 and plan['total_steps'] == 840
        adapter = sft_adapter(r)
        dataset = adapter.dataset_config(TokenizerConfig.dolma2(), 'train').build()
        dataset.prepare()
        assert len(dataset) == plan['packed_instances']['train'] == 13445
        assert len(dataset) // (r.batch // r.sequence) == 420
        assert adapter.BATCH == r.batch and adapter.GPUS == r.gpus
        print('POSTTRAIN_PACKED_DATA_VERIFIED', r.run_id, json.dumps(plan), flush=True)


def watch():
    """Submit each stage once, fail closed on missing proofs, and do not gate SFT on evals."""
    from beaker import Beaker
    from olmo_checkpoint_uploader.backend import HuggingFaceBucketBackend
    from olmo_checkpoint_uploader.models import Registration
    from olmo_checkpoint_uploader.state import StateStore

    assert MOUNT.is_mount()
    commit = os.environ['GIT_REF']
    POSTTRAIN_AUTOMATION.mkdir(parents=True, exist_ok=True)
    with (POSTTRAIN_AUTOMATION / 'LOCK').open('a') as lock, Beaker.from_env(check_for_upgrades=False) as b:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        c = control(b, commit, POSTTRAIN_AUTOMATION)
        ec = control(b, commit, POSTTRAIN_AUTOMATION / 'evals', EVAL_WORKSPACE)
        template = b.experiment.get_spec(b.workload.get(TRAIN_TEMPLATE)).to_json()
        hosts = json.loads((AUTOMATION / 'hosts.json').read_text())
        backend = HuggingFaceBucketBackend()
        backend.assert_private(BUCKET)
        store = StateStore(CONTROL, STATE)
        selected = items()
        for r in selected:
            store.register(Registration(
                run_id=r.run_id, lineage_id=r.run_id, checkpoint_root=str(r.root),
                bucket_id=BUCKET, remote_prefix=r.prefix, deletion_mode='apply',
                min_local_checkpoints=2, delete_grace_seconds=3600,
            ))
        plan = dict(runs=[r.as_dict() for r in selected],
                    dependency='native checkpoint completion, not HF conversion/eval',
                    gate=os.environ['POSTTRAIN_CONFIG_GATE'])
        saved = POSTTRAIN_AUTOMATION / 'plan.json'
        if saved.exists():
            assert json.loads(saved.read_text()) == plan, 'Refuse post-training plan drift'
        else:
            atomic_json(saved, plan)
        atomic_json(POSTTRAIN_AUTOMATION / 'deployments' / f'{commit}.json', plan)
        cached_specs = {}
        previous = None
        while True:
            rows = {}
            gate = status(b.workload.get(os.environ['POSTTRAIN_CONFIG_GATE']))
            fs = os.statvfs(MOUNT)
            admit = fs.f_bavail * fs.f_frsize >= 10_000_000_000_000
            admit = admit and status(b.workload.get(UPLOADER)) == 'STATUS_RUNNING'
            # Complete the training submission pass before spending time on eval APIs.
            for r in selected:
                try:
                    if gate != 'STATUS_SUCCEEDED':
                        rows[r.run_id] = dict(waiting='configuration/data gate', gate_status=gate)
                        continue
                    parent = Run(r.arm, 'mt' if r.stage == 'lc' else 'lc')
                    ledger = AUTOMATION if r.stage == 'lc' else POSTTRAIN_AUTOMATION
                    p = ledger / 'submissions' / f'{parent.run_id}-train.json'
                    if not p.exists():
                        rows[r.run_id] = dict(waiting='parent native training')
                        continue
                    pw = b.workload.get(json.loads(p.read_text())['experiment_id'])
                    if status(pw) != 'STATUS_SUCCEEDED':
                        rows[r.run_id] = dict(waiting='parent native training', parent_status=status(pw))
                        continue
                    verify_native(parent)
                    assert (r.source / 'model_and_optim/.metadata').is_file()
                    if not admit:
                        rows[r.run_id] = dict(waiting='storage/uploader admission')
                        continue
                    w, state = ensure_saved(c, r.run_id + '-train',
                                            lambda r=r: training_spec(template, r, commit, hosts))
                    rows[r.run_id] = dict(status=state, id=w.experiment.id if w else None)
                    if state == 'STATUS_SUCCEEDED':
                        verify_native(r)
                except Exception as exc:
                    rows[r.run_id] = dict(state='needs_attention', error=f'{type(exc).__name__}: {exc}')
            for r in selected:
                row = rows[r.run_id]
                if row.get('status') != 'STATUS_SUCCEEDED':
                    continue
                try:
                    if r.run_id not in cached_specs:
                        cached_specs[r.run_id] = eval_specs(b, r, commit)
                    estate = {}
                    for kind, spec in cached_specs[r.run_id].items():
                        w, state = ensure_saved(ec, r.run_id + '-' + kind, lambda spec=spec: spec)
                        estate[kind] = dict(status=state, id=w.experiment.id if w else None)
                        if state == 'STATUS_SUCCEEDED':
                            validate_result(r, kind)
                        if kind == 'convert' and state != 'STATUS_SUCCEEDED':
                            break
                    row['evals'] = estate
                except Exception as exc:
                    row['eval_error'] = f'{type(exc).__name__}: {exc}'
            atomic_json(POSTTRAIN_AUTOMATION / 'status.json',
                        dict(updated_at=time.time(), config_status=gate, runs=rows))
            if rows != previous:
                log('QKGAIN_POSTTRAIN_STATUS', runs=rows)
                previous = rows
            if all(len(x.get('evals', {})) == 5 and all(
                y['status'] == 'STATUS_SUCCEEDED' for y in x['evals'].values()
            ) and not x.get('eval_error') for x in rows.values()):
                log('QKGAIN_POSTTRAIN_COMPLETE')
                return
            time.sleep(60)


if __name__ == '__main__':
    import sys

    if sys.argv[1:] == ['validate']:
        validate_data()
    elif sys.argv[1:] == ['watch']:
        watch()
    else:
        raise ValueError('Expected validate or watch')
