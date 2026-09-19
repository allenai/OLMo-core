"""One 4T EMO-PT / no-EMO-MT-LC descendant: 8Mi SFT, sqrt-scaled LR, two epochs."""

import copy
import fcntl
import hashlib
import json
import os
import runpy
import sys
import time
from pathlib import Path

import olmoe3_qkgain_plan as plan
from olmoe3_lr_sweep_watch import atomic_json, log, replace_env, status

CAMPAIGN = 'olmo35-4t-emo-lc-sft8mi-20260919'
RUN_ID = CAMPAIGN + '-lr2em4-ep2'
AUTO = plan.MOUNT / 'uploader/automation' / CAMPAIGN
ROOT = plan.MOUNT / 'production-hero-small-sft' / CAMPAIGN
EVAL = plan.MOUNT / 'scratch' / CAMPAIGN
ARCHIVE = AUTO / 'sources'
SOURCE_PREFIX = 'posttrain-noemo-4t-20260916/lc100b/emo'
SOURCE_RUN = 'olmo35-small-4t-lc100b-noemo-20260916-emo'
SOURCE = ARCHIVE / SOURCE_PREFIX / 'step5961/olmo-core'
CHAIN_AUTOMATION = plan.POSTTRAIN_AUTOMATION
SCRIPT = 'src/examples/olmo_ddp/olmoe3_hero_sft8mi.py'


class HeroRun(plan.Run):
    """Preserve the original 7:1 split-gain model; change only the SFT recipe."""

    @property
    def run_id(self):
        return RUN_ID + ('-smoke' if self.smoke else '')

    @property
    def root(self):
        return ROOT / self.run_id

    @property
    def prefix(self):
        return f'{CAMPAIGN}/{self.run_id}'

    @property
    def batch(self):
        return 8_388_608

    @property
    def gpus(self):
        return 64

    @property
    def nodes(self):
        return 8

    @property
    def end(self):
        return 210

    @property
    def lr(self):
        return 2e-4

    @property
    def source(self):
        return SOURCE

    @property
    def hf(self):
        return EVAL / self.run_id / 'emo/step210/hf'

    def as_dict(self):
        result = super().as_dict()
        result.update(source_lineage=SOURCE_RUN, source_lc_step=5961,
                      source_lc_batch=16_777_216, source_lc_emo=False,
                      sft_epochs=2, sft_emo=False, lr_rule='5e-5 * sqrt(8Mi / 512Ki)')
        return result


def run(smoke=False):
    """Return the sole authorized model, never an arbitrary source."""
    return HeroRun('7to1-split', 'sft', smoke)


def install_plan():
    """Use the qualified shared trainer/exporter with a bounded campaign resolver."""
    plan.CAMPAIGN, plan.ROOT, plan.EVAL_ROOT = CAMPAIGN, ROOT, EVAL
    plan.runs = lambda smoke=False: [run(smoke)]

    def find(name):
        return next(r for r in [run(), run(True)] if r.run_id == name)

    def check():
        r = run()
        assert r.batch % (r.gpus * r.microbatch) == 0
        assert r.end * r.batch == 3360 * 524288
        assert r.end // 2 == 13445 // (r.batch // r.sequence)
        assert not r.emo and r.split and r.saves == [2, 4, 105, 210]

    plan.find_run, plan.self_test = find, check
    check()


def download():
    """Restore exactly the verified original LC checkpoint into new owned scratch."""
    import logging
    from huggingface_hub import HfApi
    import olmoe3_hero_bucket_download as downloader

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    downloader.SCRATCH = ARCHIVE
    downloader.prepare_scratch()
    api = HfApi()
    assert api.bucket_info(plan.BUCKET).private
    p = downloader.download(api, SOURCE_PREFIX, 5961, lineage_id=SOURCE_RUN)
    assert p == SOURCE
    plan.validate_checkpoint(p, 5961, 16_777_216, 64)
    atomic_json(AUTO / 'source-success.json', dict(passed=True, source=str(p),
                source_lineage=SOURCE_RUN, source_prefix=SOURCE_PREFIX, step=5961,
                metadata_sha256=hashlib.sha256((p / '.metadata.json').read_bytes()).hexdigest()))


def config_spec(template, commit):
    """Validate in the actual pinned training image, without consuming training GPUs."""
    from olmoe3_qkgain_control import cpu_spec
    spec = cpu_spec(template, commit, 'config', template)
    spec['tasks'][0]['arguments'] = ['python', SCRIPT, 'validate']
    spec['description'] = json.dumps(run().as_dict())
    return spec


def restore_spec(template, commit):
    """Use a bounded one-GPU Rhea helper for the archived checkpoint download."""
    from olmoe3_qkgain_control import cpu_spec
    spec = cpu_spec(template, commit, 'download')
    task = spec['tasks'][0]
    task['arguments'][-1] = task['arguments'][-1].replace(
        'olmoe3_qkgain_control.py download', 'olmoe3_hero_sft8mi.py download'
    )
    task['resources'] = dict(gpuCount=1, cpuCount=8, memory='64 GiB', sharedMemory='8 GiB')
    task['context'].update(priority='urgent', minRuntime='1h', autoResume=True)
    task['timeout'] = '4h'
    task['result'] = {'path': '/noop-results'}
    spec['description'] = 'Restore only original 4T EMO-ancestry LC step5961; verify HF bucket receipt and inventory'
    return spec


def watcher_spec(template, commit, gate, restore):
    """Resource-free durable automation for this one training/export/eval chain."""
    from olmoe3_qkgain_control import cpu_spec
    spec = cpu_spec(template, commit, 'watch')
    task = spec['tasks'][0]
    task['arguments'][-1] = task['arguments'][-1].replace(
        'olmoe3_qkgain_control.py watch', 'olmoe3_hero_sft8mi.py watch'
    )
    replace_env(task, {'HERO8MI_CONFIG_GATE': gate, 'HERO8MI_RESTORE': restore})
    return spec


def watch():
    """Queue after the three LC submissions, then automatically convert and evaluate."""
    from beaker import Beaker
    from olmo_checkpoint_uploader.models import Registration
    from olmo_checkpoint_uploader.state import StateStore
    from olmoe3_qkgain_control import TRAIN_TEMPLATE, control, ensure_saved, training_spec, verify_native
    from olmoe3_qkgain_eval import eval_specs, validate_result

    assert plan.MOUNT.is_mount()
    r, commit = run(), os.environ['GIT_REF']
    AUTO.mkdir(parents=True, exist_ok=True)
    with (AUTO / 'LOCK').open('a') as lock, Beaker.from_env(check_for_upgrades=False) as b:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        c = control(b, commit, AUTO)
        ec = control(b, commit, AUTO / 'evals', plan.EVAL_WORKSPACE)
        store = StateStore(plan.CONTROL, plan.STATE)
        store.register(Registration(run_id=r.run_id, lineage_id=r.run_id,
            checkpoint_root=str(r.root), bucket_id=r.bucket, remote_prefix=r.prefix,
            deletion_mode='apply', min_local_checkpoints=2, delete_grace_seconds=3600))
        template = b.experiment.get_spec(b.workload.get(TRAIN_TEMPLATE)).to_json()
        hosts = json.loads((plan.AUTOMATION / 'hosts.json').read_text())
        spec = training_spec(template, r, commit, hosts)
        task = spec['tasks'][0]
        task['arguments'] = ['python', SCRIPT, 'node', r.run_id]
        replace_env(task, {'QKGAIN_TRAIN_SCRIPT': SCRIPT})
        saved_plan = AUTO / 'plan.json'
        if saved_plan.exists():
            assert json.loads(saved_plan.read_text()) == r.as_dict()
        else:
            atomic_json(saved_plan, r.as_dict())
        previous, exports = None, None
        while True:
            row = {}
            try:
                gate = status(b.workload.get(os.environ['HERO8MI_CONFIG_GATE']))
                restored = status(b.workload.get(os.environ['HERO8MI_RESTORE']))
                chains_queued = all((CHAIN_AUTOMATION / 'submissions' /
                    f'olmo35-qkgain-20260919-{arm}-lc-4mi-w400-train.json').is_file()
                    for arm in plan.ARMS)
                fs = os.statvfs(plan.MOUNT)
                admitted = fs.f_bavail * fs.f_frsize >= 10_000_000_000_000
                admitted = admitted and status(b.workload.get(plan.UPLOADER)) == 'STATUS_RUNNING'
                if gate != 'STATUS_SUCCEEDED' or restored != 'STATUS_SUCCEEDED' or not chains_queued or not admitted:
                    row = dict(config=gate, restore=restored, chains_queued=chains_queued, admitted=admitted)
                else:
                    proof = json.loads((AUTO / 'source-success.json').read_text())
                    assert proof['passed'] and proof['source_lineage'] == SOURCE_RUN
                    assert proof['metadata_sha256'] == hashlib.sha256((SOURCE / '.metadata.json').read_bytes()).hexdigest()
                    w, state = ensure_saved(c, r.run_id + '-train', lambda: spec)
                    row = dict(status=state, id=w.experiment.id if w else None)
                    if state == 'STATUS_SUCCEEDED':
                        verify_native(r)
                        if exports is None:
                            exports = eval_specs(b, r, commit)
                            for value in exports.values():
                                t = value['tasks'][0]
                                t['arguments'][0] = t['arguments'][0].replace(
                                    'olmoe3_qkgain_eval.py ', 'olmoe3_hero_sft8mi.py eval '
                                )
                        row['evals'] = {}
                        for kind, value in exports.items():
                            ew, es = ensure_saved(ec, r.run_id + '-' + kind, lambda value=value: value)
                            row['evals'][kind] = dict(status=es, id=ew.experiment.id if ew else None)
                            if es == 'STATUS_SUCCEEDED':
                                validate_result(r, kind)
                            if kind == 'convert' and es != 'STATUS_SUCCEEDED':
                                break
            except Exception as exc:
                row['error'] = f'{type(exc).__name__}: {exc}'
            atomic_json(AUTO / 'status.json', dict(updated_at=time.time(), run=row))
            if row != previous:
                log('HERO_SFT8MI_STATUS', **row)
                previous = row
            if len(row.get('evals', {})) == 5 and not row.get('error') and all(
                e['status'] == 'STATUS_SUCCEEDED' for e in row['evals'].values()
            ):
                return
            time.sleep(60)


if __name__ == '__main__':
    install_plan()
    mode = sys.argv[1]
    if mode == 'download':
        download()
    elif mode == 'watch':
        watch()
    elif mode == 'validate':
        from olmoe3_qkgain_train import hero, sft_data_plan, validate
        hero.qualified.apply_policy()
        validate()
        p = sft_data_plan(run())
        assert p['steps_per_epoch'] == 105 and p['total_steps'] == 210
        print('HERO_SFT8MI_CONFIG_AND_EPOCHS_VERIFIED', json.dumps(run().as_dict()), flush=True)
    elif mode == 'node':
        sys.argv.pop(1)
        runpy.run_module('olmoe3_qkgain_node', run_name='__main__')
    elif mode == 'eval':
        sys.argv.pop(1)
        runpy.run_module('olmoe3_qkgain_eval', run_name='__main__')
    elif mode == 'train':
        runpy.run_module('olmoe3_qkgain_train', run_name='__main__')
    else:
        raise ValueError(mode)
