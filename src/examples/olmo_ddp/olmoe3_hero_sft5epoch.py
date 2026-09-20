"""Matched 4T LC -> five-epoch, 8Mi SFT LR sweep; EMO disabled for both PT lineages."""

import fcntl
import hashlib
import json
import os
import runpy
import sys
import time
from dataclasses import dataclass

import olmoe3_qkgain_plan as plan
from olmoe3_lr_sweep_watch import atomic_json, log, replace_env, status

CAMPAIGN = "olmo35-4t-lc-sft8mi-5epoch-20260920"
BRANCH = "codex/hero-4t-sft5epoch-20260920"
AUTO = plan.MOUNT / "uploader/automation" / CAMPAIGN
ROOT = plan.MOUNT / "production-hero-small-sft" / CAMPAIGN
EVAL = plan.MOUNT / "scratch" / CAMPAIGN
ARCHIVE = AUTO / "sources"
SCRIPT = "src/examples/olmo_ddp/olmoe3_hero_sft5epoch.py"
LRS = {"1em4": 1e-4, "2em4": 2e-4, "4em4": 4e-4}
LINEAGES = ("emo", "non-emo")
LC_CAMPAIGN = "olmo35-small-4t-lc100b-noemo-20260916"
REUSED_EMO_SOURCE = (
    plan.MOUNT
    / "uploader/automation/olmo35-4t-emo-lc-sft8mi-20260919/sources"
    / "posttrain-noemo-4t-20260916/lc100b/emo/step5961/olmo-core"
)


@dataclass(frozen=True)
class SweepRun(plan.Run):
    """An explicitly bounded source/LR/epoch trial, independent of existing two-epoch jobs."""

    lineage: str = "emo"
    lr_label: str = "2em4"
    epochs: int = 5

    def __post_init__(self):
        super().__post_init__()
        assert self.arm == "7to1-split" and self.stage == "sft"
        assert self.lineage in LINEAGES and self.lr_label in LRS and self.epochs in (2, 5)

    @property
    def run_id(self):
        return f"{CAMPAIGN}-{self.lineage}-lr{self.lr_label}-ep{self.epochs}" + (
            "-smoke" if self.smoke else ""
        )

    @property
    def root(self):
        return ROOT / self.run_id

    @property
    def prefix(self):
        return f"{CAMPAIGN}/{self.lineage}/lr{self.lr_label}/ep{self.epochs}" + (
            "/smoke" if self.smoke else ""
        )

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
        return 105 * self.epochs

    @property
    def lr(self):
        return LRS[self.lr_label]

    @property
    def source(self):
        if self.lineage == "emo":
            return REUSED_EMO_SOURCE
        return ARCHIVE / source_prefix(self.lineage) / "step5961/olmo-core"

    @property
    def hf(self):
        # The frozen export CLI names this slot emo; run_id carries the true lineage.
        return EVAL / self.run_id / "emo" / f"step{self.end}/hf"

    @property
    def saves(self):
        return [2, 4] if self.smoke else [2, 4] + [105 * n for n in range(1, self.epochs + 1)]

    def as_dict(self):
        value = super().as_dict()
        value.update(
            pt_lineage=self.lineage,
            pretrain_emo=self.lineage == "emo",
            midtrain_emo=False,
            long_context_emo=False,
            sft_emo=False,
            source_lineage=LC_CAMPAIGN + "-" + self.lineage,
            source_lc_step=5961,
            source_lc_batch=16_777_216,
            epochs=self.epochs,
            steps_per_epoch=105,
            data=str(plan.SFT_DATA),
            schedule="linear",
            warmup_fraction=0.03,
            seed=1729,
            eval_temperature=0.6,
            eval_final_only=True,
        )
        return value


def source_prefix(lineage):
    """Exact previously verified 4T LC prefix; no newer LC recipe substitution."""
    assert lineage in LINEAGES
    return f"posttrain-noemo-4t-20260916/lc100b/{lineage}"


def runs(smoke=False):
    """Six five-epoch trials plus the missing two-epoch control; reuse existing EMO control."""
    return [SweepRun("7to1-split", "sft", smoke, "non-emo", "2em4", 2)] + [
        SweepRun("7to1-split", "sft", smoke, lineage, label, 5)
        for label in ("2em4", "1em4", "4em4")
        for lineage in LINEAGES
    ]


def find_run(name):
    """Reject arbitrary run names or sources."""
    return next(r for r in runs() + runs(True) if r.run_id == name)


def self_test():
    """Validate topology, epoch boundaries and retention points without mounted data."""
    selected = runs()
    assert len(selected) == 7 and len({r.run_id for r in selected}) == 7
    assert sum(r.epochs == 5 for r in selected) == 6
    assert selected[0].lineage == "non-emo" and selected[0].epochs == 2
    for r in selected:
        assert not r.emo and r.split and r.start == 0
        assert r.batch % (r.gpus * r.microbatch) == 0
        assert r.batch // r.sequence == 128 and 13445 // 128 == 105
        assert r.end == 105 * r.epochs and r.end * r.batch == 880_803_840 * r.epochs
        assert r.saves == [2, 4] + list(range(105, r.end + 1, 105))


def install_plan():
    """Bind the qualified shared implementation only inside this campaign process."""
    plan.CAMPAIGN, plan.ROOT, plan.EVAL_ROOT, plan.BRANCH = CAMPAIGN, ROOT, EVAL, BRANCH
    plan.runs, plan.find_run, plan.self_test = runs, find_run, self_test
    self_test()


def train():
    """Use import-qualified callback types, including through config roundtrips."""
    import olmoe3_qkgain_train as adapter

    adapter.hero.qualified.apply_policy()
    adapter.main(config_builder=adapter.builder(adapter.current()))


def validate():
    """Execute every actual training dispatch without launching GPUs or writing checkpoints."""
    from unittest.mock import patch

    import olmoe3_qkgain_train as adapter
    from olmo_core.internal.experiment import CliContext, SubCmd
    from olmo_core.data import TokenizerConfig

    self_test()
    assert plan.MOUNT.is_mount()
    proofs = []
    for r in runs():
        os.environ.update(QKGAIN_RUN=r.run_id, QKGAIN_START="0", QKGAIN_STOP="2")
        os.environ.pop("QKGAIN_LOAD", None)

        def dry_main(*, config_builder):
            c = config_builder(CliContext(SCRIPT, SubCmd.dry_run, r.run_id, "ai2/holmes", []))
            c = c.merge([])
            assert c.model.num_active_params == 794233472
            assert c.model.recompute_each_block and not c.model.two_batch_overlap
            assert c.data_loader.global_batch_size == r.batch and c.data_loader.seed == 1729
            assert c.train_module.rank_microbatch_size == 65536
            assert c.train_module.optim.lr == r.lr and c.train_module.optim.weight_decay == 0
            assert not c.train_module.compile_model and c.train_module.z_loss_multiplier is None
            assert c.trainer.max_duration.value == r.end and c.trainer.hard_stop.value == 2
            assert c.trainer.load_path == str(r.source)
            assert not c.trainer.load_optim_state and not c.trainer.load_trainer_state
            assert c.trainer.callbacks["checkpointer"].fixed_steps == r.saves
            assert not c.trainer.callbacks["checkpointer"].save_async
            for key in ("qkgain_audit", "sft_validation", "finish"):
                assert c.trainer.callbacks[key].run_id == r.run_id
            for block in [c.model.block, *c.model.block_overrides.values()]:
                router = getattr(block, "routed_experts_router", None)
                assert router is None or router.emo is None
                mixer = block.sequence_mixer
                if hasattr(mixer, "qk_norm_per_head_gains"):
                    assert mixer.qk_norm_per_head_gains
                if hasattr(mixer, "use_cute_kernel"):
                    assert not mixer.use_cute_kernel
            p = adapter.sft_data_plan(r)
            assert p["steps_per_epoch"] == 105 and p["total_steps"] == r.end
            assert p["dropped_packed_instances_per_epoch"] == 5
            assert c.train_module.scheduler.get_lr(r.lr, r.end, r.end) == 0
            proofs.append(r.as_dict())
            print("FIVE_EPOCH_SFT_ENTRYPOINT_VERIFIED", r.run_id, flush=True)

        with patch.object(adapter, "main", dry_main):
            train()
    data = adapter.sft_adapter(r).dataset_config(TokenizerConfig.dolma2(), "train").build()
    data.prepare()
    assert len(data) == 13445
    atomic_json(
        AUTO / "config-proof.json",
        dict(passed=True, runs=proofs, source_commit=os.environ["GIT_REF"]),
    )


def download():
    """Reuse restored EMO weights; restore and verify only the missing non-EMO LC source."""
    import logging

    from huggingface_hub import HfApi
    import olmoe3_hero_bucket_download as downloader

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    assert plan.MOUNT.is_mount()
    api = HfApi()
    assert api.bucket_info(plan.BUCKET).private
    downloader.SCRATCH = ARCHIVE
    downloader.prepare_scratch()
    for lineage in LINEAGES:
        r = next(r for r in runs() if r.lineage == lineage)
        p = r.source
        if lineage == "non-emo":
            p = downloader.download(
                api, source_prefix(lineage), 5961, lineage_id=LC_CAMPAIGN + "-" + lineage
            )
        assert p == r.source
        plan.validate_checkpoint(p, 5961, 16_777_216, 64)
        atomic_json(
            AUTO / f"source-{lineage}.json",
            dict(
                passed=True,
                source=str(p),
                lineage=LC_CAMPAIGN + "-" + lineage,
                metadata_sha256=hashlib.sha256((p / ".metadata.json").read_bytes()).hexdigest(),
            ),
        )
        print("FIVE_EPOCH_SOURCE_VERIFIED", lineage, str(p), flush=True)


def worker_spec(template, r, commit, hosts):
    """Use the exact qualified 64-GPU runtime and allocation policy."""
    from olmoe3_qkgain_control import training_spec

    spec = training_spec(template, r, commit, hosts)
    t = spec["tasks"][0]
    t["arguments"] = ["python", SCRIPT, "node", r.run_id]
    replace_env(t, dict(GIT_BRANCH=BRANCH, QKGAIN_TRAIN_SCRIPT=SCRIPT))
    return spec


def service_spec(template, commit, role, gate=None, restore=None):
    """Gate/restore on Weka-capable Rhea; resource-free watcher on Phobos."""
    from olmoe3_qkgain_control import cpu_spec

    mode = {"validate": "config", "download": "download", "watch": "watch"}[role]
    spec = cpu_spec(template, commit, mode, template if mode == "config" else None)
    t = spec["tasks"][0]
    if role == "validate":
        t["arguments"] = ["python", SCRIPT, "validate"]
    else:
        t["arguments"][-1] = t["arguments"][-1].replace(
            f"olmoe3_qkgain_control.py {mode}", f"olmoe3_hero_sft5epoch.py {role}"
        )
    replace_env(t, dict(GIT_BRANCH=BRANCH, SFT5_CONFIG_GATE=gate, SFT5_RESTORE=restore))
    if role == "download":
        t["resources"] = dict(gpuCount=1, cpuCount=8, memory="64 GiB", sharedMemory="8 GiB")
        t["context"].update(priority="urgent", minRuntime="1h", autoResume=True)
        t["timeout"] = "4h"
    spec["description"] = f"{CAMPAIGN}: {role}; fixed source pair; no Beaker results payloads"
    return spec


def watch():
    """Durable once-only submissions followed by verified final conversions and four evals."""
    from beaker import Beaker
    from olmo_checkpoint_uploader.backend import HuggingFaceBucketBackend
    from olmo_checkpoint_uploader.models import Registration
    from olmo_checkpoint_uploader.state import StateStore
    from olmoe3_qkgain_control import (
        TRAIN_TEMPLATE,
        control,
        ensure_saved,
        verify_native,
    )
    from olmoe3_qkgain_eval import eval_specs, validate_result

    assert plan.MOUNT.is_mount()
    commit = os.environ["GIT_REF"]
    AUTO.mkdir(parents=True, exist_ok=True)
    with (AUTO / "LOCK").open("a") as lock, Beaker.from_env(check_for_upgrades=False) as b:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        c, ec = control(b, commit, AUTO), control(b, commit, AUTO / "evals", plan.EVAL_WORKSPACE)
        backend = HuggingFaceBucketBackend()
        backend.assert_private(plan.BUCKET)
        store = StateStore(plan.CONTROL, plan.STATE)
        selected = runs()
        for r in selected:
            store.register(
                Registration(
                    run_id=r.run_id,
                    lineage_id=r.run_id,
                    checkpoint_root=str(r.root),
                    bucket_id=r.bucket,
                    remote_prefix=r.prefix,
                    deletion_mode="apply",
                    min_local_checkpoints=2,
                    delete_grace_seconds=3600,
                )
            )
        payload = dict(
            runs=[r.as_dict() for r in selected], config_gate=os.environ["SFT5_CONFIG_GATE"]
        )
        saved = AUTO / "plan.json"
        if saved.exists():
            assert json.loads(saved.read_text()) == payload, "Refuse campaign drift"
        else:
            atomic_json(saved, payload)
        template = b.experiment.get_spec(b.workload.get(TRAIN_TEMPLATE)).to_json()
        hosts = json.loads((plan.AUTOMATION / "hosts.json").read_text())
        cached, previous = {}, None
        while True:
            gate = status(b.workload.get(os.environ["SFT5_CONFIG_GATE"]))
            fs = os.statvfs(plan.MOUNT)
            admitted = fs.f_bavail * fs.f_frsize >= 10_000_000_000_000
            admitted = admitted and status(b.workload.get(plan.UPLOADER)) == "STATUS_RUNNING"
            rows = {}
            for r in selected:
                try:
                    if gate != "STATUS_SUCCEEDED":
                        rows[r.run_id] = dict(waiting="configuration gate", status=gate)
                        continue
                    proof_path = AUTO / f"source-{r.lineage}.json"
                    if not proof_path.is_file():
                        rows[r.run_id] = dict(waiting="verified native LC restore")
                        continue
                    proof = json.loads(proof_path.read_text())
                    assert proof["passed"] and proof["source"] == str(r.source)
                    assert (
                        proof["metadata_sha256"]
                        == hashlib.sha256((r.source / ".metadata.json").read_bytes()).hexdigest()
                    )
                    if not admitted:
                        rows[r.run_id] = dict(waiting="storage/uploader admission")
                        continue
                    w, state = ensure_saved(
                        c, r.run_id + "-train", lambda r=r: worker_spec(template, r, commit, hosts)
                    )
                    rows[r.run_id] = dict(status=state, id=w.experiment.id if w else None)
                    if state == "STATUS_SUCCEEDED":
                        verify_native(r)
                except Exception as exc:
                    rows[r.run_id] = dict(error=f"{type(exc).__name__}: {exc}")
            for r in selected:
                row = rows[r.run_id]
                if row.get("status") != "STATUS_SUCCEEDED":
                    continue
                try:
                    if r.run_id not in cached:
                        exports = eval_specs(b, r, commit)
                        for spec in exports.values():
                            task = spec["tasks"][0]
                            task["arguments"][0] = task["arguments"][0].replace(
                                "olmoe3_qkgain_eval.py ", "olmoe3_hero_sft5epoch.py eval "
                            )
                        cached[r.run_id] = exports
                    row["evals"] = {}
                    for kind, spec in cached[r.run_id].items():
                        ew, es = ensure_saved(ec, r.run_id + "-" + kind, lambda spec=spec: spec)
                        row["evals"][kind] = dict(status=es, id=ew.experiment.id if ew else None)
                        if es == "STATUS_SUCCEEDED":
                            validate_result(r, kind)
                        if kind == "convert" and es != "STATUS_SUCCEEDED":
                            break
                except Exception as exc:
                    row["eval_error"] = f"{type(exc).__name__}: {exc}"
            atomic_json(AUTO / "status.json", dict(updated_at=time.time(), runs=rows))
            if rows != previous:
                log("FIVE_EPOCH_SFT_STATUS", runs=rows)
                previous = rows
            if all(
                len(row.get("evals", {})) == 5
                and not row.get("eval_error")
                and all(e["status"] == "STATUS_SUCCEEDED" for e in row["evals"].values())
                for row in rows.values()
            ):
                return
            time.sleep(60)


if __name__ == "__main__":
    install_plan()
    mode = sys.argv[1]
    if mode == "train":
        train()
    elif mode in ("node", "eval"):
        sys.argv.pop(1)
        runpy.run_module("olmoe3_qkgain_" + mode, run_name="__main__")
    elif mode == "validate":
        validate()
    elif mode == "download":
        download()
    elif mode == "watch":
        watch()
    elif mode == "self-test":
        self_test()
        print("FIVE_EPOCH_PLAN_VERIFIED")
    else:
        raise ValueError(mode)
