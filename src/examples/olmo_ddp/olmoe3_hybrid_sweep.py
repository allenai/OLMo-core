"""Matched 3:1 LR experiments: one job per LR, including its final 10% decay."""

import json
import os
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import ClassVar

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import the qualified hero first: it selects the exact optimized runtime policy.
import olmoe3_small_hero as hero
from olmoe3_hybrid_sweep_plan import (
    BATCH,
    BUCKET,
    CAMPAIGN,
    CONTROL,
    DATA_ROOT,
    DOLMA_MOUNT,
    MOUNT,
    WORKSPACE,
    find_run,
    runs,
    smoke_runs,
)
from olmoe3_lr_sweep_watch import atomic_json
from olmoe3_small_hybrid_3to1 import build_model_config

from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.internal.experiment import CliContext, SubCmd, build_config, main
from olmo_core.launch.beaker import BeakerWekaBucket
from olmo_core.optim.scheduler import WSD
from olmo_core.train import Duration
from olmo_core.train.callbacks import Callback
from olmo_core.train.callbacks.checkpoint_ready_notifier import (
    CheckpointReadyNotifierCallback,
)

qualified = hero.qualified


@dataclass
class Audit(qualified.IntegrationAudit):
    """Finite metrics, immutable saves and full-state same-topology restore audits."""

    priority: ClassVar[int] = 10
    run_id: str = ""

    def post_checkpoint_loaded(self, path):
        saved = json.loads((Path(path) / "resume_audit" / f"rank{get_rank()}.json").read_text())
        assert saved == hero.state_sample(self.trainer), "Resume changed sampled full state"
        atomic_json(
            Path(self.output_dir) / f"restore-step{self.step}-rank{get_rank()}.json",
            dict(source=str(path), step=self.step, sampled_state_exact=True),
        )

    def pre_train(self):
        r = find_run(self.run_id)
        reg = json.loads((CONTROL / "registrations" / f"{r.run_id}.json").read_text())
        assert reg["checkpoint_root"] == str(r.root) == str(self.trainer.save_folder)
        assert reg["run_id"] == reg["lineage_id"] == r.run_id
        assert reg["bucket_id"] == BUCKET and reg["remote_prefix"] == r.prefix
        assert reg["enabled"] and reg["deletion_mode"] == "apply"
        assert reg["min_local_checkpoints"] == 4 and reg["delete_grace_seconds"] >= 3600
        assert get_world_size() == 64 and MOUNT.is_mount() and DOLMA_MOUNT.is_mount()
        assert (
            self.trainer.global_train_tokens_seen
            == self.trainer.data_loader.tokens_processed
            == self.step * BATCH
        )
        assert 0 <= self.step <= r.end
        if r.smoke:
            assert self.step == int(os.environ["HYBRID_SMOKE_EXPECTED_START"])
        assert not (r.root / "STORAGE_PAUSED.json").exists()
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self._first_batch = True
        tm = self.trainer.train_module
        original = tm.save_state_dict_direct

        def guarded_save(directory, **kwargs):
            before = hero.state_sample(self.trainer)
            original(directory, **kwargs)
            after = hero.state_sample(self.trainer)
            assert before == after, "Synchronous save mutated live training state"
            atomic_json(Path(directory).parent / "resume_audit" / f"rank{get_rank()}.json", after)

        tm.save_state_dict_direct = guarded_save
        if get_rank() == 0:
            atomic_json(
                Path(self.output_dir)
                / f"session-{os.environ.get('BEAKER_JOB_ID')}-{self.step}.json",
                {
                    **r.as_dict(),
                    "start_step": self.step,
                    "source_commit": os.environ.get("GIT_REF"),
                    "optimization_settings": qualified.SETTINGS,
                },
            )


@dataclass
class StorageGuard(Callback):
    """Gracefully stop before exhausting the shared checkpoint mount."""

    run_id: str = ""

    def pre_train(self):
        self.post_step()

    def post_step(self):
        if get_rank() != 0 or self.step % 25:
            return
        fs = os.statvfs(MOUNT)
        free = fs.f_bavail * fs.f_frsize
        if free < 5_000_000_000_000:
            atomic_json(
                find_run(self.run_id).root / "STORAGE_PAUSED.json",
                dict(step=self.step, free_bytes=free),
            )
            self.trainer.cancel_run("Hybrid sweep: less than5TB free")


@dataclass
class Completion(Callback):
    """Publish completion only after final checkpoint callbacks finish."""

    priority: ClassVar[int] = -20
    run_id: str = ""

    def post_train(self):
        if get_rank() == 0:
            atomic_json(
                find_run(self.run_id).root / "audit" / f"completed-step{self.step}.json",
                dict(
                    step=self.step,
                    tokens=self.trainer.global_train_tokens_seen,
                    run_id=self.run_id,
                    source_commit=os.environ.get("GIT_REF"),
                ),
            )


def common_components(cli_context, **kwargs):
    """Keep initialization and data order, but use new checkpoint/run namespaces."""
    r = find_run(cli_context.run_name)
    common = qualified.common_components(cli_context, **kwargs)
    common.save_folder = str(r.root)
    if common.launch:
        common.launch.workspace = WORKSPACE
        common.launch.weka_buckets.append(BeakerWekaBucket("dolma-3p5", str(DOLMA_MOUNT)))
        common.launch.cmd = [
            "python",
            "src/examples/olmo_ddp/olmoe3_hybrid_sweep_node.py",
            r.run_id,
            cli_context.cluster,
        ]
    return common


def data_components(common):
    """Use the local mirror of the same token manifest as the earlier sweep."""
    data = qualified.base.build_data_components(common)
    data.dataset.mix_base_dir = str(DATA_ROOT)
    return data


def model_config(common):
    """Apply the exact qualified kernel policy before changing only hybridization."""
    reference = qualified.model_config(common)
    r = find_run(common.run_name)
    model = build_model_config(
        eos_token_id=common.tokenizer.eos_token_id,
        vocab_size=common.tokenizer.padded_vocab_size(),
        emo=r.emo,
    )
    assert reference.num_active_params == 794_233_472
    return model


def train_module_config(common):
    """End the same process with a linear decay; never launch a decay child."""
    r = find_run(common.run_name)
    config = qualified.train_module_config(common)
    config.optim.lr = r.lr
    config.scheduler = WSD(warmup=r.warmup, decay=r.decay, decay_fraction=None)
    return config


def trainer_config(common):
    """Save exactly the three future branch points and final10% checkpoint."""
    r = find_run(common.run_name)
    config = qualified.trainer_config(common)
    config.callbacks.pop("integration_audit")
    config.max_duration = Duration.steps(r.end)
    config.hard_stop = Duration.steps(int(os.environ["HYBRID_SMOKE_STOP"])) if r.smoke else None
    cp = config.callbacks["checkpointer"]
    cp.save_interval = None
    cp.fixed_steps = r.saves
    cp.pre_train_checkpoint = False
    cp.ephemeral_save_interval = None
    config.add_callback("audit", Audit(output_dir=str(r.root / "audit"), run_id=r.run_id))
    config.add_callback("storage", StorageGuard(run_id=r.run_id))
    config.add_callback("completion", Completion(run_id=r.run_id))
    config.add_callback(
        "checkpoint_ready",
        CheckpointReadyNotifierCallback(
            inbox_dir=str(CONTROL / "inbox"), run_id=r.run_id, lineage_id=r.run_id
        ),
    )
    if r.smoke:
        config.callbacks["lm_evaluator"].eval_interval = 4
        config.callbacks["lm_evaluator"].eval_duration = Duration.steps(1)
    wb = config.callbacks["wandb"]
    wb.group = CAMPAIGN
    wb.tags = [
        "hybrid3to1",
        r.arm,
        "small-64g",
        "16mi",
        "mb4",
        "qknorm-pr855",
        "decay10-in-process",
        f"lr:{r.lr:g}",
    ]
    wb.notes = json.dumps(r.as_dict())
    return config


def config_builder():
    """Use the qualified64GPU topology and data/eval recipes."""
    return partial(
        build_config,
        global_batch_size=BATCH,
        max_sequence_length=8192,
        num_nodes=8,
        common_config_builder=common_components,
        data_config_builder=data_components,
        model_config_builder=model_config,
        train_module_config_builder=train_module_config,
        trainer_config_builder=trainer_config,
        beaker_image=qualified.base.BEAKER_IMAGE,
        beaker_workspace=WORKSPACE,
        include_default_evals=False,
        num_execution_units=1,
    )


def validate():
    """Construct and check every config inside the production image without GPUs."""
    os.environ.setdefault("HYBRID_SMOKE_STOP", "6")
    for r in runs() + smoke_runs():
        c = config_builder()(CliContext(__file__, SubCmd.dry_run, r.run_id, "ai2/holmes", []))
        c.as_dict(json_safe=True)
        tm, tr = c.train_module, c.trainer
        assert c.model.num_active_params == 787_364_992 and c.model.num_params == 12_489_473_152
        assert c.data_loader.global_batch_size == BATCH and c.init_seed == 12536
        assert c.dataset.mix_base_dir == str(DATA_ROOT)
        assert tm.rank_microbatch_size == 4 * 8192 and tm.dp_config.use_reduce_scatter
        assert tm.ac_config is None and tm.float8_config is None
        assert tm.ep_config is None and tm.pp_config is None
        assert tr.load_optim_state and tr.load_trainer_state and not tr.save_overwrite
        cp = tr.callbacks["checkpointer"]
        assert cp.save_async is False and cp.max_checkpoints is None
        assert (
            cp.pre_train_checkpoint is False
            and cp.save_interval is None
            and cp.fixed_steps == r.saves
        )
        assert tm.scheduler.get_lr(r.lr, r.warmup, r.end) == r.lr
        assert tm.scheduler.get_lr(r.lr, r.end - r.decay, r.end) == r.lr
        assert tm.scheduler.get_lr(r.lr, r.end, r.end) == 0
        assert abs(tm.scheduler.get_lr(r.lr, r.end - r.decay // 2, r.end) / r.lr - 0.5) < 1e-10
        for b in [c.model.block, *c.model.block_overrides.values()]:
            if b.routed_experts_router is not None:
                assert (b.routed_experts_router.emo is not None) == r.emo
        print("HYBRID_CONFIG_VALIDATED", json.dumps(r.as_dict()), flush=True)


if __name__ == "__main__":
    qualified.apply_policy()
    if sys.argv[1:] == ["--validate-only"]:
        validate()
    else:
        main(config_builder=config_builder())
