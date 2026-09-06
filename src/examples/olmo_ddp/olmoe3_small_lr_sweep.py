"""Qualified production small-model LR trunks and full-state decay branches."""

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.update(
    OLMOE3_INTEGRATION_ARM="optimized",
    OLMOE3_INTEGRATION_POLICY="core-docpool-top16-wgrad-rs",
    OLMOE3_INTEGRATION_BASELINE="optimized100b",
    OLMOE3_INTEGRATION_COMMUNICATION="none",
    OLMOE3_INTEGRATION_SMOKE="0",
)

import olmoe3_small_integration as qualified
import torch
import torch.distributed as dist
from olmoe3_lr_sweep_plan import (
    BATCH,
    CONTROL,
    SWEEP,
    WORKSPACE,
    find_run,
    runs,
    smoke_runs,
)

from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.internal.experiment import CliContext, SubCmd, build_config, main
from olmo_core.optim.scheduler import WSD, ConstantWithWarmup
from olmo_core.train import Duration
from olmo_core.train.callbacks.checkpoint_ready_notifier import (
    CheckpointReadyNotifierCallback,
)
from olmo_core.train.common import LoadStrategy


@dataclass
class SweepAudit(qualified.IntegrationAudit):
    """Audit identity, full-state recovery, retention, inputs and finite metrics."""

    run_id: str = ""

    def pre_train(self):
        r = find_run(self.run_id)
        reg = json.loads((CONTROL / "registrations" / f"{r.run_id}.json").read_text())
        assert reg["run_id"] == reg["lineage_id"] == r.run_id
        assert reg["checkpoint_root"] == str(r.root) == str(self.trainer.save_folder)
        assert reg["enabled"] and reg["deletion_mode"] == "apply"
        assert reg["min_local_checkpoints"] == r.keep and reg["delete_grace_seconds"] >= 3600
        assert r.start <= self.step <= r.end, (self.step, r.as_dict())
        if r.parent:
            assert self.trainer.checkpoint_loaded, "A decay must never start from scratch"
        output = Path(self.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self._first_batch = True
        if self.step == 0:
            digest = hashlib.sha256()
            for i, (name, p) in enumerate(self.trainer.train_module.model.named_parameters()):
                if i % get_world_size() == get_rank():
                    digest.update(name.encode())
                    digest.update(
                        p.detach()
                        .contiguous()
                        .reshape(-1)
                        .view(torch.uint8)
                        .cpu()
                        .numpy()
                        .tobytes()
                    )
            hashes = [None] * get_world_size()
            dist.all_gather_object(hashes, digest.hexdigest(), group=self.trainer.bookkeeping_pg)
            if get_rank() == 0:
                qualified.write_or_verify(
                    output / "initial-weights-sha256.json", json.dumps(hashes)
                )
        if get_rank() == 0:
            data = {
                **r.as_dict(),
                "resumed_step": self.step,
                "resumed_tokens": self.trainer.global_train_tokens_seen,
                "checkpoint_loaded": self.trainer.checkpoint_loaded,
                "source_commit": os.environ.get("GIT_REF"),
                "optimization_settings": qualified.SETTINGS,
                "registration": reg,
            }
            assert data["resumed_tokens"] == self.step * BATCH
            (
                output / f"session-{os.environ.get('BEAKER_JOB_ID', 'local')}-{self.step}.json"
            ).write_text(json.dumps(data, indent=2))
            print("LR_SWEEP_START", json.dumps(data), flush=True)

    def post_train(self):
        if get_rank() == 0:
            data = {
                "run_id": self.run_id,
                "step": self.step,
                "tokens": self.trainer.global_train_tokens_seen,
                "source_commit": os.environ.get("GIT_REF"),
            }
            (Path(self.output_dir) / f"completed-step{self.step}.json").write_text(json.dumps(data))
            print("LR_SWEEP_TRAIN_COMPLETE", json.dumps(data), flush=True)


def common_components(cli_context, **kwargs):
    """Use the qualified data order and an isolated registered checkpoint root."""
    r = find_run(cli_context.run_name)
    common = qualified.common_components(cli_context, **kwargs)
    common.save_folder = str(r.root)
    if common.launch is not None:
        common.launch.workspace = WORKSPACE
        common.launch.cmd = [
            "python",
            "src/examples/olmo_ddp/olmoe3_lr_sweep_node.py",
            r.run_id,
            cli_context.cluster,
        ]
    return common


def train_module_config(common):
    """Stable trunks have no final decay; children decay on the original global clock."""
    r = find_run(common.run_name)
    config = qualified.train_module_config(common)
    config.optim.lr = r.lr
    config.scheduler = (
        WSD(warmup=r.warmup, decay=r.decay, decay_fraction=None)
        if r.parent
        else ConstantWithWarmup(warmup=r.warmup)
    )
    return config


def trainer_config(common):
    """Keep own-checkpoint-first recovery and synchronous immutable checkpoints."""
    r = find_run(common.run_name)
    config = qualified.trainer_config(common)
    config.callbacks.pop("integration_audit")
    config.max_duration = Duration.steps(r.end)
    # Only the isolated smoke uses a stop before the configured horizon.
    stop = os.environ.get("OLMOE3_LR_SMOKE_STOP")
    if stop:
        assert r in smoke_runs()
        config.hard_stop = Duration.steps(int(stop))
    config.load_path = str(r.parent_path) if r.parent else None
    config.load_strategy = LoadStrategy.always if r.parent else LoadStrategy.if_available
    cp = config.callbacks["checkpointer"]
    cp.save_interval = r.save
    config.add_callback("sweep_audit", SweepAudit(output_dir=f"{r.root}/audit", run_id=r.run_id))
    config.add_callback(
        "checkpoint_ready",
        CheckpointReadyNotifierCallback(
            inbox_dir=str(CONTROL / "inbox"), run_id=r.run_id, lineage_id=r.run_id
        ),
    )
    ev = config.callbacks["lm_evaluator"]
    if r in smoke_runs():
        ev.eval_interval = 2
        ev.eval_duration = Duration.steps(2)
    wb = config.callbacks["wandb"]
    wb.group = SWEEP
    wb.tags += [f"lr:{r.lr:g}", f"decay_steps:{r.decay}", f"retain:{r.keep}"]
    wb.notes = json.dumps(
        {
            **r.as_dict(),
            "parent_checkpoint": str(r.parent_path),
            "source_baseline": qualified.SETTINGS,
        }
    )
    return config


def config_builder():
    """Build the same model, tokenizer, data stream and compute topology as the baseline."""
    return partial(
        build_config,
        global_batch_size=BATCH,
        max_sequence_length=8192,
        num_nodes=8,
        common_config_builder=common_components,
        data_config_builder=qualified.base.build_data_components,
        model_config_builder=qualified.model_config,
        train_module_config_builder=train_module_config,
        trainer_config_builder=trainer_config,
        beaker_image=qualified.base.BEAKER_IMAGE,
        beaker_workspace=WORKSPACE,
        include_default_evals=False,
        num_execution_units=1,
    )


def validate():
    """Build every real config in the production image without allocating GPUs."""
    from olmoe3_lr_sweep_plan import validate_plan

    validate_plan()
    for r in runs() + smoke_runs():
        c = config_builder()(CliContext(__file__, SubCmd.dry_run, r.run_id, "ai2/holmes", []))
        c.as_dict(json_safe=True)
        tm, tr = c.train_module, c.trainer
        assert c.data_loader.global_batch_size == BATCH and c.init_seed == 12536
        assert tm.rank_microbatch_size == 4 * 8192 and tm.dp_config.use_reduce_scatter
        assert tm.ac_config is None and tm.float8_config is None
        assert tm.ep_config is None and tm.pp_config is None
        assert tr.load_optim_state and tr.load_trainer_state and not tr.save_overwrite
        assert tr.callbacks["checkpointer"].save_async is False
        assert tr.callbacks["checkpointer"].max_checkpoints is None
        scheduler = tm.scheduler
        assert scheduler.get_lr(r.lr, r.warmup, r.end) == r.lr
        assert scheduler.get_lr(r.lr, r.end, r.end) == (0.0 if r.parent else r.lr)
        if r.parent:
            assert scheduler.get_lr(r.lr, r.start, r.end) == r.lr
            assert abs(scheduler.get_lr(r.lr, r.start + r.decay // 2, r.end) / r.lr - 0.5) < 1e-10
        print(
            "CONFIG_VALIDATED",
            json.dumps(
                {**r.as_dict(), "active": c.model.num_active_params, "total": c.model.num_params}
            ),
            flush=True,
        )


if __name__ == "__main__":
    qualified.apply_policy()
    if sys.argv[1:] == ["--validate-only"]:
        validate()
    else:
        main(config_builder=config_builder())
