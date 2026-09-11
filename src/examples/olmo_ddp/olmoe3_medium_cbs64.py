"""Resume audited medium state, benchmark matched batches, and run bounded CBS branches."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import ClassVar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from olmoe3_medium_cbs64_plan import (
    AUTOMATION,
    CAMPAIGN,
    CONTROL,
    FORK_STEP,
    PARENT,
    PHASES,
    ROOT,
    TARGET_TOKENS,
    VARIANT,
    old_rank_for_half,
    phase_environment,
    validate,
)
from olmoe3_medium_followup_plan import sample_offsets

validate()
PHASE = PHASES[os.environ["OLMOE3_MEDIUM_CBS64_PHASE"]]
if os.environ.get("OLMOE3_MEDIUM_DIAGNOSTIC", "0") != "0":
    raise RuntimeError("No gradient-only or optimizer-disabled mode in this campaign")
phase_env = phase_environment(os.environ, PHASE)
for key in tuple(os.environ):
    if key.startswith("OLMOE3_NSYS_"):
        del os.environ[key]
os.environ.update(phase_env)

import olmoe3_medium_followup as followup
import torch
import torch.distributed as dist

from olmo_core.data import DataMix, NumpyPaddedFSLDatasetConfig
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.internal.experiment import build_config, main
from olmo_core.optim.scheduler import WSD
from olmo_core.train import Duration
from olmo_core.train.callbacks import (
    Callback,
    CheckpointerCallback,
    LMEvaluatorCallbackConfig,
)
from olmo_core.train.callbacks.checkpoint_ready_notifier import (
    CheckpointReadyNotifierCallback,
)
from olmo_core.train.callbacks.checkpointer import CheckpointRemovalStrategy
from olmo_core.train.common import LoadStrategy

base = followup.base


def atomic_json(path, value):
    """Write campaign-owned audit metadata, never original checkpoint contents."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    with tmp.open("w") as out:
        json.dump(value, out, sort_keys=True)
        out.flush()
        os.fsync(out.fileno())
    tmp.replace(path)


def live_tensors(trainer):
    """Read optimizer/master/model/buffer state without checkpoint-layout mutation."""
    tm = trainer.train_module
    result = dict(tm.optim.states)
    result.update(tm._persistent_model_buffer_state_dict())
    result.update((f"model_param/{k}", v) for k, v in tm.model.named_parameters())
    return result


def fingerprint(value):
    """Match the original 128-rank checkpoint's exact 128-element audit convention."""
    value = followup.local(value).detach().reshape(-1)
    if not value.numel():
        raise RuntimeError("Empty state in restore audit")
    offsets = torch.tensor(sample_offsets(value.numel(), 128), device=value.device)
    data = value[offsets].contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return [value.numel(), str(value.dtype), hashlib.sha256(data).hexdigest()]


def state_sample(trainer):
    """All state keys, sampled values, and exact rolling optimizer histories."""
    tm = trainer.train_module
    return {
        "gpus": get_world_size(),
        "rank": get_rank(),
        "step": trainer.global_step,
        "tokens": trainer.global_train_tokens_seen,
        "tensors": {k: fingerprint(v) for k, v in sorted(live_tensors(trainer).items())},
        "loss_history": [float(v.item()) for v in tm.optim._losses],
        "norm_history": [float(v.item()) for v in tm.optim._grad_norms],
    }


def verify_restore(trainer, path):
    """Compare each reshaped shard with both corresponding saved 128-rank halves.

    This is exact sampled-state verification, not a full-tensor equality claim.
    It checks every state key, including FP32 main params/moments and BF16 live
    weights, without allocating a second model or invoking optimizer.state_dict().
    """
    rank, world = get_rank(), get_world_size()
    cache = {}

    def saved(old_rank):
        if old_rank not in cache:
            p = Path(path).parent / "audit" / f"state-step{trainer.global_step}-rank{old_rank}.json"
            cache[old_rank] = json.loads(p.read_text())
        return cache[old_rank]

    reference = saved(rank)
    actual = state_sample(trainer)
    if reference["gpus"] == world:
        if reference != actual:
            raise RuntimeError(f"Same-topology restore audit mismatch on rank{rank}")
        return {"source_gpus": world, "gpus": world, "state_keys": len(actual["tensors"])}
    if (reference["gpus"], world, Path(path)) != (128, 64, PARENT):
        raise RuntimeError("Unapproved source/topology change")
    for key in ("step", "tokens", "loss_history", "norm_history"):
        if reference[key] != actual[key]:
            raise RuntimeError(f"Reshard restore changed {key} on rank{rank}")
    if set(reference["tensors"]) != set(actual["tensors"]):
        raise RuntimeError("Reshard restore changed state keys")
    groups = {
        f"{name}.{suffix}": group["pg"]
        for group in trainer.train_module.optim.param_groups
        for name in group["named_params"]
        for suffix in ("main", "exp_avg", "exp_avg_sq", "step")
    }
    halves = 0
    for name, tensor in sorted(live_tensors(trainer).items()):
        value = followup.local(tensor).detach().reshape(-1)
        old = reference["tensors"][name]
        if value.numel() == old[0]:
            if actual["tensors"][name] != old:
                raise RuntimeError(f"Unsharded/model/buffer restore mismatch: {name}, rank{rank}")
            continue
        if name not in groups or value.numel() != 2 * old[0]:
            raise RuntimeError(f"Unrecognized reshard geometry: {name}, {value.numel()}, {old[0]}")
        for half in (0, 1):
            old_rank = old_rank_for_half(rank, half, groups[name])
            expected = saved(old_rank)["tensors"][name]
            if fingerprint(value.narrow(0, half * old[0], old[0])) != expected:
                raise RuntimeError(f"Optimizer reshard mismatch: {name}, new{rank}, old{old_rank}")
            halves += 1
    return {
        "source_gpus": 128,
        "gpus": 64,
        "state_keys": len(actual["tensors"]),
        "verified_old_shard_halves": halves,
        "sampled_state_exact": True,
    }


@dataclass
class ResumeAudit(Callback):
    """Audit restore before updates and preserve model/optimizer state around saves."""

    priority: ClassVar[int] = 10

    def post_checkpoint_loaded(self, path):
        self.restore = verify_restore(self.trainer, path)
        self.restore.update(
            source=str(path), step=self.step, tokens=self.trainer.global_train_tokens_seen
        )
        atomic_json(PHASE.root / "audit" / f"restore-rank{get_rank()}.json", self.restore)
        if get_rank() == 0:
            print("MEDIUM_64_RESTORE_VERIFIED", json.dumps(self.restore), flush=True)

    def pre_train(self):
        if not getattr(self, "restore", None):
            raise RuntimeError("Fresh initialization is forbidden; restore audit missing")
        tm = self.trainer.train_module
        if self.step != PHASE.start or self.trainer.global_train_tokens_seen != PHASE.tokens_at(
            self.step
        ):
            raise RuntimeError("Wrong restored step or absolute token position")
        if self.trainer.data_loader.tokens_processed != PHASE.tokens_at(self.step):
            raise RuntimeError("Wrong data-loader position after batch change")
        if get_world_size() != PHASE.gpus or tm.pp_enabled or not tm.ep_enabled:
            raise RuntimeError("Wrong runtime topology")
        if dist.get_world_size(tm.ep_mp_group) != 8:
            raise RuntimeError("Expected EP8")
        hosts = [None] * PHASE.gpus
        dist.all_gather_object(
            hosts, os.environ["BEAKER_NODE_HOSTNAME"], group=self.trainer.bookkeeping_pg
        )
        base.TOPOLOGY.validate_rank_groups(tm.moe_mesh.mesh.reshape(-1, 8).tolist(), hosts)
        for group in tm.optim.param_groups:
            for key in ("lr", "initial_lr"):
                if not math.isclose(float(group[key]), PHASE.lr, rel_tol=1e-6):
                    raise RuntimeError(f"Configured {key} lost during optimizer restore")
        if any(os.environ.get(k) != v for k, v in base.FLAGS.items()):
            raise RuntimeError("Optimization flags changed")
        if PHASE.save:
            registration = json.loads(
                (CONTROL / "registrations" / f"{PHASE.run_id}.json").read_text()
            )
            if (
                registration["checkpoint_root"] != str(PHASE.root)
                or registration["deletion_mode"] != "apply"
                or registration["min_local_checkpoints"] < 2
                or not registration.get("enabled", True)
            ):
                raise RuntimeError("Missing or unsafe uploader registration")
            original = tm.save_state_dict_direct

            def guarded_save(*args, **kwargs):
                before = state_sample(self.trainer)
                result = original(*args, **kwargs)
                if state_sample(self.trainer) != before:
                    raise RuntimeError("Checkpoint save mutated live sampled state")
                return result

            tm.save_state_dict_direct = guarded_save
        self.rows = []
        self.first = True
        self.output = PHASE.root / "audit"
        self.output.mkdir(parents=True, exist_ok=True)
        if get_rank() == 0:
            record = {
                "phase": PHASE.name,
                "gpus": PHASE.gpus,
                "batch": PHASE.batch,
                "lr": PHASE.lr,
                "start": self.step,
                "end": PHASE.end,
                "tokens_start": PHASE.tokens_at(self.step),
                "tokens_end": PHASE.tokens_at(PHASE.end),
                "source": str(PHASE.load_path),
                "source_commit": os.environ.get("GIT_REF"),
                "settings": base.SETTINGS,
                "flags": base.FLAGS,
                "warmup": 2000,
                "no_decay_before_stop": True,
                "flops_per_token": tm.num_flops_per_token(8192),
            }
            atomic_json(self.output / "session.json", record)
            print("MEDIUM_CBS64_START", json.dumps(record), flush=True)

    def pre_step(self, batch):
        if self.first:
            value = batch["input_ids"].detach().contiguous().cpu()
            digest = hashlib.sha256(value.numpy().tobytes()).hexdigest()
            atomic_json(
                self.output / f"first-batch-rank{get_rank()}.json",
                {"shape": list(value.shape), "sha256": digest},
            )
            if PHASE.start == FORK_STEP:
                canonical = (
                    AUTOMATION
                    / f"first-batch-{PHASE.gpus}g-{PHASE.batch_mi}mi-rank{get_rank()}.json"
                )
                record = {"shape": list(value.shape), "sha256": digest}
                if canonical.exists() and json.loads(canonical.read_text()) != record:
                    raise RuntimeError("Same-source, same-topology first batch changed")
                if not canonical.exists():
                    atomic_json(canonical, record)
            self.first = False

    def post_checkpoint_saved(self, path):
        atomic_json(
            Path(path).parent / "audit" / f"state-step{self.step}-rank{get_rank()}.json",
            state_sample(self.trainer),
        )

    def log_metrics(self, step, metrics):
        for key in ("train/CE loss", "optim/total grad norm"):
            if key in metrics and not math.isfinite(float(metrics[key])):
                raise RuntimeError(f"Nonfinite {key} at {step}")
        if "train/CE loss" in metrics:
            self.rows.append({"step": step, **metrics})
        if get_rank() == 0:
            with (self.output / "metrics.jsonl").open("a") as out:
                out.write(json.dumps({"step": step, **metrics}) + "\n")


@dataclass
class CompletionAudit(Callback):
    """Only publish success after checkpoint/evaluation and complete timing telemetry."""

    priority: ClassVar[int] = -10

    def post_train(self):
        from olmoe3_medium_profile_plan import routing_window_summary

        # Finalize asynchronous metric callbacks before publishing a success gate.
        # This is outside the measured training window, on every rank.
        self.trainer._log_metrics()
        self.trainer._join_bookkeeping_ops()
        if self.step != PHASE.end or self.trainer.global_train_tokens_seen != PHASE.tokens_at(
            PHASE.end
        ):
            raise RuntimeError("Pass ended before the exact approved token horizon")
        if get_rank() == 0:
            # Metrics are emitted after asynchronous drains, so read the persistent leader log.
            rows = [
                json.loads(line)
                for line in (PHASE.root / "audit/metrics.jsonl").read_text().splitlines()
            ]
            train = {r["step"]: r for r in rows if "train/CE loss" in r}
            expected = list(range(PHASE.start + 1, PHASE.end + 1))
            if sorted(train) != expected:
                raise RuntimeError("Missing or duplicated training-metric updates")
            result = {
                "phase": PHASE.name,
                "start": PHASE.start,
                "end": PHASE.end,
                "gpus": PHASE.gpus,
                "batch": PHASE.batch,
                "tokens": PHASE.tokens_at(PHASE.end),
                "source_commit": os.environ.get("GIT_REF"),
                "completed": True,
            }
            if any("optim/step skipped" not in row for row in train.values()):
                raise RuntimeError("Missing optimizer skip telemetry")
            result["skipped_updates"] = sum(
                float(row["optim/step skipped"]) for row in train.values()
            )
            if not PHASE.cbs and PHASE.end - PHASE.start == 60:
                window = [train[s] for s in range(PHASE.start + 21, PHASE.end + 1)]
                tps = [float(r["throughput/device/TPS"]) for r in window]
                if any(not math.isfinite(v) or v <= 0 for v in tps):
                    raise RuntimeError("Invalid timing window")
                # Preserve every slow step and every metric-drain wait in the effective rate.
                result.update(
                    effective_tps_per_gpu=statistics.harmonic_mean(tps),
                    median_tps_per_gpu=statistics.median(tps),
                    mean_ce=statistics.mean(float(r["train/CE loss"]) for r in window),
                    measured_updates=40,
                    routing=routing_window_summary(window, [r["step"] for r in window]),
                )
                result["measured_skipped_updates"] = sum(
                    float(r["optim/step skipped"]) for r in window
                )
                if result["measured_skipped_updates"]:
                    raise RuntimeError(
                        "Timing window includes skipped optimizer updates; review before CBS"
                    )
                if not result["routing"]["telemetry_complete"]:
                    raise RuntimeError("Incomplete route-drop telemetry")
            atomic_json(PHASE.root / "audit/summary.json", result)
            print("MEDIUM_CBS64_PASS_COMPLETE", json.dumps(result), flush=True)
        dist.barrier()
        atomic_json(
            PHASE.root / "audit" / f"complete-rank{get_rank()}.json",
            {
                "step": self.step,
                "tokens": PHASE.tokens_at(self.step),
                "source_commit": os.environ.get("GIT_REF"),
            },
        )


def common_components(cli_context, **kwargs):
    """Use the same Dolma3.5 mixture and data-order cache as the existing parent."""
    if cli_context.run_name != PHASE.run_id:
        raise RuntimeError("Unexpected run name")
    common = base.common_components(cli_context, **kwargs)
    common.save_folder = str(PHASE.root)
    return common


def train_module_config(common):
    """Keep the accepted optimized BF16/FP32 bundle, with the approved scaled LR."""
    cfg = base.train_module_config(common)
    cfg.optim.lr = PHASE.lr
    cfg.scheduler = WSD(warmup=2000, decay=1, decay_fraction=None)
    return cfg


def trainer_config(common):
    """Hard-stop below the scheduler horizon; probes never alter the parent lineage."""
    cfg = base.base.build_trainer_config(common, f"medium-cbs-{PHASE.gpus}g", base.SYSTEM)
    cfg.save_folder = str(PHASE.root)
    cfg.work_dir = str(PHASE.root / "work")
    cfg.save_overwrite = False
    cfg.load_path = str(PHASE.load_path)
    cfg.load_strategy = LoadStrategy.always
    cfg.load_optim_state = cfg.load_trainer_state = True
    cfg.max_duration = Duration.tokens(2 * TARGET_TOKENS)
    cfg.hard_stop = Duration.steps(PHASE.end)
    cfg.metrics_collect_interval = 5
    # no_checkpoints=True also disables *loading* in Trainer.fit(). Disable
    # only the write callback for no-save benchmarks instead.
    cfg.no_checkpoints = False
    cfg.no_evals = PHASE.gpus == 128
    cfg.add_callback("resume_audit", ResumeAudit())
    cfg.add_callback("completion_audit", CompletionAudit())
    cfg.add_callback("full_run_memory", followup.PeakMemoryAudit(output_dir=str(PHASE.root)))
    if PHASE.save:
        cfg.add_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=PHASE.save_interval if PHASE.cbs else None,
                pre_train_checkpoint=False,
                save_async=False,
                remove=CheckpointRemovalStrategy.never,
                max_checkpoints=None,
            ),
        )
        cfg.add_callback(
            "checkpoint_ready",
            CheckpointReadyNotifierCallback(
                inbox_dir=str(CONTROL / "inbox"), run_id=PHASE.run_id, lineage_id=PHASE.run_id
            ),
        )
    else:
        cfg.add_callback("checkpointer", CheckpointerCallback(enabled=False))
    if not cfg.no_evals:
        cfg.add_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyPaddedFSLDatasetConfig.from_data_mix(
                    DataMix.v3_small_ppl_validation,
                    mix_base_dir="gs://ai2-llm",
                    sequence_length=8192,
                    tokenizer=common.tokenizer,
                    work_dir=common.work_dir,
                ),
                eval_interval=PHASE.eval_interval if PHASE.cbs else 1_000_000,
                eval_duration=Duration.epochs(1) if PHASE.cbs else Duration.steps(2),
                eval_on_finish=True,
            ),
        )
    cfg.callbacks["wandb"].group = CAMPAIGN
    cfg.callbacks["wandb"].tags = [
        "medium-cbs",
        "emo",
        "qknorm-pr855",
        "pp1",
        "ep8",
        "mb2",
        f"gpus:{PHASE.gpus}",
        f"batch:{PHASE.batch}",
        VARIANT,
        "cbs100b" if PHASE.cbs else "resume-speed-screen",
    ]
    cfg.callbacks["wandb"].notes = (
        "Same67.108864B parent, full-state restore; no new warmup or decay; no precision/kernel/model changes."
    )
    return cfg


def validate_config():
    """Build real configs in the training image before scheduling a large allocation."""
    from types import SimpleNamespace

    from olmo_core.data import TokenizerConfig

    common = SimpleNamespace(
        run_name=PHASE.run_id,
        tokenizer=TokenizerConfig.dolma2(),
        max_sequence_length=8192,
        global_batch_size=PHASE.batch,
        work_dir=base.DATA_WORK,
        save_folder=str(PHASE.root),
    )
    model, tm, cfg = base.model_config(common), train_module_config(common), trainer_config(common)
    assert (model.num_active_params, model.num_params) == (
        base.EXPECTED_ACTIVE,
        base.EXPECTED_TOTAL,
    )
    assert tm.optim.lr == PHASE.lr and tm.rank_microbatch_size == 2 * 8192
    assert (
        tm.ep_config.degree == 8
        and tm.pp_config is None
        and tm.ac_config is None
        and tm.float8_config is None
    )
    assert (
        tm.dp_config.use_reduce_scatter
        and tm.dp_config.accumulate_grads_in_fp32
        and tm.dp_config.reduce_grads_in_fp32
    )
    assert cfg.load_optim_state and cfg.load_trainer_state and cfg.load_path == str(PHASE.load_path)
    assert cfg.metrics_collect_interval == 5 and cfg.hard_stop == Duration.steps(PHASE.end)
    assert not followup.CHECKPOINT_BLOCKS
    if PHASE.save:
        assert cfg.callbacks["checkpointer"].save_async is False
        assert cfg.callbacks["checkpointer"].remove == CheckpointRemovalStrategy.never
    print("MEDIUM_CBS64_CONFIG_VALID", PHASE.name, PHASE.gpus, PHASE.batch, PHASE.lr, flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--validate-only"]:
        validate_config()
    else:
        main(
            config_builder=partial(
                build_config,
                global_batch_size=PHASE.batch,
                max_sequence_length=8192,
                num_nodes=PHASE.gpus // 8,
                common_config_builder=common_components,
                data_config_builder=base.base.build_data_components,
                model_config_builder=base.model_config,
                train_module_config_builder=train_module_config,
                trainer_config_builder=trainer_config,
                beaker_image=base.base.BEAKER_IMAGE,
                beaker_workspace="ai2/olmo3p5-training",
                include_default_evals=False,
                num_execution_units=1,
            )
        )
