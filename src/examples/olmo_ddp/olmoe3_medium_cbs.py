"""Medium CBS trajectories with synchronous saves and audited full-state branching.

The production setting is selected only after the systems qualification. This driver
does not submit jobs, register uploader policies, or prune any checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import ClassVar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from olmoe3_medium_cbs_plan import CAMPAIGN, CONTROL, TARGET_TOKENS, find_run, validate
from olmoe3_medium_followup_plan import sample_offsets

validate()
RUN = find_run(os.environ["OLMOE3_MEDIUM_CBS_RUN"])
SMOKE = "-smoke" in RUN.run_id
STOP = int(os.environ.get("OLMOE3_MEDIUM_CBS_STOP", str(RUN.end)))
EXPECTED_START = int(os.environ.get("OLMOE3_MEDIUM_CBS_EXPECTED_START", str(RUN.start)))
if not RUN.start <= EXPECTED_START < STOP <= RUN.end:
    raise ValueError((EXPECTED_START, STOP, RUN))
if not SMOKE and STOP != RUN.end:
    raise ValueError("Production CBS horizon must match the approved plan")
if os.environ.get("OLMOE3_MEDIUM_DIAGNOSTIC", "0") != "0":
    raise ValueError("Gradient-only diagnostics cannot be used for CBS training")
os.environ["OLMOE3_MEDIUM_BATCH"] = str(RUN.batch)
os.environ["OLMOE3_DEEP_PROFILE_PASS"] = "timing"

import olmoe3_medium_followup as followup
import torch

from olmo_core.data import DataMix, NumpyPaddedFSLDatasetConfig
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.internal.experiment import build_config, main
from olmo_core.optim.scheduler import WSD
from olmo_core.train import Duration
from olmo_core.train.callbacks import Callback, CheckpointerCallback, LMEvaluatorCallbackConfig
from olmo_core.train.callbacks.checkpoint_ready_notifier import CheckpointReadyNotifierCallback
from olmo_core.train.callbacks.checkpointer import CheckpointRemovalStrategy
from olmo_core.train.common import LoadStrategy

base = followup.base
if followup.BATCH != RUN.batch or base.TOPOLOGY.gpus != 128:
    raise ValueError("CBS requires the approved 128-GPU topology and exact phase batch")


def atomic_json(path, value):
    """Publish compact audit metadata without mutating checkpoint contents."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def state_sample(trainer):
    """Fingerprint fixed samples of every live optimizer shard and persistent buffer.

    This checks exact sampled values, not every element. Never call optimizer
    state_dict here: its checkpoint layout conversion temporarily swaps EP storage.
    """
    tm = trainer.train_module
    tensors = dict(tm.optim.states)
    tensors.update(tm._persistent_model_buffer_state_dict())
    fingerprints = {}
    for name, value in sorted(tensors.items()):
        value = followup.local(value).detach().reshape(-1)
        if not value.numel():
            raise RuntimeError(f"Unexpected empty live state: {name}")
        offsets = torch.tensor(sample_offsets(value.numel(), 128), device=value.device)
        sample = value[offsets].contiguous().view(torch.uint8).cpu().numpy().tobytes()
        fingerprints[name] = [value.numel(), str(value.dtype), hashlib.sha256(sample).hexdigest()]
    return {
        "gpus": get_world_size(),
        "rank": get_rank(),
        "step": trainer.global_step,
        "tokens": trainer.global_train_tokens_seen,
        "tensors": fingerprints,
        "loss_history": [float(v.item()) for v in tm.optim._losses],
        "norm_history": [float(v.item()) for v in tm.optim._grad_norms],
    }


@dataclass
class CBSAudit(Callback):
    """Fail closed on wrong lineage, unsafe retention, LR, or sampled resume state."""

    priority: ClassVar[int] = 10  # Install the save guard before Checkpointer.pre_train.

    def post_checkpoint_loaded(self, path):
        expected = Path(path).parent / "audit" / f"state-step{self.step}-rank{get_rank()}.json"
        saved = json.loads(expected.read_text())
        actual = state_sample(self.trainer)
        if saved != actual:
            changed = [key for key in actual if saved.get(key) != actual[key]]
            raise RuntimeError(f"Resume state samples changed: {expected}, fields={changed}")
        atomic_json(
            RUN.root / "audit" / f"restore-{os.environ['BEAKER_JOB_ID']}-rank{get_rank()}.json",
            {"source": str(path), "step": self.step, "sampled_state_exact": True},
        )

    def pre_train(self):
        registration = json.loads((CONTROL / "registrations" / f"{RUN.run_id}.json").read_text())
        if (
            registration.get("checkpoint_root") != str(RUN.root)
            or not registration.get("enabled", True)
            or registration.get("deletion_mode") != "apply"
            or registration.get("min_local_checkpoints", 0) < RUN.keep
            or registration.get("delete_grace_seconds", 0) < 3600
        ):
            raise RuntimeError("Missing or unsafe explicit medium uploader registration")
        if not EXPECTED_START <= self.step <= STOP:
            raise RuntimeError(f"Unexpected resume step {self.step}, expected >= {EXPECTED_START}")
        if SMOKE and self.step != EXPECTED_START:
            raise RuntimeError("Smoke must restore the explicitly tested checkpoint")
        if self.trainer.global_train_tokens_seen != RUN.tokens_at(self.step):
            raise RuntimeError("Restored absolute token position does not match CBS lineage")
        if self.trainer.data_loader.tokens_processed != self.trainer.global_train_tokens_seen:
            raise RuntimeError("Data-loader token position differs from the training trajectory")
        tm = self.trainer.train_module
        if tm.pp_enabled or not tm.ep_enabled or get_world_size() != 128:
            raise RuntimeError("CBS runtime topology differs from the qualified PP1/EP8 plan")
        for group in tm.optim.param_groups:
            for key in ("lr", "initial_lr"):
                if not math.isclose(float(group[key]), RUN.lr, rel_tol=1e-6):
                    raise RuntimeError(f"Resume changed configured peak {key}")
        original_save = tm.save_state_dict_direct

        def guarded_save(*args, **kwargs):
            before = state_sample(self.trainer)
            original_save(*args, **kwargs)
            after = state_sample(self.trainer)
            if before != after:
                raise RuntimeError("Synchronous checkpoint save changed live state samples")

        tm.save_state_dict_direct = guarded_save
        if get_rank() == 0:
            record = {
                "run": RUN.run_id,
                "step": self.step,
                "stop": STOP,
                "tokens": RUN.tokens_at(self.step),
                "target_tokens": TARGET_TOKENS,
                "batch": RUN.batch,
                "lr": RUN.lr,
                "warmup": 2000,
                "gpus": 128,
                "ep": 8,
                "pp": 1,
                "mb": followup.MB,
                "variant": followup.VARIANT,
                "settings": base.SETTINGS,
                "flags": base.FLAGS,
                "source_commit": os.environ.get("GIT_REF"),
                "checkpoint_interval": RUN.interval,
                "registration": registration,
            }
            atomic_json(RUN.root / "audit" / f"session-{os.environ['BEAKER_JOB_ID']}.json", record)
            print("MEDIUM_CBS_START", json.dumps(record), flush=True)

    def post_checkpoint_saved(self, path):
        atomic_json(
            Path(path).parent / "audit" / f"state-step{self.step}-rank{get_rank()}.json",
            state_sample(self.trainer),
        )

    def log_metrics(self, step, metrics):
        for key in ("train/CE loss", "optim/total grad norm"):
            if key in metrics and not math.isfinite(float(metrics[key])):
                raise RuntimeError(f"Nonfinite CBS metric {key} at {step}")
        if get_rank() == 0:
            with (RUN.root / "audit" / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps({"step": step, **metrics}) + "\n")

    def post_train(self):
        if self.step != STOP or self.trainer.global_train_tokens_seen != RUN.tokens_at(STOP):
            raise RuntimeError("CBS pass ended before its approved horizon")
        atomic_json(
            RUN.root / "audit" / f"complete-step{self.step}-rank{get_rank()}.json",
            {"step": self.step, "tokens": self.trainer.global_train_tokens_seen},
        )


def common_components(cli_context, **kwargs):
    """Keep Dolma3.5 data order and a disjoint durable checkpoint root."""
    if cli_context.run_name != RUN.run_id:
        raise ValueError("Run name does not match the selected CBS phase")
    common = base.common_components(cli_context, **kwargs)
    common.save_folder = str(RUN.root)
    return common


def train_module_config(common):
    """Use the chosen systems settings with the approved LR and stable WSD schedule."""
    config = base.train_module_config(common)
    config.optim.lr = RUN.lr
    config.scheduler = WSD(warmup=2000, decay=1, decay_fraction=None)
    return config


def trainer_config(common):
    """Use synchronous full checkpoints, no trainer pruning, and matched held-out evals."""
    cfg = base.base.build_trainer_config(common, "medium-cbs-128g", base.SYSTEM)
    cfg.no_checkpoints = cfg.no_evals = False
    cfg.save_folder = str(RUN.root)
    cfg.work_dir = str(RUN.root / "work")
    cfg.save_overwrite = False
    cfg.load_path = str(RUN.load_path) if RUN.load_path else None
    cfg.load_strategy = (
        LoadStrategy.always if RUN.parent or EXPECTED_START else LoadStrategy.if_available
    )
    cfg.load_optim_state = cfg.load_trainer_state = True
    cfg.max_duration = Duration.tokens(TARGET_TOKENS)
    cfg.hard_stop = Duration.steps(STOP) if SMOKE else None
    cfg.add_callback("cbs_audit", CBSAudit())
    cfg.add_callback(
        "checkpointer",
        CheckpointerCallback(
            save_interval=RUN.interval,
            pre_train_checkpoint=None,
            save_async=False,
            remove=CheckpointRemovalStrategy.never,
            max_checkpoints=None,
        ),
    )
    cfg.add_callback(
        "checkpoint_ready",
        CheckpointReadyNotifierCallback(
            inbox_dir=str(CONTROL / "inbox"),
            run_id=RUN.run_id,
            lineage_id=RUN.run_id,
        ),
    )
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
            eval_interval=2 if SMOKE else (500 if RUN.parent else 1000),
            eval_duration=Duration.steps(2) if SMOKE else Duration.epochs(1),
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
        "128gpu",
        f"mb:{followup.MB}",
        f"batch:{RUN.batch}",
        followup.VARIANT,
        "smoke" if SMOKE else "100b",
        "sync-checkpoints",
        "guarded-cleanup",
    ]
    cfg.callbacks["wandb"].notes = (
        f"Locked medium CBS; absolute token accounting; LR {RUN.lr}; WSD warmup2000. "
        "Full optimizer and data-state resume, trainer never prunes, explicit uploader policy."
    )
    return cfg


if __name__ == "__main__":
    if sys.argv[1:] == ["--validate-only"]:
        from types import SimpleNamespace

        from olmo_core.data import TokenizerConfig

        common = SimpleNamespace(
            run_name=RUN.run_id,
            tokenizer=TokenizerConfig.dolma2(),
            max_sequence_length=8192,
            global_batch_size=RUN.batch,
            work_dir=base.DATA_WORK,
            save_folder=str(RUN.root),
        )
        model, tm, trainer = (
            base.model_config(common),
            train_module_config(common),
            trainer_config(common),
        )
        assert (model.num_active_params, model.num_params) == (
            base.EXPECTED_ACTIVE,
            base.EXPECTED_TOTAL,
        )
        assert tm.optim.lr == RUN.lr and tm.rank_microbatch_size == followup.MB * 8192
        assert tm.ep_config.degree == 8 and tm.pp_config is None and tm.float8_config is None
        assert not trainer.no_checkpoints and not trainer.no_evals
        assert trainer.callbacks["checkpointer"].save_async is False
        assert trainer.load_optim_state and trainer.load_trainer_state
        assert trainer.callbacks["checkpointer"].save_interval == RUN.interval
        assert trainer.callbacks["checkpoint_ready"].run_id == RUN.run_id
        print("MEDIUM_CBS_CONFIG_VALIDATED", RUN.run_id, followup.MB, RUN.batch, RUN.lr, flush=True)
        raise SystemExit(0)
    main(
        config_builder=partial(
            build_config,
            global_batch_size=RUN.batch,
            max_sequence_length=8192,
            num_nodes=16,
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
