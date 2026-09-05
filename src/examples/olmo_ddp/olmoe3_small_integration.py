"""Matched 100B production-setting integration runs and save/restore smoke tests.

The optimization policy is explicit and recorded. Both arms use identical model,
data, optimizer, precision and schedules. No checkpoint deletion is permitted.
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

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import olmoe3_small_medium_profile as base

from olmo_core.data import DataMix, NumpyPaddedFSLDatasetConfig
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.internal.experiment import build_config, main
from olmo_core.launch.beaker import BeakerEnvSecret, BeakerEnvVar, BeakerWekaBucket
from olmo_core.optim.scheduler import WSD
from olmo_core.train import Duration
from olmo_core.train.callbacks import (
    Callback,
    CheckpointerCallback,
    LMEvaluatorCallbackConfig,
)
from olmo_core.train.callbacks.checkpointer import CheckpointRemovalStrategy
from olmo_core.train.common import LoadStrategy

ROOT = "/weka/olmo-3p5-checkpoints/production-integration"
DATA_WORK = (
    "/weka/olmo-3p5-checkpoints/production-cbs/work/"
    "olmoe3-small-cbs-8mi-100b-lr1p3em3-uploader-r1"
)
BATCH = 16_777_216
TOKENS = 6000 * BATCH
LR = 0.00185
SYSTEM = base.SYSTEMS["small-64g"]
ARM = os.environ.get("OLMOE3_INTEGRATION_ARM", "reference")
SMOKE = os.environ.get("OLMOE3_INTEGRATION_SMOKE", "0") == "1"
STOP = int(os.environ.get("OLMOE3_INTEGRATION_STOP", "4" if SMOKE else "6000"))
EXPECTED_START = int(os.environ.get("OLMOE3_INTEGRATION_EXPECTED_START", "0"))
POLICY = os.environ.get("OLMOE3_INTEGRATION_POLICY", "core-docpool")
FLAGS = {
    "OLMO_PROFILE_SAFE_NOOP_NVTX": "0",
    "OLMO_PROFILE_RS_SINGLE_PARAM_FAST_PATH": "0",
    "OLMO_PROFILE_FP32_GRAD_ADD_VECTORIZE": "0",
    "OLMO_PROFILE_SWIGLU_PAIRWISE": "0",
    "OLMO_PROFILE_EMO_DOCUMENT_POOL": "0",
    "OLMO_PROFILE_EMO_TOP16": "0",
    "OLMO_PROFILE_ROUNDED_WGRAD": "0",
}


def write_or_verify(path: Path, content: str):
    """Preserve fingerprints across retries and fail on changed inputs or initialization."""
    if path.exists():
        if path.read_text() != content:
            raise RuntimeError(f"Integration fingerprint changed on retry: {path}")
    else:
        path.write_text(content)


def apply_policy():
    """Reset every experimental switch before constructing either arm."""
    if ARM not in ("reference", "optimized") or POLICY not in (
        "core-docpool",
        "core-docpool-top16",
        "core-docpool-wgrad",
        "core-docpool-top16-wgrad",
    ):
        raise ValueError((ARM, POLICY))
    flags = dict(FLAGS)
    if ARM == "optimized":
        for key in (
            "OLMO_PROFILE_FP32_GRAD_ADD_VECTORIZE",
            "OLMO_PROFILE_SWIGLU_PAIRWISE",
            "OLMO_PROFILE_EMO_DOCUMENT_POOL",
        ):
            flags[key] = "1"
        flags["OLMO_PROFILE_EMO_TOP16"] = "1" if "top16" in POLICY else "0"
        flags["OLMO_PROFILE_ROUNDED_WGRAD"] = "1" if "wgrad" in POLICY else "0"
    os.environ.update(flags)
    return flags


@dataclass
class IntegrationAudit(Callback):
    """Fail on nonfinite metrics; record init weights and data fingerprints, not raw data."""

    output_dir: str = ""

    def pre_train(self):
        registration = json.loads(
            (
                Path("/weka/olmo-3p5-checkpoints/uploader/control/registrations")
                / f"{Path(self.trainer.save_folder).name}.json"
            ).read_text()
        )
        if (
            registration["deletion_mode"] != "report_only"
            or registration["checkpoint_root"] != str(self.trainer.save_folder)
            or not registration.get("enabled", True)
        ):
            raise RuntimeError(
                "Integration requires an enabled, matching report_only uploader registration"
            )
        # Automatic retries may resume later than the initial expected step.
        if self.step < EXPECTED_START or self.step > STOP:
            raise RuntimeError(f"Unexpected resumed step {self.step}; expected >= {EXPECTED_START}")
        output = Path(self.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self._first_batch = True
        if self.step == 0:
            digest = hashlib.sha256()
            # EP1/PP1 parameters are replicated; partition hashing work across ranks.
            for index, (name, parameter) in enumerate(
                self.trainer.train_module.model.named_parameters()
            ):
                if index % get_world_size() == get_rank():
                    digest.update(name.encode())
                    tensor = parameter.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
                    digest.update(tensor.numpy().tobytes())
            hashes = [None] * get_world_size()
            dist.all_gather_object(hashes, digest.hexdigest(), group=self.trainer.bookkeeping_pg)
            if get_rank() == 0:
                write_or_verify(output / "initial-weights-sha256.json", json.dumps(hashes))
        if get_rank() == 0:
            provenance = {
                "arm": ARM,
                "policy": POLICY,
                "flags": {key: os.environ[key] for key in FLAGS},
                "source_commit": os.environ.get("GIT_REF"),
                "start_step": self.step,
                "stop_step": STOP,
                "global_batch_tokens": BATCH,
                "total_tokens": TOKENS,
                "init_seed": 12536,
                "data_seed": 928543231,
                "gpus": 64,
                "mb_sequences": 4,
                "ga": 8,
                "lr": LR,
                "deletion": "forbidden; trainer never; uploader report_only",
            }
            path = output / f"session-{os.environ.get('BEAKER_JOB_ID', 'local')}-{self.step}.json"
            path.write_text(json.dumps(provenance, indent=2))
            print("INTEGRATION_START", json.dumps(provenance), flush=True)

    def pre_step(self, batch):
        if self._first_batch:
            tokens = batch["input_ids"].detach().contiguous().cpu()
            digest = hashlib.sha256(tokens.numpy().tobytes()).hexdigest()
            path = Path(self.output_dir) / f"input-step{self.step}-rank{get_rank()}.sha256"
            write_or_verify(path, digest)
            self._first_batch = False

    def log_metrics(self, step, metrics):
        for name, value in metrics.items():
            if name in ("train/CE loss", "optim/total grad norm") and not math.isfinite(value):
                raise RuntimeError(f"Nonfinite integration metric at step {step}: {name}={value}")
        if get_rank() == 0:
            with (Path(self.output_dir) / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps({"step": step, **metrics}) + "\n")


def common_components(cli_context, **kwargs):
    """Mount checkpoints, share the existing data order, and launch allocated urgent workers."""
    common = base.build_common_components(cli_context, "small-64g", SYSTEM, **kwargs)
    common.work_dir = DATA_WORK
    common.save_folder = f"{ROOT}/{common.run_name}"
    if (launch := common.launch) is not None:
        launch.min_runtime = "1h"
        launch.retries = 0
        launch.preemptible = False
        launch.shared_filesystem = True
        launch.weka_buckets = [
            BeakerWekaBucket("olmo-3p5-checkpoints", "/weka/olmo-3p5-checkpoints")
        ]
        launch.torchrun = False
        launch.cmd = [
            "python",
            "src/examples/olmo_ddp/olmoe3_integration_node.py",
            cli_context.run_name,
            cli_context.cluster,
        ]
        launch.post_setup = "bash src/examples/olmo_ddp/olmoe3_profile_setup.sh"
        launch.env_secrets.append(BeakerEnvSecret("GITHUB_TOKEN", "jacobm_GITHUB_TOKEN"))
        for name, value in {
            "OLMOE3_INTEGRATION_ARM": ARM,
            "OLMOE3_INTEGRATION_POLICY": POLICY,
            "OLMOE3_INTEGRATION_SMOKE": "1" if SMOKE else "0",
            "OLMOE3_INTEGRATION_STOP": str(STOP),
            "OLMOE3_INTEGRATION_EXPECTED_START": str(EXPECTED_START),
        }.items():
            launch.env_vars.append(BeakerEnvVar(name, value))
    return common


def model_config(common):
    """Keep the per-head QK-gain architecture identical in both arms."""
    apply_policy()
    from kernel_fun._common import support

    import olmo_core.ops.moe as moe_ops

    support.MIN_CTAS = 128 if ARM == "optimized" else 256
    if ARM == "optimized":
        moe_ops.pool_keep_mask = moe_ops.pool_keep_mask_inverse_scatter
    model = base.build_model_config_from_common(common, SYSTEM)
    for block in model.block_overrides.values():
        if hasattr(block.sequence_mixer, "qk_norm_per_head_gains"):
            block.sequence_mixer.qk_norm_per_head_gains = True
    return model


def train_module_config(common):
    """Freeze production optimizer, WSD, BF16, MB4 and all-reduce settings."""
    config = base.build_train_module_config(common, SYSTEM)
    config.optim.lr = LR
    config.scheduler = WSD(warmup=2000, decay=1, decay_fraction=None)
    return config


def trainer_config(common):
    """Save every checkpoint synchronously, with identical held-out evals in both arms."""
    config = base.build_trainer_config(common, "small-64g", SYSTEM)
    config.no_checkpoints = False
    config.no_evals = False
    config.save_folder = common.save_folder
    config.work_dir = f"{common.save_folder}/work"
    config.save_overwrite = False
    config.load_path = None
    config.load_strategy = LoadStrategy.if_available
    config.load_optim_state = True
    config.load_trainer_state = True
    config.max_duration = Duration.tokens(TOKENS)
    config.hard_stop = Duration.steps(STOP) if SMOKE else None
    config.add_callback(
        "checkpointer",
        CheckpointerCallback(
            save_interval=4 if SMOKE else 250,
            pre_train_checkpoint=None,
            save_async=False,
            remove=CheckpointRemovalStrategy.never,
            max_checkpoints=None,
        ),
    )
    config.add_callback(
        "integration_audit", IntegrationAudit(output_dir=f"{common.save_folder}/audit")
    )
    config.add_callback(
        "lm_evaluator",
        LMEvaluatorCallbackConfig(
            eval_dataset=NumpyPaddedFSLDatasetConfig.from_data_mix(
                DataMix.v3_small_ppl_validation,
                mix_base_dir="s3://ai2-llm",
                sequence_length=8192,
                tokenizer=common.tokenizer,
                work_dir=common.work_dir,
            ),
            eval_interval=4 if SMOKE else 1000,
            eval_duration=Duration.steps(2) if SMOKE else Duration.epochs(1),
            eval_on_finish=True,
        ),
    )
    wandb = config.callbacks["wandb"]
    wandb.name = common.run_name
    wandb.group = "small-production-integration-20260905"
    wandb.tags = [
        "small-64g",
        "16mi",
        "mb4",
        "ga8",
        "emo",
        "qknorm-pr855",
        ARM,
        POLICY,
        "smoke" if SMOKE else "100b",
        "no-deletion",
        "synchronous-checkpoints",
    ]
    wandb.notes = (
        "Matched fresh initialization, Dolma3.5, BF16, PP1 EP1 DP64 MB4 GA8; "
        "LR .00185, WSD warmup2000 decay1; 6000 updates =100.663296B tokens. "
        "No activation checkpointing/MXFP8/shared EP outputs. Uploader report_only."
    )
    return config


if __name__ == "__main__":
    apply_policy()
    main(
        config_builder=partial(
            build_config,
            global_batch_size=BATCH,
            max_sequence_length=8192,
            num_nodes=8,
            common_config_builder=common_components,
            data_config_builder=base.build_data_components,
            model_config_builder=model_config,
            train_module_config_builder=train_module_config,
            trainer_config_builder=trainer_config,
            beaker_image=base.BEAKER_IMAGE,
            beaker_workspace=base.WORKSPACE,
            include_default_evals=False,
            num_execution_units=1,
        )
    )
