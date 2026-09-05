"""Trained small-model 16Mi profiling, with isolated Nsight and PyTorch passes."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import olmoe3_small_medium_profile as base

from olmo_core.distributed.utils import get_rank
from olmo_core.internal.experiment import build_config, main
from olmo_core.launch.beaker import BeakerEnvSecret, BeakerEnvVar, BeakerWekaBucket
from olmo_core.optim.scheduler import WSD
from olmo_core.train import Duration
from olmo_core.train.callbacks import (
    Callback,
    NvidiaProfilerCallback,
    ProfilerCallback,
    TorchMemoryHistoryCallback,
)
from olmo_core.train.common import LoadStrategy

CHECKPOINT_ROOT = "/weka/olmo-3p5-checkpoints"
SOURCE_STEP = 7500
SOURCE_CHECKPOINT = (
    f"{CHECKPOINT_ROOT}/production-cbs/"
    "olmoe3-small-cbs-16mi-from-step4000-lr1p85em3-uploader-r1/step7500"
)
DATA_WORK_DIR = (
    f"{CHECKPOINT_ROOT}/production-cbs/work/" "olmoe3-small-cbs-8mi-100b-lr1p3em3-uploader-r1"
)
ARTIFACT_ROOT = f"{CHECKPOINT_ROOT}/production-profiling"
GLOBAL_BATCH_SIZE = 16 * 1024 * 1024
LEARNING_RATE = 1.85e-3
PASS = os.environ.get("OLMOE3_DEEP_PROFILE_PASS", "nsys")
VARIANT = os.environ.get("OLMOE3_DEEP_PROFILE_VARIANT", "baseline")
STEPS = int(os.environ.get("OLMOE3_DEEP_PROFILE_STEPS", "100" if PASS == "nsys" else "60"))
SYSTEM = base.SYSTEMS["small-64g"]
PROFILE_RANKS = list(range(0, 64, 8))


@dataclass
class ProfileMetrics(Callback):
    """Persist all logged metrics and exact window/provenance information outside result datasets."""

    output_dir: str = ""

    def pre_train(self):
        if self.step != SOURCE_STEP:
            raise RuntimeError(
                f"Expected trained checkpoint step {SOURCE_STEP}; loaded {self.step}"
            )
        optim = self.trainer.train_module.optim
        for group in optim.param_groups:
            for key in ("lr", "initial_lr"):
                if abs(float(group[key]) - LEARNING_RATE) > 1e-9:
                    raise RuntimeError(f"Unexpected optimizer {key}: {group[key]}")
        if get_rank() == 0:
            output = Path(self.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            provenance = {
                "source_checkpoint": SOURCE_CHECKPOINT,
                "source_step": SOURCE_STEP,
                "global_batch_tokens": GLOBAL_BATCH_SIZE,
                "gpus": 64,
                "microbatch_sequences": 4,
                "gradient_accumulation": 8,
                "lr": LEARNING_RATE,
                "pass": PASS,
                "variant": VARIANT,
                "kernel_fun_commit": "7a6983baf2beb4ec4d7fe914ec9f6670438af99b",
                "qk_norm_pr": 855,
                "clean_windows_relative_steps": [[31, 70], [81, 100]]
                if PASS == "nsys"
                else [[21, 30], [51, 60]],
                "nsys_relative_steps": [71, 73] if PASS == "nsys" else None,
                "torch_relative_steps": [36, 37] if PASS == "torch" else None,
                "memory_relative_steps": [45, 46] if PASS == "torch" else None,
            }
            (output / "provenance.json").write_text(json.dumps(provenance, indent=2))

    def log_metrics(self, step, metrics):
        if get_rank() == 0:
            with (Path(self.output_dir) / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps({"step": step, **metrics}) + "\n")


def common_components(cli_context, **kwargs):
    common = base.build_common_components(cli_context, "small-64g", SYSTEM, **kwargs)
    common.work_dir = DATA_WORK_DIR
    common.save_folder = f"{ARTIFACT_ROOT}/{common.run_name}"
    if (launch := common.launch) is not None:
        launch.min_runtime = "1h"
        launch.preemptible = False
        launch.shared_filesystem = True
        launch.weka_buckets = [BeakerWekaBucket("olmo-3p5-checkpoints", CHECKPOINT_ROOT)]
        # Gantry wraps this Python worker in torchrun. Each worker launches both passes
        # sequentially, retaining its distributed environment, with a fresh Python process.
        launch.cmd = [
            "python",
            "src/examples/olmo_ddp/olmoe3_profile_worker.py",
            cli_context.run_name,
            cli_context.cluster,
        ]
        launch.post_setup = "bash src/examples/olmo_ddp/olmoe3_profile_setup.sh"
        launch.env_secrets.append(BeakerEnvSecret("GITHUB_TOKEN", "jacobm_GITHUB_TOKEN"))
        launch.env_vars.extend(
            [
                BeakerEnvVar("OLMOE3_DEEP_PROFILE_VARIANT", VARIANT),
                BeakerEnvVar(
                    "OLMOE3_DEEP_PROFILE_PASSES",
                    os.environ.get("OLMOE3_DEEP_PROFILE_PASSES", "nsys,torch"),
                ),
            ]
        )
    return common


def model_config(common):
    model = base.build_model_config_from_common(common, SYSTEM)
    for block in model.block_overrides.values():
        mixer = block.sequence_mixer
        if hasattr(mixer, "qk_norm_per_head_gains"):
            mixer.qk_norm_per_head_gains = True
    return model


def train_module_config(common):
    config = base.build_train_module_config(common, SYSTEM)
    config.optim.lr = LEARNING_RATE
    config.scheduler = WSD(warmup=2000, decay=1, decay_fraction=None)
    config.expand_shared_qk_norm_on_load = True
    if VARIANT == "reduce-scatter":
        config.dp_config.use_reduce_scatter = True
    elif VARIANT != "baseline":
        raise ValueError(f"Unknown variant: {VARIANT}")
    return config


def trainer_config(common):
    config = base.build_trainer_config(common, "small-64g", SYSTEM)
    config.save_folder = common.save_folder
    config.work_dir = common.save_folder
    config.save_overwrite = False
    config.load_path = SOURCE_CHECKPOINT
    config.load_strategy = LoadStrategy.always
    config.load_optim_state = True
    config.load_trainer_state = True
    config.max_duration = Duration.tokens(100_663_296_000)
    config.hard_stop = Duration.steps(SOURCE_STEP + STEPS)
    config.callbacks["wandb"].tags = [
        "small-64g",
        "16mi",
        "mb4",
        "ga8",
        "emo",
        "qknorm-pr855",
        "kernel-fun-7a6983b",
        f"pass:{PASS}",
        f"variant:{VARIANT}",
        "trained-routing",
        "no-checkpoints",
    ]
    config.callbacks["wandb"].notes = (
        "Small 794M active / 12.496B total; 16 layers, d=1024, latent=512, "
        "14 KDA/2 FA; trained step7500 with shared QK gains and Adam moments expanded; "
        "BF16, FA4/scalable-softmax, EMO16->512, PP1 EP1 DP64 MB4 GA8; "
        "no recomputation/MXFP8/shared EP outputs; lr=.00185 constant WSD. "
        "Use only clean windows in provenance.json for throughput."
    )
    config.add_callback("profile_metrics", ProfileMetrics(output_dir=common.save_folder))
    if PASS == "nsys":
        config.add_callback(
            "nsys_capture",
            NvidiaProfilerCallback(
                start=SOURCE_STEP + 71,
                end=SOURCE_STEP + 73,
                profile_ranks=list(range(64)),
            ),
        )
    elif PASS == "torch":
        config.add_callback(
            "torch_capture",
            ProfilerCallback(
                skip_first=30,
                wait=4,
                warmup=1,
                active=2,
                repeat=1,
                with_stack=False,
                profile_memory=False,
                enable_cuda_sync_events=True,
                export_distributed_event_summary=True,
                ranks=PROFILE_RANKS,
            ),
        )
        config.add_callback(
            "memory_capture",
            TorchMemoryHistoryCallback(
                start=SOURCE_STEP + 45,
                end=SOURCE_STEP + 46,
                profile_ranks=[0],
                max_entries=100_000,
                output_dir=f"{common.save_folder}/memory",
            ),
        )
    elif PASS != "timing":
        raise ValueError(f"Unknown pass: {PASS}")
    return config


if __name__ == "__main__":
    main(
        config_builder=partial(
            build_config,
            global_batch_size=GLOBAL_BATCH_SIZE,
            max_sequence_length=base.SEQUENCE_LENGTH,
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
