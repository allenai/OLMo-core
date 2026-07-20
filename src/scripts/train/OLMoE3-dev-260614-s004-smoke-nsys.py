"""Short Gantry/Beaker reproduction of the s004 GDN run with Nsight Systems.

The target W&B run was allocated 8 nodes (64 B300 GPUs). This smoke test uses
4 nodes (32 GPUs), keeps EP=8 and the per-rank microbatch unchanged, and uses
the 18M-token global batch that was active when the loss spike occurred.

Dry run::

    python src/scripts/train/OLMoE3-dev-260614-s004-smoke-nsys.py \
        dry_run OLMoE3-dev-260614-s004-smoke-nsys ai2/holmes

Launch (choose the priority explicitly)::

    python src/scripts/train/OLMoE3-dev-260614-s004-smoke-nsys.py \
        launch OLMoE3-dev-260614-s004-smoke-nsys ai2/holmes \
        --launch.priority=normal

The launch is asynchronous. Nsight reports and NCCL flight-recorder dumps are
written to Weka under ``/weka/oe-training-default/robertb/profiling``.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import ModuleType

# This must be set before loading the base script because NVTX_DISABLE is read
# when NVTX is first imported.
os.environ["OLMO_USE_NV_PROFILE"] = "1"

from olmo_core.internal.experiment import CliContext, ExperimentConfig, build_config, main
from olmo_core.launch.beaker import BeakerEnvVar
from olmo_core.train import Duration
from olmo_core.train.callbacks import NvidiaProfilerCallback, WandBCallback


BASE_SCRIPT = Path(__file__).with_name("OLMoE3-dev-260614-s004.py")
DEFAULT_NUM_NODES = 4
PROFILE_START_STEP = 31
PROFILE_END_STEP = 35
SMOKE_STEPS = 40
NSYS_WRAPPER = "src/scripts/train/nsys-profile-rank.sh"
BEAKER_IMAGE = "tianhuat/olmo-core-torch212-2404-cu130"
BEAKER_WORKSPACE = "ai2/OLMo-3-moe-experiments"
WANDB_ENTITY = "ai2-llm"
WANDB_PROJECT = "robertb-moe-tests"
WEKA_PROFILE_BASE = "/weka/oe-training-default/robertb/profiling"


def _load_base_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("olmoe3_s004_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base training script at {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base_script()

# The spike segment used this stage-3 batch setting as an uncommitted change at
# Git commit 3aff38004. It was committed later in e6abbbb56.
base.GLOBAL_BATCH_SIZE_SEQ = (8 * 8 * 2) * 18
base.GLOBAL_BATCH_SIZE = base.GLOBAL_BATCH_SIZE_SEQ * base.SEQUENCE_LENGTH
base.GLOBAL_BATCH_TOKENS_IN_M = base.GLOBAL_BATCH_SIZE // 1024 // 1024

# Recompute values derived from global batch size exactly as the base script
# does. The smoke duration itself is capped separately with hard_stop below.
base.SCHED_WARMUP_TOKENS = int((10e9 // base.GLOBAL_BATCH_SIZE) * base.GLOBAL_BATCH_SIZE)
base.SCHED_FAST_DECAY_TOKENS = 0
base.SCHED_LONG_DECAY_TOKENS = int((19990e9 // base.GLOBAL_BATCH_SIZE) * base.GLOBAL_BATCH_SIZE)
base.LR = 2e-4 * (base.GLOBAL_BATCH_SIZE / (base.LR_REF_BSZ_IN_M * 1024 * 1024)) ** base.LR_ALPHA
base.EXPERT_LR = base.LR
base.MONKEY_PATCH_DECAY_DURATION_TOKENS = int(
    (200e9 // base.GLOBAL_BATCH_SIZE) * base.GLOBAL_BATCH_SIZE
)


def build_smoke_trainer_config(common):
    config = base.build_trainer_config(common)
    config.save_folder = common.save_folder
    config.hard_stop = Duration.steps(SMOKE_STEPS)
    config.no_checkpoints = True

    wandb_callback = config.callbacks["wandb"]
    if not isinstance(wandb_callback, WandBCallback):
        raise TypeError("Expected the smoke trainer to have a W&B callback")
    wandb_callback.entity = WANDB_ENTITY
    wandb_callback.project = WANDB_PROJECT

    num_nodes = int(os.environ.get("NUM_NODES", DEFAULT_NUM_NODES))
    config.callbacks["profiler"] = NvidiaProfilerCallback(
        enabled=True,
        profile_ranks=list(range(0, num_nodes * 8, 8)),
        start=PROFILE_START_STEP,
        end=PROFILE_END_STEP,
    )
    return config


def finalize_smoke_config(config: ExperimentConfig) -> None:
    base.finalize_config(config)
    profile_root = f"{WEKA_PROFILE_BASE}/{config.run_name}"
    if config.launch is not None:
        config.launch.env_secrets = [
            secret for secret in config.launch.env_secrets if secret.required
        ]
        config.launch.google_credentials_secret = None
        config.launch.env_vars.extend(
            [
                BeakerEnvVar(name="NSYS_OUTPUT_DIR", value=f"{profile_root}/nsight"),
                BeakerEnvVar(name="PYTHONPATH", value="/gantry-runtime/src"),
            ]
        )
        config.launch.post_setup = (
            "python -m olmo_core.kernels.build_symm_mem_vdev2d_ext "
            "--inplace --backend cmake"
        )
        for env_var in config.launch.env_vars:
            if env_var.name == "TORCH_FR_DUMP_TEMP_FILE":
                env_var.value = f"{profile_root}/flightrecorder/nccl_trace_rank_"
        config.launch.cmd = [NSYS_WRAPPER, *config.launch.cmd]
        config.launch.follow = False
        config.launch.step_timeout = None
        config.launch.step_soft_timeout = None
        config.launch.retries = 0


_config_builder = partial(
    build_config,
    global_batch_size=base.GLOBAL_BATCH_SIZE,
    max_sequence_length=base.SEQUENCE_LENGTH,
    data_config_builder=base.build_data_components,
    model_config_builder=base.build_model_config,
    train_module_config_builder=base.build_train_module_config,
    trainer_config_builder=build_smoke_trainer_config,
    num_nodes=DEFAULT_NUM_NODES,
    beaker_image=BEAKER_IMAGE,
    beaker_workspace=BEAKER_WORKSPACE,
    flight_recorder=True,
    include_instance_filter=True,
    include_default_evals=False,
    finalize_config=finalize_smoke_config,
)


def build_smoke_config(cli_context: CliContext) -> ExperimentConfig:
    """Build without round-tripping the specialized model through Config.merge."""
    launch_overrides: dict[str, str] = {}
    for override in cli_context.overrides:
        if not override.startswith("--launch.") or "=" not in override:
            raise ValueError(
                "This exact-reproduction launcher only accepts launch overrides in "
                "'--launch.FIELD=VALUE' form."
            )
        key, value = override.removeprefix("--launch.").split("=", 1)
        if key not in {"priority", "num_nodes"}:
            raise ValueError(f"Unsupported smoke launch override: {key!r}")
        launch_overrides[key] = value

    config = _config_builder(replace(cli_context, overrides=[]))
    if config.launch is not None:
        if priority := launch_overrides.get("priority"):
            config.launch.priority = priority
        if num_nodes := launch_overrides.get("num_nodes"):
            config.launch.num_nodes = int(num_nodes)
    return config


if __name__ == "__main__":
    main(config_builder=build_smoke_config)
