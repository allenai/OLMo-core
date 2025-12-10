"""
An example of how to launch the training script on Beaker.
Run this with:

    python src/examples/llama/train_launch.py run_name [OVERRIDES...]
"""

import sys
from typing import List, cast
import os

from beaker import Priority

from olmo_core.launch.beaker import BeakerLaunchConfig, BeakerEnvVar, BeakerWekaBucket, BeakerEnvSecret, DEFAULT_SETUP_STEPS
from olmo_core.launch.utils import GIT_BRANCH_ENV_VAR, GIT_REPO_URL_ENV_VAR, GIT_REF_ENV_VAR
from olmo_core.internal.common import get_beaker_username
from olmo_core.utils import generate_uuid, prepare_cli_environment


def build_config(run_name: str, overrides: List[str]) -> tuple[BeakerLaunchConfig, bool]:
    cluster = os.environ.get("BEAKER_CLUSTER", "ai2/jupiter-cirrascale-2")
    stage = os.environ.get("BEAKER_STAGE", "stage1")

    env_vars = []
    weka_buckets = []
    shared_filesystem = False

    for transparent_env_var in [
        "SEQUENCE_LENGTH",
        "STAGE1_CKPT_PATH",
        "EMBEDDING_INIT_PATH",
        "OLMO_CKPT_PATH",
        "OLMO_ARCH",
        "TOKENIZER",
        "TRAIN_MODE",
        "DATA_SOURCE",
        "DTYPE",
        "SAVE_FOLDER",
        "BYTE_EXPANSION_FACTOR",
        "LOCAL_MODEL_STYLE",
        "MODEL_STYLE",
        "LOCAL_MODEL_BLOCKS",
        "LR_SCHEDULE",
        "TOKEN_NOISE_STR",
        "ADD_HASH_EMBEDDINGS",
        "ADD_EXPANDED_EMBEDDINGS",
        "TEACHER_MODE",
        "GLOBAL_MODEL_LEARNING_RATE",
        "NUM_WORKERS",
        "PREFILL_LENGTH",
        "GENERATE_LENGTH",
        "N_BATCHES",
        "BATCH_SIZE",
        "AVG_BYTES_PER_TOKEN",
    ]:
        if transparent_env_var in os.environ:
            env_vars.append(BeakerEnvVar(name=transparent_env_var, value=os.environ[transparent_env_var]))

    if cluster not in {"ai2/augusta-google-1", "ai2/augusta"}:
        env_vars.append(BeakerEnvVar(name="HAS_WEKA", value="1"))
        weka_buckets = [
            BeakerWekaBucket(
                bucket="oe-training-default",
                mount="/weka/oe-training-default",
            ),
            BeakerWekaBucket(
                bucket="oe-adapt-default",
                mount="/weka/oe-adapt-default",
            ),
        ]
        shared_filesystem = True
    else:
        shared_filesystem = False

    # fast setup for clusters were we have a prebuilt image
    if cluster == "ai2/titan-cirrascale":
        image = "benjaminm/titan_blt_train"
        setup_steps = [
            f'if [[ -z "${GIT_BRANCH_ENV_VAR}" ]]; then',
            f'  git clone "${GIT_REPO_URL_ENV_VAR}" .',
            "else",
            f'  git clone -b "${GIT_BRANCH_ENV_VAR}" --single-branch "${GIT_REPO_URL_ENV_VAR}" .',
            "fi",
            f'git checkout "${GIT_REF_ENV_VAR}"',
            "git submodule update --init --recursive",
            "pip freeze",
        ]
    else:
        image = "benjaminm/titan_blt_train"
        # slow setup (need appropriate torch / cuda build)
        setup_steps = list(DEFAULT_SETUP_STEPS)

    setup_steps += ["pip install flash-attn==2.8.3 --no-build-isolation"]
    setup_steps += ["pip install triton==3.5.1"]
    setup_steps += ["pip install xlstm==2.0.5"]
    setup_steps += ["pip install flash-linear-attention==0.3.1"]
    setup_steps += ["pip install --upgrade huggingface-hub==0.35.3"]
    setup_steps += ["pip install bettermap==1.3.1"]

    if stage == "stage1":
        launch_script = "src/examples/bolmo/train_stage1.py"
        torchrun = True
    elif stage == "stage2":
        launch_script = "src/examples/bolmo/train_stage2.py"
        torchrun = True
    elif stage == "baseline":
        launch_script = "src/examples/bolmo/train_baseline.py"
        torchrun = True
    elif stage == "compute_entropies":
        launch_script = "src/examples/bolmo/compute_entropies.py"
        torchrun = True
    elif stage == "benchmark_generation":
        launch_script = "src/examples/bolmo/benchmark_generation.py"
        torchrun = False
    elif stage == "generate_ifeval":
        launch_script = "src/examples/bolmo/generate_ifeval.py"
        torchrun = True
    else:
        raise ValueError(f"Unknown stage: {stage}. Must be 'stage1', 'stage2', 'baseline', 'compute_entropies' or 'benchmark_generation'.")

    beaker_username = get_beaker_username()

    return (
        BeakerLaunchConfig(
            name=f"blt-distill-{run_name}-{generate_uuid()[:4]}",
            budget="ai2/oe-base",
            cmd=[launch_script, run_name, *overrides],
            task_name="train",
            workspace=os.environ.get("BEAKER_WORKSPACE", "ai2/benjaminm"),
            description="Distilling OLMo from and to BLT.",
            clusters=[cluster],
            env_vars=env_vars,
            num_nodes=int(os.environ.get("BEAKER_NUM_NODES", "1")),
            num_gpus=int(os.environ.get("BEAKER_NUM_GPUS", "1")),
            priority=cast(Priority, os.environ.get("BEAKER_PRIORITY", "normal")),
            preemptible=os.environ.get("BEAKER_PREEMPTIBLE", "true").lower() in {"1", "true", "yes"},
            shared_filesystem=shared_filesystem,
            allow_dirty=True,
            beaker_image=image,
            weka_buckets=weka_buckets,
            env_secrets=[
                BeakerEnvSecret(
                    name="WANDB_API_KEY",
                    secret=f"{beaker_username}_WANDB_API_KEY",
                ),
            ],
            setup_steps=setup_steps,  # type: ignore
        ),
        torchrun
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} run_name [OVERRIDES...]")
        sys.exit(1)

    run_name, *overrides = sys.argv[1:]

    prepare_cli_environment()

    config, torchrun = build_config(run_name, overrides)
    config.launch(follow=False, torchrun=torchrun)
