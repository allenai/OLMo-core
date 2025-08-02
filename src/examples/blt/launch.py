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


def build_config(run_name: str, overrides: List[str]) -> BeakerLaunchConfig:
    cluster = os.environ.get("BEAKER_CLUSTER", "ai2/jupiter-cirrascale-2")
    stage = os.environ.get("BEAKER_STAGE", "stage1")

    env_vars = []
    weka_buckets = []
    shared_filesystem = False

    for transparent_env_var in [
        "EMBEDDING_INIT_PATH",
        "OLMO_CKPT_PATH",
        "OLMO_ARCH",
        "TRAIN_MODE",
        "DATA_SOURCE",
        "SAVE_FOLDER",
        "BYTE_EXPANSION_FACTOR",
        "LOCAL_MODEL_STYLE",
        "ADD_HASH_EMBEDDINGS",
        "GLOBAL_MODEL_LEARNING_RATE",
    ]:
        if transparent_env_var in os.environ:
            env_vars.append(BeakerEnvVar(name=transparent_env_var, value=os.environ[transparent_env_var]))

    if cluster != "ai2/augusta-google-1":
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
        image = "ai2/cuda12.8-dev-ubuntu22.04-torch2.7.0"
        # slow setup (need appropriate torch / cuda build)
        setup_steps = list(DEFAULT_SETUP_STEPS)

    if stage == "stage1":
        launch_script = "src/examples/blt/train_stage1.py"
    elif stage == "stage2":
        launch_script = "src/examples/blt/train_stage2.py"
    else:
        raise ValueError(f"Unknown stage: {stage}. Must be 'stage1' or 'stage2'.")

    beaker_username = get_beaker_username()

    return BeakerLaunchConfig(
        name=f"blt-distill-{run_name}-{generate_uuid()[:4]}",
        budget="ai2/oe-training",
        cmd=[launch_script, run_name, *overrides],
        task_name="train",
        workspace="ai2/benjaminm",
        description="Distilling OLMo from and to BLT.",
        clusters=[cluster],
        env_vars=env_vars,
        num_nodes=int(os.environ.get("BEAKER_NUM_NODES", "1")),
        num_gpus=int(os.environ.get("BEAKER_NUM_GPUS", "1")),
        priority=cast(Priority, os.environ.get("BEAKER_PRIORITY", "normal")),
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
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} run_name [OVERRIDES...]")
        sys.exit(1)

    run_name, *overrides = sys.argv[1:]

    prepare_cli_environment()

    build_config(run_name, overrides).launch(follow=False)
