"""
An example of how to launch the training script on Beaker.
Run this with:

    python src/examples/llama/train_launch.py run_name [OVERRIDES...]
"""

import sys
from typing import List
import os

from olmo_core.launch.beaker import BeakerLaunchConfig, BeakerEnvVar, BeakerWekaBucket, BeakerEnvSecret
from olmo_core.internal.common import get_beaker_username
from olmo_core.utils import generate_uuid, prepare_cli_environment


def build_config(run_name: str, overrides: List[str]) -> BeakerLaunchConfig:
    cluster = os.environ.get("BEAKER_CLUSTER", "ai2/jupiter-cirrascale-2")

    env_vars = []
    weka_buckets = []
    shared_filesystem = False

    if "EMBEDDING_INIT_PATH" in os.environ:
        env_vars.append(BeakerEnvVar(name="EMBEDDING_INIT_PATH", value=os.environ["EMBEDDING_INIT_PATH"]))

    if "OLMO_CKPT_PATH" in os.environ:
        env_vars.append(BeakerEnvVar(name="OLMO_CKPT_PATH", value=os.environ["OLMO_CKPT_PATH"]))

    if "TRAIN_MODE" in os.environ:
        env_vars.append(BeakerEnvVar(name="TRAIN_MODE", value=os.environ["TRAIN_MODE"]))

    if cluster != "ai2/augusta-google-1":
        env_vars.append(BeakerEnvVar(name="HAS_WEKA", value="1"))
        weka_buckets = [
            BeakerWekaBucket(
                bucket="oe-training-default",
                mount="/weka/oe-training-default",
            )
        ]
        shared_filesystem = True

    beaker_username = get_beaker_username()

    return BeakerLaunchConfig(
        name=f"blt-distill-{generate_uuid()[:8]}",
        budget="ai2/oe-training",
        cmd=["src/examples/blt/distill.py", run_name, *overrides],
        task_name="train",
        workspace="ai2/benjaminm",
        description="Distilling OLMo from and to BLT.",
        clusters=[cluster],
        env_vars=env_vars,
        num_nodes=1,
        num_gpus=1,
        shared_filesystem=shared_filesystem,
        allow_dirty=True,
        beaker_image="ai2/cuda12.8-ubuntu22.04-torch2.7.0",
        weka_buckets=weka_buckets,
        env_secrets=[
            BeakerEnvSecret(
                name="WANDB_API_KEY",
                secret=f"{beaker_username}_WANDB_API_KEY",
            ),
        ]
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} run_name [OVERRIDES...]")
        sys.exit(1)

    run_name, *overrides = sys.argv[1:]

    prepare_cli_environment()

    build_config(run_name, overrides).launch(follow=False)
