"""
An example of how to launch the training script on Beaker.
Run this with:

    python src/examples/train_with_mixture_launch.py run_name [OVERRIDES...]
"""

import sys

from olmo_core.launch.beaker import BeakerLaunchConfig, BeakerEnvSecret
from olmo_core.utils import generate_uuid, prepare_cli_environment


def build_config(run_name: str) -> BeakerLaunchConfig:
    return BeakerLaunchConfig(
        name=f"olmo-core-test-{generate_uuid()[:8]}",
        budget="ai2/oe-training",
        cmd=["src/examples/train_with_mixture.py", run_name],
        task_name="train",
        workspace="ai2/OLMo-core",
        description="Testing OLMo-core launch utilities",
        clusters=["ai2/allennlp-elanding-a100-40g"],
        env_secrets=[
            BeakerEnvSecret("AWS_ACCESS_KEY_ID", "AWS_ACCESS_KEY_ID"),
            BeakerEnvSecret("AWS_SECRET_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY"),
        ],
        num_nodes=1,
        num_gpus=4,
        shared_filesystem=True,
        nfs=False,
        allow_dirty=True,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} run_name [OVERRIDES...]")
        sys.exit(1)

    run_name = sys.argv[1]

    prepare_cli_environment()

    build_config(run_name).launch(follow=True)
