"""
An example of how to launch the training script on Beaker.
Run this with:

    python src/examples/train_with_mixture_launch.py run_name [OVERRIDES...]
"""

import sys

from beaker import Beaker

from olmo_core.launch.beaker import BeakerLaunchConfig, BeakerEnvSecret
from olmo_core.utils import generate_uuid, prepare_cli_environment


def build_config(run_name: str) -> BeakerLaunchConfig:
    beaker_user = (Beaker.from_env().account.whoami().name).upper()
    return BeakerLaunchConfig(
        name=f"olmo-core-test-{generate_uuid()[:8]}",
        budget="ai2/oe-training",
        cmd=["src/examples/train_with_mixture.py", run_name],
        task_name="train",
        workspace="ai2/OLMo-core",
        description="Testing OLMo-core launch utilities",
        clusters=["ai2/allennlp-elanding-a100-40g"],
        env_secrets=[
            BeakerEnvSecret(name="BEAKER_TOKEN", secret=f"{beaker_user}_BEAKER_TOKEN"),
            BeakerEnvSecret(name="WANDB_API_KEY", secret=f"{beaker_user}_WANDB_API_KEY"),
            BeakerEnvSecret(name="COMET_API_KEY", secret=f"{beaker_user}_COMET_API_KEY"),
            BeakerEnvSecret(name="AWS_CONFIG", secret=f"{beaker_user}_AWS_CONFIG"),
            BeakerEnvSecret(name="AWS_CREDENTIALS", secret=f"{beaker_user}_AWS_CREDENTIALS"),
            BeakerEnvSecret(name="R2_ENDPOINT_URL", secret="R2_ENDPOINT_URL"),
            BeakerEnvSecret(name="WEKA_ENDPOINT_URL", secret="WEKA_ENDPOINT_URL"),
        ],
        setup_steps=[
            # Clone repo.
            'git clone "$REPO_URL" .',
            'git checkout "$GIT_REF"',
            "git submodule update --init --recursive",
            # Setup python environment.
            "conda shell.bash activate base",
            "pip install -e '.[all]'",
            "pip freeze",
            # Move AWS credentials from env to relevant files
            "mkdir -p ~/.aws",
            "printenv AWS_CONFIG > ~/.aws/config",
            "printenv AWS_CREDENTIALS > ~/.aws/credentials",
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
