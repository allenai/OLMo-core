"""
Launch tests on Beaker.
"""

import sys
from typing import List

from rich import print

from olmo_core.launch.beaker import (
    BeakerEnvSecret,
    BeakerLaunchConfig,
    OLMoCoreBeakerImage,
)
from olmo_core.utils import generate_uuid, prepare_cli_environment


def build_config(command: List[str], overrides: List[str]) -> BeakerLaunchConfig:
    return BeakerLaunchConfig(
        name=f"olmo-core-test-{generate_uuid()[:8]}",
        budget="ai2/oe-base",
        cmd=command,
        task_name="test",
        workspace="ai2/OLMo-core",
        beaker_image=OLMoCoreBeakerImage.stable,
        clusters=[
            "ai2/jupiter",
            "ai2/augusta",
            "ai2/ceres",
        ],
        num_nodes=1,
        num_gpus=2,
        shared_filesystem=True,
        #  host_networking=False,
        env_secrets=[
            BeakerEnvSecret(name="HF_TOKEN", secret="HF_TOKEN"),
        ],
    ).merge(overrides)


if __name__ == "__main__":
    if len(sys.argv) < 3 or "--" not in sys.argv:
        print(f"Usage: python {sys.argv[0]} [OVERRIDES...] -- [CMD...]")
        sys.exit(1)

    sep_index = sys.argv.index("--")
    overrides = sys.argv[1:sep_index]
    entrypoint = sys.argv[sep_index + 1]
    command = sys.argv[sep_index + 2 :]

    prepare_cli_environment()

    config = build_config(command, overrides)
    print(config)
    config.launch(follow=True, torchrun=False, entrypoint=entrypoint)
