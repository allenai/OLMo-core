"""
Launch a command on Beaker.
"""

import sys
from typing import List

from olmo_core.launch.beaker import BeakerLaunchConfig, OLMoCoreBeakerImage
from olmo_core.utils import generate_uuid, prepare_cli_environment


def build_config(cmd: List[str]) -> BeakerLaunchConfig:
    return BeakerLaunchConfig(
        name=f"olmo-core-test-{generate_uuid()[:8]}",
        budget="ai2/oe-training",
        cmd=cmd,
        task_name="test",
        workspace="ai2/OLMo-core",
        beaker_image=OLMoCoreBeakerImage.stable,
        clusters=[
            "ai2/jupiter-cirrascale-2",
            "ai2/augusta-google-1",
            "ai2/ceres-cirrascale",
        ],
        num_nodes=1,
        num_gpus=2,
        shared_filesystem=True,
        host_networking=False,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} CMD ...")
        sys.exit(1)

    prepare_cli_environment()

    build_config(sys.argv[1:]).launch(follow=True, torchrun=False)
