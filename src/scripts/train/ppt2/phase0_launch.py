"""
An example of how to launch the training script on Beaker.
Run this with:

    python src/scripts/train/ppt2/phase0_launch.py run_name [OVERRIDES...]
"""

import sys
from typing import List

from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.utils import generate_uuid, prepare_cli_environment


def build_config(run_name: str, cluster: str, overrides: List[str]) -> BeakerLaunchConfig:
    return BeakerLaunchConfig(
        name=f"phase0-train-{generate_uuid()[:8]}",
        budget="ai2/willm-ppt2",
        cmd=["src/scripts/train/ppt2/phase0.py", run_name, *overrides],
        task_name="train",
        workspace="ai2/OLMo-core",
        description="PPT2 phase0",
        clusters=[cluster],
        num_nodes=1,
        num_gpus=8,
        shared_filesystem=True,
        nfs=True,
        allow_dirty=True,
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} run_name cluster [OVERRIDES...]")
        sys.exit(1)

    run_name, cluster, *overrides = sys.argv[1:]

    prepare_cli_environment()

    build_config(run_name, cluster, overrides).launch(follow=True)