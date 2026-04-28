"""
Launch the Qwen3 OLMo-core ↔ HF parity probe on Beaker as a single-GPU job.

Usage: uv run python src/scripts/beaker/qwen3_parity_launch.py [OVERRIDES...]
"""

import sys

from rich import print

from olmo_core.launch.beaker import (
    BeakerEnvSecret,
    BeakerLaunchConfig,
    OLMoCoreBeakerImage,
)
from olmo_core.utils import generate_uuid, prepare_cli_environment


def build_config(overrides: list[str]) -> BeakerLaunchConfig:
    return BeakerLaunchConfig(
        name=f"qwen3-parity-{generate_uuid()[:8]}",
        budget="ai2/oe-base",
        cmd=["python", "src/scripts/beaker/qwen3_parity.py"],
        task_name="qwen3-parity",
        workspace="ai2/open-instruct-dev",
        beaker_image=OLMoCoreBeakerImage.stable,
        clusters=["ai2/jupiter", "ai2/ceres"],
        num_nodes=1,
        num_gpus=1,
        shared_filesystem=True,
        torchrun=False,
        priority="urgent",
        preemptible=False,
        env_secrets=[BeakerEnvSecret(name="HF_TOKEN", secret="HF_TOKEN")],
    ).merge(overrides)


if __name__ == "__main__":
    prepare_cli_environment()
    overrides = sys.argv[1:]
    config = build_config(overrides)
    print(config)
    config.launch(follow=True, torchrun=False)
