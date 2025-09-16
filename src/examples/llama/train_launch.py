"""
An example of how to launch the training script on Beaker.
Run this script without arguments to see the usage.
"""

import argparse
import textwrap
from typing import List

from beaker import Priority as BeakerJobPriority

from olmo_core.launch.beaker import (
    BeakerEnvVar,
    BeakerLaunchConfig,
    BeakerWekaBucket,
    OLMoCoreBeakerImage,
)
from olmo_core.utils import generate_uuid, prepare_cli_environment


def build_config(opts, overrides: List[str]) -> BeakerLaunchConfig:
    env_vars: List[BeakerEnvVar] = []
    if opts.debug:
        env_vars.append(BeakerEnvVar(name="CUDA_LAUNCH_BLOCKING", value="1"))
        env_vars.append(BeakerEnvVar(name="NCCL_DEBUG", value="INFO"))
    return BeakerLaunchConfig(
        name=f"{opts.run_name}-{generate_uuid()[:8]}",
        budget=opts.budget,
        cmd=["src/examples/llama/train.py", opts.run_name, *overrides],
        env_vars=env_vars,
        task_name="train",
        description=opts.description,
        clusters=opts.cluster,
        num_nodes=opts.nodes,
        num_gpus=opts.gpus,
        preemptible=opts.preemptible,
        priority=opts.priority,
        beaker_image=opts.beaker_image,
        workspace=opts.workspace,
        allow_dirty=opts.allow_dirty,
        weka_buckets=[
            BeakerWekaBucket(bucket=bucket, mount=f"/weka/{bucket}") for bucket in (opts.weka or [])
        ],
    )


def parse_args():
    parser = argparse.ArgumentParser(
        "train_launch",
        usage="python src/examples/llama/train_launch.py RUN_NAME [OPTIONS...] [CONFIG_OVERRIDES...]",
        description="Launch example training run on Beaker",
        epilog=textwrap.dedent(
            """
            examples:
              ❯ python src/examples/llama/train_launch.py test-run-01 --gpus=2 --trainer.hard_stop='{value: 100, unit: steps}'
            """
        ),
        formatter_class=type(  # type: ignore[arg-type]
            "CustomFormatter",
            (
                argparse.ArgumentDefaultsHelpFormatter,
                argparse.RawDescriptionHelpFormatter,
            ),
            {},
        ),
    )
    parser.add_argument("run_name", type=str, help="The name of the run.")
    parser.add_argument("--gpus", type=int, default=2, help="The number of GPUs per node/replica.")
    parser.add_argument("--nodes", type=int, default=1, help="The number of nodes/replicas.")
    parser.add_argument("--budget", type=str, help="The Beaker budget account to use.")
    parser.add_argument("--workspace", type=str, help="The Beaker workspace to use.")
    parser.add_argument(
        "--description", type=str, help="A description to assign to the Beaker experiment."
    )
    parser.add_argument(
        "--cluster",
        type=str,
        nargs="*",
        default=["ai2/jupiter", "ai2/ceres", "ai2/saturn", "ai2/prometheus"],
        help="""Clusters to launch on (multiple allowed).""",
    )
    parser.add_argument(
        "--priority",
        choices=[p.value for p in BeakerJobPriority],
        default=BeakerJobPriority.normal,
        help="The priority level.",
    )
    parser.add_argument(
        "--preemptible",
        action="store_true",
        help="""If the job should be preemptible.""",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="""Allow launching with uncommitted changes.""",
        default=False,
    )
    parser.add_argument(
        "--beaker-image",
        type=str,
        default=OLMoCoreBeakerImage.stable,
        help="""The Beaker image to use.""",
    )
    parser.add_argument("--weka", type=str, nargs="*", help="Weka buckets to mount at '/weka/'.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="""Set debugging env vars, like `CUDA_LAUNCH_BLOCKING=1`.""",
    )
    return parser.parse_known_args()


def main():
    prepare_cli_environment()
    opts, overrides = parse_args()
    print(build_config(opts, overrides))  # .launch(follow=True)


if __name__ == "__main__":
    main()
