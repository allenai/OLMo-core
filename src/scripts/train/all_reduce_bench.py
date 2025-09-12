"""
Run an all-reduce benchmark. Run this script without any arguments to see usage info.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import List

import torch
import torch.distributed as dist

from olmo_core.config import Config, StrEnum
from olmo_core.distributed.utils import get_local_rank, get_world_size
from olmo_core.launch.beaker import BeakerLaunchConfig, OLMoCoreBeakerImage
from olmo_core.train import prepare_training_environment, teardown_training_environment
from olmo_core.utils import generate_uuid, prepare_cli_environment

log = logging.getLogger(__name__)


TRIALS = 5

# these emulate the payload which will become a M * N * 4-sized tensor below
N = 500000
M = 2000


class SubCmd(StrEnum):
    launch = "launch"
    run = "run"
    dry_run = "dry_run"

    def prepare_environment(self):
        if self in (SubCmd.launch, SubCmd.dry_run):
            prepare_cli_environment()
        elif self == SubCmd.run:
            prepare_training_environment()
        else:
            raise NotADirectoryError(self)

    def execute(self, config: BenchmarkConfig):
        log.info(config)
        if self == SubCmd.launch:
            config.launch.launch(follow=True)
        elif self == SubCmd.dry_run:
            pass
        elif self == SubCmd.run:
            try:
                # Show env vars for debugging.
                for var_name in sorted(os.environ.keys()):
                    var_val = os.environ[var_name]
                    log.info(f"Env var {var_name} set to '{var_val}'")

                mat = torch.rand(N, M, dtype=torch.float32).cuda(get_local_rank())

                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                # do a few warm up iterations
                for i in range(2):
                    timed_allreduce(mat, start_event, end_event)

                # real benchmark
                algbw_gather = []
                for i in range(TRIALS):
                    log.info(f"{i + 1}")
                    algbw_gather += timed_allreduce(mat, start_event, end_event)

                algbw = torch.mean(torch.stack(algbw_gather))

                # the 2*(n-1)/n busbw correction factor specific to all-reduce is explained here:
                # https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#allreduce
                # busbw reflects how optimally the hardware is used
                n = dist.get_world_size()
                busbw = algbw * (2 * (n - 1) / n)

                log.info(
                    f"The average bandwidth of all_reduce with a {M * N * 4 / 1e9}GB payload ({TRIALS} trials, {n} ranks):\n"
                    f"algbw: {algbw / 1e9:.3f} GBps ({algbw * 8 / 1e9:.1f} Gbps)\n"
                    f"busbw: {busbw / 1e9:.3f} GBps ({busbw * 8 / 1e9:.1f} Gbps)\n"
                )
            finally:
                teardown_training_environment()
        else:
            raise NotADirectoryError(self)


@dataclass
class BenchmarkConfig(Config):
    launch: BeakerLaunchConfig


def build_config(script: str, run_name: str, cluster: str, overrides: List[str]) -> BenchmarkConfig:
    launch_config = BeakerLaunchConfig(
        name=f"{run_name}-{generate_uuid()[:8]}",
        budget="ai2/oe-base",
        cmd=[script, SubCmd.run, run_name, cluster, *overrides],
        task_name="benchmark",
        workspace="ai2/OLMo-core",
        clusters=[cluster],
        beaker_image=OLMoCoreBeakerImage.stable,
        num_nodes=1,
        num_gpus=8,
        allow_dirty=False,
        setup_steps=[
            # Clone repo.
            'git clone "$REPO_URL" .',
            'git checkout "$GIT_REF"',
            "git submodule update --init --recursive",
            # Setup python environment.
            "conda shell.bash activate base",
            "pip install -e '.[all]'",
            "pip freeze",
        ],
    )

    return BenchmarkConfig(launch=launch_config).merge(overrides)


def timed_allreduce(mat, start_event, end_event):
    dist.barrier()
    start_event.record()
    dist.all_reduce(mat)
    end_event.record()

    torch.cuda.synchronize()
    duration = start_event.elapsed_time(end_event) / 1000

    size = M * N * 4  # 4 is 4 bytes in fp32
    # note that this is following the same math as NVIDIA/nccl-tests
    algbw = torch.tensor([size / duration]).cuda(get_local_rank())

    # calculate mean across all ranks
    dist.reduce(algbw, dst=0, op=dist.ReduceOp.SUM)
    algbw /= get_world_size()

    return algbw


def main():
    usage = f"""
[yellow]Usage:[/] [i blue]python[/] [i cyan]{sys.argv[0]}[/] [i b magenta]{"|".join(SubCmd)}[/] [i b]RUN_NAME CLUSTER[/] [i][OVERRIDES...][/]

[b]Subcommands[/]
[b magenta]launch:[/]      Launch the benchmark on Beaker with the [b magenta]run[/] subcommand.
[b magenta]run:[/]         Run the benchmark. You usually shouldn't invoke the script with this subcommand directly.
             Instead use [b magenta]launch[/] or run it with torchrun.
[b magenta]dry_run:[/]     Pretty print the config and exit.

[b]Examples[/]
$ [i]python {sys.argv[0]} {SubCmd.launch} run01 ai2/pluto-cirrascale --launch.num_nodes=2[/]
    """.strip()

    if len(sys.argv) < 4 or sys.argv[1] not in set(SubCmd):
        import rich

        rich.get_console().print(usage, highlight=False)
        sys.exit(1)

    script, cmd, run_name, cluster, *overrides = sys.argv

    cmd = SubCmd(cmd)
    cmd.prepare_environment()

    config = build_config(
        script,
        run_name,
        cluster,
        overrides,
    )

    cmd.execute(config)


if __name__ == "__main__":
    main()
