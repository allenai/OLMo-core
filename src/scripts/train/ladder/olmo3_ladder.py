import argparse
import logging
import sys
import textwrap
from pathlib import Path

import rich

import olmo_core.io as io
from olmo_core.data import DataMix, TokenizerConfig
from olmo_core.data.composable import *
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.internal.common import build_launch_config, get_gpu_type, get_root_dir
from olmo_core.launch.beaker import (
    BeakerLaunchConfig,
    BeakerPriority,
    OLMoCoreBeakerImage,
)
from olmo_core.model_ladder2 import *
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    commands = [
        "dry_run",
        "benchmark",
        "launch_benchmark",
        "run",
        "launch_run",
        "status",
        "metrics",
    ]
    command = sys.argv[1] if len(sys.argv) >= 2 else "help"

    parser = argparse.ArgumentParser(
        sys.argv[0],
        usage=f"python {sys.argv[0]} [CMD] [OPTIONS...]",
        description=textwrap.dedent(
            """
            Launch and manage a ladder experiment on Beaker.
            """
        ),
        epilog=textwrap.dedent(
            f"""
            examples:
              ❯ python {sys.argv[0]} dry_run --size=190M
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
    parser.add_argument(
        "cmd",
        type=str,
        choices=commands,
        help="The command to execute.",
    )
    parser.add_argument(
        "--size",
        choices=list(TransformerSize),
        required=command
        in {"dry_run", "benchmark", "launch_benchmark", "run", "launch_run", "metrics"},
        help="The model size.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="olmo3-ladder",
        help="A name to assign to the ladder experiment.",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        choices=["ai2/augusta", "ai2/jupiter", "ai2/titan"],
        default="ai2/augusta",
        help="The Beaker cluster to launch each run on.",
    )
    parser.add_argument(
        "--max-gpus",
        type=int,
        default=64,
        help="The maximum number of GPUs to use for the ladder.",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="ai2/oe-t-ladder",
        help="The Beaker workspace to use.",
    )
    parser.add_argument(
        "--budget",
        type=str,
        default="ai2/oe-base",
        help="The Beaker budget to use.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=8 * 1024,
        help="The sequence length to configure the ladder with.",
    )
    parser.add_argument(
        "--chinchilla-multiple",
        type=float,
        default=4.0,
        help="The Chinchilla multiple to use for the ladder.",
    )
    parser.add_argument(
        "--beaker-image",
        choices=list(OLMoCoreBeakerImage),
        default=OLMoCoreBeakerImage.stable,
        help="The Beaker image to use.",
    )
    parser.add_argument(
        "--priority",
        choices=[p.value for p in BeakerPriority],
        default=BeakerPriority.normal,
        help="The priority level.",
    )
    parser.add_argument(
        "--preemptible",
        action=argparse.BooleanOptionalAction,
        help="""If the job should be preemptible.""",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="""Allow launching with uncommitted changes.""",
        default=False,
    )
    parser.add_argument(
        "--show-model",
        action="store_true",
        help="Show the model config.",
        default=False,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("~/Downloads").expanduser(),
        help="""A local directory to store artifacts in like metrics.""",
    )

    # Make sure the command is in the right position, otherwise the way we build the launch
    # config would fail.
    if command not in commands:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()


def configure_ladder(args: argparse.Namespace) -> ModelLadder:
    tokenizer = TokenizerConfig.dolma2()
    instance_sources: list[InstanceSourceConfig] = [
        ConcatAndChunkInstanceSourceConfig(
            sources=[
                NumpyDocumentSourceMixConfig(
                    tokenizer=tokenizer,
                    mix=DataMix.OLMo_mix_0925,
                    mix_base_dir=get_root_dir(args.cluster),
                )
            ],
            sequence_length=args.sequence_length,
        ),
    ]
    ladder = ModelLadder(
        name=args.name,
        dir=str(io.join_path(get_root_dir(args.cluster), "model-ladders", args.name)),
        sizes=list(TransformerSize),
        max_devices=args.max_gpus,
        device_type=get_gpu_type(args.cluster),
        model_configurator=TransformerModelConfigurator(),
        run_configurator=WSDSChinchillaRunConfigurator(
            chinchilla_multiple=args.chinchilla_multiple
        ),
        sequence_length=args.sequence_length,
        tokenizer=tokenizer,
        instance_sources=instance_sources,
        data_loader=ComposableDataLoaderConfig(num_workers=8),
    )
    if args.show_model:
        log.info("Model config:")
        log.info(ladder.get_model_config(args.size))
    return ladder


def configure_launcher(
    args: argparse.Namespace, ladder: ModelLadder, cmd: str
) -> BeakerLaunchConfig:
    ladder = configure_ladder(args)
    num_gpus = ladder.get_num_devices(args.size)
    assert (num_gpus % 8 == 0) or num_gpus < 8
    launch_config = build_launch_config(
        cmd=[sys.argv[0], cmd] + sys.argv[2:],
        name=f"{args.name}-{args.size}",
        num_nodes=max(num_gpus // 8, 1),
        cluster=args.cluster,
        workspace=args.workspace,
        beaker_image=args.beaker_image,
        budget=args.budget,
        nccl_debug="VERSION",
    )
    if num_gpus < 8:
        launch_config.num_gpus = num_gpus
    launch_config.priority = BeakerPriority(args.priority)
    if args.preemptible is not None:
        launch_config.preemptible = args.preemptible
    launch_config.allow_dirty = args.allow_dirty
    return launch_config


def main():
    args = parse_args()
    if args.cmd == "dry_run":
        dry_run(args)
    elif args.cmd == "benchmark":
        benchmark(args)
    elif args.cmd == "launch_benchmark":
        launch_benchmark(args)
    elif args.cmd == "run":
        run(args)
    elif args.cmd == "launch_run":
        launch_run(args)
    elif args.cmd == "status":
        status(args)
    elif args.cmd == "metrics":
        metrics(args)
    else:
        raise NotImplementedError(f"Command '{args.cmd}' is not implemented.")


def dry_run(args: argparse.Namespace):
    prepare_cli_environment()
    ladder = configure_ladder(args)
    ladder.dry_run(args.size)


def benchmark(args: argparse.Namespace):
    ladder = configure_ladder(args)
    ladder.run_benchmark(args.size)


def launch_benchmark(args: argparse.Namespace):
    prepare_cli_environment()
    ladder = configure_ladder(args)
    launcher = configure_launcher(args, ladder, "benchmark")
    launcher.launch(follow=True, slack_notifications=False)


def run(args: argparse.Namespace):
    ladder = configure_ladder(args)
    ladder.run(args.size)


def launch_run(args: argparse.Namespace):
    prepare_cli_environment()
    ladder = configure_ladder(args)

    # Check status of run. Don't do anything if final checkpoint already exist.
    checkpoints = ladder.get_checkpoints(args.size)
    if not checkpoints:
        raise OLMoConfigurationError(
            f"Run for size {args.size} has no configured checkpoint intervals."
        )
    elif checkpoints[-1].exists:
        rich.get_console().print(
            f"[b green]✔[/] Run for size [green]{args.size}[/] already complete. "
            f"Final checkpoint can be found at [u blue]{checkpoints[-1].path}[/]",
            highlight=False,
        )
        return

    launcher = configure_launcher(args, ladder, "run")
    log.info(f"Launching ladder run for size {args.size}...")
    log.info(f"Results will be saved to {ladder.get_save_folder(args.size)}")
    launcher.launch(follow=True, slack_notifications=False)


def status(args: argparse.Namespace):
    prepare_cli_environment()
    ladder = configure_ladder(args)
    sizes = [args.size] if args.size else ladder.sizes
    for size in sizes:
        print()
        checkpoints = ladder.get_checkpoints(size)
        if not checkpoints:
            rich.get_console().print(
                f"[b yellow]Run for size {size} has no configured checkpoint intervals.[/]",
                highlight=False,
            )
            continue

        max_step_completed = 0
        max_step = -1
        checkpoint_displays = []
        for ckpt in checkpoints:
            max_step = max(max_step, ckpt.step)
            if ckpt.exists:
                max_step_completed = max(max_step_completed, ckpt.step)
            checkpoint_displays.append(ckpt.display())

        assert max_step > 0
        pct_complete = round((max_step_completed / max_step) * 100.0)
        color = "green" if pct_complete == 100 else "yellow"
        completion_display = f"[b {color}]Run for size {size}, {pct_complete}% complete:[/]"
        rich.get_console().print(
            f"{completion_display}\n" + "\n".join(checkpoint_displays),
            highlight=False,
        )


def metrics(args: argparse.Namespace):
    prepare_cli_environment()
    ladder = configure_ladder(args)
    df = ladder.get_metrics(args.size)
    if df is not None:
        path = io.join_path(args.output_dir, f"metrics_{args.size}.pkl")
        df.to_pickle(path)
        rich.get_console().print(
            f"[b green]✔[/] Metrics for size [green]{args.size}[/] saved to [u blue]{path}[/]\n"
            f"Use pandas to load and analyze the metrics, e.g.:\n\n"
            f"    import pandas as pd\n"
            f"    df = pd.read_pickle('{path}')\n",
            highlight=False,
        )
    else:
        rich.get_console().print(
            f"[b yellow]Run for size {args.size} has no metrics yet.[/]",
            highlight=False,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
