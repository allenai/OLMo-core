import argparse
import logging
import sys
import textwrap
from pathlib import Path
from typing import Callable, Type

import rich

import olmo_core.io as io
from olmo_core.data import DataMix, TokenizerConfig
from olmo_core.data.composable import *
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.launch.beaker import (
    BeakerLaunchConfig,
    BeakerPriority,
    OLMoCoreBeakerImage,
    is_running_in_beaker_batch_job,
)
from olmo_core.model_ladder import *
from olmo_core.utils import prepare_cli_environment

from .common import build_launch_config, get_gpu_type, get_root_dir

log = logging.getLogger(__name__)


def parse_args(
    configure_ladder: Callable[[argparse.Namespace], ModelLadder],
    *,
    size_enum: Type[TransformerSize] = TransformerSize,
    add_additional_args: Callable[[str, argparse.ArgumentParser], None] | None = None,
) -> argparse.Namespace:
    formatter_class = type(
        "CustomFormatter",
        (
            argparse.ArgumentDefaultsHelpFormatter,
            argparse.RawDescriptionHelpFormatter,
        ),
        {},
    )

    base_parser = argparse.ArgumentParser(
        sys.argv[0],
        usage=f"python {sys.argv[0]} [CMD] [OPTIONS...]",
        description=textwrap.dedent(
            f"""
            Launch and manage a ladder experiment on Beaker.

            examples:
            • See a description of all options for a certain command:
              ❯ python {sys.argv[0]} dry-run --help
            • Run a dry run for the {list(size_enum)[0]} model size:
              ❯ python {sys.argv[0]} dry-run --size={list(size_enum)[0]}
            """
        ),
        epilog=textwrap.dedent(
            """
            notes:
            • The command (e.g. 'dry-run', 'launch', etc) must always be the first argument.
            """
        ),
        formatter_class=formatter_class,  # type: ignore[arg-type]
    )
    sub_parsers = base_parser.add_subparsers(dest="cmd")

    def add_general_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--name",
            type=str,
            default="olmo3-ladder",
            help="A name to assign to the ladder experiment.",
        )
        parser.add_argument(
            "--project",
            type=str,
            default=None,
            help="""A project name to associate with the ladder runs.
            Used by the W&B logger, for example.""",
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
            "--lr-multiplier",
            type=float,
            default=1.0,
            help="A multiplier to apply to the default learning rate.",
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
            "--rank-mbz",
            type=int,
            default=None,
            help="Override the rank micro-batch size, in instances.",
        )

    def add_launch_args(parser: argparse.ArgumentParser):
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
            "--beaker-image",
            default=OLMoCoreBeakerImage.stable,
            type=str,
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
            help="If the job should be preemptible.",
        )
        parser.add_argument(
            "--allow-dirty",
            action="store_true",
            help="Allow launching with uncommitted changes.",
            default=False,
        )
        parser.add_argument(
            "--slack-notifications",
            action="store_true",
            help="Enable Slack notifications for job status updates.",
            default=False,
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Print the launch config without launching the run.",
            default=False,
        )

    sub_commands: dict[str, argparse.ArgumentParser] = {}

    def add_sub_command(name: str, func: Callable[[argparse.Namespace], None], help: str):
        sub_commands[name] = sub_parsers.add_parser(
            name,
            help=help,
            formatter_class=formatter_class,  # type: ignore[arg-type]
        )
        sub_commands[name].set_defaults(
            func=func, configure_ladder=configure_ladder, size_enum=size_enum
        )

    add_sub_command(
        "dry-run",
        dry_run,
        "Simulate the ladder run for a given model size, showing the LR schedule plot.",
    )
    add_sub_command(
        "benchmark",
        benchmark,
        """
        Run a benchmark for a given model size locally.
        This usually isn't invoked directly, but rather via 'launch-benchmark' or 'torchrun'.
        """,
    )
    add_sub_command(
        "launch-benchmark",
        launch_benchmark,
        """
        Launch a Beaker job to run a benchmark for a given model size.
        """,
    )
    add_sub_command(
        "run",
        run,
        """
        Run the ladder for a given model size locally.
        This usually isn't invoked directly, but rather via 'launch' or 'torchrun'.
        """,
    )
    add_sub_command(
        "launch",
        launch,
        "Launch a Beaker job to run the ladder for a given model size.",
    )
    add_sub_command(
        "launch-all",
        launch_all,
        "Launch Beaker jobs to run the all model sizes of the ladder.",
    )
    add_sub_command(
        "status",
        status,
        "Check the status of the ladder run for a given model size.",
    )
    add_sub_command(
        "metrics",
        metrics,
        "Download the metrics for a given model size.",
    )
    add_sub_command(
        "metrics-all",
        metrics_all,
        "Download the metrics for all ladder rung sizes.",
    )

    for cmd, parser in sub_commands.items():
        add_general_args(parser)

        if cmd != "launch-all":
            parser.add_argument(
                "--size",
                choices=list(size_enum),
                required=cmd
                in {"dry-run", "benchmark", "launch-benchmark", "run", "launch", "metrics"},
                help="The model size.",
            )

        if cmd in {"launch-all", "status", "metrics-all"}:
            parser.add_argument(
                "--max-size",
                choices=list(size_enum),
                default=None,
                help="The maximum model size. If not specified, status/metrics for all sizes will be shown.",
            )

        if cmd in {"dry-run", "metrics", "metrics-all"}:
            parser.add_argument(
                "--output-dir",
                type=Path,
                default=Path("~/Downloads").expanduser(),
                help="""A local directory to store artifacts in like metrics or the LR schedule plot.""",
            )

        if cmd == "dry-run":
            parser.add_argument(
                "--show-plot",
                action="store_true",
                help="""Open a live display of the LR schedule plot.""",
            )
            parser.add_argument(
                "--show-model",
                action="store_true",
                help="Show the model config.",
                default=False,
            )

        if "launch" in cmd:
            add_launch_args(parser)

        if add_additional_args is not None:
            add_additional_args(cmd, parser)

    # Make sure the command is in the right position, otherwise the way we build the launch
    # config would fail.
    if len(sys.argv) < 2 or sys.argv[1] not in sub_commands:
        base_parser.print_help()
        sys.exit(1)

    # If running in a Beaker batch job only parse known args to ignore extra args from the launch command,
    # otherwise parse strictly.
    if is_running_in_beaker_batch_job():
        args, _ = base_parser.parse_known_args()
        return args
    else:
        return base_parser.parse_args()


def get_default_ladder_factory(
    configure_model: Callable[[argparse.Namespace], ModelConfigurator],
    configure_run: Callable[[argparse.Namespace], RunConfigurator] | None = None,
) -> Callable[[argparse.Namespace], ModelLadder]:
    def factory(args: argparse.Namespace) -> ModelLadder:
        tokenizer = TokenizerConfig.dolma2()
        instance_sources: list[InstanceSourceConfig] = [
            ConcatAndChunkInstanceSourceConfig(
                sources=[
                    NumpyDocumentSourceMixConfig(
                        tokenizer=tokenizer,
                        mix=DataMix.OLMo_mix_0925,
                        mix_base_dir="gs://ai2-llm/",
                    )
                ],
                sequence_length=args.sequence_length,
            ),
        ]
        ladder = ModelLadder(
            name=args.name,
            project=args.project,
            dir=str(io.join_path(get_root_dir(args.cluster), "model-ladders", args.name)),
            sizes=list(TransformerSize),
            max_devices=args.max_gpus,
            device_type=get_gpu_type(args.cluster),
            model_configurator=configure_model(args),
            run_configurator=configure_run(args)
            if configure_run is not None
            else WSDSChinchillaRunConfigurator(
                chinchilla_multiple=args.chinchilla_multiple,
                lr_multiplier=args.lr_multiplier,
            ),
            sequence_length=args.sequence_length,
            tokenizer=tokenizer,
            instance_sources=instance_sources,
            data_loader=ComposableDataLoaderConfig(
                num_workers=8, instance_filter_config=InstanceFilterConfig()
            ),
        )
        return ladder

    return factory


def main(
    configure_ladder: Callable[[argparse.Namespace], ModelLadder] | None | None = None,
    configure_model: Callable[[argparse.Namespace], ModelConfigurator] | None = None,
    configure_run: Callable[[argparse.Namespace], RunConfigurator] | None = None,
    size_enum: Type[TransformerSize] = TransformerSize,
    add_additional_args: Callable[[str, argparse.ArgumentParser], None] | None = None,
):
    if configure_ladder is None:
        assert configure_model is not None, (
            "configure_model is required if configure_ladder is unspecified"
        )

        configure_ladder = get_default_ladder_factory(configure_model, configure_run)
    else:
        assert configure_model is None and configure_run is None, (
            "configure_model / configure_run and mutually exclusive with configure_ladder"
        )

    args = parse_args(
        configure_ladder, size_enum=size_enum, add_additional_args=add_additional_args
    )
    args.func(args)


def configure_launcher(
    args: argparse.Namespace, ladder: ModelLadder, cmd: str, size: str | None = None
) -> BeakerLaunchConfig:
    cmd_to_run = [sys.argv[0], cmd] + sys.argv[2:]

    if size is not None:
        cmd_to_run += ["--size", str(size)]
    else:
        assert args.size
        size = args.size
    assert size is not None

    num_gpus = ladder.get_num_devices(size)
    assert (num_gpus % 8 == 0) or num_gpus < 8

    launch_config = build_launch_config(
        cmd=cmd_to_run,
        name=f"{args.name}-{size}",
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


def dry_run(args: argparse.Namespace):
    prepare_cli_environment()
    ladder = args.configure_ladder(args)
    if args.show_model:
        log.info("Model config:")
        log.info(ladder.get_model_config(args.size))
    ladder.dry_run(
        args.size,
        show_plot=args.show_plot,
        save_plot=io.join_path(args.output_dir / f"lr_schedule_{args.size}.png"),
    )


def benchmark(args: argparse.Namespace):
    ladder = args.configure_ladder(args)
    ladder.run_benchmark(args.size)


def launch_benchmark(args: argparse.Namespace):
    prepare_cli_environment()
    ladder = args.configure_ladder(args)
    launcher = configure_launcher(args, ladder, "benchmark")
    launcher.launch(follow=True, slack_notifications=args.slack_notifications)


def run(args: argparse.Namespace):
    ladder = args.configure_ladder(args)
    ladder.run(args.size)


def _launch_run(
    ladder: ModelLadder,
    launcher: BeakerLaunchConfig,
    size: str,
    follow: bool = True,
    slack_notifications: bool = False,
    dry_run: bool = False,
):
    # Check status of run. Don't do anything if final checkpoint already exist.
    checkpoints = ladder.get_checkpoints(size)
    if not checkpoints or checkpoints[-1].step == 0:
        raise OLMoConfigurationError(f"Run for size {size} has no configured checkpoint intervals.")
    elif checkpoints[-1].exists:
        rich.get_console().print(
            f"[b green]✔[/] Run for size [green]{size}[/] already complete. "
            f"Final checkpoint can be found at [u blue]{checkpoints[-1].path}[/]",
            highlight=False,
        )
        return

    if dry_run:
        log.info(f"Launch dry run for size {size}...")
        log.info(f"Results would be saved to {ladder.get_save_folder(size)}")
        rich.get_console().print(launcher)
    else:
        log.info(f"Launching ladder run for size {size}...")
        log.info(f"Results will be saved to {ladder.get_save_folder(size)}")
        launcher.launch(follow=follow, slack_notifications=slack_notifications)


def launch(args: argparse.Namespace):
    prepare_cli_environment()
    ladder = args.configure_ladder(args)
    launcher = configure_launcher(args, ladder, "run")
    _launch_run(
        ladder,
        launcher,
        args.size_enum(args.size),
        follow=True,
        slack_notifications=args.slack_notifications,
        dry_run=args.dry_run,
    )


def launch_all(args: argparse.Namespace):
    prepare_cli_environment()
    ladder = args.configure_ladder(args)
    sizes = [args.size_enum(s) for s in ladder.sizes]
    if args.max_size:
        sizes = [s for s in sizes if s <= args.size_enum(args.max_size)]

    for size in sizes:
        launcher = configure_launcher(args, ladder, "run", size=size)
        _launch_run(
            ladder,
            launcher,
            size,
            follow=False,
            slack_notifications=args.slack_notifications,
            dry_run=args.dry_run,
        )


def status(args: argparse.Namespace):
    prepare_cli_environment()
    ladder = args.configure_ladder(args)

    sizes: list[str]
    if args.size:
        sizes = [args.size_enum(args.size)]
    else:
        sizes = [args.size_enum(s) for s in ladder.sizes]
        if args.max_size:
            sizes = [s for s in sizes if s <= args.size_enum(args.max_size)]

    for size in sizes:
        print()
        checkpoints = ladder.get_checkpoints(size)
        if not checkpoints or checkpoints[-1].step == 0:
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
    ladder = args.configure_ladder(args)
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


def metrics_all(args: argparse.Namespace):
    prepare_cli_environment()
    ladder = args.configure_ladder(args)
    sizes = [args.size_enum(s) for s in ladder.sizes]
    if args.max_size:
        sizes = [s for s in sizes if s <= args.size_enum(args.max_size)]

    saved_paths: list[str] = []
    missing_sizes: list[str] = []

    for size in sizes:
        df = ladder.get_metrics(size)
        if df is not None:
            path = io.join_path(args.output_dir, f"metrics_{size}.pkl")
            df.to_pickle(path)
            rich.get_console().print(
                f"[b green]✔[/] Metrics for size [green]{size}[/] saved to [u blue]{path}[/]",
                highlight=False,
            )
            saved_paths.append(str(path))
        else:
            rich.get_console().print(
                f"[b yellow]Run for size {size} has no metrics yet.[/]",
                highlight=False,
            )
            missing_sizes.append(str(size))

    print()
    if saved_paths:
        rich.get_console().print(
            f"Use pandas to load and analyze the metrics, e.g.:\n\n"
            f"    import pandas as pd\n"
            f"    df = pd.read_pickle('{saved_paths[0]}')\n",
            highlight=False,
        )

    if missing_sizes and not saved_paths:
        sys.exit(1)
