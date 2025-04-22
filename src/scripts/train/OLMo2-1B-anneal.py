"""
This script can be used to launch an annealing run for the 1B model on Beaker.

Unlike the 32B script, this one doesn't change the data mix.

Run the script without any arguments to see usage info.
"""

import importlib
import json
import logging
import sys
from pathlib import Path

import rich
import torch

from olmo_core.internal.experiment import SubCmd, build_config
from olmo_core.io import resource_path
from olmo_core.optim import CosWithWarmup, LinearWithWarmup
from olmo_core.train import Duration, LoadStrategy

olmo1b = importlib.import_module("OLMo2-1B")

log = logging.getLogger(__name__)

if __name__ == "__main__":
    USAGE = f"""
Anneal a 1B model.

[yellow]Usage:[/] [i blue]python[/] [i cyan]{sys.argv[0]}[/] [i b magenta]launch|train|dry_run[/] [i b]RUN_NAME PRETRAIN_CHECKPOINT CLUSTER[/] [i][OVERRIDES...][/]

[b]Subcommands[/]
[b magenta]launch:[/]      Launch the script on Beaker with the [b magenta]train[/] subcommand.
[b magenta]train:[/]       Run the trainer. You usually shouldn't invoke the script with this subcommand directly.
             Instead use the [b magenta]launch[/] cmd to submit it to Beaker or run it via torchrun if you know what you're doing.
[b magenta]train_single:[/]       Run the trainer on a single device (GPU, CPU, MPS). num_nodes is ignored.
[b magenta]dry_run:[/]     Print the config for debugging.

[b]Examples[/]
$ [i]python {sys.argv[0]} launch run01 gs://ai2-llm/checkpoints/dirkg/baseline27/step290601/ ai2/augusta-google-1 --launch.num_nodes=2[/]
""".strip()

    # Parse command line arguments.
    if len(sys.argv) < 5 or sys.argv[1] not in ("launch", "train", "train_single", "dry_run"):
        rich.get_console().print(USAGE, highlight=False)
        sys.exit(1)

    script, cmd, run_name, checkpoint, cluster, *overrides = sys.argv
    cmd = SubCmd(cmd)
    cmd.prepare_environment()

    # Get step number and max steps to infer where the learning rate left off.
    checkpoint = checkpoint.rstrip("/")
    checkpoint_train_state = torch.load(
        resource_path(f"{checkpoint}/train", "rank0.pt"), weights_only=False
    )
    last_pretrain_step: int = checkpoint_train_state["global_step"]
    max_pretrain_steps: int = checkpoint_train_state["max_steps"]
    with resource_path(checkpoint, "config.json").open() as f:
        checkpoint_config = json.load(f)
    run_name = f"{checkpoint_config['run_name']}-from{last_pretrain_step}--{run_name}"

    config = build_config(
        script,
        cmd,
        run_name,
        cluster,
        overrides,
        global_batch_size=checkpoint_config["data_loader"]["global_batch_size"],
        model_config_builder=olmo1b.build_model_config,
        train_module_config_builder=olmo1b.build_train_module_config,
        trainer_config_builder=olmo1b.build_trainer_config,
        sequence_length=checkpoint_config["train_module"]["max_sequence_length"],
        include_default_evals=False,
        intra_document_masking=False,
        include_instance_filter=False,
    )

    config.trainer.load_path = checkpoint
    config.trainer.load_strategy = LoadStrategy.always

    # Now infer the learning rate.
    base_lr = checkpoint_config["train_module"]["optim"]["lr"]
    scheduler_config = checkpoint_config["train_module"]["scheduler"]
    assert scheduler_config.pop("_CLASS_") == f"{CosWithWarmup.__module__}.{CosWithWarmup.__name__}"
    scheduler = CosWithWarmup(**scheduler_config)
    starting_lr = float(scheduler.get_lr(base_lr, last_pretrain_step, max_pretrain_steps))
    config.train_module.optim.lr = starting_lr
    config.train_module.scheduler = LinearWithWarmup(warmup_steps=0, alpha_f=0.0)
    config.trainer.max_duration = Duration.tokens(int(50e9))
    log.info(
        f"Will anneal from checkpoint at step {last_pretrain_step:,d} with an lr of {starting_lr:.6f}"
    )

    # fix up the launch config
    script_path = Path(config.launch.cmd[0])
    olmo1_path = Path(olmo1b.__file__)
    config.launch.cmd[0] = str(script_path.with_stem(olmo1_path.stem))

    cmd.run(config)
