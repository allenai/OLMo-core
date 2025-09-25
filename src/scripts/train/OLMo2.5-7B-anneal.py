import importlib.util
import json
import logging
import sys
from math import ceil
from pathlib import Path
from typing import Dict, Optional

import torch

from olmo_core.distributed.checkpoint import load_state_dict
from olmo_core.internal.experiment import (
    CliContext,
    CommonComponents,
    SubCmd,
    build_config,
)
from olmo_core.io import join_path, resource_path
from olmo_core.launch.beaker import OLMoCoreBeakerImage
from olmo_core.nn.transformer import TransformerActivationCheckpointingMode
from olmo_core.optim import SchedulerUnits
from olmo_core.optim.scheduler import WSD
from olmo_core.train import Duration, LoadStrategy, TrainerConfig
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerTrainModuleConfig,
)

log = logging.getLogger(__name__)


if __name__ == "__main__":
    usage = f"""
Performs anneals on original data. Resumes data loader state and optimizer state.

[yellow]Usage:[/] [i blue]python[/] [i cyan]{sys.argv[0]}[/] [i b magenta]{"|".join(SubCmd)}[/] [i b]ORIGINAL_CHECKPOINT LENGTH CLUSTER[/] [i][OVERRIDES...][/]

[b]Subcommands[/]
[b magenta]launch:[/]      Launch the script on Beaker with the [b magenta]train[/] subcommand.
[b magenta]train:[/]       Run the trainer. You usually shouldn't invoke the script with this subcommand directly.
             Instead use [b magenta]launch[/] or run it with torchrun.
[b magenta]train_single:[/]       Run the trainer on a single device (GPU, CPU, MPS). num_nodes is ignored.
[b magenta]dry_run:[/]     Pretty print the config and exit.

[b]Examples[/]
$ [i]python {sys.argv[0]} {SubCmd.launch} gs://ai2-llm/checkpoints/OLMo25/step238000/ 100e9 ai2/augusta --launch.num_nodes=2[/]
    """.strip()

    if len(sys.argv) < 5 or sys.argv[1] not in set(SubCmd):
        import rich

        rich.get_console().print(usage, highlight=False)
        sys.exit(1)

    script, cmd, original_checkpoint, length, cluster, *overrides = sys.argv

    length_in_tokens = int(float(length))
    log.info(f"Training for {length_in_tokens} tokens ({length_in_tokens / 1_000_000_000}B)")

    # Load OLMo 2.5 7B module
    o25_spec = importlib.util.spec_from_file_location(
        "OLMo2.5-7B", Path(__file__).parent / "OLMo2.5-7B.py"
    )
    assert o25_spec is not None and o25_spec.loader is not None, "Failed to load OLMo2.5-7B module"
    o25_module = importlib.util.module_from_spec(o25_spec)
    assert o25_module is not None, "Failed to create OLMo2.5-7B module"
    sys.modules["OLMo2.5-7B"] = o25_module
    o25_spec.loader.exec_module(o25_module)
    batch_size = o25_module.GLOBAL_BATCH_SIZE

    # load state from the original training run
    trainer_state_file = resource_path(join_path(original_checkpoint, "train"), "rank0.pt")
    trainer_state = torch.load(trainer_state_file, weights_only=False)
    config_file = resource_path(original_checkpoint, "config.json")
    with open(config_file, "rb") as f:
        config = json.load(f)

    global_step = trainer_state["global_step"]
    run_name = f"{config['run_name']}-from{global_step}"
    sequence_length = trainer_state["data_loader"]["sequence_length"]
    length_in_steps = ceil(length_in_tokens / batch_size)

    # determine last learning rate
    param = "embeddings.weight"
    key = f"optim.param_groups.{param}.lr"
    state_dict: Dict[str, Optional[float]] = {key: None}
    load_state_dict(join_path(original_checkpoint, "model_and_optim"), state_dict)
    assert state_dict[key] is not None
    lr = float(state_dict[key])  # type: ignore
    log.info(f"Starting learning rate is {lr}")

    cmd = SubCmd(cmd)
    cli_context = CliContext(script, cmd, run_name, cluster, overrides)

    def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
        config = o25_module.build_train_module_config(common)

        # configure lr
        config.optim.lr = lr
        config.scheduler = WSD(
            units=SchedulerUnits.steps,
            warmup=global_step,
            warmup_fraction=None,
            decay=length_in_steps,
            decay_fraction=None,
        )

        # performance settings
        config.rank_microbatch_size = 8192 * 2
        config.dp_config.shard_degree = 32
        config.ac_config = TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget, activation_memory_budget=0.85
        )

        return config

    def build_trainer_config(common: CommonComponents) -> TrainerConfig:
        config = o25_module.build_trainer_config(common)

        config.load_path = original_checkpoint
        config.load_strategy = LoadStrategy.always
        config.max_duration = Duration.steps(global_step + length_in_steps)
        config.hard_stop = None

        config.callbacks["checkpointer"].save_interval = 10000
        config.callbacks["checkpointer"].ephemeral_save_interval = 500

        # performance settings
        config.metrics_collect_interval = 50

        return config

    config = build_config(
        cli_context,
        global_batch_size=batch_size,
        max_sequence_length=sequence_length,
        model_config_builder=o25_module.build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        finalize_config=None,
        include_default_evals=False,
        intra_document_masking=False,
        include_instance_filter=False,
        beaker_image=OLMoCoreBeakerImage.stable,
        num_nodes=16,
        beaker_workspace="ai2/OLMo_3",
    )
    assert config.launch is not None
    config.launch.cmd = [script, "train", original_checkpoint, length, cluster] + overrides

    cmd.prepare_environment(config)
    cmd.run(config)
