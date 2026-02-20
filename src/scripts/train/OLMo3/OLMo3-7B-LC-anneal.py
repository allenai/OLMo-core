import importlib.util
import json
import logging
import sys
from math import ceil
from pathlib import Path
from typing import Dict, Optional

import torch

from olmo_core.data import NumpyDatasetConfig, NumpyDataLoaderConfig, NumpyPackedFSLDatasetConfig, DataMix
from olmo_core.distributed.checkpoint import load_state_dict
from olmo_core.internal.experiment import (
    CliContext,
    CommonComponents,
    SubCmd,
    build_config, DataComponents,
)
from olmo_core.io import join_path, resource_path
from olmo_core.launch.beaker import OLMoCoreBeakerImage
from olmo_core.nn.rope import YaRNRoPEScalingConfig
from olmo_core.nn.transformer import TransformerActivationCheckpointingMode, TransformerConfig
from olmo_core.optim import SchedulerUnits
from olmo_core.optim.scheduler import WSD
from olmo_core.train import Duration, LoadStrategy, TrainerConfig
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerTrainModuleConfig, TransformerContextParallelConfig,
)

log = logging.getLogger(__name__)


if __name__ == "__main__":
    usage = f"""
Performs anneals with LC data. Resumes optimizer state.

[yellow]Usage:[/] [i blue]python[/] [i cyan]{sys.argv[0]}[/] [i b magenta]{"|".join(SubCmd)}[/] [i b]ORIGINAL_CHECKPOINT LENGTH CLUSTER[/] [i][OVERRIDES...][/]

[b]Subcommands[/]
[b magenta]launch:[/]      Launch the script on Beaker with the [b magenta]train[/] subcommand.
[b magenta]train:[/]       Run the trainer. You usually shouldn't invoke the script with this subcommand directly.
             Instead use [b magenta]launch[/] or run it with torchrun.
[b magenta]train_single:[/]       Run the trainer on a single device (GPU, CPU, MPS). num_nodes is ignored.
[b magenta]dry_run:[/]     Pretty print the config and exit.

[b]Examples[/]
$ [i]python {sys.argv[0]} {SubCmd.launch} gs://ai2-llm/checkpoints/OLMo25/step238000/ 100e9 ai2/jupiter --launch.num_nodes=2[/]
    """.strip()

    if len(sys.argv) < 5 or sys.argv[1] not in set(SubCmd):
        import rich

        rich.get_console().print(usage, highlight=False)
        sys.exit(1)

    script, cmd, original_checkpoint, length, cluster, *overrides = sys.argv

    length_in_tokens = int(float(length))
    log.info(f"Training for {length_in_tokens} tokens ({length_in_tokens / 1_000_000_000}B)")

    # Load OLMo 3 7B module
    olmo3_spec = importlib.util.spec_from_file_location(
        "OLMo3-7B", Path(__file__).parent / "OLMo3-7B.py"
    )
    assert (
        olmo3_spec is not None and olmo3_spec.loader is not None
    ), "Failed to load OLMo3-7B module"
    olmo3_module = importlib.util.module_from_spec(olmo3_spec)
    assert olmo3_module is not None, "Failed to create OLMo3-7B module"
    sys.modules["OLMo3-7B"] = olmo3_module
    olmo3_spec.loader.exec_module(olmo3_module)
    batch_size = olmo3_module.GLOBAL_BATCH_SIZE

    # load state from the original training run
    trainer_state_file = resource_path(join_path(original_checkpoint, "train"), "rank0.pt")
    trainer_state = torch.load(trainer_state_file, weights_only=False)
    config_file = resource_path(original_checkpoint, "config.json")
    with open(config_file, "rb") as f:
        config = json.load(f)

    global_step = trainer_state["global_step"]
    run_name = f"{config['run_name']}-from{global_step}-LC"
    sequence_length = 64 * 1024
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
        config = olmo3_module.build_train_module_config(common)

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
        config.max_sequence_length = sequence_length
        config.rank_microbatch_size = sequence_length
        config.dp_config.shard_degree = 1
        config.dp_config.wrapping_strategy = 'full'
        config.cp_config = (
            TransformerContextParallelConfig.llama3(degree=8, head_stride=1)
        )
        config.ac_config = TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget, activation_memory_budget=0.8
        )

        return config

    def build_data_config(common: CommonComponents) -> DataComponents:
        return DataComponents(
            dataset=NumpyPackedFSLDatasetConfig.from_data_mix(
                DataMix.OLMo_longmino_mix_0925,
                tokenizer=common.tokenizer,
                sequence_length=sequence_length,
                generate_doc_lengths=True,  # enables intra-document masking
                source_group_size=8,
                source_permutation_seed=123,
                work_dir=common.work_dir
            ),
            data_loader=NumpyDataLoaderConfig(
                global_batch_size=batch_size, seed=4123, num_workers=12
            )
        )

    def build_model_config(common: CommonComponents) -> TransformerConfig:
        config = olmo3_module.build_model_config(common).with_rope_scaling(
            YaRNRoPEScalingConfig(factor=8, beta_fast=32, beta_slow=1, old_context_len=8192)
        )

        return config

    def build_trainer_config(common: CommonComponents) -> TrainerConfig:
        config = olmo3_module.build_trainer_config(common)

        config.load_path = original_checkpoint
        config.load_strategy = LoadStrategy.always
        config.max_duration = Duration.steps(global_step + length_in_steps)
        config.hard_stop = None

        config.callbacks["checkpointer"].save_interval = 10000
        config.callbacks["checkpointer"].ephemeral_save_interval = 500

        # performance settings
        config.metrics_collect_interval = 50
        config.callbacks["garbage_collector"].gc_interval = 200

        return config

    config = build_config(
        cli_context,
        global_batch_size=batch_size,
        max_sequence_length=sequence_length,
        data_config_builder=build_data_config,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        finalize_config=None,
        include_default_evals=False,
        num_nodes=16,
        beaker_workspace="ai2/OLMo_3",
    )
    assert config.launch is not None
    config.launch.cmd = [script, "train", original_checkpoint, length, cluster] + overrides

    cmd.prepare_environment(config)
    cmd.run(config)
