import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Dict

import torch
from olmo_core.data import NumpyDatasetConfig, TokenizerConfig

from olmo_core.distributed.checkpoint import load_state_dict
from olmo_core.internal.experiment import SubCmd, build_config, CommonComponents
from olmo_core.io import resource_path, join_path
from olmo_core.launch.beaker import OLMoCoreBeakerImage
from olmo_core.optim import SchedulerUnits
from olmo_core.optim.scheduler import WSD
from olmo_core.train import TrainerConfig, Duration, LoadStrategy
from olmo_core.train.train_module import TransformerTrainModuleConfig

log = logging.getLogger(__name__)


DATASET_NAME = "round5"


if __name__ == "__main__":
    usage = f"""
Performs mid-training. Uses midtraining data mix '{DATASET_NAME}'. Does not resume trainer and optimizer state.
    
[yellow]Usage:[/] [i blue]python[/] [i cyan]{sys.argv[0]}[/] [i b magenta]{'|'.join(SubCmd)}[/] [i b]ORIGINAL_CHECKPOINT LENGTH CLUSTER[/] [i][OVERRIDES...][/]

[b]Subcommands[/]
[b magenta]launch:[/]      Launch the script on Beaker with the [b magenta]train[/] subcommand.
[b magenta]train:[/]       Run the trainer. You usually shouldn't invoke the script with this subcommand directly.
             Instead use [b magenta]launch[/] or run it with torchrun.
[b magenta]train_single:[/]       Run the trainer on a single device (GPU, CPU, MPS). num_nodes is ignored.
[b magenta]dry_run:[/]     Pretty print the config and exit.

[b]Examples[/]
$ [i]python {sys.argv[0]} {SubCmd.launch} gs://ai2-llm/checkpoints/dirkg/OLMo2-7B-retrofit2-lowlr/step17882/ ai2/augusta-google-1 --launch.num_nodes=2[/]
    """.strip()

    if len(sys.argv) < 4 or sys.argv[1] not in set(SubCmd):
        import rich

        rich.get_console().print(usage, highlight=False)
        sys.exit(1)

    script, cmd, original_checkpoint, cluster, *overrides = sys.argv

    # Load OLMo 2 retrofit module
    retrofit_spec = importlib.util.spec_from_file_location("OLMo2-7B-retrofit", Path(__file__).parent / "OLMo2-7B-retrofit.py")
    retrofit_module = importlib.util.module_from_spec(retrofit_spec)
    sys.modules["OLMo2-7B-retrofit"] = retrofit_module
    retrofit_spec.loader.exec_module(retrofit_module)

    batch_size = 2 * 1024 * 1024

    # load state from the original training run
    trainer_state_file = resource_path(join_path(original_checkpoint, "train"), "rank0.pt")
    trainer_state = torch.load(trainer_state_file, weights_only=False)
    config_file = resource_path(original_checkpoint, "config.json")
    with open(config_file, "rb") as f:
        config = json.load(f)
    global_step = trainer_state["global_step"]

    run_name = f"{config['run_name']}-from{global_step}-{DATASET_NAME}"
    sequence_length = trainer_state["data_loader"]["sequence_length"]

    # determine last learning rate
    param = "embeddings.weight"
    key = f"optim.param_groups.{param}.lr"
    state_dict: Dict[str, Optional[float]] = {key: None}
    load_state_dict(join_path(original_checkpoint, "model_and_optim"), state_dict)
    assert state_dict[key] is not None
    lr = float(state_dict[key])
    log.info(f"Starting learning rate is {lr}")

    cmd = SubCmd(cmd)
    cmd.prepare_environment()

    def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
        config = retrofit_module.build_train_module_config(common)

        # configure lr
        config.optim.lr = lr
        config.scheduler = WSD(
            units=SchedulerUnits.steps,
            warmup_fraction=0,
            decay_fraction=1,
        )

        return config

    def build_trainer_config(common: CommonComponents) -> TrainerConfig:
        config: TrainerConfig = retrofit_module.build_trainer_config(common)

        config.load_path = original_checkpoint
        config.load_strategy = LoadStrategy.always
        config.load_trainer_state = False
        config.max_duration = Duration.tokens(int(100e9))
        config.hard_stop = None

        config.callbacks["checkpointer"].save_interval = 10000
        config.callbacks.pop("batchwup", None)

        # performance settings
        config.metrics_collect_interval = 50

        return config

    config = build_config(
        script,
        cmd,
        run_name,
        cluster,
        overrides,
        global_batch_size=batch_size,
        model_config_builder=retrofit_module.build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        finalize_config=None,
        sequence_length=sequence_length,
        include_default_evals=False,
        intra_document_masking=False,
        include_instance_filter=False,
        beaker_image=OLMoCoreBeakerImage.stable,
        num_nodes=16,
        beaker_workspace="ai2/OLMo_3",
        init_seed=1337,
    )
    config.launch.cmd = [script, "train", original_checkpoint, cluster] + overrides

    # Set dataset
    from olmo_core.data.named_source_mixtures import SOURCE_MIXTURES
    tokenizer_config, source_mixture_config = SOURCE_MIXTURES[DATASET_NAME]
    config.dataset = NumpyDatasetConfig(
        source_mixture_config=source_mixture_config,
        tokenizer=tokenizer_config,
        generate_doc_lengths=True,
        expand_glob=True
    )

    assert config.dataset.tokenizer == config.trainer.callbacks['downstream_evaluator'].tokenizer
    assert config.dataset.tokenizer == config.trainer.callbacks['lm_evaluator'].eval_dataset.tokenizer

    cmd.run(config)

# TODO:
# set run name properly
# assert that tokenizers match
