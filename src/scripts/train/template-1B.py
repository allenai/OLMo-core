import logging
import sys
from dataclasses import dataclass
from typing import List, Optional, cast

from olmo_core.float8 import Float8Config

from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyDatasetType,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.common import (
    build_launch_config,
    get_beaker_username,
    get_root_dir,
    get_work_dir,
)
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import (
    TrainerConfig,
    prepare_cli_environment,
    prepare_training_environment,
    teardown_training_environment,
    Duration
)
from olmo_core.train.callbacks import (
    BeakerCallback,
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

log = logging.getLogger(__name__)

#######################
#### CONFIGURATION ####
#######################

# TODO: update these settings for your use case

# Set this if you want to start from an existing checkpoint (like for annealing).
# NOTE: You do NOT need to set on a restart as long as the trainer's 'save_folder' is the same.
CHECKPOINT: Optional[str] = None

# Data configuration.
SEQUENCE_LENGTH = 2048
TOKENIZER_CONFIG = TokenizerConfig.dolma2()
DATA_PATHS: List[str] = []
DATA_PATHS_FILE = "datadelve_dclm_sample_expanded.txt"
GLOBAL_BATCH_SIZE = 1024 * SEQUENCE_LENGTH
INTRA_DOCUMENT_MASKING = False

MAX_DURATION = 113184153600 #int(4e12)

# Model and optim.
MODEL_CONFIG = TransformerConfig.olmo2_1B_v2(vocab_size=TOKENIZER_CONFIG.padded_vocab_size())
LEARNING_RATE = 8e-4 # 4e-4

# Beaker.
BEAKER_CLUSTER = "ai2/augusta-google-1" # "ai2/jupiter-cirrascale-2"
NUM_NODES = 4
BEAKER_WORKSPACE = "ai2/oe-data"
BEAKER_BUDGET = "ai2/oe-data"

# Logging.
COMET_PROJECT: Optional[str] = None  # set this to enable Comet logging
WANDB_PROJECT: Optional[str] = "regmixer"  # set this to enable W&B logging

###########################
#### END CONFIGURATION ####
###########################

# NOTE: in most cases you shouldn't need to edit below this line.


@dataclass
class ExperimentConfig(Config):
    launch: BeakerLaunchConfig
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = 12536


def build_config(script: str, run_name: str, overrides: List[str]) -> ExperimentConfig:
    root_dir = get_root_dir(BEAKER_CLUSTER)
    work_dir = get_work_dir(BEAKER_CLUSTER)
    beaker_user = get_beaker_username()
    assert beaker_user is not None

    if len(DATA_PATHS) == 0:
        with open(DATA_PATHS_FILE, "r") as f:
            data_paths = sorted([row_ for row in f if not (row_ := row.strip()).startswith("#")])
    else:
        data_paths = DATA_PATHS

    dataset_config = NumpyDatasetConfig(
        paths=data_paths,
        name=NumpyDatasetType.fsl,
        sequence_length=SEQUENCE_LENGTH,
        tokenizer=TOKENIZER_CONFIG,
        work_dir=work_dir,
        generate_doc_lengths=INTRA_DOCUMENT_MASKING,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,
        num_workers=4,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=8 * SEQUENCE_LENGTH,
        max_sequence_length=dataset_config.effective_sequence_length,
        optim=SkipStepAdamWConfig(
            lr=LEARNING_RATE,
            weight_decay=0.033,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=2000, t_max=int(5e12 / GLOBAL_BATCH_SIZE)),

    )

    cancel_check_interval = 50

    trainer_config = (
        TrainerConfig(
            save_folder=f"{root_dir}/checkpoints/{beaker_user.lower()}/{run_name}",
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=cancel_check_interval,
            max_duration=Duration.tokens(MAX_DURATION),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=100_000,
                ephemeral_save_interval=5_000,
                save_async=True,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=run_name,
                workspace="ai2",
                project=COMET_PROJECT,
                cancel_check_interval=10,
                enabled=COMET_PROJECT is not None,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                entity="ai2-llm",
                project="OLMo-core-1B",
                cancel_check_interval=cancel_check_interval,
                enabled=WANDB_PROJECT is not None,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("garbage_collector", GarbageCollectorCallback())
        .with_callback("beaker", BeakerCallback())
        .with_recommended_evals(TOKENIZER_CONFIG, SEQUENCE_LENGTH, BEAKER_CLUSTER)
    )

    return ExperimentConfig(
        model=MODEL_CONFIG,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
        launch=build_launch_config(
            name=run_name,
            root_dir=root_dir,
            cmd=[script, "train", run_name, *overrides],
            cluster=BEAKER_CLUSTER,
            workspace=BEAKER_WORKSPACE,
            budget=BEAKER_BUDGET,
            num_nodes=NUM_NODES,
        ),
    ).merge(overrides)


def train(config: ExperimentConfig):
    # Set RNG states on all devices.
    seed_all(config.init_seed)

    # Build components.
    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)

    # Save config to W&B and each checkpoint dir.
    config_dict = config.as_config_dict()
    cast(CometCallback, trainer.callbacks["comet"]).config = config_dict
    cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    # Maybe load from existing checkpoint.
    if CHECKPOINT is not None and not trainer.maybe_load_checkpoint(trainer.save_folder):
        trainer.load_checkpoint(CHECKPOINT)

    # Train.
    trainer.fit()


def launch(config: ExperimentConfig):
    config.launch.launch(follow=True)


if __name__ == "__main__":
    usage = f"Usage: python {sys.argv[0]} train|launch|dry_run RUN_NAME [OVERRIDES...]"
    if len(sys.argv) < 3:
        print(usage)
        sys.exit(1)

    script, cmd, run_name, *overrides = sys.argv

    if cmd == "train":
        prepare_training_environment()
    else:
        prepare_cli_environment()

    config = build_config(script, run_name, overrides)
    log.info(config)

    if cmd == "train":
        try:
            train(config)
        finally:
            teardown_training_environment()
    elif cmd == "launch":
        launch(config)
    elif cmd == "dry_run":
        pass
    else:
        print(usage)
        sys.exit(1)
