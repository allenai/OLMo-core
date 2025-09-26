from datetime import datetime
from typing import Optional

from olmo_core.internal.common import (
    get_root_dir,
    get_work_dir,
    build_launch_config,
)
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.internal.experiment import (
    main,
    CliContext,
    ExperimentConfig,
)
from olmo_core.internal import cookbook
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import Duration, TrainerConfig, LoadStrategy
from olmo_core.train.train_module import (
    TransformerTrainModuleConfig,
)
from olmo_core.optim.scheduler import WSD, SchedulerUnits
from olmo_core.data import (
    DataMix,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)

SEQ_LENGTH = 8192
GLOBAL_BATCH_SIZE = 4096


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    run_name_with_suffix = (
        f"{cli_context.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    )
    root_dir = get_root_dir(cli_context.cluster)
    work_dir = get_work_dir(root_dir)
    save_dir = f"{root_dir}/checkpoints/{run_name_with_suffix}"

    beaker_launch_config: Optional[BeakerLaunchConfig] = build_launch_config(
        name=cli_context.run_name,
        root_dir=root_dir,
        cmd=cli_context.remote_cmd,
        cluster=cli_context.cluster,
        workspace="ai2/oe-data",
        num_nodes=4,
    )

    tokenizer_config = TokenizerConfig.dolma2()

    model_config = TransformerConfig.olmo2_1B(
        vocab_size=tokenizer_config.padded_vocab_size(),
    )

    train_module_config: TransformerTrainModuleConfig = cookbook.configure_train_module(
        max_sequence_length=SEQ_LENGTH,
        global_batch_size=GLOBAL_BATCH_SIZE,
        instances_per_rank_microbatch=4,
        optim=SkipStepAdamWConfig(
            lr=4.4e-5 * (4**0.5),
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=WSD(
            units=SchedulerUnits.steps,
            warmup=2000,
            decay=2000,
            decay_fraction=None,
        ),
    )

    trainer_config = TrainerConfig(
        max_duration=Duration.chinchilla_tokens(
            2.0, model_params=model_config.num_active_non_embedding_params
        ),
        # max_duration=Duration.tokens(500000),
        save_folder=save_dir,
        save_overwrite=True,
        load_strategy=LoadStrategy.always,
    )
    callbacks = cookbook.configure_default_callbacks(
        run_name=cli_context.run_name, launch_config=beaker_launch_config
    )
    for name, callback in callbacks.items():
        trainer_config.add_callback(name, callback)

    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        DataMix.OLMoE_mix_0824,
        tokenizer=tokenizer_config,
        mix_base_dir=root_dir,
        work_dir=work_dir,
        sequence_length=SEQ_LENGTH,
        generate_doc_lengths=True,  # providing doc lenghts enables intra-document masking
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE, seed=34521, num_workers=4
    )

    return ExperimentConfig(
        run_name=cli_context.run_name,
        launch=beaker_launch_config,
        model=model_config,
        train_module=train_module_config,
        trainer=trainer_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        init_seed=12536,
    )


if __name__ == "__main__":
    main(config_builder=build_experiment_config)
