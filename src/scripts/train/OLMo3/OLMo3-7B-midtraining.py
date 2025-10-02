from datetime import datetime
from typing import Optional

from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.data.source_mixture import SourceMixtureDatasetConfig, SourceMixtureList
from olmo_core.internal import cookbook
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim.scheduler import LinearWithWarmup, SchedulerUnits
from olmo_core.train import Duration
from olmo_core.train.train_module import TransformerTrainModuleConfig

SEQ_LENGTH = 8192
GLOBAL_BATCH_SIZE = 2**21  # ~2M tokens
SEED = 1337


# TODO: address midtraining dtata issue where items in batch dont have same shape
# 1Z Original Traceback (most recent call last):
# 2025-10-01T21:40:47.121Z   File "/opt/conda/lib/python3.11/site-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
# 2025-10-01T21:40:47.121Z     data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
# 2025-10-01T21:40:47.121Z            ^^^^^^^^^^^^^^^^^^^^
# 2025-10-01T21:40:47.121Z   File "/opt/conda/lib/python3.11/site-packages/torch/utils/data/_utils/fetch.py", line 42, in fetch
# 2025-10-01T21:40:47.121Z     data = next(self.dataset_iter)
# 2025-10-01T21:40:47.121Z            ^^^^^^^^^^^^^^^^^^^^^^^
# 2025-10-01T21:40:47.121Z   File "/olmo-core-runtime/src/olmo_core/data/data_loader.py", line 1045, in <genexpr>
# 2025-10-01T21:40:47.121Z     return (
# 2025-10-01T21:40:47.121Z            ^
# 2025-10-01T21:40:47.121Z   File "/olmo-core-runtime/src/olmo_core/data/utils.py", line 401, in iter_batched
# 2025-10-01T21:40:47.121Z     raise RuntimeError(
# 2025-10-01T21:40:47.121Z RuntimeError: Items in batch don't have the same shape! Expected (8192,), got (5081,)
def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    run_name_with_ts = (
        f"{cli_context.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    )
    root_dir = get_root_dir(cli_context.cluster)
    work_dir = get_work_dir(root_dir)
    save_dir = f"{root_dir}/checkpoints/{run_name_with_ts}"

    beaker_launch_config: Optional[BeakerLaunchConfig] = build_launch_config(
        name=cli_context.run_name,
        cmd=cli_context.remote_cmd,
        cluster=cli_context.cluster,
        root_dir=root_dir,
        beaker_image="petew/olmo-core-tch270cu128",
        # workspace="ai2/olmo-3-microanneals",
        workspace="ai2/OLMo_3",
        num_nodes=16,
        nccl_debug=True,
        # override priority from the CLI eg `--launch.priority=high`
    )

    tokenizer_config = TokenizerConfig.dolma2()
    model_config = TransformerConfig.olmo3_7B(vocab_size=tokenizer_config.padded_vocab_size())

    max_tokens = 100_000_000_000  # 100B
    max_duration = Duration.tokens(max_tokens)

    train_module_config: TransformerTrainModuleConfig = cookbook.configure_train_module(
        max_sequence_length=SEQ_LENGTH,
        rank_microbatch_size=SEQ_LENGTH * 2,
        learning_rate=0.00020712352850360292,
        scheduler=LinearWithWarmup(units=SchedulerUnits.steps, warmup=0, alpha_f=0.0),
        activation_memory_budget=0.7,
    )

    source_list = SourceMixtureList.from_yaml(
        "src/olmo_core/data/source_mixtures/OLMo3-7B-midtraining.yaml"
    )
    source_list.validate()
    dataset_config = NumpyFSLDatasetConfig.from_src_mix(
        src_mix=SourceMixtureDatasetConfig(
            source_list=source_list,
            requested_tokens=max_tokens,
            global_batch_size=GLOBAL_BATCH_SIZE,
            processes=16,
            seed=SEED,
        ),
        tokenizer=tokenizer_config,
        work_dir=work_dir,
        sequence_length=SEQ_LENGTH,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE, seed=SEED, num_workers=4
    )

    trainer_config = cookbook.configure_trainer(
        load_path="gs://ai2-llm/checkpoints/OLMo25/step1413814",
        load_trainer_state=False,
        load_optim_state=True,
        max_duration=max_duration,
        checkpoint_dir=save_dir,
        work_dir=work_dir,
    ).with_callbacks(
        cookbook.configure_default_callbacks(
            run_name=run_name_with_ts, wandb_group_name=cli_context.run_name
        )
    )

    experiment_config = ExperimentConfig(
        run_name=cli_context.run_name,
        launch=beaker_launch_config,
        model=model_config,
        train_module=train_module_config,
        trainer=trainer_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        init_seed=SEED,
    )
    experiment_config = experiment_config.merge(cli_context.overrides)
    return experiment_config


if __name__ == "__main__":
    """
    Invoke this script directly to access the internal experiment CLI, which
    supports launch, train, dry_run, and other subcommands.

    Examples:
        To render the config and exit:
            python src/scripts/train/olmo3-midtraining/cookbook-example.py dry_run debug_run ai2/augusta

        To launch a training run on Augusta w/ 8 nodes:
        python src/scripts/train/olmo3-midtraining/cookbook-example.py launch my_run ai2/augusta \
            --launch.num_nodes=8 \
            --launch.priority=high
    """
    main(config_builder=build_experiment_config)
