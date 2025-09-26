from datetime import datetime
from functools import partial

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import CLUSTER_TO_GPU_TYPE
from olmo_core.internal.experiment import CommonComponents, build_config, main, DataComponents
from olmo_core.nn.attention import SlidingWindowAttentionConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import CheckpointerCallback, CometCallback, WandBCallback
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
    TransformerActivationCheckpointingConfig,
    TransformerActivationCheckpointingMode,
)
from olmo_core.float8 import AOFloat8LinearConfig, Float8Config


# def build_data_components(
#     common: CommonComponents,
#     intra_document_masking: bool = False,
#     include_instance_filter: bool = False,
# ) -> DataComponents:
#     """
#     Default dataset and data loader configurations. Constructs a simple FSL dataset and data loader
#     configuration with default settings.
#     """
#     dataset_config = NumpyFSLDatasetConfig.from_data_mix(
#         DataMix.OLMoE_mix_0824,
#         tokenizer=common.tokenizer,
#         mix_base_dir=common.root_dir,
#         work_dir=common.work_dir,
#         sequence_length=common.max_sequence_length,
#         # max target sequence length doesn't affect how the data is loaded, just how it's cached behind the scenes
#         max_target_sequence_length=max(common.max_sequence_length, 8192),
#         generate_doc_lengths=intra_document_masking,
#         instance_filter_config=None
#         if not include_instance_filter
#         else InstanceFilterConfig(
#             repetition_max_period=13, repetition_min_period=1, repetition_max_count=32
#         ),
#     )

#     data_loader_config = NumpyDataLoaderConfig(
#         global_batch_size=common.global_batch_size, seed=34521, num_workers=4
#     )

#     return DataComponents(dataset=dataset_config, data_loader=data_loader_config)


# def build_model_config(common: CommonComponents) -> TransformerConfig:
#     config = TransformerConfig.olmo2_7B(
#         vocab_size=common.tokenizer.padded_vocab_size(),
#         n_kv_heads=8,
#         hidden_size_multiplier=1.2,
#         hidden_size_multiple_of=1024,
#     )
#     #  config.block.name = TransformerBlockType.default
#     #  config.block.attention.qk_norm = None
#     config.block.attention.sliding_window = SlidingWindowAttentionConfig(
#         force_full_attention_on_first_layer=False,
#         force_full_attention_on_last_layer=True,
#         # NOTE: 4097 instead of 4096 to reproduce with the off-by-one bug.
#         pattern=[4097, 4097, 4097, -1],
#     )
#     config.block.attention.use_flash = True
#     config.block.attention.use_head_qk_norm = True

#     return config


# def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
#     rank_microbatch_size = 2 * common.max_sequence_length
#     if common.launch is not None:
#         gpus = {CLUSTER_TO_GPU_TYPE.get(c, "unknown") for c in common.launch.clusters}
#         if all("B200" in g for g in gpus):
#             rank_microbatch_size *= 2

#     return TransformerTrainModuleConfig(
#         rank_microbatch_size=rank_microbatch_size,
#         max_sequence_length=common.max_sequence_length,
#         optim=SkipStepAdamWConfig(
#             lr=LR,
#             weight_decay=0.1,
#             betas=(0.9, 0.95),
#             group_overrides=[
#                 OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
#             ],
#             compile=False,
#             foreach=True,
#             step_increment_bugfix=False,
#         ),
#         compile_model=True,
#         dp_config=TransformerDataParallelConfig(
#             name=DataParallelType.hsdp,
#             param_dtype=DType.bfloat16,
#             reduce_dtype=DType.float32,
#             wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
#             shard_degree=32,
#         ),
#         ac_config=TransformerActivationCheckpointingConfig(
#             mode=TransformerActivationCheckpointingMode.selected_modules,
#             modules=["blocks.*.feed_forward"],
#         ),
#         float8_config=Float8Config(
#             enabled=True,
#             ao=AOFloat8LinearConfig(
#                 enable_fsdp_float8_all_gather=True,
#                 force_recompute_fp8_weight_in_bwd=True,
#                 round_scales_to_power_of_2=True,
#             ),
#         ),
#         z_loss_multiplier=1e-5,
#         max_grad_norm=1.0,
#         scheduler=WSD(
#             units=SchedulerUnits.steps,
#             warmup=2000,
#             decay=(
#                 int(ANNEAL_TOKENS / (4 * common.global_batch_size))
#             ),  # * 4 because we're doubling the batch size twice with batch size warmup
#             decay_fraction=None,
#         ),
#     )


# def build_trainer_config(common: CommonComponents) -> TrainerConfig:
#     cancel_check_interval = 50

#     assert common.launch is not None
#     assert len(common.launch.clusters) == 1
#     cluster = common.launch.clusters[0]

#     run_name = f"{common.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"

#     config = (
#         TrainerConfig(
#             # save_folder=common.save_folder,
#             save_folder=f"gs://ai2-llm/checkpoints/{common.run_name}/",
#             save_overwrite=True,
#             load_strategy=LoadStrategy.always,
#             metrics_collect_interval=10,
#             cancel_check_interval=cancel_check_interval,
#             max_duration=Duration.tokens(MAX_DURATION),
#         )
#         .with_callback(
#             "checkpointer",
#             CheckpointerCallback(
#                 save_interval=1000,
#                 ephemeral_save_interval=None,
#                 save_async=False,
#             ),
#         )
#         .with_callback(
#             "comet",
#             CometCallback(
#                 name=run_name,
#                 workspace="ai2",
#                 project="olmo3",
#                 enabled=True,
#                 cancel_check_interval=cancel_check_interval,
#             ),
#         )
#         .with_callback(
#             "wandb",
#             WandBCallback(
#                 name=run_name,
#                 group=common.run_name,
#                 entity="ai2-llm",
#                 project="olmo3",
#                 enabled=True,
#                 cancel_check_interval=cancel_check_interval,
#             ),
#         )
#         .with_callback("monkey_patcher", MonkeyPatcherCallback())
#         .with_recommended_evals(
#             common.tokenizer,
#             common.max_sequence_length,
#             cluster,
#             task_set="fast",
#             eval_interval=EVAL_INTERVAL,
#         )
#     )

#     # batch size warmup
#     config.callbacks["batchwup"] = BatchSizeSchedulerCallback(
#         batch_sizes=[
#             # common.global_batch_size,
#             # common.global_batch_size * 2,
#             common.global_batch_size * 4,
#         ],
#         schedule=[
#             Duration.tokens(0),
#             # Duration.tokens(167_772_160_000),
#             # Duration.tokens(503_316_480_000),
#         ],
#         enabled=True,
#     )

#     return config


# trainer_config = CookbookTrainerConfig(
#     # one of the two following must be provided
#     chinchilla_multiplier=5,
#     max_tokens=None,
#     # if the following are not provided, error out if not resuming from a checkpoint;
#     # otherwise import from checkpoint
#     batch_size=1024 * 4096,
#     sequence_length=4096,
#     scheduler="cosine",
#     learning_rate=1.8e-3,
# )

# eval_config = CookbookEvalConfig(
#     eval_interval=250,  # every 250 steps
#     checkpoint_interval=1000,  # every 1000 steps
#     downstream_evaluators=["olmo2_dev_1b"],  # by default empty list,  no downstream evals.
# )

# # users don't assign name or groups. the name is experiment_name from CookbookExperimentConfig,
# # suffixed with a 8 hexidecimal character id. Group is the same of experiment_name, but with no suffix.
# logging_config = CookbookWandbLoggingConfig(project="olmo3", entity="ai2-llm")


# cookbook_sources = [
#     CookbookSourceConfig(
#         name="social_life",
#         target_ratio=0.01865278057032931,
#         repetition_factor=2.0,
#         paths=[
#             "s3://ai2-llm/preprocessed/dclm/baseline_topic_ft_lr05_ng2_n3M6_ova_20pct/allenai/dolma2-tokenizer/social_life/**/*.npy"
#         ],
#     ),
#     CookbookSourceConfig(
#         name="software",
#         target_ratio=0.00024003225357094727,
#         repetition_factor=2.0,
#         paths=[
#             "s3://ai2-llm/preprocessed/dclm/baseline_topic_ft_lr05_ng2_n3M6_ova_20pct/allenai/dolma2-tokenizer/software/**/*.npy"
#         ],
#     ),
#     CookbookSourceConfig(
#         name="software_development",
#         target_ratio=0.2546200267035965,
#         repetition_factor=2.0,
#         paths=[
#             "s3://ai2-llm/preprocessed/dclm/baseline_topic_ft_lr05_ng2_n3M6_ova_20pct/allenai/dolma2-tokenizer/software_development/**/*.npy"
#         ],
#     ),
#     # ...
#     # could do more ofc.
# ]


# experiment_config = CookbookExperimentConfig(
#     experiment_name="my-first-experiment",
#     model_config=model_config,
#     trainer_config=trainer_config,
#     eval_config=eval_config,
#     logging_config=logging_config,
#     beaker_config=beaker_config,
#     sources=cookbook_sources,
# )


if __name__ == "__main__":
    config_builder = partial(
        build_config,
        global_batch_size=GLOBAL_BATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        data_config_builder=build_data_config,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        include_default_evals=False,
        include_instance_filter=False,  # We use SkipStepOptimizer for this problem.
    )
    main(config_builder=config_builder)
