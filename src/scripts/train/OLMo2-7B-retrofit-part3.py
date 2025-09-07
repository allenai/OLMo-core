import math
from datetime import datetime

from olmo_core.config import DType
from olmo_core.data import DataMix
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import CLUSTER_TO_GPU_TYPE
from olmo_core.internal.experiment import CommonComponents, main, build_common_components
from olmo_core.nn.transformer import TransformerConfig, TransformerActivationCheckpointingMode
from olmo_core.nn.rope import YaRNRoPEScalingConfig
from olmo_core.optim import OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.optim.scheduler import ConstantScheduler
from olmo_core.train import Duration, TrainerConfig, LoadStrategy
from olmo_core.train.callbacks import CheckpointerCallback, CometCallback, WandBCallback
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig, TransformerActivationCheckpointingConfig,
)

SEQUENCE_LENGTH = 32 * 1024
GLOBAL_BATCH_SIZE = 8 * 1024 * 1024


def build_model_config(common: CommonComponents) -> TransformerConfig:
    config = TransformerConfig.olmo2_7B(vocab_size=common.tokenizer.padded_vocab_size())

    # We don't do SWA because Amanda found it doesn't retrofit well.

    #config.block.attention.sliding_window = SlidingWindowAttentionConfig(
    #    force_full_attention_on_first_layer=False,
    #    force_full_attention_on_last_layer=True,
    #    pattern=[4096, 4096, 4096, -1],
    #)
    #config.block.attention.use_flash = True


    # We don't do RoPE scaling because the old Retrofit didn't have it due to a bug.

    # RoPE scaling
    #OLD_SEQUENCE_LENGTH = 4096
    #config.block.attention.rope.scaling = RoPEScalingConfig(
    #    old_context_len=OLD_SEQUENCE_LENGTH,
    #    factor=SEQUENCE_LENGTH / OLD_SEQUENCE_LENGTH
    #)


    # We cannot use headwise QK norm or GQA, because those can't be retrofit.

    config.block.attention.use_flash = True
    # old_context_len needs to be the original context len from part 1
    config.block.attention.rope.scaling = YaRNRoPEScalingConfig(
        factor=8, beta_fast=32, beta_slow=1, old_context_len=8192
    )
    return config


def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    rank_microbatch_size = SEQUENCE_LENGTH
    last_lr_of_olmo2 = 6.135113558011711e-05
    batch_size_of_olmo2 = 4 * 1024 * 1024
    lr = last_lr_of_olmo2 * math.sqrt(GLOBAL_BATCH_SIZE / batch_size_of_olmo2)
    lr *= 1.5     # fudge factor because it seems to work

    return TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_size,
        max_sequence_length=common.dataset.effective_sequence_length,
        optim=SkipStepAdamWConfig(
            lr=lr,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
            shard_degree=32
        ),
        ac_config= TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget,
            activation_memory_budget=0.1
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=ConstantScheduler(),
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    cancel_check_interval = 10

    assert common.launch is not None
    assert len(common.launch.clusters) == 1
    cluster = common.launch.clusters[0]

    run_name = f"{common.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%z')}"

    config = (
        TrainerConfig(
            save_folder=common.save_folder,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=cancel_check_interval,
            max_duration=Duration.tokens(int(250e9)),
            hard_stop=Duration.tokens(int(200e9)),
            load_path="gs://ai2-llm/checkpoints/dirkg/OLMo2-7B-retrofit3-part2/step23842/",
            load_strategy=LoadStrategy.always,
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=10_000,
                ephemeral_save_interval=250,
                save_async=False,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=common.run_name,
                workspace="ai2",
                project="olmo3",
                enabled=False,
                cancel_check_interval=cancel_check_interval,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                entity="ai2-llm",
                project="olmo3",
                group=common.run_name,
                enabled=True,
                cancel_check_interval=cancel_check_interval,
            ),
        )
        .with_recommended_evals(
            common.tokenizer,
            SEQUENCE_LENGTH,
            cluster,
            task_set="fast"
        )
    )
    config.callbacks["lm_evaluator"].eval_interval = 1000
    config.callbacks["downstream_evaluator"].eval_interval = 1000
    return config


def build_common_config(*args, **kwargs):
    components = build_common_components(*args, **kwargs)
    components.data_loader.num_workers = 8
    components.dataset.max_target_sequence_length = 64 * 1024
    components.dataset.mix = DataMix.OLMo_mix_0625
    return components


if __name__ == "__main__":
    main(
        global_batch_size=GLOBAL_BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        common_config_builder=build_common_config,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        include_instance_filter=False,  # We use SkipStepOptimizer for this problem.
        include_default_evals=False,
    )
