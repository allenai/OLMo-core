"""
Train a 1B OLMo model. Run this script without any arguments to see usage info.

TODO: Point to custom data: gs://allennlp-willm/ppt2/shuffle-dyck.npy
"""

from datetime import datetime

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import CLUSTER_TO_GPU_TYPE
from olmo_core.internal.experiment import CommonComponents, main
from olmo_core.nn.attention import SlidingWindowAttentionConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import CheckpointerCallback, CometCallback, WandBCallback, ConfigSaverCallback
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

from olmo_core.utils import seed_all
from typing import List, cast
import sys
from olmo_core.train import (
    prepare_training_environment,
    teardown_training_environment,
)

# === willm: taken from https://arxiv.org/abs/2502.19249 ===
SEQUENCE_LENGTH = 2048
GLOBAL_BATCH_SIZE = 32 * SEQUENCE_LENGTH
WARMUP_STEPS = 1000
N_TOKENS = 500 * GLOBAL_BATCH_SIZE  # 35M tokens
# === willm: original values ===
# SEQUENCE_LENGTH = 8 * 1024
# GLOBAL_BATCH_SIZE = 4 * 1024 * 1024
# WARMUP_STEPS = 2000

def build_model_config(common: CommonComponents) -> TransformerConfig:
    config = TransformerConfig.olmo2_1B_v2(vocab_size=common.tokenizer.padded_vocab_size())
    config.block.attention.sliding_window = SlidingWindowAttentionConfig(
        force_full_attention_on_first_layer=False,
        force_full_attention_on_last_layer=True,
        pattern=[4096, 4096, 4096, -1],
    )
    config.block.attention.use_flash = True
    return config


def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    rank_microbatch_size = 4 * SEQUENCE_LENGTH
    if common.launch is not None:
        gpus = {CLUSTER_TO_GPU_TYPE.get(c, "unknown") for c in common.launch.clusters}
        if all("B200" in g for g in gpus):
            rank_microbatch_size *= 2

    return TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_size,
        max_sequence_length=common.dataset.effective_sequence_length,
        optim=SkipStepAdamWConfig(
            lr=4e-4 * 2,
            weight_decay=0.033,
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
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=WARMUP_STEPS),
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    cancel_check_interval = 50

    if common.launch is None:
        cluster = "local"
    else:
        assert len(common.launch.clusters) == 1
        cluster = common.launch.clusters[0]

    run_name = f"{common.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%z')}"

    return (
        TrainerConfig(
            save_folder=f"gs://ai2-llm/checkpoints/{common.run_name}/",
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=cancel_check_interval,
            max_duration=Duration.tokens(int(10 * N_TOKENS)),  # willm: 1 * N_TOKENS is original
            hard_stop=Duration.tokens(int(2.5e12 + GLOBAL_BATCH_SIZE * (WARMUP_STEPS / 2))), # After this, we switch to a longer cosine to reach 6T.
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=250,  # willm: 500 corresponds to original paper
                ephemeral_save_interval=None,
                save_async=True,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=run_name,
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
                group=common.run_name,
                entity="ai2-llm",
                project="olmo3",
                enabled=True,
                cancel_check_interval=cancel_check_interval,
            ),
        )
        .with_recommended_evals(
            common.tokenizer,
            SEQUENCE_LENGTH,
            cluster,
            task_set="fast",
            eval_interval=1000
        )
    )


def main(run_name: str, overrides: List[str]):
    """Custom main training function.
    
    Cf. https://github.com/allenai/OLMo-core/blob/willm/ppt2/src/examples/llama/train.py
    """
    # TODO: Fix this
    config = build_config(run_name, overrides)

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
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    # Train.
    trainer.fit()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} run_name [OVERRIDES...]")
        sys.exit(1)

    run_name, *overrides = sys.argv[1:]

    prepare_training_environment()
    try:
        main(run_name, overrides=overrides)
    finally:
        teardown_training_environment()

# if __name__ == "__main__":
#     main(
#         global_batch_size=GLOBAL_BATCH_SIZE,
#         sequence_length=SEQUENCE_LENGTH,
#         model_config_builder=build_model_config,
#         train_module_config_builder=build_train_module_config,
#         trainer_config_builder=build_trainer_config,
#         include_instance_filter=False,  # We use SkipStepOptimizer for this problem.
#         include_default_evals=False,
#         intra_document_masking=False,
#         beaker_workspace="ai2/willm-ppt2",
#     )
