"""
Train a 1B OLMo model. Run this script without any arguments to see usage info.
"""

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.experiment import CommonComponents, main
from olmo_core.nn.transformer import TransformerConfig, TransformerDataParallelConfig
from olmo_core.optim import AdamWConfig, OptimGroupOverride
from olmo_core.train import TrainerConfig
from olmo_core.train.callbacks import CheckpointerCallback, CometCallback, WandBCallback
from olmo_core.train.callbacks.evaluator_callback import (
    DownstreamEvaluatorCallbackConfig,
)


def build_model_config(common: CommonComponents) -> TransformerConfig:
    return TransformerConfig.olmo2_1B(
        vocab_size=common.tokenizer.padded_vocab_size(),
        compile=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
    )


def build_optim_config(common: CommonComponents) -> AdamWConfig:
    del common
    return AdamWConfig(
        lr=4e-4,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        group_overrides=[
            OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
        ],
        fused=True,
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    return (
        TrainerConfig(
            save_folder=common.save_folder,
            rank_microbatch_size=8 * 4096,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=1,
            z_loss_multiplier=1e-5,
            compile_loss=True,
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=10_000,
                ephemeral_save_interval=1000,
                save_async=True,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=common.run_name,
                workspace="ai2",
                project="OLMo-core-1B",
                enabled=True,
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                entity="ai2-llm",
                project="OLMo-core-1B",
                enabled=False,
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=[
                    "arc_challenge_val_rc_5shot",
                    "arc_challenge_val_mc_5shot",
                    "arc_challenge_test_rc_5shot",
                    "arc_challenge_test_mc_5shot",
                    "arc_easy_val_rc_5shot",
                    "arc_easy_val_mc_5shot",
                    "arc_easy_test_rc_5shot",
                    "arc_easy_test_mc_5shot",
                    "boolq_val_rc_5shot",
                    "boolq_val_mc_5shot",
                    "csqa_val_rc_5shot",
                    "csqa_val_mc_5shot",
                    "hellaswag_val_rc_5shot",
                    "hellaswag_val_mc_5shot",
                    "openbookqa_val_rc_5shot",
                    "openbookqa_val_mc_5shot",
                    "openbookqa_test_rc_5shot",
                    "openbookqa_test_mc_5shot",
                    "piqa_val_rc_5shot",
                    "piqa_val_mc_5shot",
                    "socialiqa_val_rc_5shot",
                    "socialiqa_val_mc_5shot",
                    "winogrande_val_rc_5shot",
                    "winogrande_val_mc_5shot",
                    "mmlu_stem_val_rc_5shot",
                    "mmlu_stem_val_mc_5shot",
                    "mmlu_humanities_val_rc_5shot",
                    "mmlu_humanities_val_mc_5shot",
                    "mmlu_social_sciences_val_rc_5shot",
                    "mmlu_social_sciences_val_mc_5shot",
                    "mmlu_other_val_rc_5shot",
                    "mmlu_other_val_mc_5shot",
                ],
                tokenizer=common.tokenizer,
                eval_batch_size=1024 * 4096,
                eval_interval=1000,
            ),
        )
    )


if __name__ == "__main__":
    main(
        global_batch_size=1024 * 4096,
        model_config_builder=build_model_config,
        optim_config_builder=build_optim_config,
        trainer_config_builder=build_trainer_config,
    )
