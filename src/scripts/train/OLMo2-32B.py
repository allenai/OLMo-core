"""
Train a 32B OLMo model. Run this script without any arguments to see usage info.
"""

import logging

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.experiment import CommonComponents, main, ExperimentConfig
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingConfig,
    TransformerActivationCheckpointingMode,
    TransformerConfig,
    TransformerDataParallelConfig,
)
from olmo_core.optim import OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import Duration, DurationUnit, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    DownstreamEvaluatorCallbackConfig,
    WandBCallback,
)
from olmo_core.train.checkpoint import CheckpointerConfig

log = logging.getLogger(__name__)

NUM_NODES = 16

def build_model_config(common: CommonComponents) -> TransformerConfig:
    compile = True
    return TransformerConfig.olmo2_32B(
        vocab_size=common.tokenizer.padded_vocab_size(),
        compile=compile,
        fused_ops=False,
        use_flash=not compile,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            num_replicas=NUM_NODES // 2,
        ),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_modules,
            modules=[
                "embeddings",
                "blocks.*.attention",
                "blocks.*.attention_norm",
                "blocks.*.feed_forward",
                "blocks.*.feed_forward_norm"
            ]
        ),
        float8_config=Float8Config(compile=compile, enabled=False),
    )


def build_optim_config(common: CommonComponents) -> SkipStepAdamWConfig:
    del common
    return SkipStepAdamWConfig(
        lr=6e-4,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        group_overrides=[
            OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
        ],
        # fused=True,
        compile=False,
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    project_name = "peteish32-hybrid"
    return (
        TrainerConfig(
            save_folder=f"gs://ai2-llm/checkpoints/{project_name}/",
            rank_microbatch_size=2 * 4096,
            checkpointer=CheckpointerConfig(save_thread_count=1, load_thread_count=32),
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            z_loss_multiplier=1e-5,
            compile_loss=False,
            fused_loss=True,
            max_duration=Duration(int(6.5e12), DurationUnit.tokens),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=250,
                save_async=True,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=common.run_name,
                workspace="ai2",
                project="peteish32",
                enabled=True,
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                entity="ai2-llm",
                project="peteish32",
                enabled=False,
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=[
                    # MMLU for backwards compatibility
                    "mmlu_stem_mc_5shot",
                    "mmlu_humanities_mc_5shot",
                    "mmlu_social_sciences_mc_5shot",
                    "mmlu_other_mc_5shot",
                    # MMLU test
                    "mmlu_stem_mc_5shot_test",
                    "mmlu_humanities_mc_5shot_test",
                    "mmlu_social_sciences_mc_5shot_test",
                    "mmlu_other_mc_5shot_test",
                    # Core 12 tasks for backwards compatibility
                    "arc_challenge",
                    "arc_easy",
                    "basic_arithmetic",
                    "boolq",
                    "commonsense_qa",
                    "copa",
                    "hellaswag",
                    "openbook_qa",
                    "piqa",
                    "sciq",
                    "social_iqa",
                    "winogrande",
                    # Core 12 tasks 5-shot
                    "arc_challenge_rc_5shot",
                    "arc_easy_rc_5shot",
                    # "basic_arithmetic_rc_5shot",  # doesn't exist
                    # "boolq_rc_5shot",  # we don't like it
                    "csqa_rc_5shot",
                    # "copa_rc_5shot",  # doesn't exist
                    "hellaswag_rc_5shot",
                    "openbookqa_rc_5shot",
                    "piqa_rc_5shot",
                    # "sciq_rc_5shot",  # doesn't exist
                    "socialiqa_rc_5shot",
                    "winogrande_rc_5shot",
                    # New in-loop evals
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
                eval_interval=1000,
            ),
        )
    )


def finalize_config(config: ExperimentConfig) -> None:
    if config.trainer.load_path is not None and config.trainer.load_path.startswith("gs://"):
        source_path = config.trainer.load_path.rstrip("/")
        assert len(source_path) > 0
        final_path_component = source_path.rsplit("/", maxsplit=1)[-1]
        assert len(final_path_component) > 0
        target_path = f"/data/olmo_core/{final_path_component}"
        assert len(target_path) > 0  # just to be extra sure, because we're rm'ing it below
        config.launch.setup_steps.extend([
            f"rm -rf {target_path}",
            f"mkdir -p {target_path}",
            f"gsutil -q -m rsync -r {source_path} {target_path}"
        ])
        config.trainer.load_path = target_path


if __name__ == "__main__":
    main(
        global_batch_size=2 * 4096 * NUM_NODES * 8,
        model_config_builder=build_model_config,
        optim_config_builder=build_optim_config,
        trainer_config_builder=build_trainer_config,
        finalize_config=finalize_config,
    )
