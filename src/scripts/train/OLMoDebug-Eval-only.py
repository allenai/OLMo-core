""" Backfill evals """

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.experiment import CommonComponents, main
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    DownstreamEvaluatorCallbackConfig,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)


def build_model_config(common: CommonComponents) -> TransformerConfig:
    return TransformerConfig.olmo2_1B(
        vocab_size=common.tokenizer.padded_vocab_size()
    )


def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    return TransformerTrainModuleConfig(
        rank_microbatch_size=2 * 8 * 4096, # doubled for titan!
        max_sequence_length=common.dataset.effective_sequence_length,
        optim=AdamWConfig(
            # lr=4e-4,
            lr=1.8e-3,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
            fused=True,
        ),
        compile_model=False, # To fix errors on Titan
        dp_config=TransformerDataParallelConfig(
            # name=DataParallelType.hsdp, 
            name=DataParallelType.fsdp, 
            param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
        # float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        # max_grad_norm=1.0,
        # scheduler=CosWithWarmup(warmup_steps=2000),
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    tokenizer_config = TokenizerConfig.dolma2()

    return (
        TrainerConfig(
            # save_folder=common.save_folder,
            # save_folder="/root/ai2/in_loop/workspace",
            # save_folder="/root/ai2/in_loop/workspace/olmo-cookbook-1b-5xC-dclm-baseline-natural-9a234fde/step53000",
            save_folder="/oe-eval-default/davidh/in_loop/workspace/olmo-cookbook-1b-5xC-dclm-baseline-natural-9a234fde/step53971",
            # rank_microbatch_size=8 * 4096,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=1,
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
                project="in-loop-debug",
                enabled=False,
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                entity="ai2-llm",
                project="in-loop-debug",
                enabled=True,
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=[
                    # OLMES Core 9 RC
                    "arc_challenge_test_rc_5shot",
                    "arc_easy_test_rc_5shot",
                    "hellaswag_rc_5shot",  # 1K subset of HellaSwag
                    "winogrande_val_rc_5shot",  # Helpful after 750M-5xC scale
                    "csqa_val_rc_5shot",
                    "piqa_val_rc_5shot",
                    "socialiqa_val_rc_5shot",
                    # Too noisy to be worth tracking
                    # "boolq_val_rc_5shot",
                    # "openbookqa_test_rc_5shot",
                    # MMLU RC
                    "mmlu_stem_val_rc_5shot",
                    "mmlu_humanities_val_rc_5shot",
                    "mmlu_social_sciences_val_rc_5shot",
                    "mmlu_other_val_rc_5shot",
                    "mmlu_stem_test_rc_5shot",
                    "mmlu_humanities_test_rc_5shot",
                    "mmlu_social_sciences_test_rc_5shot",
                    "mmlu_other_test_rc_5shot",
                    # Gen tasks BPB
                    "gsm8k_gold_bpb_5shot",
                    "minerva_math_algebra_gold_bpb_0shot",
                    "minerva_math_counting_and_probability_gold_bpb_0shot",
                    "minerva_math_geometry_gold_bpb_0shot",
                    "minerva_math_intermediate_algebra_gold_bpb_0shot",
                    "minerva_math_number_theory_gold_bpb_0shot",
                    "minerva_math_prealgebra_gold_bpb_0shot",
                    "minerva_math_precalculus_gold_bpb_0shot",
                    "codex_humaneval_gold_bpb_0shot",
                    "codex_mbpp_gold_bpb_0shot",
                    # Sanity check for MCQA ability
                    "copycolors_10way",
                    # OLMES Core 9 MC
                    "arc_challenge_test_mc_5shot",
                    "arc_easy_test_mc_5shot",
                    "hellaswag_rc_5shot",  # 1K subset of HellaSwag
                    # "csqa_val_mc_5shot",
                    "piqa_val_mc_5shot",
                    "socialiqa_val_mc_5shot",
                    "winogrande_val_rc_5shot",
                    # Too noisy to be worth tracking
                    # "boolq_val_mc_5shot",
                    # "openbookqa_test_mc_5shot",
                    # MMLU MC BPB
                    "mmlu_stem_val_mc_5shot",
                    "mmlu_humanities_val_mc_5shot",
                    "mmlu_social_sciences_val_mc_5shot",
                    "mmlu_other_val_mc_5shot",
                    "mmlu_stem_test_mc_5shot",
                    "mmlu_humanities_test_mc_5shot",
                    "mmlu_social_sciences_test_mc_5shot",
                    "mmlu_other_test_mc_5shot",
                    # Gen tasks BPB
                    "gsm8k_gold_bpb_5shot",
                    "minerva_math_algebra_gold_bpb_0shot",
                    "minerva_math_counting_and_probability_gold_bpb_0shot",
                    "minerva_math_geometry_gold_bpb_0shot",
                    "minerva_math_intermediate_algebra_gold_bpb_0shot",
                    "minerva_math_number_theory_gold_bpb_0shot",
                    "minerva_math_prealgebra_gold_bpb_0shot",
                    "minerva_math_precalculus_gold_bpb_0shot",
                    "codex_humaneval_gold_bpb_0shot",
                    "codex_mbpp_gold_bpb_0shot",
                    # Sanity check for MCQA ability
                    "copycolors_10way",
                ],
                tokenizer=tokenizer_config,
                # eval_interval=250,
                eval_interval=1,
                enabled=True,
            ),
        )
    )


if __name__ == "__main__":
    main(
        global_batch_size=1024 * 4096,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,

        include_default_evals=False,
        sequence_length=2048
    )
