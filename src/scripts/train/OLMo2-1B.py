"""
Train a 1B OLMo model. Run this script without any arguments to see usage info.
"""
from olmo_core.config import DType
from olmo_core.data import NumpyDatasetConfig, DataMix, NumpyDatasetType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import CLUSTER_TO_GPU_TYPE, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CommonComponents, main
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import TrainerConfig, Duration
from olmo_core.train.callbacks import CheckpointerCallback, CometCallback, WandBCallback, \
    DownstreamEvaluatorCallbackConfig, LMEvaluatorCallbackConfig
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)


SEQUENCE_LENGTH = 4096
GLOBAL_BATCH_SIZE = 512 * SEQUENCE_LENGTH
MAX_DURATION = int(4e12)


def build_model_config(common: CommonComponents) -> TransformerConfig:
    return TransformerConfig.olmo2_1B(vocab_size=common.tokenizer.padded_vocab_size())


def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    rank_microbatch_size = 8 * SEQUENCE_LENGTH
    gpus = {CLUSTER_TO_GPU_TYPE.get(c, "unknown") for c in common.launch.clusters}
    if all("B200" in g for g in gpus):
        rank_microbatch_size *= 2

    return TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_size,
        max_sequence_length=common.dataset.effective_sequence_length,
        optim=SkipStepAdamWConfig(
            lr=4e-4,
            weight_decay=0.033,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
            fused=True,
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=2000, t_max=int(4e12 / GLOBAL_BATCH_SIZE)),
    )

def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    cancel_check_interval = 50

    # For training runs where we don't expect the model to acquire MC (e.g., 1B-5xC, short 7B training runs)
    tasks_small_compute = [
        # OLMES Core 9(-ish) RC
        "arc_challenge_test_rc_5shot",
        "arc_easy_test_rc_5shot",
        "hellaswag_rc_5shot", # 1K subset of HellaSwag
        "winogrande_val_rc_5shot", # Helpful after 750M-5xC scale
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
    ]

    # For training runs where we expect the model to acquire MC
    tasks_large_compute = [
        # OLMES Core 9(-ish) MC
        "arc_challenge_test_mc_5shot",
        "arc_easy_test_mc_5shot",
        "hellaswag_rc_5shot", # 1K subset of HellaSwag
        "csqa_val_mc_5shot",
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
    ]
    # Unfortunately we need the same metrics for everything, so we run them all.
    tasks = list(set(tasks_small_compute + tasks_large_compute))
    tasks.sort()

    assert len(common.launch.clusters) == 1
    cluster = common.launch.clusters[0]

    return (
        TrainerConfig(
            save_folder=common.save_folder,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=cancel_check_interval,
            max_duration=Duration.tokens(MAX_DURATION)
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=100_000,
                ephemeral_save_interval=10_000,
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
                cancel_check_interval=cancel_check_interval,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                entity="ai2-llm",
                project="OLMo-core-1B",
                enabled=False,
                cancel_check_interval=cancel_check_interval,
            ),
        )
        .with_callback(
            'downstream_evaluator',
            DownstreamEvaluatorCallbackConfig(
                tasks=tasks,
                tokenizer=common.tokenizer,
                eval_interval=10000,
            )
        ).with_callback(
        "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyDatasetConfig.from_data_mix(
                    DataMix.v3_small_ppl_validation,
                    name=NumpyDatasetType.padded_fsl,
                    mix_base_dir=get_root_dir(cluster),
                    sequence_length=SEQUENCE_LENGTH,
                    tokenizer=common.tokenizer,
                    work_dir=get_work_dir(get_root_dir(cluster)),
                ),
                eval_interval=10000,
            ),
        )
    )


if __name__ == "__main__":
    main(
        global_batch_size=GLOBAL_BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        include_instance_filter=False,  # We use SkipStepOptimizer for this problem.
        include_default_evals=False,
    )

