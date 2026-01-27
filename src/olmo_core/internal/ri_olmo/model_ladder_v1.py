"""
Ricursive-Olmo v1 model series.

The goal for this model series is to define the initial model baseline.
All additional proposed features will be run on top of this baseline.


#################### v1 ####################
* Switch to Gemma 3N-like vanilla model architecture
* Assumes StepFun optimal batch size and learning rate schedule [Li 2025]
* Use sigmoid gated attention [Qiu 2025]
* Add additional model configs up to 60B
* Use pre- and post- RMSNorm for attention and FFN
* Enable QK norm
* Assume grouped attention (kv_heads=8) for all model sizes
* Increase MLP hidden dimension to 8x
* Change sliding window pattern to [1K*4, global]

"""

import argparse
import ast
import enum
import math
from typing import List

from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.ri_olmo import RicursiveTransformerConfig
from olmo_core.nn.transformer import TransformerActivationCheckpointingMode
from olmo_core.optim import (
    CosWithWarmup,
    OptimGroupOverride,
    SchedulerUnits,
    SkipStepAdamWConfig,
)
from olmo_core.script_utils import ExperimentConfig, main, parser
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    LMEvaluatorCallbackConfig,
    MonkeyPatcherCallback,
    SpeedMonitorCallback,
    StabilityMonitorCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

DEFAULT_SEQUENCE_LENGTH = 8192


class RicursiveOlmoV1(enum.Enum):
    RI_OLMO_260M = (enum.auto(),)
    RI_OLMO_709M = (enum.auto(),)
    RI_OLMO_1p3B = (enum.auto(),)
    RI_OLMO_2B = (enum.auto(),)
    RI_OLMO_4B = (enum.auto(),)
    RI_OLMO_8B = (enum.auto(),)
    RI_OLMO_15B = (enum.auto(),)
    RI_OLMO_34B = (enum.auto(),)
    RI_OLMO_65B = (enum.auto(),)


def _add_args():
    parser.add_argument(
        "--auto-resume",
        type=lambda x: ast.literal_eval(x),  # Hack to avoid bool("False") => True
        default=False,
        help=(
            "Automatically resume experiment in Comet if the experiment name"
            "matches a prior experiment. Currently unsupported for WandB."
        ),
    )
    parser.add_argument(
        "--enable-comet-logging",
        type=lambda x: ast.literal_eval(x),
        default=True,
        help="Enable experiment logging in Comet (https://www.comet.com).",
    )
    parser.add_argument(
        "--enable-wandb-logging",
        type=lambda x: ast.literal_eval(x),
        default=False,
        help="Enable experiment logging in Weights and Biases.",
    )
    parser.add_argument(
        "--lr-multiplier",
        type=float,
        default=1.0,
        help="Peak learning rate multiplier, useful for quickly sweeping LR.",
    )
    parser.add_argument(
        "--batch-override-instances",
        type=int,
        default=None,
        help="Batch size override in instances, useful for quickly sweeping batch.",
    )
    parser.add_argument(
        "--batch-multiplier",
        type=float,
        default=1.0,
        help="Batch size multiplier, useful for quickly sweeping batch.",
    )
    parser.add_argument(
        "--model",
        type=lambda x: RicursiveOlmoV1[x],
        choices=list(RicursiveOlmoV1),
        default=RicursiveOlmoV1.RI_OLMO_260M,
        help="Model to pretrain.",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="ricursive-olmo",
        help="Workspace name used in Comet/WandB.",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default="ri-olmo-v1",
        help="Project name used in Comet/WandB. Experiments are grouped under projects.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=2000,
        help="Learning rate warmup steps. Increases linearly from 0 to peak LR.",
    )


def get_learning_rate(model_params: int, training_tokens: int) -> float:
    # Using step law from [Li 2025]: https://arxiv.org/pdf/2503.04715v1

    n = model_params
    d = training_tokens

    lr = 1.79 * pow(n, -0.713) * pow(d, 0.307)

    print(f"Model size: {n}, training tokens: {d}, opt_lr: {lr}")

    return lr


# Returns batch size in tokens
def get_global_batch_size(
    model_params: int,
    training_tokens: int,
    sequence_length: int,
    round_nearest: int,
) -> int:
    # Using step law from [Li 2025]: https://arxiv.org/pdf/2503.04715v1

    n = model_params
    d = training_tokens

    global_bsz = 0.58 * pow(d, 0.571)

    print(f"Model size: {n}, training tokens: {d}, opt_global_bsz: {global_bsz}")

    # Calculate instance batch size
    instance_bsz = global_bsz / sequence_length

    # Round batch size to (round_nearest * seqlen), clamping up
    rounded_instance_bsz = int(math.ceil(instance_bsz / round_nearest) * round_nearest)
    print(f"Rounding instance bsz from {instance_bsz} to {rounded_instance_bsz}")

    rounded_global_bsz = sequence_length * rounded_instance_bsz
    print(f"Rounding global bsz from {global_bsz} to {rounded_global_bsz}")

    return rounded_global_bsz


def get_optimal_training_tokens(model_params: int) -> int:
    # Assume Chinchilla-like rule-of-thumb of 20 tok/param

    return int(20 * model_params)


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    sequence_length = opts.sequence_length or DEFAULT_SEQUENCE_LENGTH
    tokenizer_config = TokenizerConfig.dolma2()
    vocab_size = tokenizer_config.padded_vocab_size()

    round_nearest = 16
    activation_memory_budget = 1.0

    match opts.model:
        case RicursiveOlmoV1.RI_OLMO_260M:
            model_config = RicursiveTransformerConfig.ri_olmo_v1_260M(vocab_size)
        case RicursiveOlmoV1.RI_OLMO_709M:
            model_config = RicursiveTransformerConfig.ri_olmo_v1_709M(vocab_size)
        case RicursiveOlmoV1.RI_OLMO_1p3B:
            model_config = RicursiveTransformerConfig.ri_olmo_v1_1p3B(vocab_size)
        case RicursiveOlmoV1.RI_OLMO_2B:
            model_config = RicursiveTransformerConfig.ri_olmo_v1_2B(vocab_size)
        case RicursiveOlmoV1.RI_OLMO_4B:
            model_config = RicursiveTransformerConfig.ri_olmo_v1_4B(vocab_size)
            round_nearest = 32
        case RicursiveOlmoV1.RI_OLMO_8B:
            model_config = RicursiveTransformerConfig.ri_olmo_v1_8B(vocab_size)
            round_nearest = 64
            activation_memory_budget = 0.9
        case RicursiveOlmoV1.RI_OLMO_15B:
            model_config = RicursiveTransformerConfig.ri_olmo_v1_15B(vocab_size)
            # Support up to 16 hosts, bsz8 per host
            round_nearest = 128
            activation_memory_budget = 0.4
        case RicursiveOlmoV1.RI_OLMO_34B:
            model_config = RicursiveTransformerConfig.ri_olmo_v1_34B(vocab_size)
            # Currently does not work, OOMs!!!
            activation_memory_budget = 0.1
        case RicursiveOlmoV1.RI_OLMO_65B:
            model_config = RicursiveTransformerConfig.ri_olmo_v1_65B(vocab_size)

        case _:
            raise ValueError(
                f"Model not in list! Valid models: {[m.name for m in RicursiveOlmoV1]}\n\n"
            )

    model_active_params = model_config.num_active_params
    training_tokens = get_optimal_training_tokens(model_active_params)

    learning_rate = get_learning_rate(model_active_params, training_tokens)
    global_batch_size = get_global_batch_size(
        model_params=model_active_params,
        training_tokens=training_tokens,
        sequence_length=sequence_length,
        round_nearest=round_nearest,
    )

    old_global_batch_size = global_batch_size
    if opts.batch_override_instances is not None:
        global_batch_size = opts.batch_override_instances * sequence_length
    else:
        global_batch_size = int(global_batch_size * opts.batch_multiplier)

    if global_batch_size is not old_global_batch_size:
        print(f"Modified global batch from {old_global_batch_size} to {global_batch_size}")

    # Used for stats tracking. Global batch size is seqlen * batch.
    # max_steps = 1 + training_tokens // global_batch_size

    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        DataMix.OLMo_mix_0925_official,
        tokenizer=tokenizer_config,
        mix_base_dir=opts.data_root,
        sequence_length=sequence_length,
        max_target_sequence_length=max(8192, sequence_length),
        work_dir=opts.work_dir,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=global_batch_size,
        seed=34521,
        num_workers=8,
    )

    adjusted_learning_rate = learning_rate * opts.lr_multiplier

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=sequence_length,
        max_sequence_length=sequence_length,
        optim=SkipStepAdamWConfig(
            lr=adjusted_learning_rate,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=CosWithWarmup(
            units=SchedulerUnits.tokens,
            warmup=opts.warmup_steps * global_batch_size,
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
        ),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget,
            activation_memory_budget=activation_memory_budget,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=opts.save_folder,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.tokens(training_tokens),
        )
        .with_callback("monkey_patcher", MonkeyPatcherCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=None,
                save_async=True,
            ),
        )
        .with_callback(
            "speed_monitor",
            SpeedMonitorCallback(
                # max_steps=max_steps,
            ),
        )
        .with_callback("stability_monitor", StabilityMonitorCallback(enabled=True))
        .with_callback(
            "comet",
            CometCallback(
                name=opts.name,
                workspace=opts.workspace,
                project=opts.project_name,
                cancel_check_interval=10,
                auto_resume=opts.auto_resume,
                enabled=opts.enable_comet_logging,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=opts.name,
                project=opts.project_name,
                cancel_check_interval=10,
                enabled=opts.enable_wandb_logging,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyPaddedFSLDatasetConfig.from_data_mix(
                    DataMix.v3_small_ppl_validation,
                    mix_base_dir=opts.data_root,
                    sequence_length=sequence_length,
                    tokenizer=tokenizer_config,
                    work_dir=opts.work_dir,
                ),
                # eval_on_finish=True,
                # max_steps=max_steps,
                log_interval=10,
                eval_interval=2_500,
            ),
        )
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=sorted(
                    [  # "fast" task set
                        # Subset of OLMES
                        "arc_challenge_test_bpb_5shot",
                        "arc_challenge_test_mc_5shot_fast",
                        "arc_easy_test_bpb_5shot",
                        "arc_easy_test_mc_5shot_fast",
                        "hellaswag_bpb_5shot",
                        "mmlu_humanities_test_bpb_5shot",
                        "mmlu_humanities_test_mc_5shot_fast",
                        "mmlu_other_test_bpb_5shot",
                        "mmlu_other_test_mc_5shot_fast",
                        "mmlu_social_sciences_test_bpb_5shot",
                        "mmlu_social_sciences_test_mc_5shot_fast",
                        "mmlu_stem_test_bpb_5shot",
                        "mmlu_stem_test_mc_5shot_fast",
                        # Basic Skills
                        "basic_skills_arithmetic_rc_5shot",
                        "basic_skills_coding_rc_5shot",
                        "basic_skills_common_knowledge_rc_5shot",
                        "basic_skills_logical_reasoning_rc_5shot",
                        "basic_skills_pattern_rc_5shot",
                        "basic_skills_string_operations_rc_5shot",
                        # Gen tasks BPB
                        "codex_humaneval_gold_bpb_3shot",
                        "codex_mbpp_gold_bpb_3shot",
                        "minerva_math_500_gold_bpb_0shot",
                        "mt_mbpp_cpp_gold_bpb_3shot",
                        "mt_mbpp_java_gold_bpb_3shot",
                        "mt_mbpp_rust_gold_bpb_3shot",
                        # Sanity check for MCQA ability
                        "copycolors_10way_fast",
                    ]
                ),
                tokenizer=tokenizer_config,
                # eval_on_finish=True,
                # max_steps=max_steps,
                eval_interval=5_000,
            ),
        )
    )

    return ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
    ).merge(overrides)


if __name__ == "__main__":
    _add_args()
    main(build_config)
