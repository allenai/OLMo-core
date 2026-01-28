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

Usage:
    python src/olmo_core/internal/ri_olmo/model_ladder_v1.py dry_run ri-olmo-v1-260m ai2/jupiter
    python src/olmo_core/internal/ri_olmo/model_ladder_v1.py launch ri-olmo-v1-260m ai2/jupiter
    python src/olmo_core/internal/ri_olmo/model_ladder_v1.py train ri-olmo-v1-260m ai2/jupiter

"""

import math
from typing import Optional, Tuple

from olmo_core.config import DType, StrEnum
from olmo_core.data import (
    DataMix,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.internal.ri_olmo.ri_olmo_config import RicursiveTransformerConfig
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.nn.transformer import TransformerActivationCheckpointingMode, TransformerConfig
from olmo_core.optim import (
    CosWithWarmup,
    OptimGroupOverride,
    SchedulerUnits,
    SkipStepAdamWConfig,
)
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


class RicursiveOlmoV1(StrEnum):
    RI_OLMO_260M = "260M"
    RI_OLMO_709M = "709M"
    RI_OLMO_1p3B = "1.3B"
    RI_OLMO_2B = "2B"
    RI_OLMO_4B = "4B"
    RI_OLMO_8B = "8B"
    RI_OLMO_15B = "15B"
    RI_OLMO_34B = "34B"
    RI_OLMO_65B = "65B"

    def get_num_nodes(self) -> int:
        """Get the number of nodes required for this model size."""
        match self:
            case (
                RicursiveOlmoV1.RI_OLMO_260M
                | RicursiveOlmoV1.RI_OLMO_709M
                | RicursiveOlmoV1.RI_OLMO_1p3B
                | RicursiveOlmoV1.RI_OLMO_2B
                | RicursiveOlmoV1.RI_OLMO_4B
                | RicursiveOlmoV1.RI_OLMO_8B
            ):
                return 1  # 8 GPUs
            case RicursiveOlmoV1.RI_OLMO_15B:
                return 2  # 16 GPUs
            case RicursiveOlmoV1.RI_OLMO_34B:
                return 4  # 32 GPUs
            case RicursiveOlmoV1.RI_OLMO_65B:
                return 8  # 64 GPUs
            case _:
                raise ValueError(f"Invalid model size: {self}")

    def get_model_config_and_settings(
        self, vocab_size: int
    ) -> Tuple[TransformerConfig, int, float]:
        """Get the model config, round_nearest, and activation_memory_budget for this model size."""
        match self:
            case RicursiveOlmoV1.RI_OLMO_260M:
                return (RicursiveTransformerConfig.ri_olmo_v1_260M(vocab_size), 16, 1.0)
            case RicursiveOlmoV1.RI_OLMO_709M:
                return (RicursiveTransformerConfig.ri_olmo_v1_709M(vocab_size), 16, 1.0)
            case RicursiveOlmoV1.RI_OLMO_1p3B:
                return (RicursiveTransformerConfig.ri_olmo_v1_1p3B(vocab_size), 16, 1.0)
            case RicursiveOlmoV1.RI_OLMO_2B:
                return (RicursiveTransformerConfig.ri_olmo_v1_2B(vocab_size), 16, 1.0)
            case RicursiveOlmoV1.RI_OLMO_4B:
                return (RicursiveTransformerConfig.ri_olmo_v1_4B(vocab_size), 32, 1.0)
            case RicursiveOlmoV1.RI_OLMO_8B:
                return (RicursiveTransformerConfig.ri_olmo_v1_8B(vocab_size), 64, 0.9)
            case RicursiveOlmoV1.RI_OLMO_15B:
                # Support up to 16 hosts, bsz8 per host
                return (RicursiveTransformerConfig.ri_olmo_v1_15B(vocab_size), 128, 0.4)
            case RicursiveOlmoV1.RI_OLMO_34B:
                # Currently does not work, OOMs!!!
                return (RicursiveTransformerConfig.ri_olmo_v1_34B(vocab_size), 16, 0.1)
            case RicursiveOlmoV1.RI_OLMO_65B:
                return (RicursiveTransformerConfig.ri_olmo_v1_65B(vocab_size), 16, 1.0)
            case _:
                raise ValueError(
                    f"Model not in list! Valid models: {[m.name for m in RicursiveOlmoV1]}\n\n"
                )


def get_learning_rate(model_params: int, training_tokens: int) -> float:
    """
    Get optimal learning rate using step law from Li 2025.
    https://arxiv.org/pdf/2503.04715v1
    """
    n = model_params
    d = training_tokens
    lr = 1.79 * pow(n, -0.713) * pow(d, 0.307)

    print(f"Model size: {n}, training tokens: {d}, opt_lr: {lr}")

    return lr


def get_global_batch_size(
    model_params: int,
    training_tokens: int,
    sequence_length: int,
    round_nearest: int,
) -> int:
    """
    Get optimal global batch size in tokens using step law from Li 2025.
    https://arxiv.org/pdf/2503.04715v1
    """
    n = model_params
    d = training_tokens
    global_bsz = 0.58 * pow(d, 0.571)

    print(f"Model size: {n}, training tokens: {d}, opt_global_bsz: {global_bsz}")
    instance_bsz = global_bsz / sequence_length

    # Round batch size to (round_nearest * seqlen), clamping up
    rounded_instance_bsz = int(math.ceil(instance_bsz / round_nearest) * round_nearest)
    print(f"Rounding instance bsz from {instance_bsz} to {rounded_instance_bsz}")

    rounded_global_bsz = sequence_length * rounded_instance_bsz
    print(f"Rounding global bsz from {global_bsz} to {rounded_global_bsz}")

    return rounded_global_bsz


def parse_model_size(run_name: str) -> RicursiveOlmoV1:
    """
    Parse model size from run name.
    The run name must contain one of the enum values (e.g., "260M", "1.3B", "8B").
    Examples: "260m", "ri-olmo-v1-260m", "1.3b", "1p3b" (normalized to "1.3b").
    """
    normalized = run_name.lower().strip().replace("1p3b", "1.3b").replace("1p3", "1.3")

    for size in RicursiveOlmoV1:
        if size.value.lower() in normalized:
            return size

    raise ValueError(
        f"Could not parse model size from run name '{run_name}'. "
        f"Valid sizes: {[s.value for s in RicursiveOlmoV1]}. "
        f"Examples: '260m', 'ri-olmo-v1-260m', '1.3b'"
    )


def _extract_and_remove_overrides(
    overrides: list[str], prefix: str
) -> tuple[list[str], Optional[str]]:
    """Extract override with given prefix and remove it from the list."""
    value = None
    remaining = []
    for override in overrides:
        if override.startswith(f"{prefix}="):
            value = override.split("=", 1)[1]
        else:
            remaining.append(override)
    return remaining, value


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    """
    Build experiment config for RI-OLMo v1.

    Model size can be specified as just the size (e.g., "260m") or with prefix
    (e.g., "ri-olmo-v1-260m"). The model size is parsed from the run name.

    Hyperparameters are computed using StepFun optimal schedules [Li 2025], but can be
    overridden using standard config override syntax:

        --train_module.optim.lr=0.001              Override learning rate
        --data_loader.global_batch_size=1000       Override batch size
        --train_module.scheduler.warmup=1000000    Override warmup (in tokens)

    Convenience multipliers (for quick sweeps):
        --lr-multiplier=2.0          Multiply computed LR
        --batch-multiplier=0.5       Multiply computed batch size

    Other settings can be overridden via standard config paths, e.g.:
        --trainer.callbacks.comet.enabled=false
        --trainer.callbacks.comet.auto_resume=true
        --trainer.callbacks.wandb.enabled=true
        --launch.num_nodes=2
    """
    # Parse model size from run name
    model = parse_model_size(cli_context.run_name)
    print(f"Parsed model size: {model} from run name: {cli_context.run_name}")

    # Extract convenience multipliers from overrides (remove them from override list)
    overrides = list(cli_context.overrides)
    overrides, lr_multiplier_str = _extract_and_remove_overrides(overrides, "--lr-multiplier")
    overrides, batch_multiplier_str = _extract_and_remove_overrides(overrides, "--batch-multiplier")
    overrides, chinchilla_multiple_str = _extract_and_remove_overrides(
        overrides, "--chinchilla-multiple"
    )

    lr_multiplier = float(lr_multiplier_str) if lr_multiplier_str else 1.0
    batch_multiplier = float(batch_multiplier_str) if batch_multiplier_str else 1.0
    chinchilla_multiple = float(chinchilla_multiple_str) if chinchilla_multiple_str else 1.0

    sequence_length = DEFAULT_SEQUENCE_LENGTH
    root_dir = get_root_dir(cli_context.cluster)
    work_dir = get_work_dir(root_dir)
    save_folder = f"{root_dir}/checkpoints/{cli_context.run_name}"

    tokenizer_config = TokenizerConfig.dolma2()
    vocab_size = tokenizer_config.padded_vocab_size()

    # Get model config and size-specific settings
    model_config, round_nearest, activation_memory_budget = model.get_model_config_and_settings(
        vocab_size
    )

    # Compute hyperparameters
    model_active_params = model_config.num_active_params
    training_tokens = Duration.chinchilla_tokens(
        chinchilla_multiple, model_params=model_active_params
    ).value

    learning_rate = get_learning_rate(model_active_params, training_tokens)
    base_global_batch_size = get_global_batch_size(
        model_params=model_active_params,
        training_tokens=training_tokens,
        sequence_length=sequence_length,
        round_nearest=round_nearest,
    )

    # Apply multipliers
    adjusted_learning_rate = learning_rate * lr_multiplier
    global_batch_size = int(base_global_batch_size * batch_multiplier)

    if lr_multiplier != 1.0:
        print(
            f"Applied LR multiplier: {lr_multiplier}, LR: {learning_rate} -> {adjusted_learning_rate}"
        )
    if batch_multiplier != 1.0:
        print(
            f"Applied batch multiplier: {batch_multiplier}, batch: {base_global_batch_size} -> {global_batch_size}"
        )

    # Build Beaker launch config
    beaker_launch_config: Optional[BeakerLaunchConfig] = build_launch_config(
        name=cli_context.run_name,
        cmd=cli_context.remote_cmd,
        cluster=cli_context.cluster,
        root_dir=root_dir,
        workspace="ai2/oe-t-ladder",
        num_nodes=model.get_num_nodes(),
        nccl_debug=True,
    )

    # Dataset config
    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        DataMix.OLMo_mix_0925_official,
        tokenizer=tokenizer_config,
        mix_base_dir=root_dir,
        sequence_length=sequence_length,
        max_target_sequence_length=max(8192, sequence_length),
        work_dir=work_dir,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=global_batch_size, seed=34521, num_workers=8
    )

    # Train module config
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
            warmup=2000 * global_batch_size,
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

    # Trainer config
    trainer_config = (
        TrainerConfig(
            save_folder=save_folder,
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
        .with_callback("speed_monitor", SpeedMonitorCallback())
        .with_callback("stability_monitor", StabilityMonitorCallback(enabled=True))
        .with_callback(
            "comet",
            CometCallback(
                name=cli_context.run_name,
                project="ri-olmo-v1",
                workspace="ricursive-olmo",
                cancel_check_interval=10,
                auto_resume=False,
                enabled=False,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=cli_context.run_name,
                project="ri-olmo-v1",
                cancel_check_interval=10,
                enabled=False,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyPaddedFSLDatasetConfig.from_data_mix(
                    DataMix.v3_small_ppl_validation,
                    mix_base_dir=root_dir,
                    sequence_length=sequence_length,
                    tokenizer=tokenizer_config,
                    work_dir=work_dir,
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

    experiment_config = ExperimentConfig(
        run_name=cli_context.run_name,
        launch=beaker_launch_config,
        model=model_config,
        train_module=train_module_config,
        trainer=trainer_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
    )

    # Merge remaining overrides (multipliers have been removed)
    return experiment_config.merge(overrides)


if __name__ == "__main__":
    """
    Invoke this script directly to access the internal experiment CLI, which
    supports launch, train, dry_run, and other subcommands.

    Examples:
        To render the config and exit:
            python src/olmo_core/internal/ri_olmo/model_ladder_v1.py dry_run ri-olmo-v1-260m ai2/jupiter

        To start a training run with torchrun:
            torchrun --nproc-per-node=8 src/olmo_core/internal/ri_olmo/model_ladder_v1.py train ri-olmo-v1-260m ai2/jupiter

        To launch a training run on Jupiter:
            python src/olmo_core/internal/ri_olmo/model_ladder_v1.py launch ri-olmo-v1-260m ai2/jupiter

        To launch with custom hyperparameters:
            python src/olmo_core/internal/ri_olmo/model_ladder_v1.py launch ri-olmo-v1-260m ai2/jupiter \\
                --lr-multiplier=2.0 \\
                --warmup-steps=1000

        To override num_nodes:
            python src/olmo_core/internal/ri_olmo/model_ladder_v1.py launch ri-olmo-v1-8b ai2/jupiter \\
                --launch.num_nodes=2
    """
    main(config_builder=build_experiment_config)
