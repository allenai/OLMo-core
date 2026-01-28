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

import argparse
import math
from dataclasses import dataclass
from datetime import datetime
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
from olmo_core.eval.task_groups import TASK_GROUPS
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.cookbook import configure_required_callbacks
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.internal.ri_olmo.ri_olmo_config import RicursiveTransformerConfig
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerConfig,
)
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
    DownstreamEvaluatorCallbackConfig,
    LMEvaluatorCallbackConfig,
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


@dataclass
class _ModelSizeSettings:
    """Training settings for a specific model size."""

    size: str
    num_nodes: int
    batch_size_round_nearest: int
    activation_memory_budget: float


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

    def get_settings(self, vocab_size: int) -> Tuple[TransformerConfig, _ModelSizeSettings]:
        """Get the model config and all settings for this model size."""
        # Mapping: (size, num_nodes, round_nearest, activation_memory_budget)
        settings_map = {
            RicursiveOlmoV1.RI_OLMO_260M: _ModelSizeSettings("260M", 1, 16, 1.0),
            RicursiveOlmoV1.RI_OLMO_709M: _ModelSizeSettings("709M", 2, 16, 1.0),
            RicursiveOlmoV1.RI_OLMO_1p3B: _ModelSizeSettings("1p3B", 2, 16, 1.0),
            RicursiveOlmoV1.RI_OLMO_2B: _ModelSizeSettings("2B", 4, 16, 1.0),
            RicursiveOlmoV1.RI_OLMO_4B: _ModelSizeSettings("4B", 4, 32, 1.0),
            RicursiveOlmoV1.RI_OLMO_8B: _ModelSizeSettings("8B", 8, 64, 0.9),
            RicursiveOlmoV1.RI_OLMO_15B: _ModelSizeSettings(
                "15B", 8, 128, 0.4
            ),  # Support up to 16 hosts, bsz8 per host
            RicursiveOlmoV1.RI_OLMO_34B: _ModelSizeSettings(
                "34B", 8, 16, 0.1
            ),  # Currently does not work, OOMs!!!
            RicursiveOlmoV1.RI_OLMO_65B: _ModelSizeSettings("65B", 16, 16, 1.0),
        }
        if self not in settings_map:
            raise ValueError(
                f"Model not in list! Valid models: {[m.name for m in RicursiveOlmoV1]}\n\n"
            )

        settings = settings_map[self]
        config_method = getattr(RicursiveTransformerConfig, f"ri_olmo_v1_{settings.size}")
        model_config = config_method(vocab_size)
        return (model_config, settings)


def handle_custom_args(
    overrides: list[str],
) -> tuple[list[str], argparse.Namespace]:
    """Extract multiplier override values using argparse and remove them from the list."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr-multiplier", type=float, default=1.0)
    parser.add_argument("--batch-multiplier", type=float, default=1.0)
    parser.add_argument("--chinchilla-multiple", type=float, default=4.0)  # Default is 4xC

    # Extract argument names from parser
    arg_prefixes = [
        option_string
        for action in parser._actions
        if isinstance(action, argparse._StoreAction)
        for option_string in action.option_strings
    ]

    # Remove custom args from overrides
    multiplier_args = []
    remaining = []
    for override in overrides:
        if any(override.startswith(f"{prefix}=") for prefix in arg_prefixes):
            # Split "key=value" into ["--key", "value"] for argparse
            key, value = override.split("=", 1)
            multiplier_args.extend([key, value])
        else:
            remaining.append(override)

    # Parse multiplier args
    args = parser.parse_args(multiplier_args)
    return remaining, args


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


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    """
    Build experiment config for RI-OLMo v1.

    Model size can be specified as just the size (e.g., "260m") or with prefix
    (e.g., "ri-olmo-v1-260m"). The model size is parsed from the run name.

    Hyperparameters are computed using StepFun optimal schedules [Li 2025], but can be
    overridden using standard config override syntax:

    Standard config overrides:
        --train_module.optim.lr=0.001                     Override learning rate
        --data_loader.global_batch_size=1000              Override batch size
        --train_module.scheduler.warmup=1000000           Override warmup (in tokens)
        --trainer.callbacks.comet.enabled=false           Disable Comet logging
        --trainer.callbacks.wandb.enabled=true            Enable WandB logging
        --launch.num_nodes=2                              Override node count

    Convenience multipliers (for quick hyperparameter sweeps):
        --lr-multiplier=2.0                                Multiply computed learning rate
        --batch-multiplier=0.5                             Multiply computed batch size
        --chinchilla-multiple=1                          Multiply Chinchilla training tokens

    """
    # Parse model size from run name
    model = parse_model_size(cli_context.run_name)
    print(f"Parsed model size: {model} from run name: {cli_context.run_name}")

    # Add timestamp to run name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name_with_timestamp = f"{cli_context.run_name}-{timestamp}"

    # Extract convenience multipliers from overrides (remove them from override list)
    overrides = list(cli_context.overrides)
    overrides, custom_args = handle_custom_args(overrides)
    lr_multiplier = custom_args.lr_multiplier
    batch_multiplier = custom_args.batch_multiplier
    chinchilla_multiple = custom_args.chinchilla_multiple

    sequence_length = DEFAULT_SEQUENCE_LENGTH
    root_dir = get_root_dir(cli_context.cluster)
    data_root = "gs://ai2-llm"
    work_dir = get_work_dir(root_dir)
    save_folder = f"{root_dir}/checkpoints/{cli_context.run_name}"

    tokenizer_config = TokenizerConfig.dolma2()
    model_config, model_size_settings = model.get_settings(tokenizer_config.padded_vocab_size())

    # Compute hyperparameters
    model_active_params = model_config.num_active_params
    train_duration = Duration.chinchilla_tokens(
        chinchilla_multiple, model_params=model_active_params
    )
    training_tokens = train_duration.value

    learning_rate = get_learning_rate(model_active_params, training_tokens)
    base_global_batch_size = get_global_batch_size(
        model_params=model_active_params,
        training_tokens=training_tokens,
        sequence_length=sequence_length,
        round_nearest=model_size_settings.batch_size_round_nearest,
    )

    # Apply custom multipliers
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

    beaker_launch_config: Optional[BeakerLaunchConfig] = build_launch_config(
        name=cli_context.run_name,
        cmd=cli_context.remote_cmd,
        cluster=cli_context.cluster,
        root_dir=root_dir,
        workspace="ai2/oe-t-ladder",
        num_nodes=model_size_settings.num_nodes,
        nccl_debug=True,
    )

    # Dataset config
    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        # DataMix.OLMo_mix_0925_official,
        DataMix.OLMo_mix_0925,
        tokenizer=tokenizer_config,
        mix_base_dir=data_root,
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
            activation_memory_budget=model_size_settings.activation_memory_budget,
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
            max_duration=train_duration,
        )
        .with_callbacks(configure_required_callbacks(cli_context.run_name))
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
                name=run_name_with_timestamp,
                group=cli_context.run_name,
                project="ri-olmo",
                entity="ai2-llm",
                cancel_check_interval=10,
                enabled=False,
            ),
        )
        .with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyPaddedFSLDatasetConfig.from_data_mix(
                    DataMix.v3_small_ppl_validation,
                    mix_base_dir=data_root,
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
                tasks=sorted(TASK_GROUPS["fast"]),
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
    Invoke this script directly to access the internal experiment CLI.

    The CLI supports several subcommands: launch, train, dry_run, and others.
    See the main() function documentation for full details.

    Examples:
        Render the config and exit (dry run):
            python src/olmo_core/internal/ri_olmo/model_ladder_v1.py dry_run ri-olmo-v1-260m ai2/jupiter

        Start a local training run with torchrun:
            torchrun --nproc-per-node=8 src/olmo_core/internal/ri_olmo/model_ladder_v1.py train ri-olmo-v1-260m ai2/jupiter

        Launch a training run on Beaker (Jupiter cluster):
            python src/olmo_core/internal/ri_olmo/model_ladder_v1.py launch ri-olmo-v1-260m ai2/jupiter

        Launch with custom hyperparameters using multipliers:
            python src/olmo_core/internal/ri_olmo/model_ladder_v1.py launch ri-olmo-v1-260m ai2/jupiter \\
                --lr-multiplier=2.0 \\
                --batch-multiplier=0.5

        Override specific config values:
            python src/olmo_core/internal/ri_olmo/model_ladder_v1.py launch ri-olmo-v1-8b ai2/jupiter \\
                --launch.num_nodes=2 \\
                --train_module.scheduler.warmup=5000000

        Enable logging callbacks:
            python src/olmo_core/internal/ri_olmo/model_ladder_v1.py launch ri-olmo-v1-260m ai2/jupiter \\
                --trainer.callbacks.wandb.enabled=true \\
                --trainer.callbacks.comet.enabled=true
    """
    main(config_builder=build_experiment_config)
