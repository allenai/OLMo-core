"""
Ricursive-Olmo v1 model series with model merging.

This is a copy of model_ladder_v1.py with model merging added.
Merged checkpoints are saved and evaluated at eval intervals and before the final decay phase.

Usage:
    python src/olmo_core/internal/ri_olmo/model_ladder_v1_merging.py dry_run ri-olmo-v1-260m ai2/jupiter
    python src/olmo_core/internal/ri_olmo/model_ladder_v1_merging.py launch ri-olmo-v1-260m ai2/jupiter
    python src/olmo_core/internal/ri_olmo/model_ladder_v1_merging.py train ri-olmo-v1-260m ai2/jupiter

"""

import argparse
import math
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

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
    CosWithWarmupAndLinearDecay,
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
from olmo_core.train.callbacks.model_merger_v2 import ModelMergeCallbackV2
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

DEFAULT_SEQUENCE_LENGTH = 8192

# Model merging configuration
MERGE_LAST_N_STEPS = 250


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
    parser.add_argument("--mix-base-dir", type=str, default="gs://ai2-llm")
    parser.add_argument("--root-dir", type=str, default="")
    parser.add_argument("--work-dir", type=str, default="")
    parser.add_argument("--save-folder", type=str, default="")
    parser.add_argument("--lr-multiplier", type=float, default=1.0)
    parser.add_argument("--batch-multiplier", type=float, default=1.0)
    parser.add_argument("--chinchilla-multiple", type=float, default=4.0)  # Default is 4xC
    parser.add_argument("--no-beaker-launch", action="store_true", default=False)
    parser.add_argument(
        "--data-mix",
        type=str,
        choices=list(DataMix),
        default=str(DataMix.OLMo_mix_0925),
    )

    # Extract argument names from parser (both value-based and boolean flags)
    arg_prefixes = []
    boolean_flags = []
    for action in parser._actions:
        if isinstance(action, argparse._StoreAction):
            arg_prefixes.extend(action.option_strings)
        elif isinstance(action, argparse._StoreTrueAction):
            boolean_flags.extend(action.option_strings)

    # Remove custom args from overrides
    custom_args_list = []
    remaining = []
    for override in overrides:
        matched = False
        # Check for value-based args (--key=value)
        if any(override.startswith(f"{prefix}=") for prefix in arg_prefixes):
            # Split "key=value" into ["--key", "value"] for argparse
            key, value = override.split("=", 1)
            custom_args_list.extend([key, value])
            matched = True
        # Check for boolean flags (--flag)
        elif override in boolean_flags:
            custom_args_list.append(override)
            matched = True

        if not matched:
            remaining.append(override)

    # Parse custom args
    args = parser.parse_args(custom_args_list)
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


def compute_merge_steps(
    max_steps: int,
    decay_steps: int,
    eval_interval: int,
    merge_last_n_steps: int,
) -> List[int]:
    """
    Compute merge steps: at eval intervals + before final decay.

    If the pre-decay merge would overlap with the last eval merge, the last
    eval merge is replaced with the pre-decay merge to avoid overlapping windows.
    """
    if eval_interval <= 0:
        raise ValueError(f"eval_interval must be positive, got {eval_interval}")

    decay_start_step = max_steps - decay_steps
    if decay_start_step < eval_interval:
        raise ValueError(
            f"decay_start_step ({decay_start_step}) must be >= eval_interval ({eval_interval})"
        )

    merge_steps = list(range(eval_interval, decay_start_step, eval_interval))

    # Handle pre-decay merge, avoiding overlap with last eval merge
    if merge_steps and (decay_start_step - merge_steps[-1]) < merge_last_n_steps:
        # Would overlap: replace last eval merge with pre-decay merge
        merge_steps[-1] = decay_start_step
    else:
        merge_steps.append(decay_start_step)

    return merge_steps


def compute_merge_window_starts(
    merge_steps: List[int],
    merge_last_n_steps: int,
) -> List[int]:
    """
    Compute window start steps where checkpoints must be saved.
    """
    return [max(0, step - merge_last_n_steps + 1) for step in merge_steps]


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    """
    Build experiment config for RI-OLMo v1 with model merging.

    Model size can be specified as just the size (e.g., "260m") or with prefix
    (e.g., "ri-olmo-v1-260m"). The model size is parsed from the run name.

    Hyperparameters are computed using StepFun optimal schedules [Li 2025], but can be
    overridden using standard config override syntax.
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
    mix_base_dir = custom_args.mix_base_dir
    data_mix = custom_args.data_mix
    lr_multiplier = custom_args.lr_multiplier
    batch_multiplier = custom_args.batch_multiplier
    chinchilla_multiple = custom_args.chinchilla_multiple
    no_beaker_launch = custom_args.no_beaker_launch

    sequence_length = DEFAULT_SEQUENCE_LENGTH
    root_dir = custom_args.root_dir or get_root_dir(cli_context.cluster)
    work_dir = custom_args.work_dir or get_work_dir(root_dir)
    save_folder = custom_args.save_folder or f"{root_dir}/checkpoints/{cli_context.run_name}"

    print(f"mix_base_dir (dataset location): {mix_base_dir}")
    print(f"root_dir (checkpoint location): {root_dir}")
    print(f"work_dir (local path for temp files): {work_dir}")
    print(f"save_folder (checkpoint location): {save_folder}")

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

    # Compute training steps and decay
    max_steps = training_tokens // global_batch_size
    decay_steps = 2000  # From scheduler: decay=2000 * global_batch_size tokens

    # Eval intervals
    lm_eval_interval = 2500
    downstream_eval_interval = 5000

    # Compute merge steps (at LM eval intervals + before decay)
    merge_steps = compute_merge_steps(
        max_steps=max_steps,
        decay_steps=decay_steps,
        eval_interval=lm_eval_interval,
        merge_last_n_steps=MERGE_LAST_N_STEPS,
    )
    window_starts = compute_merge_window_starts(merge_steps, MERGE_LAST_N_STEPS)

    print(f"Model merging configuration:")
    print(f"  max_steps: {max_steps}")
    print(f"  decay_steps: {decay_steps} (decay starts at step {max_steps - decay_steps})")
    print(f"  merge_steps: {merge_steps}")
    print(f"  merge_last_n_steps: {MERGE_LAST_N_STEPS}")
    print(f"  window_starts (checkpoints required): {window_starts}")

    beaker_launch_config = None
    if not no_beaker_launch:
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
        mix=data_mix,
        tokenizer=tokenizer_config,
        mix_base_dir=mix_base_dir,
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
        scheduler=CosWithWarmupAndLinearDecay(
            units=SchedulerUnits.tokens,
            warmup=2000 * global_batch_size,
            decay=2000 * global_batch_size,
            decay_fraction=None,
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
                fixed_steps=window_starts,  # Ensure checkpoints at merge window starts
                ephemeral_save_interval=None,
                save_async=True,
            ),
        )
        .with_callback(
            "model_merger",
            ModelMergeCallbackV2(
                merge_step=merge_steps,
                merge_last_n_steps=MERGE_LAST_N_STEPS,
                enabled=True,
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
                    mix_base_dir=mix_base_dir,
                    sequence_length=sequence_length,
                    tokenizer=tokenizer_config,
                    work_dir=work_dir,
                ),
                eval_on_finish=True,
                log_interval=10,
                eval_interval=lm_eval_interval,
            ),
        )
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=sorted(TASK_GROUPS["fast"]),
                tokenizer=tokenizer_config,
                eval_on_finish=True,
                eval_interval=downstream_eval_interval,
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
    main(config_builder=build_experiment_config)
