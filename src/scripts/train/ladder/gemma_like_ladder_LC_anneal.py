"""
Long-context anneal for gemma-like ladder checkpoints.

Takes a completed ladder run checkpoint (from gemma_like_ladder.py) and creates a
long-context anneal training run with:
- 64K sequence length (up from 8K)
- YaRN RoPE scaling on global attention layers
- LR warmup to 50% of original peak, then linear decay to 0
- Anneal length proportional to original training (calibrated: 8B @ 4xC -> 10B tokens)

Usage:
    python gemma_like_ladder_LC_anneal.py <SUBCOMMAND> <ORIGINAL_CHECKPOINT> <CLUSTER> [OVERRIDES...]

Examples:
    # Dry run to check config
    python gemma_like_ladder_LC_anneal.py dry_run gs://ai2-llm/checkpoints/gl-v2-8b/step100000/ ai2/jupiter

    # Launch on Beaker
    python gemma_like_ladder_LC_anneal.py launch gs://ai2-llm/checkpoints/gl-v2-8b/step100000/ ai2/jupiter

    # Override CP degree or node count
    python gemma_like_ladder_LC_anneal.py launch <checkpoint> ai2/jupiter --cp-degree=4 --launch.num_nodes=8
"""

import argparse
import importlib.util
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch

from olmo_core.config import DType
from olmo_core.data import DataMix, NumpyDataLoaderConfig, NumpyPackedFSLDatasetConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import get_root_dir, get_work_dir
from olmo_core.internal.cookbook import configure_required_callbacks
from olmo_core.internal.experiment import (
    CliContext,
    CommonComponents,
    DataComponents,
    SubCmd,
    build_config,
)
from olmo_core.io import add_cached_path_clients, join_path, resource_path
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.rope import YaRNRoPEScalingConfig
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerConfig,
    TransformerDataParallelWrappingStrategy,
)
from olmo_core.optim import OptimGroupOverride, SchedulerUnits, SkipStepAdamWConfig
from olmo_core.optim.scheduler import WSD
from olmo_core.train import Duration, LoadStrategy, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    SpeedMonitorCallback,
    StabilityMonitorCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)

log = logging.getLogger(__name__)

# Target 64K sequence length for LC anneal.
TARGET_SEQUENCE_LENGTH = 64 * 1024

# Original ladder default sequence length.
DEFAULT_OLD_SEQUENCE_LENGTH = 8192

# Anneal token fraction, calibrated so 8B v2 (8,331,358,720 non-embedding params)
# at 4xC (666.5B tokens) gets a 10B token anneal.
ANNEAL_TOKEN_FRACTION = 10_000_000_000 / (4 * 20 * 8_331_358_720)


@dataclass
class _LCAnnealSettings:
    """LC anneal settings for a specific model size."""

    cp_degree: int
    num_nodes: int
    dp_shard_degree: int
    activation_memory_budget: float


def _load_ladder_module():
    """Load the gemma_like_ladder module from the same directory."""
    spec = importlib.util.spec_from_file_location(
        "gemma_like_ladder", Path(__file__).parent / "gemma_like_ladder.py"
    )
    assert spec is not None and spec.loader is not None, "Failed to load gemma_like_ladder module"
    module = importlib.util.module_from_spec(spec)
    sys.modules["gemma_like_ladder"] = module
    spec.loader.exec_module(module)
    return module


def handle_lc_anneal_args(
    overrides: list[str],
) -> tuple[list[str], argparse.Namespace]:
    """Extract LC-anneal-specific args from overrides, returning remaining overrides and parsed args."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp-degree", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=TARGET_SEQUENCE_LENGTH)
    parser.add_argument("--lr-scale", type=float, default=0.5)
    parser.add_argument("--mix-base-dir", type=str, default="r2://olmo-data")
    parser.add_argument("--root-dir", type=str, default="")
    parser.add_argument("--work-dir", type=str, default="")
    parser.add_argument("--save-folder", type=str, default="")
    parser.add_argument("--use-gdn", action="store_true", default=False)

    # Build lists of arg prefixes and boolean flags for filtering.
    arg_prefixes: List[str] = []
    boolean_flags: List[str] = []
    for action in parser._actions:
        if isinstance(action, argparse._StoreAction):
            arg_prefixes.extend(action.option_strings)
        elif isinstance(action, argparse._StoreTrueAction):
            boolean_flags.extend(action.option_strings)

    custom_args_list: List[str] = []
    remaining: List[str] = []
    for override in overrides:
        matched = False
        if any(override.startswith(f"{prefix}=") for prefix in arg_prefixes):
            key, value = override.split("=", 1)
            custom_args_list.extend([key, value])
            matched = True
        elif override in boolean_flags:
            custom_args_list.append(override)
            matched = True
        if not matched:
            remaining.append(override)

    args = parser.parse_args(custom_args_list)
    return remaining, args


def apply_yarn_to_global_layers(
    model_config: TransformerConfig,
    sequence_length: int,
    old_sequence_length: int,
) -> None:
    """
    Apply YaRN RoPE scaling to global attention layers only.

    ``TransformerConfig.with_rope_scaling()`` raises when ``block_overrides`` are already set,
    which is always the case for v2 models (global attention every 5th layer). We apply
    YaRN manually to only the global layers. Local SWA layers (1024 window, theta=10K)
    and GDN layers don't need scaling.
    """
    yarn = YaRNRoPEScalingConfig(
        factor=sequence_length / old_sequence_length,
        beta_fast=32,
        beta_slow=1,
        old_context_len=old_sequence_length,
    )

    if model_config.block_overrides:
        for block_override in model_config.block_overrides.values():
            if isinstance(block_override.sequence_mixer, AttentionConfig):
                rope = block_override.sequence_mixer.rope
                if rope is not None:
                    new_rope = rope.copy()
                    new_rope.scaling = yarn
                    block_override.sequence_mixer.rope = new_rope


if __name__ == "__main__":
    usage = f"""
Performs LC anneal on a completed gemma-like ladder checkpoint.
Anneal length is computed automatically from the original training duration.

[yellow]Usage:[/] [i blue]python[/] [i cyan]{sys.argv[0]}[/] [i b magenta]{"|".join(SubCmd)}[/] [i b]ORIGINAL_CHECKPOINT CLUSTER[/] [i][OVERRIDES...][/]

[b]Subcommands[/]
[b magenta]launch:[/]       Launch the script on Beaker with the [b magenta]train[/] subcommand.
[b magenta]train:[/]        Run the trainer. Usually invoked by [b magenta]launch[/] or torchrun.
[b magenta]train_single:[/] Run the trainer on a single device.
[b magenta]dry_run:[/]      Pretty print the config and exit.

[b]LC anneal options[/]
  --cp-degree=N          Override context parallelism degree
  --sequence-length=N    Target sequence length (default: 65536)
  --lr-scale=F           Fraction of original peak LR (default: 0.5)
  --mix-base-dir=PATH    Base directory for LC data mix (default: r2://olmo-data)
  --use-gdn              Force GDN model variant (auto-detected from checkpoint)

[b]Examples[/]
$ [i]python {sys.argv[0]} {SubCmd.launch} gs://ai2-llm/checkpoints/gl-v2-8b/step100000/ ai2/jupiter[/]
$ [i]python {sys.argv[0]} {SubCmd.dry_run} /weka/oe-training-default/ai2-llm/checkpoints/gl-v2-260m/step50000/ ai2/jupiter[/]
    """.strip()

    if len(sys.argv) < 4 or sys.argv[1] not in set(SubCmd):
        import rich

        rich.get_console().print(usage, highlight=False)
        sys.exit(1)

    script, cmd, original_checkpoint, cluster, *all_overrides = sys.argv

    # Parse LC anneal custom args (remaining overrides are standard config overrides).
    remaining_overrides, custom_args = handle_lc_anneal_args(all_overrides)

    sequence_length = custom_args.sequence_length
    lr_scale = custom_args.lr_scale
    mix_base_dir = custom_args.mix_base_dir
    use_gdn = custom_args.use_gdn

    # Load the ladder module for model configs and utilities.
    ladder_module = _load_ladder_module()
    GemmaLikeOlmoV2 = ladder_module.GemmaLikeOlmoV2

    # Per-size LC anneal settings: (cp_degree, num_nodes).
    LC_ANNEAL_SETTINGS: Dict[str, _LCAnnealSettings] = {
        GemmaLikeOlmoV2.GL_260M: _LCAnnealSettings(
            cp_degree=1, num_nodes=1, dp_shard_degree=1, activation_memory_budget=1.0
        ),
        GemmaLikeOlmoV2.GL_709M: _LCAnnealSettings(
            cp_degree=1, num_nodes=2, dp_shard_degree=1, activation_memory_budget=1.0
        ),
        GemmaLikeOlmoV2.GL_1p3B: _LCAnnealSettings(
            cp_degree=2, num_nodes=3, dp_shard_degree=1, activation_memory_budget=1.0
        ),
        GemmaLikeOlmoV2.GL_2B: _LCAnnealSettings(
            cp_degree=4, num_nodes=8, dp_shard_degree=1, activation_memory_budget=1.0
        ),
        GemmaLikeOlmoV2.GL_4B: _LCAnnealSettings(
            cp_degree=4, num_nodes=12, dp_shard_degree=8, activation_memory_budget=0.9
        ),
        GemmaLikeOlmoV2.GL_8B: _LCAnnealSettings(
            cp_degree=8, num_nodes=16, dp_shard_degree=8, activation_memory_budget=0.8
        ),
        GemmaLikeOlmoV2.GL_15B: _LCAnnealSettings(
            cp_degree=8, num_nodes=16, dp_shard_degree=8, activation_memory_budget=0.4
        ),
        GemmaLikeOlmoV2.GL_34B: _LCAnnealSettings(
            cp_degree=8, num_nodes=32, dp_shard_degree=8, activation_memory_budget=0.1
        ),
    }

    # Register weka:// and other custom URL schemes before reading checkpoint.
    add_cached_path_clients()

    # Load checkpoint config and trainer state.
    log.info(f"Loading checkpoint config from {original_checkpoint}")
    config_file = resource_path(original_checkpoint, "config.json")
    with open(config_file, "rb") as f:
        checkpoint_config = json.load(f)

    trainer_state_file = resource_path(join_path(original_checkpoint, "train"), "rank0.pt")
    trainer_state = torch.load(trainer_state_file, weights_only=False)
    global_step = trainer_state["global_step"]
    old_sequence_length = trainer_state["data_loader"].get(
        "sequence_length", DEFAULT_OLD_SEQUENCE_LENGTH
    )

    # Extract parameters from checkpoint.
    original_run_name = checkpoint_config["run_name"]
    original_lr = checkpoint_config["train_module"]["optim"]["lr"]
    original_global_batch_size = checkpoint_config["data_loader"]["global_batch_size"]
    total_training_tokens = checkpoint_config["trainer"]["max_duration"]["value"]

    # Auto-detect GDN from checkpoint config if not explicitly set.
    if not use_gdn:
        block_mixer = checkpoint_config.get("model", {}).get("block", {}).get("sequence_mixer", {})
        if "GatedDeltaNet" in str(block_mixer.get("CLASS", "")):
            use_gdn = True
            log.info("Auto-detected GDN model from checkpoint config")

    # Parse model size and get settings.
    parse_model_size = ladder_module.parse_model_size
    model_size = parse_model_size(original_run_name)
    log.info(f"Model size: {model_size.value}, original run: {original_run_name}")

    # Compute anneal parameters.
    anneal_tokens = int(ANNEAL_TOKEN_FRACTION * total_training_tokens)
    anneal_peak_lr = lr_scale * original_lr
    global_batch_size = original_global_batch_size

    lc_settings = LC_ANNEAL_SETTINGS[model_size]
    cp_degree = (
        custom_args.cp_degree if custom_args.cp_degree is not None else lc_settings.cp_degree
    )
    num_nodes = lc_settings.num_nodes

    anneal_run_name = f"{original_run_name}-from{global_step}-LC"

    root_dir = custom_args.root_dir or get_root_dir(cluster)
    work_dir = custom_args.work_dir or get_work_dir(root_dir)
    save_folder = custom_args.save_folder or f"{root_dir}/checkpoints/{anneal_run_name}"

    log.info(
        f"LC anneal: {anneal_tokens / 1e9:.1f}B tokens, "
        f"LR: {anneal_peak_lr:.2e} (={lr_scale}x of {original_lr:.2e}), "
        f"seq_len: {sequence_length}, CP: {cp_degree}, nodes: {num_nodes}"
    )

    cmd = SubCmd(cmd)
    cli_context = CliContext(script, cmd, anneal_run_name, cluster, remaining_overrides)

    # --- Config builders (closures over the computed parameters) ---

    def build_model_config(common: CommonComponents) -> TransformerConfig:
        model_config, _ = model_size.get_settings(
            common.tokenizer.padded_vocab_size(), use_gdn=use_gdn
        )
        apply_yarn_to_global_layers(model_config, sequence_length, old_sequence_length)
        return model_config

    def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
        config = TransformerTrainModuleConfig(
            rank_microbatch_size=sequence_length,
            max_sequence_length=sequence_length,
            optim=SkipStepAdamWConfig(
                lr=anneal_peak_lr,
                weight_decay=0.1,
                betas=(0.9, 0.95),
                group_overrides=[
                    OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
                ],
            ),
            scheduler=WSD(
                units=SchedulerUnits.tokens,
                warmup_fraction=0.1,
                decay_fraction=0.9,
            ),
            compile_model=True,
            dp_config=TransformerDataParallelConfig(
                name=DataParallelType.hsdp,
                param_dtype=DType.bfloat16,
                reduce_dtype=DType.float32,
                shard_degree=lc_settings.dp_shard_degree,
                wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            ),
            ac_config=TransformerActivationCheckpointingConfig(
                mode=TransformerActivationCheckpointingMode.budget,
                activation_memory_budget=lc_settings.activation_memory_budget,
            ),
            float8_config=Float8Config(enabled=False),
            z_loss_multiplier=1e-5,
            max_grad_norm=1.0,
        )

        if cp_degree > 1:
            config.cp_config = TransformerContextParallelConfig.llama3(
                degree=cp_degree, head_stride=1
            )

        return config

    def build_data_config(common: CommonComponents) -> DataComponents:
        return DataComponents(
            dataset=NumpyPackedFSLDatasetConfig.from_data_mix(
                DataMix.OLMo_longmino_mix_0925,
                mix_base_dir=mix_base_dir,
                tokenizer=common.tokenizer,
                sequence_length=sequence_length,
                generate_doc_lengths=True,
                source_group_size=8,
                source_permutation_seed=123,
                work_dir=common.work_dir,
            ),
            data_loader=NumpyDataLoaderConfig(
                global_batch_size=global_batch_size,
                seed=4123,
                num_workers=12,
            ),
        )

    def build_trainer_config(common: CommonComponents) -> TrainerConfig:
        return (
            TrainerConfig(
                save_folder=save_folder,
                work_dir=work_dir,
                save_overwrite=True,
                metrics_collect_interval=50,
                cancel_check_interval=10,
                max_duration=Duration.tokens(anneal_tokens),
                load_path=original_checkpoint,
                load_strategy=LoadStrategy.always,
                load_trainer_state=False,
            )
            .with_callbacks(configure_required_callbacks(anneal_run_name))
            .with_callback(
                "checkpointer",
                CheckpointerCallback(
                    save_interval=10000,
                    ephemeral_save_interval=500,
                    save_async=True,
                ),
            )
            .with_callback("speed_monitor", SpeedMonitorCallback())
            .with_callback("stability_monitor", StabilityMonitorCallback(enabled=True))
            .with_callback(
                "comet",
                CometCallback(
                    name=anneal_run_name,
                    project="gl-olmo-lc-anneal",
                    workspace="oe-t-ladder",
                    cancel_check_interval=10,
                    auto_resume=False,
                    enabled=False,
                ),
            )
            .with_callback(
                "wandb",
                WandBCallback(
                    name=anneal_run_name,
                    group=anneal_run_name,
                    project="oe-t-ladder",
                    entity="ai2-llm",
                    cancel_check_interval=10,
                    enabled=True,
                ),
            )
            # No lm_evaluator or downstream_evaluator: CP is incompatible.
        )

    # --- Assemble config ---

    config = build_config(
        cli_context,
        global_batch_size=global_batch_size,
        max_sequence_length=sequence_length,
        data_config_builder=build_data_config,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        finalize_config=None,
        include_default_evals=False,
        num_nodes=num_nodes,
        beaker_workspace="ai2/oe-t-ladder",
    )

    # Fix launch cmd so Beaker re-runs with the correct arguments (including custom args).
    assert config.launch is not None
    config.launch.cmd = [script, "train", original_checkpoint, cluster] + all_overrides

    cmd.prepare_environment(config)
    cmd.run(config)
