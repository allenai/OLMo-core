"""
Gemma-like ladder.

This uses the `gemma3_like()` function from `TransformerConfig`, but puts in some stuff from Qwen on top of it.
"""

import argparse
import math
from datetime import datetime
from typing import List, Optional, Tuple, Dict

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
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMLossImplementation, LMHeadConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType
from olmo_core.nn.transformer import TransformerActivationCheckpointingMode, TransformerBlockConfig, \
    TransformerBlockType
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


from dataclasses import dataclass

from olmo_core.nn.attention import AttentionBackendName, GateConfig, GateGranularity, GatedDeltaNetConfig, \
    AttentionConfig, AttentionType, SlidingWindowAttentionConfig
from olmo_core.nn.feed_forward import ActivationFunction, FeedForwardConfig
from olmo_core.nn.transformer import TransformerConfig


@dataclass
class GemmaLikeTransformerConfig(TransformerConfig):
    @classmethod
    def v1(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        The v1 model series baseline.
        """
        return cls.gemma3_like(
            vocab_size=vocab_size,
            n_kv_heads=6,
            head_dim=128,
            global_layer_interval=5,
            activation=ActivationFunction.silu,
            gate=GateConfig(
                granularity=GateGranularity.elementwise,
                full_precision=True,
            ),
            attn_backend=kwargs.pop(
                "attn_backend",
                AttentionBackendName.flash_3,
            ),
            **kwargs,
        )

    @classmethod
    def v1_250M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 250M model config.

        251,359,360 total params
        187,134,080 non-embedding params
        """
        return cls.v1(
            d_model=640,
            hidden_size=640 * 8,
            n_layers=10,
            n_heads=6,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def v1_680M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 680M model config.

        677,446,400 total params
        574,685,952 non-embedding params
        """
        return cls.v1(
            d_model=1024,
            hidden_size=1024 * 8,
            n_layers=15,
            n_heads=12,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def v1_1p2B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 1.2B model config.

        1,200,728,320 total params
        1,072,277,760 non-embedding params
        """
        return cls.v1(
            d_model=1280,
            hidden_size=1280 * 8,
            n_layers=20,
            n_heads=12,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def v1_2B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 2B model config.

        2,048,423,680 total params
        1,894,283,008 non-embedding params
        """
        return cls.v1(
            d_model=1536,
            hidden_size=1536 * 8,
            n_layers=25,
            n_heads=18,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def v1_4B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 4B model config.

        4,091,799,040 total params
        3,886,278,144 non-embedding params
        """
        return cls.v1(
            d_model=2048,
            hidden_size=2048 * 8,
            n_layers=30,
            n_heads=24,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def v1_8B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        An 8B model config.

        8,142,615,040 total params
        7,885,713,920 non-embedding params
        """
        return cls.v1(
            d_model=2560,
            hidden_size=2560 * 8,
            n_layers=40,
            n_heads=30,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def v1_14B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 14B model config.

        14,301,109,760 total params
        13,992,828,416 non-embedding params
        """
        return cls.v1(
            d_model=3072,
            hidden_size=3072 * 8,
            n_layers=50,
            n_heads=36,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def v1_32B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 32B model config.

        32,311,906,560 total params
        31,900,864,768 non-embedding params
        """
        return cls.v1(
            d_model=4096,
            hidden_size=4096 * 8,
            n_layers=65,
            n_heads=48,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def v2(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        # Too many changes from the original gemma_like, so we need to import a lot of stuff into here.

        d_model: int = kwargs.pop("d_model")
        n_layers: int = kwargs.pop("n_layers")
        n_heads: int = kwargs.pop("n_heads")
        hidden_size: int = kwargs.pop("hidden_size")
        n_kv_heads = 8
        head_dim = 128
        global_layer_interval = 5
        global_rope_theta = 1_000_000
        layer_norm_eps = 1e-6
        dtype = DType.float32
        attn_backend = kwargs.pop("attn_backend", AttentionBackendName.flash_3)
        lm_head_loss_impl = kwargs.pop("lm_head_loss_impl", LMLossImplementation.default)
        use_gdn = kwargs.pop("use_gdn", False)

        local_rope_theta = 10_000
        local_window_size = 1024

        layer_norm = LayerNormConfig(
            name=LayerNormType.rms,
            eps=layer_norm_eps,
            bias=False,
            dtype=dtype,
        )

        feed_forward = FeedForwardConfig(
            hidden_size=hidden_size,
            bias=False,
            dtype=dtype,
            activation=ActivationFunction.silu,
        )

        if use_gdn:
            # Default block uses GatedDeltaNet (replaces sliding window attention layers).
            block = TransformerBlockConfig(
                name=TransformerBlockType.peri_norm,
                sequence_mixer=GatedDeltaNetConfig(
                    n_heads=n_heads,
                    n_v_heads=n_heads,  # for GDN, we intentionally match the number of heads.
                    head_dim=head_dim,
                    expand_v=1.0,
                    dtype=dtype,
                ),
                feed_forward=feed_forward,
                layer_norm=layer_norm,
            )
        else:
            # Default block uses sliding window attention (like v1/gemma3_like).
            block = TransformerBlockConfig(
                name=TransformerBlockType.peri_norm,
                sequence_mixer=AttentionConfig(
                    name=AttentionType.default,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    bias=False,
                    rope=RoPEConfig(name=RoPEType.default, theta=local_rope_theta),
                    gate=GateConfig(
                        granularity=GateGranularity.elementwise,
                        full_precision=True,
                    ),
                    qk_norm=layer_norm,
                    use_head_qk_norm=True,
                    backend=attn_backend,
                    sliding_window=SlidingWindowAttentionConfig(
                        pattern=[local_window_size] * (global_layer_interval - 1) + [-1],
                        force_full_attention_on_first_layer=False,
                        force_full_attention_on_last_layer=False,
                    ),
                    dtype=dtype,
                ),
                feed_forward=feed_forward,
                layer_norm=layer_norm,
            )

        # Override every `global_layer_interval`-th layer with full global attention.
        block_overrides: Dict[int, TransformerBlockConfig] = {}
        for layer_idx in range(n_layers):
            if layer_idx % global_layer_interval == (global_layer_interval - 1):
                global_block = TransformerBlockConfig(
                    name=TransformerBlockType.peri_norm,
                    sequence_mixer=AttentionConfig(
                        name=AttentionType.default,
                        n_heads=n_heads,
                        n_kv_heads=n_kv_heads,
                        head_dim=head_dim,
                        bias=False,
                        rope=RoPEConfig(name=RoPEType.default, theta=global_rope_theta),
                        gate=GateConfig(
                            granularity=GateGranularity.elementwise,
                            full_precision=True,
                        ),
                        qk_norm=layer_norm,
                        use_head_qk_norm=True,
                        backend=attn_backend,
                        dtype=dtype,
                    ),
                    feed_forward=feed_forward,
                    layer_norm=layer_norm,
                )
                block_overrides[layer_idx] = global_block

        return cls(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=n_layers,
            block=block,
            lm_head=LMHeadConfig(
                loss_implementation=lm_head_loss_impl,
                layer_norm=layer_norm,
                bias=False,
                dtype=dtype,
            ),
            dtype=dtype,
            block_overrides=block_overrides if block_overrides else None,
            embed_scale=math.sqrt(d_model),
            **kwargs,
        )

    @classmethod
    def v2_260M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 260M model config.

        259,551,360 total params
        195,326,080 non-embedding params
        """
        return cls.v2(
            d_model=640,
            hidden_size=640 * 8,
            n_layers=10,
            n_heads=8,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def v2_709M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 709M model config.

        708,903,680 total params
        606,143,232 non-embedding params
        """
        return cls.v2(
            d_model=1024,
            hidden_size=1024 * 8,
            n_layers=15,
            n_heads=16,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def v2_1p3B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 1.3B model config.

        1,253,157,120 total params
        1,124,706,560 non-embedding params
        """
        return cls.v2(
            d_model=1280,
            hidden_size=1280 * 8,
            n_layers=20,
            n_heads=16,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def v2_2B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 2.2B model config.

        2,156,558,080 total params
        2,002,417,408 non-embedding params
        """
        return cls.v2(
            d_model=1536,
            hidden_size=1536 * 8,
            n_layers=25,
            n_heads=24,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def v2_4B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 4.3B model config.

        4,312,000,000 total params
        4,106,479,104 non-embedding params
        """
        return cls.v2(
            d_model=2048,
            hidden_size=2048 * 8,
            n_layers=30,
            n_heads=32,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def v2_8B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        An 8B model config.

        8,588,259,840 total params
        8,331,358,720 non-embedding params
        """
        return cls.v2(
            d_model=2560,
            hidden_size=2560 * 8,
            n_layers=40,
            n_heads=40,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def v2_15B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 15B model config.

        15,087,541,760 total params
        14,779,260,416 non-embedding params
        """
        return cls.v2(
            d_model=3072,
            hidden_size=3072 * 8,
            n_layers=50,
            n_heads=48,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def v2_34B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 34B model config.

        34,084,000,000 total params
        33,672,958,208 non-embedding params
        """
        return cls.v2(
            d_model=4096,
            hidden_size=4096 * 8,
            n_layers=65,
            n_heads=64,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def v2_65B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 65B model config.

        64,782,689,280 total params
        64,268,887,040 non-embedding params
        """
        return cls.v2(
            d_model=5120,
            hidden_size=5120 * 8,
            n_layers=80,
            n_heads=80,
            vocab_size=vocab_size,
            **kwargs,
        )


@dataclass
class _ModelSizeSettings:
    """Training settings for a specific model size."""

    size: str
    num_nodes: int
    batch_size_round_nearest: int
    activation_memory_budget: float


class GemmaLikeOlmoV2(StrEnum):
    GL_260M = "260M"
    GL_709M = "709M"
    GL_1p3B = "1.3B"
    GL_2B = "2B"
    GL_4B = "4B"
    GL_8B = "8B"
    GL_15B = "15B"
    GL_34B = "34B"

    def get_settings(self, vocab_size: int, use_gdn: bool = False) -> Tuple[TransformerConfig, _ModelSizeSettings]:
        """Get the model config and all settings for this model size."""
        # Mapping: (size, num_nodes, round_nearest, activation_memory_budget)
        settings_map = {
            GemmaLikeOlmoV2.GL_260M: _ModelSizeSettings("260M", 1, 16, 1.0),
            GemmaLikeOlmoV2.GL_709M: _ModelSizeSettings("709M", 2, 16, 1.0),
            GemmaLikeOlmoV2.GL_1p3B: _ModelSizeSettings("1p3B", 3, 16, 1.0),
            GemmaLikeOlmoV2.GL_2B: _ModelSizeSettings("2B", 8, 16, 1.0),
            GemmaLikeOlmoV2.GL_4B: _ModelSizeSettings("4B", 9, 32, 1.0),
            GemmaLikeOlmoV2.GL_8B: _ModelSizeSettings("8B", 14, 64, 0.9),
            GemmaLikeOlmoV2.GL_15B: _ModelSizeSettings(
                "15B", 16, 128, 0.4
            ),  # Support up to 16 hosts, bsz8 per host
            GemmaLikeOlmoV2.GL_34B: _ModelSizeSettings(
                "34B", 16, 16, 0.1
            ),  # Currently does not work, OOMs!!!
        }
        if self not in settings_map:
            raise ValueError(
                f"Model not in list! Valid models: {[m.name for m in GemmaLikeOlmoV2]}\n\n"
            )

        settings = settings_map[self]
        config_method = getattr(GemmaLikeTransformerConfig, f"v2_{settings.size}")
        model_config = config_method(vocab_size, use_gdn=use_gdn)
        return model_config, settings


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
    parser.add_argument("--use-gdn", action="store_true", default=False)
    parser.add_argument(
        "--data-mix",
        type=DataMix,
        choices=list(DataMix),
        default=str(DataMix.OLMo_mix_0925),
    )

    # Extract argument names from parser (both value-based and boolean flags)
    arg_prefixes: List[str] = []
    boolean_flags: List[str] = []
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


def parse_model_size(run_name: str) -> GemmaLikeOlmoV2:
    """
    Parse model size from run name.
    The run name must contain one of the enum values (e.g., "260M", "1.3B", "8B").
    Examples: "260m", "gl-v2-260m", "1.3b", "1p3b" (normalized to "1.3b").
    """
    normalized = run_name.lower().strip().replace("1p3b", "1.3b").replace("1p3", "1.3")

    # Sort by value length descending so longer matches are tried first,
    # e.g. "32b" is matched before "2b", "14b" before "4b".
    for size in sorted(GemmaLikeOlmoV2, key=lambda s: len(s.value), reverse=True):
        if size.value.lower() in normalized:
            return size

    raise ValueError(
        f"Could not parse model size from run name '{run_name}'. "
        f"Valid sizes: {[s.value for s in GemmaLikeOlmoV2]}. "
        f"Examples: '260m', 'gl-v1-260m', '1.3b'"
    )


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    """
    Build experiment config for GL-OLMo v2.

    Model size can be specified as just the size (e.g., "260m") or with prefix
    (e.g., "gl-v2-260m"). The model size is parsed from the run name.

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
        --chinchilla-multiple=1                            Multiply Chinchilla training tokens
        --no-beaker-launch                                 Skip setting beaker launch config

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
    use_gdn = custom_args.use_gdn

    sequence_length = DEFAULT_SEQUENCE_LENGTH
    root_dir = custom_args.root_dir or get_root_dir(cli_context.cluster)
    work_dir = custom_args.work_dir or get_work_dir(root_dir)
    save_folder = custom_args.save_folder or f"{root_dir}/checkpoints/{cli_context.run_name}"

    print(f"mix_base_dir (dataset location): {mix_base_dir}")
    print(f"root_dir (checkpoint location): {root_dir}")
    print(f"work_dir (local path for temp files): {work_dir}")
    print(f"save_folder (checkpoint location): {save_folder}")

    tokenizer_config = TokenizerConfig.dolma2()
    model_config, model_size_settings = model.get_settings(tokenizer_config.padded_vocab_size(), use_gdn=use_gdn)

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

    beaker_launch_config: Optional[BeakerLaunchConfig] = None
    if not no_beaker_launch:
        beaker_launch_config = build_launch_config(
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
                project="gl-olmo-v1",
                workspace="oe-t-ladder",
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
                project="oe-t-ladder",
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
                eval_interval=2_500,
            ),
        )
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=sorted(TASK_GROUPS["fast"]),
                tokenizer=tokenizer_config,
                eval_on_finish=True,
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
            python src/scripts/train/ladder/gemma_like_ladder.py dry_run gl-v2-260m ai2/jupiter

        Start a local training run with torchrun:
            torchrun --nproc-per-node=8 src/scripts/train/ladder/gemma_like_ladder.py train gl-v2-260m ai2/jupiter

        Launch a training run on Beaker (Jupiter cluster):
            python src/scripts/train/ladder/gemma_like_ladder.py launch gl-v2-260m ai2/jupiter

        Launch with custom hyperparameters using multipliers:
            python src/scripts/train/ladder/gemma_like_ladder.py launch gl-v2-260m ai2/jupiter \
                --lr-multiplier=2.0 \
                --batch-multiplier=0.5
        Override specific config values:
            python src/scripts/train/ladder/gemma_like_ladder.py launch gl-v2-8b ai2/jupiter \
                --launch.num_nodes=2 \
                --train_module.scheduler.warmup=5000000

        Enable logging callbacks:
            python src/scripts/train/ladder/gemma_like_ladder.py launch gl-v2-260m ai2/jupiter \
                --trainer.callbacks.wandb.enabled=true \
                --trainer.callbacks.comet.enabled=true

        Override config without setting launch (uses default node count):
            python src/scripts/train/ladder/gemma_like_ladder.py launch gl-v2-260m ai2/jupiter \
                --train_module.optim.lr=0.001 \
                --data_loader.global_batch_size=1000 \
                --train_module.scheduler.warmup=1000000
    """
    main(config_builder=build_experiment_config)
