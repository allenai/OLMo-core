"""
OLMo hybrid small suite — 275M, 810M, and 1.4B Cx100 training runs.

Architecture:
  - Gemma3-like hybrid: 4 GDN layers + 1 global attention layer (repeating)
  - GDN with expand_v=2 (double value head dimension)
  - NoPE: no positional embeddings on global attention layers
  - RMS embedding norm

Hyperparameters:
  - LR: tuned per model size
  - BSZ: set per model size (derived from Li 2025 step law, rounded to nearest num_gpus)
  - Scheduler: cosine with 2000-step warmup and 2000-step linear decay
  - Duration: 100x Chinchilla (20 * N * 100 tokens per non-embedding param)

Model sizes:
  275M:  275,493,760 total /  211,268,480 non-embedding params, D=422B tokens  (4 nodes)
  810M:  810,354,816 total /  707,594,368 non-embedding params, D=1.4T tokens  (8 nodes)
  1.4B: 1,422,110,720 total / 1,293,660,160 non-embedding params, D=2.6T tokens (32 nodes)

Usage:
  # Dry run (print config without launching):
  python src/scripts/train/OLMo_hybrid/OLMo-hybrid-small-suite.py dry_run \
      hybrid-small-275M ai2/jupiter

  # Launch on Beaker (model size is parsed from run name):
  python src/scripts/train/OLMo_hybrid/OLMo-hybrid-small-suite.py launch \
      hybrid-small-1.4B ai2/jupiter
"""

import math
from datetime import datetime
from functools import partial
from typing import Dict

from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.experiment import (
    CommonComponents,
    DataComponents,
    build_config,
    main,
)
from olmo_core.nn.attention import (
    AttentionBackendName,
    AttentionConfig,
    AttentionType,
    GateConfig,
    GatedDeltaNetConfig,
    GateGranularity,
)
from olmo_core.nn.feed_forward import ActivationFunction, FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig, LMLossImplementation
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
)
from olmo_core.optim import (
    CosWithWarmupAndLinearDecay,
    OptimGroupOverride,
    SchedulerUnits,
    SkipStepAdamWConfig,
)
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    SpeedMonitorCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

SEQUENCE_LENGTH = 8192
CHINCHILLA_MULTIPLE = 100
DATA_MIX = DataMix.OLMo_mix_0925

# Per-model-size settings.
MODEL_CONFIGS = {
    "275m": dict(
        # 275,493,760 total params / 211,268,480 non-embedding params
        d_model=640,
        hidden_size=640 * 8,
        n_layers=10,
        n_heads=8,
        num_nodes=4,
        global_batch_size=2_621_440,
        rank_microbatch_size=5 * SEQUENCE_LENGTH,
        lr=0.008,
    ),
    "810m": dict(
        # 810,354,816 total params / 707,594,368 non-embedding params
        d_model=1024,
        hidden_size=1024 * 8,
        n_layers=15,
        n_heads=16,
        num_nodes=8,
        global_batch_size=5_242_880,
        rank_microbatch_size=2 * SEQUENCE_LENGTH,
        lr=0.002,
    ),
    "1.4b": dict(
        # 1,422,110,720 total params / 1,293,660,160 non-embedding params
        d_model=1280,
        hidden_size=1280 * 8,
        n_layers=20,
        n_heads=16,
        num_nodes=32,
        global_batch_size=8_388_608,
        rank_microbatch_size=2 * SEQUENCE_LENGTH,
        lr=0.0028,  # TODO: update after sweep completes
    ),
}


def build_model_config(common: CommonComponents, model_size: str) -> TransformerConfig:
    cfg = MODEL_CONFIGS[model_size]

    d_model = cfg["d_model"]
    hidden_size = cfg["hidden_size"]
    n_layers = cfg["n_layers"]
    n_heads = cfg["n_heads"]

    n_kv_heads = 8
    head_dim = 128
    global_layer_interval = 5
    layer_norm_eps = 1e-6
    dtype = DType.float32
    expand_v = 2.0

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

    # Default block: GDN (replaces sliding window attention layers)
    block = TransformerBlockConfig(
        name=TransformerBlockType.peri_norm,
        sequence_mixer=GatedDeltaNetConfig(
            n_heads=n_heads,
            n_v_heads=n_heads,
            head_dim=head_dim,
            expand_v=expand_v,
            dtype=dtype,
        ),
        feed_forward=feed_forward,
        layer_norm=layer_norm,
    )

    # Override every global_layer_interval-th layer with full global attention.
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
                    rope=None,
                    gate=GateConfig(
                        granularity=GateGranularity.elementwise,
                        full_precision=True,
                    ),
                    qk_norm=layer_norm,
                    use_head_qk_norm=True,
                    backend=AttentionBackendName.flash_3,
                    dtype=dtype,
                ),
                feed_forward=feed_forward,
                layer_norm=layer_norm,
            )
            block_overrides[layer_idx] = global_block

    model_config = TransformerConfig(
        d_model=d_model,
        vocab_size=common.tokenizer.padded_vocab_size(),
        n_layers=n_layers,
        block=block,
        lm_head=LMHeadConfig(
            loss_implementation=LMLossImplementation.default,
            layer_norm=layer_norm,
            bias=False,
            dtype=dtype,
        ),
        dtype=dtype,
        block_overrides=block_overrides if block_overrides else None,
        embed_scale=math.sqrt(d_model),
        embedding_norm=LayerNormConfig(
            name=LayerNormType.rms,
            eps=1e-6,
            bias=False,
        ),
    )

    return model_config


def build_train_module_config(common: CommonComponents, model_size: str) -> TransformerTrainModuleConfig:
    cfg = MODEL_CONFIGS[model_size]

    return TransformerTrainModuleConfig(
        rank_microbatch_size=cfg["rank_microbatch_size"],
        max_sequence_length=SEQUENCE_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=cfg["lr"],
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=CosWithWarmupAndLinearDecay(
            units=SchedulerUnits.steps,
            warmup=2000,
            decay=2000,
            decay_fraction=None,
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget,
            activation_memory_budget=1.0,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )


def build_data_components(
    common: CommonComponents,
) -> DataComponents:
    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        mix=DATA_MIX,
        tokenizer=common.tokenizer,
        mix_base_dir="gs://ai2-llm",
        sequence_length=common.max_sequence_length,
        max_target_sequence_length=max(8192, common.max_sequence_length),
        work_dir=common.work_dir,
        instance_filter_config=InstanceFilterConfig(),
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=common.global_batch_size, seed=34521, num_workers=8
    )

    return DataComponents(dataset=dataset_config, data_loader=data_loader_config)


def build_trainer_config(common: CommonComponents, model_size: str) -> TrainerConfig:
    cancel_check_interval = 25

    assert common.launch is not None
    assert len(common.launch.clusters) == 1
    cluster = common.launch.clusters[0]

    # Compute training duration
    model_config = build_model_config(common, model_size)
    model_params = model_config.num_non_embedding_params
    train_duration = Duration.chinchilla_tokens(
        CHINCHILLA_MULTIPLE, model_params=model_params
    )

    run_name = f"{common.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"

    return (
        TrainerConfig(
            save_folder=common.save_folder,
            work_dir=common.work_dir,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=cancel_check_interval,
            max_duration=train_duration,
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=None,
                save_async=True,
            ),
        )
        .with_callback("speed_monitor", SpeedMonitorCallback())
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                group=common.run_name,
                project="hybrid-small-suite",
                entity="ai2-llm",
                cancel_check_interval=cancel_check_interval,
                enabled=True,
            ),
        )
        .with_recommended_evals(common.tokenizer, SEQUENCE_LENGTH, cluster, task_set="fast")
    )


if __name__ == "__main__":
    import sys

    # Parse model size from the run name (2nd arg), which is forwarded to Beaker as-is.
    # e.g. "hybrid-small-275M" → "275m", "hybrid-small-1.4B" → "1.4b"
    if len(sys.argv) < 3:
        raise SystemExit(
            f"Usage: {sys.argv[0]} <subcmd> <run_name> <cluster> [overrides...]\n"
            f"Run name must contain a model size: {list(MODEL_CONFIGS.keys())}"
        )

    run_name = sys.argv[2].lower()
    model_size = None
    # Try longest keys first so "1.4b" matches before partial overlaps.
    for key in sorted(MODEL_CONFIGS.keys(), key=len, reverse=True):
        if key in run_name:
            model_size = key
            break

    if model_size is None:
        raise SystemExit(
            f"Error: could not parse model size from run name '{sys.argv[2]}'. "
            f"Run name must contain one of: {list(MODEL_CONFIGS.keys())}"
        )

    cfg = MODEL_CONFIGS[model_size]

    config_builder = partial(
        build_config,
        global_batch_size=cfg["global_batch_size"],
        max_sequence_length=SEQUENCE_LENGTH,
        num_nodes=cfg["num_nodes"],
        data_config_builder=build_data_components,
        model_config_builder=partial(build_model_config, model_size=model_size),
        train_module_config_builder=partial(build_train_module_config, model_size=model_size),
        trainer_config_builder=partial(build_trainer_config, model_size=model_size),
        include_default_evals=False,
        beaker_workspace="ai2/linear-rnns",
        num_execution_units=1,
    )
    main(config_builder=config_builder)
