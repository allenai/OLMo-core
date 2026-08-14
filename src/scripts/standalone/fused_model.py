"""Build the smallest or largest OLMoE3 ladder model with production fused kernels.

Setup in an empty directory on a Linux CUDA machine::

    python -m venv .venv
    . .venv/bin/activate
    pip install 'ai2-olmo-core[fa4,fla] @ git+https://github.com/allenai/OLMo-core.git@f2cf93839a823b88955e94a851c808829c5201ba'
    python fused_model.py

PyTorch supplies its compatible Triton build; ``fa4`` installs FlashAttention 4 and
``fla`` installs flash-linear-attention. NVSHMEM is additionally required only when
running the configured multi-GPU rowwise expert-parallel path.

Unlike ``standalone_model.py``, this script deliberately uses OLMo-core's native
implementations:

* Kimi Delta Attention -> flash-linear-attention's fused KDA kernel
* full attention -> FlashAttention 4
* SwiGLU shared/routed experts -> OLMo-core fused MoE v2 kernels
* expert dispatch/combine -> the configured rowwise NVSHMEM EP path after EP setup

By default the script validates and reports the config without building modules.
Pass ``--device meta`` to construct shapes without parameter storage, or ``--device
cuda`` on appropriately sharded hardware. Module construction requires the production
FLA, FlashAttention 4, Triton, and (for expert parallelism) NVSHMEM environment.
Use ``--model-size 30m`` for the single-GPU smoke configuration; the default is
the 3.5B-active / 63B-stored target configuration.
"""

from __future__ import annotations

import argparse
import math
from copy import deepcopy
from dataclasses import dataclass

from olmo_core.config import DType
from olmo_core.nn.attention import (
    AttentionBackendName,
    AttentionConfig,
    AttentionType,
    GateConfig,
    GateGranularity,
    KimiDeltaAttention,
    KimiDeltaAttentionConfig,
)
from olmo_core.nn.ddp import (
    OLMoDDPTransformerBlock,
    OLMoDDPTransformerBlockConfig,
)
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.moe import (
    EmoRouterConfig,
    LatentMoEConfig,
    MoELoadBalancingLossGranularity,
    MoERouterGatingFunction,
)
from olmo_core.nn.moe.v2.emo_router import EmoRouterV2
from olmo_core.nn.moe.v2.ep_config import ExpertParallelConfig, ExpertParallelPath
from olmo_core.nn.moe.v2.fp8 import MoERowwiseFP8Config
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2, MoERouterV2
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig
from olmo_core.nn.transformer import (
    OLMoDDPModelConfig,
    Transformer,
    TransformerBlockType,
    TransformerType,
)

VOCAB_SIZE = 100_352
TOP_K = 16


@dataclass(frozen=True)
class Geometry:
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    expert_hidden_size: int
    num_routed_experts: int
    latent_dim: int
    expected_total_params: int
    expected_active_params: int

    @property
    def full_attention_layers(self) -> tuple[int, ...]:
        return tuple(range(4, self.n_layers, 5))


GEOMETRIES = {
    # Left unaligned: a fast plumbing/correctness check, not performance-representative.
    "30m": Geometry(128, 5, 1, 1, 128, 192, 32, 64, 32_323_588, 29_964_292),
    # head_dim/expert_hidden_size/latent_dim are all multiples of 256 for TPU MXU
    # alignment (tpu-optimizations-guide.md, Principle 1); see PR description for
    # the parameter-count trade-off this implies relative to the prior geometry.
    "3p5b": Geometry(1792, 30, 8, 4, 256, 1792, 512, 768, 62_864_102_080, 3_475_903_168),
}


@dataclass(frozen=True)
class FusedModelOptions:
    model_size: str = "3p5b"
    emo_enabled: bool = True
    global_load_balancing: bool = True
    device: str | None = None


@dataclass(frozen=True)
class ParameterCounts:
    total: int
    active: int
    embedding: int
    non_embedding: int
    active_non_embedding: int


def build_fused_config(options: FusedModelOptions) -> OLMoDDPModelConfig:
    """Build an exact ladder config without repository-local imports."""
    try:
        geometry = GEOMETRIES[options.model_size]
    except KeyError as exc:
        raise ValueError(f"model_size must be one of {tuple(GEOMETRIES)}") from exc
    norm = LayerNormConfig(
        name=LayerNormType.rms, eps=1e-6, bias=False, dtype=DType.float32
    )
    kda = KimiDeltaAttentionConfig(
        n_heads=geometry.n_heads,
        n_v_heads=geometry.n_heads,
        head_dim=geometry.head_dim,
        expand_v=2.0,
        allow_neg_eigval=True,
        conv_size=4,
        conv_bias=False,
        norm_eps=1e-5,
        dtype=DType.float32,
    )
    full_attention = AttentionConfig(
        name=AttentionType.default,
        n_heads=geometry.n_heads,
        n_kv_heads=geometry.n_kv_heads,
        head_dim=geometry.head_dim,
        bias=False,
        gate=GateConfig(granularity=GateGranularity.elementwise, full_precision=True),
        rope=None,
        qk_norm=deepcopy(norm),
        backend=AttentionBackendName.flash_4,
        dtype=DType.float32,
        use_head_qk_norm=True,
    )
    shared = SharedExpertsConfig(
        d_model=geometry.d_model,
        hidden_size=geometry.expert_hidden_size,
        num_experts=1,
        bias=False,
        dtype=DType.float32,
    )
    routed = RoutedExpertsConfig(
        d_model=geometry.latent_dim,
        hidden_size=geometry.expert_hidden_size,
        num_experts=geometry.num_routed_experts,
        bias=False,
        dtype=DType.float32,
        rowwise_fp8=MoERowwiseFP8Config(enabled=False),
    )
    router = MoERouterConfigV2(
        d_model=geometry.d_model,
        num_experts=geometry.num_routed_experts,
        top_k=TOP_K,
        bias=False,
        normalize_expert_weights=1.0,
        gating_function=MoERouterGatingFunction.softmax,
        dtype=DType.float32,
        lb_loss_weight=0.01,
        lb_loss_granularity=(
            MoELoadBalancingLossGranularity.local_batch
            if options.global_load_balancing
            else MoELoadBalancingLossGranularity.instance
        ),
        z_loss_weight=1e-5,
        restore_weight_scale=True,
        use_recompute_fp32_cast=False,
        global_load_balancing=options.global_load_balancing,
        emo=(
            EmoRouterConfig(
                eos_token_id=100_257,
                min_document_expert_pool=TOP_K,
                max_document_expert_pool=geometry.num_routed_experts,
                eval_document_expert_pool=geometry.num_routed_experts,
            )
            if options.emo_enabled
            else None
        ),
    )

    def block(sequence_mixer) -> OLMoDDPTransformerBlockConfig:
        return OLMoDDPTransformerBlockConfig(
            name=TransformerBlockType.moe_fused_v2,
            sequence_mixer=sequence_mixer,
            layer_norm=deepcopy(norm),
            shared_experts=deepcopy(shared),
            routed_experts=deepcopy(routed),
            routed_experts_router=deepcopy(router),
            latent_moe=LatentMoEConfig(
                latent_dim=geometry.latent_dim, up_proj_input_norm_enabled=False
            ),
            use_peri_norm=True,
            use_pre_norm=False,
            checkpoint_attn=False,
            checkpoint_permute_moe_unpermute=False,
            checkpoint_second_unpermute=False,
            ep=ExpertParallelConfig(path=ExpertParallelPath.rowwise_nvshmem),
            rowwise_fp8=MoERowwiseFP8Config(enabled=False),
        )

    dense_first = OLMoDDPTransformerBlockConfig(
        name=TransformerBlockType.moe_fused_v2,
        sequence_mixer=deepcopy(kda),
        layer_norm=deepcopy(norm),
        shared_experts=SharedExpertsConfig(
            d_model=geometry.d_model,
            hidden_size=8 * geometry.d_model,
            num_experts=1,
            bias=False,
            dtype=DType.float32,
        ),
        use_peri_norm=True,
        use_pre_norm=False,
        checkpoint_attn=False,
        checkpoint_permute_moe_unpermute=False,
        checkpoint_second_unpermute=False,
    )
    config = OLMoDDPModelConfig(
        name=TransformerType.moe_fused_v2,
        d_model=geometry.d_model,
        vocab_size=VOCAB_SIZE,
        n_layers=geometry.n_layers,
        block=block(kda),
        block_overrides={
            0: dense_first,
            **{
                idx: block(deepcopy(full_attention))
                for idx in geometry.full_attention_layers
            },
        },
        lm_head=LMHeadConfig(layer_norm=deepcopy(norm), bias=False, dtype=DType.float32),
        embedding_norm=deepcopy(norm),
        dtype=DType.float32,
        init_method="normal",
        init_seed=0,
        init_std=0.02,
        embed_scale=math.sqrt(geometry.d_model),
        tie_word_embeddings=False,
        two_batch_overlap=False,
        recompute_all_blocks_by_chunk=False,
        recompute_each_block=False,
    )
    config.validate()
    return config


def parameter_counts(config) -> ParameterCounts:
    """Use OLMo-core's own config properties, including top-k active experts."""
    embedding = config.d_model * config.vocab_size
    return ParameterCounts(
        total=config.num_params,
        active=config.num_active_params,
        embedding=embedding,
        non_embedding=config.num_non_embedding_params,
        active_non_embedding=config.num_active_non_embedding_params,
    )


def verify_fused_config(config, options: FusedModelOptions) -> None:
    """Fail early if the canonical ladder stops selecting the fused implementations."""
    geometry = GEOMETRIES[options.model_size]
    assert config.num_params == geometry.expected_total_params
    assert config.num_active_params == geometry.expected_active_params

    for layer_idx, block in enumerate(config.resolved_block_configs):
        if layer_idx in geometry.full_attention_layers:
            assert isinstance(block.sequence_mixer, AttentionConfig)
            assert block.sequence_mixer.backend == AttentionBackendName.flash_4
        else:
            assert isinstance(block.sequence_mixer, KimiDeltaAttentionConfig)
        if layer_idx == 0:
            assert block.routed_experts is None
            continue
        assert block.routed_experts is not None
        assert block.routed_experts_router is not None
        assert block.routed_experts_router.top_k == TOP_K
        assert block.routed_experts_router.global_load_balancing is options.global_load_balancing
        assert (block.routed_experts_router.emo is not None) is options.emo_enabled


def build_fused_model(options: FusedModelOptions) -> tuple[object, Transformer | None]:
    config = build_fused_config(options)
    verify_fused_config(config, options)
    model = config.build(init_device=options.device) if options.device is not None else None
    return config, model


def verify_fused_modules(model: Transformer, options: FusedModelOptions) -> None:
    """Verify that config resolution produced the intended runtime module classes."""
    geometry = GEOMETRIES[options.model_size]
    for layer_idx, block in enumerate(model.blocks):
        assert isinstance(block, OLMoDDPTransformerBlock)
        if layer_idx not in geometry.full_attention_layers:
            assert isinstance(block.attention, KimiDeltaAttention)
        if layer_idx > 0:
            expected_router = EmoRouterV2 if options.emo_enabled else MoERouterV2
            assert type(block.routed_experts_router) is expected_router


def print_parameter_counts(config) -> None:
    counts = parameter_counts(config)
    print(f"total params:                {counts.total:,}")
    print(f"active params:               {counts.active:,}")
    print(f"embedding params:            {counts.embedding:,}")
    print(f"non-embedding params:        {counts.non_embedding:,}")
    print(f"active non-embedding params: {counts.active_non_embedding:,}")


def parse_args() -> FusedModelOptions:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=tuple(GEOMETRIES), default="3p5b")
    parser.add_argument(
        "--device",
        help="Build fused runtime modules on this device (e.g. meta or cuda)",
    )
    parser.add_argument(
        "--emo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable EMo document-level expert pools",
    )
    parser.add_argument(
        "--global-load-balancing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Globally balance routed-expert assignments across DP ranks",
    )
    args = parser.parse_args()
    return FusedModelOptions(
        model_size=args.model_size,
        emo_enabled=args.emo,
        global_load_balancing=args.global_load_balancing,
        device=args.device,
    )


if __name__ == "__main__":
    options = parse_args()
    fused_config, fused_model = build_fused_model(options)
    if fused_model is not None:
        verify_fused_modules(fused_model, options)
    print_parameter_counts(fused_config)
    print(f"model size:                  {options.model_size}")
    print(f"EMo:                        {options.emo_enabled}")
    print(f"global load balancing:      {options.global_load_balancing}")
    print(f"initialization device:      {options.device or 'config only'}")
