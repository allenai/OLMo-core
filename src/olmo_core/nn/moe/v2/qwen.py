from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from olmo_core.config import DType
from olmo_core.nn.attention import (
    AttentionBackendName,
    AttentionConfig,
    AttentionType,
    GateConfig,
    GateGranularity,
)
from olmo_core.nn.attention.recurrent import GatedDeltaNetConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig, LMLossImplementation
from olmo_core.nn.moe import MoELoadBalancingLossGranularity, MoERouterGatingFunction
from olmo_core.nn.moe.v2.block import MoEFusedV2TransformerBlockConfig
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType
from olmo_core.nn.transformer import TransformerBlockType, TransformerType
from olmo_core.nn.transformer.config import MoEFusedV2TransformerConfig


QWEN3_MOE_LAYER_PATTERN = (
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
)

QWEN3_DENSE_MOE_LAYER_TYPE = "full_attention"


def _text_config(hf_config: Mapping[str, Any]) -> Mapping[str, Any]:
    return hf_config.get("text_config", hf_config)


def _num_experts(text_config: Mapping[str, Any]) -> int:
    for key in ("num_experts", "num_local_experts", "num_routed_experts", "n_routed_experts"):
        if key in text_config:
            return int(text_config[key])
    raise KeyError("Qwen MoE config is missing num_experts/num_local_experts")


def get_qwen3_moe_text_config_overrides(hf_config: Mapping[str, Any]) -> dict[str, Any]:
    text_config = _text_config(hf_config)
    head_dim = int(text_config["head_dim"])
    partial_rotary_factor = float(text_config.get("partial_rotary_factor", 1.0))
    common = {
        "vocab_size": text_config["vocab_size"],
        "d_model": text_config["hidden_size"],
        "n_layers": text_config["num_hidden_layers"],
        "num_attention_heads": text_config["num_attention_heads"],
        "num_key_value_heads": text_config["num_key_value_heads"],
        "attention_head_dim": head_dim,
        "attention_rotary_dim": int(head_dim * partial_rotary_factor),
        "rope_theta": text_config.get("rope_parameters", {}).get(
            "rope_theta", text_config.get("rope_theta", 10_000_000)
        ),
        "num_experts": _num_experts(text_config),
        "num_experts_per_tok": text_config["num_experts_per_tok"],
        "moe_intermediate_size": text_config["moe_intermediate_size"],
        "rms_norm_eps": text_config["rms_norm_eps"],
        "router_aux_loss_weight": text_config.get("router_aux_loss_coef", 0.001),
    }

    if "linear_num_key_heads" in text_config:
        common.update(
            {
                "linear_num_key_heads": text_config["linear_num_key_heads"],
                "linear_num_value_heads": text_config["linear_num_value_heads"],
                "linear_key_head_dim": text_config["linear_key_head_dim"],
                "linear_value_head_dim": text_config["linear_value_head_dim"],
                "linear_conv_kernel_dim": text_config["linear_conv_kernel_dim"],
                "shared_expert_intermediate_size": text_config["shared_expert_intermediate_size"],
                "layer_types": tuple(text_config["layer_types"]),
                "attention_output_gate": text_config.get("attn_output_gate", True),
                "norm_type": LayerNormType.qwen_rms,
            }
        )
    else:
        n_layers = int(text_config["num_hidden_layers"])
        common.update(
            {
                "shared_expert_intermediate_size": text_config.get("shared_expert_intermediate_size"),
                "layer_types": (QWEN3_DENSE_MOE_LAYER_TYPE,) * n_layers,
                "attention_output_gate": False,
                "norm_type": LayerNormType.rms,
            }
        )

    return common


def _require_linear_attention_kwargs(
    *,
    linear_num_key_heads: int | None,
    linear_num_value_heads: int | None,
    linear_key_head_dim: int | None,
    linear_value_head_dim: int | None,
    linear_conv_kernel_dim: int | None,
) -> tuple[int, int, int, int, int]:
    missing = [
        name
        for name, value in (
            ("linear_num_key_heads", linear_num_key_heads),
            ("linear_num_value_heads", linear_num_value_heads),
            ("linear_key_head_dim", linear_key_head_dim),
            ("linear_value_head_dim", linear_value_head_dim),
            ("linear_conv_kernel_dim", linear_conv_kernel_dim),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            "linear attention layer_types require Qwen3.5/3.6 linear attention "
            f"parameters: {', '.join(missing)}"
        )
    return (
        int(linear_num_key_heads),
        int(linear_num_value_heads),
        int(linear_key_head_dim),
        int(linear_value_head_dim),
        int(linear_conv_kernel_dim),
    )


def _resolve_block_config(
    blocks: dict[str, MoEFusedV2TransformerBlockConfig],
    layer_types: Sequence[str],
) -> tuple[MoEFusedV2TransformerBlockConfig | dict[str, MoEFusedV2TransformerBlockConfig], list[str] | None]:
    if len(set(layer_types)) == 1:
        return blocks[layer_types[0]], None
    return blocks, list(layer_types)


def _resolve_layer_types(layer_types: Sequence[str], n_layers: int) -> tuple[str, ...]:
    if n_layers <= 0:
        raise ValueError(f"n_layers must be positive, got {n_layers}")
    if len(layer_types) == n_layers:
        return tuple(layer_types)
    if len(layer_types) == 0:
        raise ValueError("layer_types must be non-empty")
    if len(layer_types) > n_layers:
        return tuple(layer_types[:n_layers])
    if n_layers % len(layer_types) != 0:
        raise ValueError(
            f"n_layers ({n_layers}) must be divisible by layer_types length ({len(layer_types)})"
        )
    repeats = n_layers // len(layer_types)
    return tuple(layer_types) * repeats


def _build_shared_expert_configs(
    *,
    d_model: int,
    shared_expert_intermediate_size: int | None,
    dtype: DType,
) -> tuple[SharedExpertsConfig | None, MoERouterConfigV2 | None]:
    if shared_expert_intermediate_size is None:
        return None, None
    shared_experts = SharedExpertsConfig(
        d_model=d_model,
        hidden_size=shared_expert_intermediate_size,
        num_experts=1,
        bias=False,
        dtype=dtype,
    )
    shared_router = MoERouterConfigV2(
        d_model=d_model,
        num_experts=1,
        top_k=1,
        gating_function=MoERouterGatingFunction.sigmoid,
        lb_loss_weight=None,
        z_loss_weight=None,
        lb_loss_granularity=MoELoadBalancingLossGranularity.instance,
        dtype=dtype,
    )
    return shared_experts, shared_router


def _make_full_attention_block(
    *,
    common_block_kwargs: dict[str, Any],
    num_attention_heads: int,
    num_key_value_heads: int,
    attention_head_dim: int,
    attention_rotary_dim: int,
    rope_theta: int | float,
    attention_output_gate: bool,
    attention_backend: AttentionBackendName,
    layer_norm: LayerNormConfig,
    dtype: DType,
) -> MoEFusedV2TransformerBlockConfig:
    return MoEFusedV2TransformerBlockConfig(
        sequence_mixer=AttentionConfig(
            name=AttentionType.default,
            n_heads=num_attention_heads,
            n_kv_heads=num_key_value_heads,
            head_dim=attention_head_dim,
            bias=False,
            gate=(
                GateConfig(granularity=GateGranularity.elementwise, full_precision=True)
                if attention_output_gate
                else None
            ),
            rope=RoPEConfig(
                name=RoPEType.default,
                theta=rope_theta,
                rotary_dim=attention_rotary_dim,
                full_precision=True,
            ),
            qk_norm=layer_norm,
            use_head_qk_norm=True,
            backend=attention_backend,
            dtype=dtype,
            d_attn=num_attention_heads * attention_head_dim,
        ),
        **{k: deepcopy(v) for k, v in common_block_kwargs.items()},
    )


def _make_linear_attention_block(
    *,
    common_block_kwargs: dict[str, Any],
    linear_num_key_heads: int,
    linear_num_value_heads: int,
    linear_key_head_dim: int,
    linear_value_head_dim: int,
    linear_conv_kernel_dim: int,
    rms_norm_eps: float,
    dtype: DType,
) -> MoEFusedV2TransformerBlockConfig:
    linear_expand_v = linear_value_head_dim / linear_key_head_dim
    return MoEFusedV2TransformerBlockConfig(
        sequence_mixer=GatedDeltaNetConfig(
            n_heads=linear_num_key_heads,
            n_v_heads=linear_num_value_heads,
            head_dim=linear_key_head_dim,
            expand_v=linear_expand_v,
            allow_neg_eigval=False,
            conv_size=linear_conv_kernel_dim,
            conv_bias=False,
            norm_eps=rms_norm_eps,
            dtype=dtype,
        ),
        **common_block_kwargs,
    )


def build_qwen3_moe_config(
    *,
    vocab_size: int = 248_320,
    d_model: int = 2048,
    n_layers: int = 40,
    num_attention_heads: int = 16,
    num_key_value_heads: int = 2,
    attention_head_dim: int = 256,
    attention_rotary_dim: int = 64,
    rope_theta: int = 10_000_000,
    linear_num_key_heads: int | None = 16,
    linear_num_value_heads: int | None = 32,
    linear_key_head_dim: int | None = 128,
    linear_value_head_dim: int | None = 128,
    linear_conv_kernel_dim: int | None = 4,
    num_experts: int = 256,
    num_experts_per_tok: int = 8,
    moe_intermediate_size: int = 512,
    shared_expert_intermediate_size: int | None = 512,
    rms_norm_eps: float = 1e-6,
    layer_types: Sequence[str] = QWEN3_MOE_LAYER_PATTERN,
    attention_output_gate: bool = True,
    norm_type: LayerNormType = LayerNormType.qwen_rms,
    dtype: DType = DType.bfloat16,
    init_seed: int = 2026,
    init_std: float = 0.02,
    attention_backend: AttentionBackendName = AttentionBackendName.flash_4,
    router_aux_loss_weight: float | None = 0.001,
    router_z_loss_weight: float | None = None,
    compile_friendly_recompute: bool = False,
    use_ep_no_sync: bool = False,
    use_rowwise_all_to_all: bool = False,
    ep_no_sync_capacity_factor: float = 1.25,
) -> MoEFusedV2TransformerConfig:
    layer_types = _resolve_layer_types(layer_types, n_layers)
    layer_norm = LayerNormConfig(
        name=norm_type,
        eps=rms_norm_eps,
        bias=False,
        dtype=dtype,
    )

    routed_experts = RoutedExpertsConfig(
        d_model=d_model,
        hidden_size=moe_intermediate_size,
        num_experts=num_experts,
        bias=False,
        dtype=dtype,
    )
    shared_experts, shared_router = _build_shared_expert_configs(
        d_model=d_model,
        shared_expert_intermediate_size=shared_expert_intermediate_size,
        dtype=dtype,
    )

    routed_router = MoERouterConfigV2(
        d_model=d_model,
        num_experts=num_experts,
        top_k=num_experts_per_tok,
        gating_function=MoERouterGatingFunction.softmax,
        normalize_expert_weights=1.0,
        lb_loss_weight=router_aux_loss_weight,
        z_loss_weight=router_z_loss_weight,
        lb_loss_granularity=MoELoadBalancingLossGranularity.instance,
        dtype=dtype,
    )
    common_block_kwargs = dict(
        name=TransformerBlockType.moe_fused_v2,
        use_pre_norm=True,
        use_peri_norm=False,
        ep_no_sync=use_ep_no_sync,
        ep_no_sync_use_rowwise_all_to_all=use_rowwise_all_to_all,
        ep_no_sync_capacity_factor=ep_no_sync_capacity_factor,
        checkpoint_attn=compile_friendly_recompute,
        checkpoint_permute_moe_unpermute=compile_friendly_recompute,
        checkpoint_second_unpermute=compile_friendly_recompute,
        attention_norm=layer_norm,
        feed_forward_norm=layer_norm,
        routed_experts=deepcopy(routed_experts),
        routed_experts_router=deepcopy(routed_router),
        shared_experts=deepcopy(shared_experts),
        shared_experts_router=deepcopy(shared_router),
    )

    blocks: dict[str, MoEFusedV2TransformerBlockConfig] = {}
    if QWEN3_DENSE_MOE_LAYER_TYPE in set(layer_types):
        blocks[QWEN3_DENSE_MOE_LAYER_TYPE] = _make_full_attention_block(
            common_block_kwargs=common_block_kwargs,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            attention_head_dim=attention_head_dim,
            attention_rotary_dim=attention_rotary_dim,
            rope_theta=rope_theta,
            attention_output_gate=attention_output_gate,
            attention_backend=attention_backend,
            layer_norm=layer_norm,
            dtype=dtype,
        )
    if "linear_attention" in set(layer_types):
        (
            linear_num_key_heads,
            linear_num_value_heads,
            linear_key_head_dim,
            linear_value_head_dim,
            linear_conv_kernel_dim,
        ) = _require_linear_attention_kwargs(
            linear_num_key_heads=linear_num_key_heads,
            linear_num_value_heads=linear_num_value_heads,
            linear_key_head_dim=linear_key_head_dim,
            linear_value_head_dim=linear_value_head_dim,
            linear_conv_kernel_dim=linear_conv_kernel_dim,
        )
        blocks["linear_attention"] = _make_linear_attention_block(
            common_block_kwargs=common_block_kwargs,
            linear_num_key_heads=linear_num_key_heads,
            linear_num_value_heads=linear_num_value_heads,
            linear_key_head_dim=linear_key_head_dim,
            linear_value_head_dim=linear_value_head_dim,
            linear_conv_kernel_dim=linear_conv_kernel_dim,
            rms_norm_eps=rms_norm_eps,
            dtype=dtype,
        )
    missing = set(layer_types) - set(blocks)
    if missing:
        raise ValueError(f"Unsupported Qwen MoE layer_types: {sorted(missing)}")
    block, block_pattern = _resolve_block_config(blocks, layer_types)

    config = MoEFusedV2TransformerConfig(
        init_seed=init_seed,
        init_std=init_std,
        d_model=d_model,
        vocab_size=vocab_size,
        n_layers=n_layers,
        block=block,
        block_pattern=block_pattern,
        embedding_norm=None,
        lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
        name=TransformerType.moe_fused_v2,
        dtype=dtype,
        two_batch_overlap=False,
        recompute_each_block=compile_friendly_recompute,
        recompute_all_blocks_by_chunk=False,
    )
    config.lm_head.loss_implementation = LMLossImplementation.default
    return config


def build_qwen3_moe_config_from_hf_config(
    hf_config: Mapping[str, Any],
    **overrides: Any,
) -> MoEFusedV2TransformerConfig:
    kwargs = get_qwen3_moe_text_config_overrides(hf_config)
    kwargs.update(overrides)
    return build_qwen3_moe_config(**kwargs)


def build_debug_qwen3_moe_config(
    *,
    vocab_size: int,
    n_layers: int = 4,
    d_model: int = 256,
    num_experts: int = 8,
    num_experts_per_tok: int = 2,
    moe_intermediate_size: int = 128,
    shared_expert_intermediate_size: int = 128,
    dtype: DType = DType.bfloat16,
    attention_backend: AttentionBackendName = AttentionBackendName.flash_4,
    **kwargs: Any,
) -> MoEFusedV2TransformerConfig:
    return build_qwen3_moe_config(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=1,
        attention_head_dim=64,
        attention_rotary_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_intermediate_size=moe_intermediate_size,
        shared_expert_intermediate_size=shared_expert_intermediate_size,
        dtype=dtype,
        attention_backend=attention_backend,
        layer_types=tuple(QWEN3_MOE_LAYER_PATTERN[:n_layers])
        if n_layers <= len(QWEN3_MOE_LAYER_PATTERN)
        else QWEN3_MOE_LAYER_PATTERN,
        **kwargs,
    )
