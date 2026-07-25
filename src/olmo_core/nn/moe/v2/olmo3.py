"""Olmo3Moe builders and correctness-first OLMoDDP/HF weight interchange."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import torch
from transformers import PretrainedConfig

from olmo_core.config import DType
from olmo_core.nn.attention import (
    AttentionBackendName,
    AttentionConfig,
    AttentionType,
    SlidingWindowAttentionConfig,
)
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig, LMLossImplementation
from olmo_core.nn.moe import MoELoadBalancingLossGranularity, MoERouterGatingFunction
from olmo_core.nn.moe.v2.checkpoint import (
    gather_olmo_ddp_hf_state,
    load_olmo_ddp_hf_state,
)
from olmo_core.nn.moe.v2.ep_config import ExpertParallelConfig, ExpertParallelPath
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType
from olmo_core.nn.transformer import TransformerBlockType, TransformerType
from olmo_core.nn.transformer.config import OLMoDDPModelConfig, TransformerBlockConfig

OLMO3_FULL_ATTENTION = "full_attention"
OLMO3_SLIDING_ATTENTION = "sliding_attention"


def _as_mapping(config: PretrainedConfig | Mapping[str, Any]) -> Mapping[str, Any]:
    return config.to_dict() if isinstance(config, PretrainedConfig) else config


def build_olmo3_moe_config_from_hf_config(
    hf_config: PretrainedConfig | Mapping[str, Any],
    *,
    dtype: DType = DType.bfloat16,
    attention_backend: AttentionBackendName = AttentionBackendName.flash_4,
    ep_path: ExpertParallelPath | str = ExpertParallelPath.sync_1d,
    ep_capacity_factor: float = 1.25,
    router_aux_loss_weight: float | None = None,
    router_z_loss_weight: float | None = None,
    init_seed: int = 2026,
) -> OLMoDDPModelConfig:
    """Build an OLMoDDP model from a supported Olmo3Moe checkpoint config."""
    config = _as_mapping(hf_config)
    if config.get("model_type") != "olmo3moe":
        raise ValueError(f"Expected model_type='olmo3moe', got {config.get('model_type')!r}.")
    rope_parameters = config.get("rope_parameters") or config.get("rope_scaling") or {}
    if rope_parameters and rope_parameters.get("rope_type", "default") != "default":
        raise NotImplementedError("Scaled RoPE is not supported by this stage-one factory.")
    if config.get("attention_bias", False):
        raise NotImplementedError("Biased Olmo3Moe attention is not supported.")
    if config.get("hidden_act", "silu") != "silu":
        raise NotImplementedError("Only SwiGLU Olmo3Moe experts are supported.")
    if not config.get("use_head_qk_norm", False):
        raise NotImplementedError("Olmo3Moe conversion requires head-wise QK norm.")

    n_layers = int(config["num_hidden_layers"])
    dense_layers = {int(idx) for idx in config.get("dense_layers_indices") or ()}
    invalid_dense_layers = sorted(idx for idx in dense_layers if idx < 0 or idx >= n_layers)
    if invalid_dense_layers:
        raise ValueError(
            f"dense_layers_indices must be in [0, {n_layers}), got {invalid_dense_layers}."
        )
    dense_hidden = config.get("dense_mlp_intermediate_size")
    if dense_layers and dense_hidden is None:
        raise ValueError(
            "dense_mlp_intermediate_size must be set when dense_layers_indices is non-empty."
        )
    layer_types = tuple(config.get("layer_types") or (OLMO3_FULL_ATTENTION,) * n_layers)
    if len(layer_types) != n_layers:
        raise ValueError(f"Expected {n_layers} layer_types, got {len(layer_types)}.")
    unsupported = set(layer_types) - {OLMO3_FULL_ATTENTION, OLMO3_SLIDING_ATTENTION}
    if unsupported:
        raise ValueError(f"Unsupported Olmo3Moe layer types: {sorted(unsupported)}")

    d_model = int(config["hidden_size"])
    num_experts = int(config["n_routed_experts"])
    layer_norm = LayerNormConfig(
        name=LayerNormType.rms,
        eps=float(config["rms_norm_eps"]),
        bias=False,
        dtype=dtype,
    )
    routed_experts = RoutedExpertsConfig(
        d_model=d_model,
        hidden_size=int(config["moe_intermediate_size"]),
        num_experts=num_experts,
        bias=False,
        dtype=dtype,
    )
    routed_router = MoERouterConfigV2(
        d_model=d_model,
        num_experts=num_experts,
        top_k=int(config["num_experts_per_tok"]),
        gating_function=MoERouterGatingFunction(config.get("gating_function", "softmax")),
        normalize_expert_weights=config.get("normalize_expert_weights"),
        restore_weight_scale=bool(config.get("restore_weight_scale", False)),
        original_top_k=config.get("original_num_experts_per_tok"),
        lb_loss_weight=router_aux_loss_weight,
        z_loss_weight=router_z_loss_weight,
        lb_loss_granularity=MoELoadBalancingLossGranularity.instance,
        dtype=dtype,
    )
    shared_hidden = config.get("shared_expert_intermediate_size")
    shared_experts = (
        None
        if shared_hidden is None
        else SharedExpertsConfig(
            d_model=d_model,
            hidden_size=int(shared_hidden),
            num_experts=1,
            bias=False,
            dtype=dtype,
        )
    )
    ep = ExpertParallelConfig(
        path=ExpertParallelPath(ep_path),
        capacity_factor=ep_capacity_factor,
    )
    ep.validate()
    common = dict(
        name=TransformerBlockType.moe_fused_v2,
        use_pre_norm=False,
        use_peri_norm=bool(config.get("use_peri_ln", False)),
        layer_norm=layer_norm,
        shared_experts_router=None,
    )

    def make_block(sliding: bool, *, dense: bool) -> OLMoDDPTransformerBlockConfig:
        window = int(config["sliding_window"]) - 1
        block_shared_experts = (
            SharedExpertsConfig(
                d_model=d_model,
                hidden_size=int(dense_hidden),
                num_experts=1,
                bias=False,
                dtype=dtype,
            )
            if dense
            else shared_experts
        )
        return OLMoDDPTransformerBlockConfig(
            sequence_mixer=AttentionConfig(
                name=AttentionType.default,
                n_heads=int(config["num_attention_heads"]),
                n_kv_heads=int(config["num_key_value_heads"]),
                head_dim=int(config["head_dim"]),
                bias=False,
                dropout=float(config.get("attention_dropout", 0.0)),
                rope=RoPEConfig(
                    name=RoPEType.default,
                    theta=int(config.get("rope_theta", 10_000)),
                    full_precision=True,
                ),
                qk_norm=layer_norm,
                use_head_qk_norm=True,
                backend=attention_backend,
                dtype=dtype,
                sliding_window=(
                    SlidingWindowAttentionConfig(
                        pattern=[window],
                        force_full_attention_on_first_layer=False,
                        force_full_attention_on_last_layer=False,
                    )
                    if sliding
                    else None
                ),
            ),
            ep=None if dense else ep,
            routed_experts=None if dense else routed_experts,
            routed_experts_router=None if dense else routed_router,
            shared_experts=block_shared_experts,
            **{key: deepcopy(value) for key, value in common.items()},
        )

    def block_name(layer_type: str, *, dense: bool) -> str:
        return f"{layer_type}_dense" if dense else layer_type

    block_pattern = [
        block_name(layer_type, dense=layer_idx in dense_layers)
        for layer_idx, layer_type in enumerate(layer_types)
    ]
    blocks = {}
    for layer_idx, layer_type in enumerate(layer_types):
        dense = layer_idx in dense_layers
        name = block_name(layer_type, dense=dense)
        if name not in blocks:
            blocks[name] = make_block(layer_type == OLMO3_SLIDING_ATTENTION, dense=dense)

    block: TransformerBlockConfig | dict[str, TransformerBlockConfig]
    resolved_block_pattern: list[str] | None
    if len(blocks) == 1:
        block = next(iter(blocks.values()))
        resolved_block_pattern = None
    else:
        block = dict(blocks)
        resolved_block_pattern = block_pattern

    model_config = OLMoDDPModelConfig(
        init_seed=init_seed,
        init_std=float(config.get("initializer_range", 0.02)),
        d_model=d_model,
        vocab_size=int(config["vocab_size"]),
        n_layers=n_layers,
        block=block,
        block_pattern=resolved_block_pattern,
        embedding_norm=layer_norm if config.get("embed_norm", False) else None,
        embed_scale=float(config.get("embed_scale", 1.0)),
        tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
        lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
        name=TransformerType.moe_fused_v2,
        dtype=dtype,
        two_batch_overlap=False,
        recompute_each_block=False,
        recompute_all_blocks_by_chunk=False,
    )
    model_config.lm_head.loss_implementation = LMLossImplementation.default
    return model_config


def load_olmo3_moe_hf_state(
    model: torch.nn.Module, hf_config: PretrainedConfig, hf_state: Mapping[str, torch.Tensor]
) -> None:
    """Backward-compatible Olmo3Moe wrapper for :func:`load_olmo_ddp_hf_state`."""
    load_olmo_ddp_hf_state(model, hf_config, hf_state)


def gather_olmo3_moe_hf_state(
    model: torch.nn.Module, hf_config: PretrainedConfig
) -> dict[str, torch.Tensor]:
    """Backward-compatible Olmo3Moe wrapper for :func:`gather_olmo_ddp_hf_state`."""
    return gather_olmo_ddp_hf_state(model, hf_config)
