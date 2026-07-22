"""Olmo3Moe builders and correctness-first OLMoDDP/HF weight interchange."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import PretrainedConfig

from olmo_core.config import DType
from olmo_core.distributed.utils import get_local_tensor
from olmo_core.nn.attention import (
    AttentionBackendName,
    AttentionConfig,
    AttentionType,
    SlidingWindowAttentionConfig,
)
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
from olmo_core.nn.hf.convert import convert_state_from_hf, convert_state_to_hf
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig, LMLossImplementation
from olmo_core.nn.moe import MoELoadBalancingLossGranularity, MoERouterGatingFunction
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
    router_aux_loss_weight: float | None = None,
    router_z_loss_weight: float | None = None,
    init_seed: int = 2026,
) -> OLMoDDPModelConfig:
    """Build the stage-one OLMoDDP model for an all-MoE Olmo3Moe checkpoint."""
    config = _as_mapping(hf_config)
    if config.get("model_type") != "olmo3moe":
        raise ValueError(f"Expected model_type='olmo3moe', got {config.get('model_type')!r}.")
    if config.get("dense_layers_indices"):
        raise NotImplementedError(
            "Stage-one OLMoDDP Olmo3Moe loading requires dense_layers_indices=[]; "
            "dense Olmo3Moe layers use a different native parameter layout."
        )
    if config.get("use_peri_ln", False):
        raise NotImplementedError("Olmo3Moe peri-LN is not supported by this factory.")
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
    ep = ExpertParallelConfig(path=ExpertParallelPath(ep_path))
    ep.validate()
    common = dict(
        name=TransformerBlockType.moe_fused_v2,
        use_pre_norm=False,
        use_peri_norm=False,
        ep=ep,
        layer_norm=layer_norm,
        routed_experts=routed_experts,
        routed_experts_router=routed_router,
        shared_experts=shared_experts,
        shared_experts_router=None,
    )

    def make_block(sliding: bool) -> OLMoDDPTransformerBlockConfig:
        window = int(config["sliding_window"]) - 1
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
            **{key: deepcopy(value) for key, value in common.items()},
        )

    blocks = {
        OLMO3_FULL_ATTENTION: make_block(False),
        OLMO3_SLIDING_ATTENTION: make_block(True),
    }
    block: TransformerBlockConfig | dict[str, TransformerBlockConfig]
    block_pattern: list[str] | None
    if len(set(layer_types)) == 1:
        block = blocks[layer_types[0]]
        block_pattern = None
    else:
        block = dict(blocks)
        block_pattern = list(layer_types)

    model_config = OLMoDDPModelConfig(
        init_seed=init_seed,
        init_std=float(config.get("initializer_range", 0.02)),
        d_model=d_model,
        vocab_size=int(config["vocab_size"]),
        n_layers=n_layers,
        block=block,
        block_pattern=block_pattern,
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


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    wrapped = getattr(model, "module", None)
    return wrapped if isinstance(wrapped, torch.nn.Module) else model


def load_olmo3_moe_hf_state(
    model: torch.nn.Module, hf_config: PretrainedConfig, hf_state: Mapping[str, torch.Tensor]
) -> None:
    """Load a full HF state into an unsharded or EP-sharded OLMoDDP model."""
    model = _unwrap_model(model)
    native_state = convert_state_from_hf(hf_config, dict(hf_state), model_type="olmo3moe")
    parameters = dict(model.named_parameters())
    missing = set(parameters) - set(native_state)
    if missing:
        raise RuntimeError(f"Converted Olmo3Moe state is missing parameters: {sorted(missing)}")
    unexpected = set(native_state) - set(parameters)
    if unexpected:
        raise RuntimeError(
            f"Converted Olmo3Moe state has unexpected parameters: {sorted(unexpected)}"
        )

    with torch.no_grad():
        for name, target in parameters.items():
            source = native_state[name]
            owner_name = name.rsplit(".", 1)[0]
            owner = model.get_submodule(owner_name)
            if getattr(owner, "_ep_sharded", False):
                local_experts = int(owner.num_local_experts)
                start = int(owner.ep_rank) * local_experts
                source = source[start : start + local_experts]
            if tuple(source.shape) != tuple(target.shape):
                raise RuntimeError(
                    f"Shape mismatch for {name}: converted={tuple(source.shape)}, "
                    f"model={tuple(target.shape)}"
                )
            target.copy_(source.to(device=target.device, dtype=target.dtype))


def gather_olmo3_moe_hf_state(
    model: torch.nn.Module, hf_config: PretrainedConfig
) -> dict[str, torch.Tensor]:
    """Collect an EP-sharded OLMoDDP model and return a full HF state on every rank."""
    model = _unwrap_model(model)
    native_state: dict[str, torch.Tensor] = {}
    for name, value in model.state_dict().items():
        local = get_local_tensor(value) if isinstance(value, DTensor) else value
        owner_name = name.rsplit(".", 1)[0]
        owner = model.get_submodule(owner_name)
        if getattr(owner, "_ep_sharded", False):
            group = owner.ep_mesh["ep_mp"].get_group()
            gathered = [torch.empty_like(local) for _ in range(dist.get_world_size(group))]
            dist.all_gather(gathered, local.contiguous(), group=group)
            local = torch.cat(gathered, dim=0)
        native_state[name] = local
    return convert_state_to_hf(hf_config, native_state)
