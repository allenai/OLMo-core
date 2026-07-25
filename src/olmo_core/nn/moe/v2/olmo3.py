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
from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType
from olmo_core.nn.transformer import TransformerBlockType, TransformerType
from olmo_core.nn.transformer.config import OLMoDDPModelConfig, TransformerBlockConfig

OLMO3_FULL_ATTENTION = "full_attention"
OLMO3_SLIDING_ATTENTION = "sliding_attention"


def build_olmo3_moe_hf_config_from_native_config(
    model_config: OLMoDDPModelConfig,
    *,
    max_position_embeddings: int,
    pad_token_id: int | None,
    bos_token_id: int | None,
    eos_token_id: int | list[int] | None,
) -> Olmo3MoeConfig:
    """Build an exact serving config from a supported native OLMoDDP Olmo3MoE config."""
    blocks = model_config.resolved_block_configs
    if not blocks:
        raise ValueError("An Olmo3Moe model must contain at least one transformer block.")
    if not all(isinstance(block, OLMoDDPTransformerBlockConfig) for block in blocks):
        raise NotImplementedError("Olmo3Moe export requires OLMoDDP transformer blocks.")

    typed_blocks = [block for block in blocks if isinstance(block, OLMoDDPTransformerBlockConfig)]
    moe_blocks = [block for block in typed_blocks if block.routed_experts is not None]
    dense_layers_indices = [
        idx for idx, block in enumerate(typed_blocks) if block.routed_experts is None
    ]
    if not moe_blocks:
        raise NotImplementedError("Olmo3Moe export requires at least one routed MoE block.")

    representative = moe_blocks[0]
    attention = representative.sequence_mixer
    router = representative.routed_experts_router
    routed_experts = representative.routed_experts
    if not isinstance(attention, AttentionConfig):
        raise NotImplementedError("Olmo3Moe export requires attention sequence mixers.")
    if router is None or routed_experts is None:
        raise NotImplementedError("Olmo3Moe blocks require routed experts and a router.")
    if representative.layer_norm is None:
        raise NotImplementedError("Olmo3Moe export requires RMS layer norms.")
    if attention.rope is None or attention.rope.scaling is not None:
        raise NotImplementedError("Olmo3Moe export requires unscaled RoPE.")
    if attention.bias:
        raise NotImplementedError("Olmo3Moe export does not support attention bias.")
    if not attention.use_head_qk_norm or attention.qk_norm is None:
        raise NotImplementedError("Olmo3Moe export requires head-wise QK norm.")
    if any(block.use_pre_norm for block in typed_blocks):
        raise NotImplementedError("Olmo3Moe export does not support pre-norm blocks.")
    use_peri_ln = representative.use_peri_norm
    if any(block.use_peri_norm != use_peri_ln for block in typed_blocks):
        raise ValueError("All Olmo3Moe blocks must use the same peri-norm setting.")

    def architecture_signature(block: OLMoDDPTransformerBlockConfig) -> tuple[Any, ...]:
        block_attention = block.sequence_mixer
        if not isinstance(block_attention, AttentionConfig):
            raise NotImplementedError("Olmo3Moe export requires attention sequence mixers.")
        return (
            block_attention.n_heads,
            block_attention.n_kv_heads,
            block_attention.head_dim,
            block_attention.bias,
            block_attention.dropout,
            block_attention.use_head_qk_norm,
            block_attention.rope.theta if block_attention.rope is not None else None,
            block.layer_norm.eps if block.layer_norm is not None else None,
        )

    expected_attention = architecture_signature(representative)
    if any(architecture_signature(block) != expected_attention for block in typed_blocks):
        raise ValueError("Olmo3Moe attention architecture must be consistent across layers.")

    routed_signature = (
        routed_experts.hidden_size,
        routed_experts.num_experts,
        routed_experts.bias,
        routed_experts.activation,
        router.num_experts,
        router.top_k,
        router.bias,
        router.gating_function,
        router.normalize_expert_weights,
        router.restore_weight_scale,
        router.original_top_k,
    )
    for block in moe_blocks[1:]:
        block_router = block.routed_experts_router
        block_experts = block.routed_experts
        if block_router is None or block_experts is None:
            raise ValueError("Every MoE block must have routed experts and a router.")
        if (
            block_experts.hidden_size,
            block_experts.num_experts,
            block_experts.bias,
            block_experts.activation,
            block_router.num_experts,
            block_router.top_k,
            block_router.bias,
            block_router.gating_function,
            block_router.normalize_expert_weights,
            block_router.restore_weight_scale,
            block_router.original_top_k,
        ) != routed_signature:
            raise ValueError("Routed expert architecture must be consistent across MoE layers.")

    if routed_experts.bias or router.bias:
        raise NotImplementedError("Olmo3Moe export does not support expert or router bias.")
    if routed_experts.activation.value != "swiglu":
        raise NotImplementedError("Olmo3Moe export only supports SwiGLU experts.")

    shared_hidden_sizes = {
        block.shared_experts.hidden_size
        for block in moe_blocks
        if block.shared_experts is not None
    }
    if any(block.shared_experts is None for block in moe_blocks) and shared_hidden_sizes:
        raise ValueError("Shared experts must be present in either every or no MoE layer.")
    if len(shared_hidden_sizes) > 1:
        raise ValueError("Shared expert width must be consistent across MoE layers.")
    if any(
        block.shared_experts is not None
        and (
            block.shared_experts.num_experts != 1
            or block.shared_experts.bias
            or block.shared_experts.activation.value != "swiglu"
        )
        for block in moe_blocks
    ):
        raise NotImplementedError(
            "Olmo3Moe export supports one bias-free SwiGLU shared expert per MoE layer."
        )

    dense_blocks = [block for block in typed_blocks if block.routed_experts is None]
    dense_hidden_sizes = {
        block.shared_experts.hidden_size
        for block in dense_blocks
        if block.shared_experts is not None
    }
    if dense_blocks and (
        len(dense_hidden_sizes) != 1
        or any(block.shared_experts is None for block in dense_blocks)
    ):
        raise ValueError("Dense Olmo3Moe layers must have one consistent shared-expert width.")
    if any(
        block.shared_experts is not None
        and (
            block.shared_experts.num_experts != 1
            or block.shared_experts.bias
            or block.shared_experts.activation.value != "swiglu"
        )
        for block in dense_blocks
    ):
        raise NotImplementedError(
            "Dense Olmo3Moe layers require one bias-free SwiGLU shared expert."
        )

    layer_types: list[str] = []
    window_sizes: set[int] = set()
    for layer_idx, block in enumerate(typed_blocks):
        block_attention = block.sequence_mixer
        assert isinstance(block_attention, AttentionConfig)
        sliding = block_attention.sliding_window
        if sliding is not None and sliding.should_use_swa(layer_idx, model_config.n_layers):
            layer_types.append(OLMO3_SLIDING_ATTENTION)
            window_sizes.add(sliding.get_window_size(layer_idx, model_config.n_layers))
        else:
            layer_types.append(OLMO3_FULL_ATTENTION)
    if len(window_sizes) > 1:
        raise ValueError(f"Olmo3Moe HF export supports one sliding window size, got {window_sizes}.")

    head_dim = attention.head_dim or model_config.d_model // attention.n_heads
    return Olmo3MoeConfig(
        vocab_size=model_config.vocab_size,
        hidden_size=model_config.d_model,
        attention_hidden_size=attention.n_heads * head_dim,
        head_dim=head_dim,
        dense_mlp_intermediate_size=(
            next(iter(dense_hidden_sizes)) if dense_hidden_sizes else None
        ),
        moe_intermediate_size=routed_experts.hidden_size,
        shared_expert_intermediate_size=(
            next(iter(shared_hidden_sizes)) if shared_hidden_sizes else None
        ),
        n_routed_experts=routed_experts.num_experts,
        num_experts_per_tok=router.top_k,
        original_num_experts_per_tok=router.original_top_k,
        num_hidden_layers=model_config.n_layers,
        num_attention_heads=attention.n_heads,
        num_key_value_heads=attention.n_kv_heads,
        hidden_act="silu",
        gating_function=str(router.gating_function),
        normalize_expert_weights=router.normalize_expert_weights,
        restore_weight_scale=router.restore_weight_scale,
        max_position_embeddings=max_position_embeddings,
        initializer_range=model_config.init_std,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        tie_word_embeddings=model_config.tie_word_embeddings,
        rope_theta=attention.rope.theta,
        attention_bias=False,
        attention_dropout=attention.dropout or 0.0,
        rms_norm_eps=representative.layer_norm.eps,
        sliding_window=(
            next(iter(window_sizes)) + 1 if window_sizes else max_position_embeddings
        ),
        use_head_qk_norm=True,
        layer_types=layer_types,
        dense_layers_indices=dense_layers_indices,
        embed_scale=(model_config.embed_scale if model_config.embed_scale is not None else 1.0),
        embed_norm=model_config.embedding_norm is not None,
        use_peri_ln=use_peri_ln,
    )


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
