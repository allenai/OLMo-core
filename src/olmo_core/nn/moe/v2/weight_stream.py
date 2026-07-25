"""Low-memory HF weight streaming for expert-parallel OLMoDDP models."""

import re
from collections.abc import Iterable, Iterator

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import PretrainedConfig

from olmo_core.distributed.utils import get_local_tensor

HFWeight = tuple[str, torch.Tensor]
HFWeightMetadata = tuple[str, torch.dtype, tuple[int, ...]]


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    wrapped = getattr(model, "module", None)
    return wrapped if isinstance(wrapped, torch.nn.Module) else model


def iter_olmo3moe_tensor_to_hf(
    config: PretrainedConfig, name: str, value: torch.Tensor
) -> Iterator[HFWeight]:
    """Convert one unsharded Olmo3MoE tensor into its HF tensor or tensors."""
    root_mapping = {
        "embeddings.weight": "model.embed_tokens.weight",
        "lm_head.norm.weight": "model.norm.weight",
        "lm_head.w_out.weight": "lm_head.weight",
        "embedding_norm.weight": "model.embed_norm.weight",
    }
    if name in root_mapping:
        yield root_mapping[name], value
        return

    match = re.fullmatch(r"blocks\.(\d+)\.(.+)", name)
    if match is None:
        raise RuntimeError(f"Cannot convert unsupported olmo3moe state key '{name}'")
    layer_idx = int(match.group(1))
    suffix = match.group(2)
    prefix = f"model.layers.{layer_idx}."

    layer_mapping = {
        "attention.w_q.weight": "self_attn.q_proj.weight",
        "attention.w_k.weight": "self_attn.k_proj.weight",
        "attention.w_v.weight": "self_attn.v_proj.weight",
        "attention.w_out.weight": "self_attn.o_proj.weight",
        "attention.q_norm.weight": "self_attn.q_norm.weight",
        "attention.k_norm.weight": "self_attn.k_norm.weight",
        "attention_norm.weight": "post_attention_layernorm.weight",
        "feed_forward_norm.weight": "post_feedforward_layernorm.weight",
        "attention_input_norm.weight": "pre_attention_layernorm.weight",
        "feed_forward_input_norm.weight": "pre_feedforward_layernorm.weight",
    }
    if suffix == "attention.w_qkv.weight":
        q_dim = int(config.num_attention_heads) * int(config.head_dim)
        kv_dim = int(config.num_key_value_heads) * int(config.head_dim)
        q, k, v = value.split((q_dim, kv_dim, kv_dim), dim=0)
        yield f"{prefix}self_attn.q_proj.weight", q
        yield f"{prefix}self_attn.k_proj.weight", k
        yield f"{prefix}self_attn.v_proj.weight", v
        return

    if suffix in layer_mapping:
        yield f"{prefix}{layer_mapping[suffix]}", value
        return

    dense_indices = set(getattr(config, "dense_layers_indices", None) or ())
    if layer_idx in dense_indices:
        if suffix == "shared_experts.w_up_gate":
            dense_hidden = value.shape[1] // 2
            yield f"{prefix}mlp.up_proj.weight", value[:, :dense_hidden].T.contiguous()
            yield f"{prefix}mlp.gate_proj.weight", value[:, dense_hidden:].T.contiguous()
        elif suffix == "shared_experts.w_down":
            yield f"{prefix}mlp.down_proj.weight", value.squeeze(0).T.contiguous()
        else:
            raise RuntimeError(f"Cannot convert unsupported dense olmo3moe state key '{name}'")
        return

    n_experts = int(config.n_routed_experts)
    d_model = int(config.hidden_size)
    moe_hidden = int(config.moe_intermediate_size)
    shared_hidden = getattr(config, "shared_expert_intermediate_size", None)

    if suffix == "routed_experts_router.weight":
        yield f"{prefix}mlp.router.gate.weight", value.reshape(n_experts, d_model)
    elif suffix == "routed_experts.w_up_gate":
        w_up_gate = value.reshape(n_experts, 2 * moe_hidden, d_model)
        for expert_idx in range(n_experts):
            yield (
                f"{prefix}mlp.experts.{expert_idx}.up_proj.weight",
                w_up_gate[expert_idx, :moe_hidden, :].contiguous(),
            )
            yield (
                f"{prefix}mlp.experts.{expert_idx}.gate_proj.weight",
                w_up_gate[expert_idx, moe_hidden:, :].contiguous(),
            )
    elif suffix == "routed_experts.w_down":
        w_down = value.reshape(n_experts, moe_hidden, d_model)
        for expert_idx in range(n_experts):
            yield (
                f"{prefix}mlp.experts.{expert_idx}.down_proj.weight",
                w_down[expert_idx].T.contiguous(),
            )
    elif suffix == "shared_experts.w_up_gate" and shared_hidden is not None:
        shared_up_gate = value.reshape(d_model, 2 * shared_hidden)
        yield (
            f"{prefix}mlp.shared_expert.up_proj.weight",
            shared_up_gate[:, :shared_hidden].T.contiguous(),
        )
        yield (
            f"{prefix}mlp.shared_expert.gate_proj.weight",
            shared_up_gate[:, shared_hidden:].T.contiguous(),
        )
    elif suffix == "shared_experts.w_down" and shared_hidden is not None:
        shared_down = value.reshape(shared_hidden, d_model)
        yield f"{prefix}mlp.shared_expert.down_proj.weight", shared_down.T.contiguous()
    else:
        raise RuntimeError(f"Cannot convert unsupported MoE olmo3moe state key '{name}'")


def _local_state(model: torch.nn.Module) -> Iterable[tuple[str, torch.Tensor, torch.nn.Module]]:
    for name, value in model.state_dict().items():
        local = get_local_tensor(value) if isinstance(value, DTensor) else value
        owner = model.get_submodule(name.rsplit(".", 1)[0])
        yield name, local, owner


def get_olmo_ddp_hf_weight_metadata(
    model: torch.nn.Module, hf_config: PretrainedConfig
) -> list[HFWeightMetadata]:
    """Return streamed HF weight metadata without allocating real tensor storage."""
    if hf_config.model_type != "olmo3moe":
        raise NotImplementedError(
            f"Streaming OLMoDDP HF weights does not support model_type={hf_config.model_type!r}"
        )

    model = _unwrap_model(model)
    metadata: list[HFWeightMetadata] = []
    for name, local, owner in _local_state(model):
        shape = tuple(local.shape)
        if getattr(owner, "_ep_sharded", False):
            group = owner.ep_mesh["ep_mp"].get_group()
            shape = (shape[0] * dist.get_world_size(group), *shape[1:])
        meta_value = torch.empty(shape, dtype=local.dtype, device="meta")
        for hf_name, hf_value in iter_olmo3moe_tensor_to_hf(hf_config, name, meta_value):
            metadata.append((hf_name, hf_value.dtype, tuple(hf_value.shape)))
    return metadata


def iter_olmo_ddp_hf_weights(
    model: torch.nn.Module,
    hf_config: PretrainedConfig,
    *,
    target_rank: int = 0,
) -> Iterator[HFWeight]:
    """Gather, convert, and yield one HF weight at a time on ``target_rank``.

    Ranks in the target's expert-parallel group participate in each gather but only
    the target rank emits weights. Other data-parallel replicas skip the gathers.
    """
    if hf_config.model_type != "olmo3moe":
        raise NotImplementedError(
            f"Streaming OLMoDDP HF weights does not support model_type={hf_config.model_type!r}"
        )

    model = _unwrap_model(model)
    rank = dist.get_rank() if dist.is_initialized() else 0
    for name, local, owner in _local_state(model):
        if getattr(owner, "_ep_sharded", False):
            if not dist.is_initialized():
                raise RuntimeError("Expert-parallel weight gathering requires torch.distributed")
            group = owner.ep_mesh["ep_mp"].get_group()
            if target_rank not in dist.get_process_group_ranks(group):
                continue
            full_shape = (local.shape[0] * dist.get_world_size(group), *local.shape[1:])
            full_value = torch.empty(full_shape, dtype=local.dtype, device=local.device)
            dist.all_gather_into_tensor(full_value, local.contiguous(), group=group)
        else:
            if rank != target_rank:
                continue
            full_value = local

        if rank == target_rank:
            yield from iter_olmo3moe_tensor_to_hf(hf_config, name, full_value)


def gather_olmo_ddp_hf_state_to_cpu(
    model: torch.nn.Module,
    hf_config: PretrainedConfig,
    *,
    target_rank: int = 0,
) -> dict[str, torch.Tensor]:
    """Stream a complete HF state to CPU on ``target_rank`` for final saving."""
    return {
        name: value.detach().cpu()
        for name, value in iter_olmo_ddp_hf_weights(model, hf_config, target_rank=target_rank)
    }
