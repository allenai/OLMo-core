"""Correctness-first HF weight interchange for OLMoDDP MoE models."""

from collections.abc import Mapping

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import PretrainedConfig

from olmo_core.distributed.utils import get_local_tensor
from olmo_core.nn.hf.convert import convert_state_from_hf, convert_state_to_hf


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    wrapped = getattr(model, "module", None)
    return wrapped if isinstance(wrapped, torch.nn.Module) else model


def load_olmo_ddp_hf_state(
    model: torch.nn.Module,
    hf_config: PretrainedConfig,
    hf_state: Mapping[str, torch.Tensor],
) -> None:
    """Load a full supported HF state into an unsharded or EP-sharded OLMoDDP model."""
    model = _unwrap_model(model)
    model_type = getattr(hf_config, "model_type", None)
    if not model_type:
        raise ValueError("The Hugging Face config must define model_type.")
    native_state = convert_state_from_hf(hf_config, dict(hf_state), model_type=model_type)
    parameters = dict(model.named_parameters())
    missing = set(parameters) - set(native_state)
    if missing:
        raise RuntimeError(f"Converted {model_type} state is missing parameters: {sorted(missing)}")
    unexpected = set(native_state) - set(parameters)
    if unexpected:
        raise RuntimeError(
            f"Converted {model_type} state has unexpected parameters: {sorted(unexpected)}"
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


def gather_olmo_ddp_hf_state(
    model: torch.nn.Module, hf_config: PretrainedConfig
) -> dict[str, torch.Tensor]:
    """Collect a supported EP-sharded OLMoDDP model into full HF state on every rank."""
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
