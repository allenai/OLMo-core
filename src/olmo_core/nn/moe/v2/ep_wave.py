from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, Tuple, Union, cast

import torch
import torch.distributed as dist

from olmo_core.kernels.symm_mem_vdev2d import (
    rowwise_bf16_mega_moe_local_umma_compute,
    rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_peer_group,
    rowwise_bf16_mega_moe_standard_ep_workspace_config,
)

from ...moe.utils import wait_stream_no_compile
from .comm import _DispatchRowwiseAutograd, _RowwiseCombineWeightedAutograd
from .ep_no_sync_buffers import (
    compute_ep_no_sync_rank_capacity,
    get_cached_ep_no_sync_buffers,
    get_ep_no_sync_buffers,
    get_ep_no_sync_group_name,
)
from .ep_no_sync_common import (
    padded_local_expert_splits_for_capacity,
    sync_tail_drop_allowed_splits_single_a2a,
)
from .ep_no_sync_rowwise_helpers import (
    accumulate_ep_no_sync_rowwise_metrics,
    build_rowwise_route_maps,
)
from .routed_experts import (
    ExpertActivation,
    requires_host_side_split_sizes,
)

if TYPE_CHECKING:
    from olmo_core.nn.ddp.block import OLMoDDPTransformerBlock


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _check_ep_wave_supported(block: "OLMoDDPTransformerBlock", x: torch.Tensor) -> None:
    if not block.ep_no_sync:
        raise RuntimeError("combined_forward_ep_wave requires ep_no_sync=True")
    if not block.ep_no_sync_use_rowwise_all_to_all:
        raise RuntimeError(
            "combined_forward_ep_wave uses the rowwise NVSHMEM backend; "
            "set ep_no_sync_use_rowwise_all_to_all=True"
        )
    if block.ep_no_sync_use_2d_all_to_all:
        raise RuntimeError(
            "ep_no_sync_use_2d_all_to_all=True is not supported by combined_forward_ep_wave"
        )
    if block.ep_pg is None:
        raise RuntimeError("combined_forward_ep_wave requires block.apply_ep(...) first")
    if x.ndim != 3:
        raise RuntimeError(f"combined_forward_ep_wave expects input [B, S, D], got {tuple(x.shape)}")
    if block.routed_experts is None or block.routed_experts_router is None:
        raise RuntimeError("combined_forward_ep_wave requires routed experts and router")
    if block.num_local_routed_experts is None:
        raise RuntimeError("combined_forward_ep_wave requires local routed expert metadata")
    if requires_host_side_split_sizes():
        raise RuntimeError(
            "combined_forward_ep_wave does not support host-side split size communication"
        )
    if block.routed_experts.activation != ExpertActivation.swiglu:
        raise NotImplementedError("combined_forward_ep_wave currently supports swiglu routed experts only")
    if block.routed_experts.b_up_gate is not None or block.routed_experts.b_down is not None:
        raise NotImplementedError("combined_forward_ep_wave currently supports bias=False routed experts only")
    rowwise_fp8_cfg = block.rowwise_fp8
    if (
        rowwise_fp8_cfg is not None
        and rowwise_fp8_cfg.enabled
        and x.device.type == "cuda"
    ):
        raise NotImplementedError(
            "combined_forward_ep_wave currently targets BF16 only. "
            "MXFP8 training needs a dedicated fused wave autograd path."
        )


def _use_bf16_persistent_mega_forward(
    block: "OLMoDDPTransformerBlock",
    x: torch.Tensor,
    top_k: int,
) -> bool:
    if not (
        bool(getattr(block, "ep_no_sync_wave_use_bf16_persistent_mega_forward", False))
        or _env_flag("OLMO_EP_WAVE_USE_BF16_PERSISTENT_MEGA")
    ):
        return False
    if x.device.type != "cuda":
        raise RuntimeError("The BF16 persistent MegaMoE wave path requires CUDA inputs")
    if block.routed_experts is None:
        raise RuntimeError("BF16 persistent MegaMoE wave forward requires routed experts")
    if top_k <= 0 or top_k > 64:
        raise RuntimeError(f"BF16 persistent MegaMoE wave forward requires 0 < top_k <= 64, got {top_k}")
    if x.dtype != torch.bfloat16:
        raise RuntimeError(f"BF16 persistent MegaMoE wave forward requires bf16 inputs, got {x.dtype}")
    try:
        from olmo_core.kernels import olmo_symm_mem
    except Exception as e:
        raise RuntimeError("BF16 persistent MegaMoE wave forward requires OLMo symmetric memory") from e
    if not olmo_symm_mem.is_enabled():
        raise RuntimeError("BF16 persistent MegaMoE wave forward requires OLMo symmetric memory enabled")
    return True


def _can_use_standard_ep_peer_group_forward(
    block: "OLMoDDPTransformerBlock",
    moe_inp: torch.Tensor,
    top_k: int,
) -> bool:
    return _standard_ep_peer_group_forward_rejection_reason(block, moe_inp, top_k) is None


def _standard_ep_peer_group_forward_rejection_reason(
    block: "OLMoDDPTransformerBlock",
    moe_inp: torch.Tensor,
    top_k: int,
) -> Optional[str]:
    if _env_flag("OLMO_EP_WAVE_DISABLE_STANDARD_EP_PEER_GROUP") or _env_flag(
        "OLMO_EP_WAVE_DISABLE_STANDARD_EP_COLLECTIVE"
    ):
        return "standard EP peer-group path disabled by environment"
    if block.ep_pg is None or not dist.is_available() or not dist.is_initialized():
        return "block.apply_ep(...) and torch.distributed initialization are required"
    if dist.get_world_size(block.ep_pg) != 4:
        return f"EP process group size must be 4, got {dist.get_world_size(block.ep_pg)}"
    assert block.routed_experts is not None
    if block.routed_experts.num_experts != 32:
        return f"total routed experts must be 32, got {block.routed_experts.num_experts}"
    if block.num_local_routed_experts != 8:
        return f"local routed experts must be 8, got {block.num_local_routed_experts}"
    if top_k != 4:
        return f"top_k must be 4, got {top_k}"
    if moe_inp.shape[0] > 16384:
        return f"tokens per rank must be <= 16384, got {moe_inp.shape[0]}"
    if moe_inp.shape[1] % 16 != 0 or block.routed_experts.hidden_size % 16 != 0:
        return (
            "hidden and intermediate dimensions must be divisible by 16, got "
            f"{moe_inp.shape[1]} and {block.routed_experts.hidden_size}"
        )
    if block.routed_experts.w_up_gate.dtype != torch.bfloat16:
        return f"w_up_gate must be BF16, got {block.routed_experts.w_up_gate.dtype}"
    if block.routed_experts.w_down.dtype != torch.bfloat16:
        return f"w_down must be BF16, got {block.routed_experts.w_down.dtype}"
    return None


def _standard_ep_peer_group_cache_key(
    *,
    num_tokens: int,
    hidden: int,
    intermediate: int,
    device: torch.device,
) -> tuple[int, int, int, int]:
    return (
        int(num_tokens),
        int(hidden),
        int(intermediate),
        int(device.index if device.index is not None else torch.cuda.current_device()),
    )


def _get_standard_ep_peer_group_cache(
    block: "OLMoDDPTransformerBlock",
    *,
    num_tokens: int,
    hidden: int,
    intermediate: int,
    device: torch.device,
) -> dict[str, torch.Tensor | dict[str, int] | tuple[int, int, int, int]]:
    if block.ep_pg is None:
        raise RuntimeError("standard EP peer-group forward requires block.apply_ep(...) first")
    from olmo_core.kernels import olmo_symm_mem

    key = _standard_ep_peer_group_cache_key(
        num_tokens=num_tokens,
        hidden=hidden,
        intermediate=intermediate,
        device=device,
    )
    cache = getattr(block, "_ep_wave_standard_ep_peer_group_cache", None)
    if cache is not None and cache.get("key") == key:
        return cast(dict[str, torch.Tensor | dict[str, int] | tuple[int, int, int, int]], cache)

    workspace_config = rowwise_bf16_mega_moe_standard_ep_workspace_config(
        num_tokens=num_tokens,
        hidden=hidden,
        intermediate=intermediate,
    )
    workspace = olmo_symm_mem.empty(
        (workspace_config["workspace_stride_bytes"],),
        dtype=torch.uint8,
        device=device,
        group=block.ep_pg,
    )
    rank_workspace_bases = olmo_symm_mem.peer_base_ptrs(workspace, group=block.ep_pg)
    local_packed_capacity = workspace_config["local_packed_capacity"]
    num_route_slots = workspace_config["num_route_slots"]
    tensor_cache: dict[str, torch.Tensor | dict[str, int] | tuple[int, int, int, int]] = {
        "key": key,
        "workspace_config": workspace_config,
        "workspace": workspace,
        "rank_workspace_bases": rank_workspace_bases,
        "gathered_out": torch.empty(
            (num_tokens, workspace_config["top_k"], hidden),
            device=device,
            dtype=torch.bfloat16,
        ),
        "out": torch.empty((num_tokens, hidden), device=device, dtype=torch.bfloat16),
        "global_counts": torch.empty(
            (workspace_config["num_total_experts"],),
            device=device,
            dtype=torch.long,
        ),
        "global_offsets": torch.empty(
            (workspace_config["num_total_experts"] + 1,),
            device=device,
            dtype=torch.long,
        ),
        "expert_cursors": torch.empty(
            (workspace_config["num_total_experts"],),
            device=device,
            dtype=torch.long,
        ),
        "packed_route": torch.empty((local_packed_capacity,), device=device, dtype=torch.long),
        "route_to_slot": torch.empty((num_route_slots,), device=device, dtype=torch.long),
        "packed_input": torch.empty(
            (local_packed_capacity, hidden),
            device=device,
            dtype=torch.bfloat16,
        ),
        "h": torch.empty(
            (local_packed_capacity, intermediate),
            device=device,
            dtype=torch.bfloat16,
        ),
        "packed_expert_out": torch.empty(
            (local_packed_capacity, hidden),
            device=device,
            dtype=torch.bfloat16,
        ),
        "barrier_state": torch.empty(
            (workspace_config["barrier_state_len"],),
            device=device,
            dtype=torch.int32,
        ),
    }
    setattr(block, "_ep_wave_standard_ep_peer_group_cache", tensor_cache)
    return tensor_cache


def _standard_ep_peer_group_forward(
    block: "OLMoDDPTransformerBlock",
    moe_inp: torch.Tensor,
    route_expert_indices: torch.Tensor,
    route_probs: torch.Tensor,
) -> torch.Tensor:
    assert block.routed_experts is not None
    assert block.ep_pg is not None
    if moe_inp.dtype != torch.bfloat16:
        raise RuntimeError("standard EP peer-group forward requires BF16 input")
    num_tokens, hidden = moe_inp.shape
    intermediate = block.routed_experts.hidden_size
    cache = _get_standard_ep_peer_group_cache(
        block,
        num_tokens=num_tokens,
        hidden=hidden,
        intermediate=intermediate,
        device=moe_inp.device,
    )
    rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_peer_group(
        moe_inp.contiguous(),
        cast(torch.Tensor, cache["gathered_out"]),
        cast(torch.Tensor, cache["out"]),
        route_expert_indices.to(dtype=torch.long).contiguous(),
        route_probs.to(dtype=torch.float32).contiguous(),
        block.routed_experts.w_up_gate.contiguous(),
        block.routed_experts.w_down.contiguous(),
        cast(torch.Tensor, cache["workspace"]),
        cast(torch.Tensor, cache["rank_workspace_bases"]),
        cast(torch.Tensor, cache["global_counts"]),
        cast(torch.Tensor, cache["global_offsets"]),
        cast(torch.Tensor, cache["expert_cursors"]),
        cast(torch.Tensor, cache["packed_route"]),
        cast(torch.Tensor, cache["route_to_slot"]),
        cast(torch.Tensor, cache["packed_input"]),
        cast(torch.Tensor, cache["h"]),
        cast(torch.Tensor, cache["packed_expert_out"]),
        cast(torch.Tensor, cache["barrier_state"]),
        caller_rank_idx=dist.get_rank(block.ep_pg),
    )
    return cast(torch.Tensor, cache["out"])


def combined_forward_ep_wave(
    block: "OLMoDDPTransformerBlock",
    x: torch.Tensor,
    *,
    activation_checkpointing: Optional[bool] = None,
    accumulate_routed_aux_loss_metrics: Optional[bool] = None,
    loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
    **kwargs,
) -> torch.Tensor:
    """Forward-only BF16 wave EP path using the standard EP peer-group megakernel."""
    _check_ep_wave_supported(block, x)
    assert block.routed_experts_router is not None
    assert block.routed_experts is not None
    top_k = block.routed_experts_router.top_k
    if not _use_bf16_persistent_mega_forward(block, x, top_k):
        raise RuntimeError(
            "OLMo wave EP requires the fused BF16 MegaMoE target. Set "
            "ep_no_sync_wave_use_bf16_persistent_mega_forward=True or "
            "OLMO_EP_WAVE_USE_BF16_PERSISTENT_MEGA=1; otherwise use rowwise EP."
        )

    if torch.is_grad_enabled():
        raise NotImplementedError(
            "combined_forward_ep_wave currently implements BF16 forward only. "
            "Run it under torch.no_grad() for inference/profiling, or use rowwise "
            "EP for training until the fused wave backward is implemented."
        )
    if activation_checkpointing is None:
        activation_checkpointing = False
    if accumulate_routed_aux_loss_metrics is None:
        accumulate_routed_aux_loss_metrics = False
    if activation_checkpointing:
        raise NotImplementedError(
            "combined_forward_ep_wave is forward-only and does not support activation checkpointing yet"
        )

    self = block
    group_name = get_ep_no_sync_group_name(self)
    B, S, D = x.shape

    block_inp = x
    del x

    attn_res_out = self._checkpointed_res_norm_attn(block_inp, **kwargs)
    kwargs.pop("max_doc_len", None)
    kwargs.pop("cu_doc_lens", None)
    moe_inp = self._prepare_moe_input(attn_res_out)

    (
        local_x_global_routed_expert_weights,
        local_x_global_routed_expert_indices,
        local_batch_size_per_global_routed_expert,
        routed_expert_router_aux_loss_info,
    ) = self.routed_experts_router(
        moe_inp,
        False,
        loss_div_factor=loss_div_factor,
    )

    wait_stream_no_compile(
        this_stream=self.get_dense_stream(),
        other_stream=torch.cuda.current_stream(),
    )
    with torch.cuda.stream(self.get_dense_stream()):
        if self.shared_experts_router:
            (
                local_x_global_shared_expert_weights,
                _,
                _,
                _,
            ) = self.shared_experts_router(
                moe_inp,
                True,
                loss_div_factor=loss_div_factor,
            )
        else:
            local_x_global_shared_expert_weights = None

    in_shape = moe_inp.size()
    moe_inp = moe_inp.view(-1, in_shape[-1])
    num_out_tokens = local_x_global_routed_expert_indices.numel()
    num_input_tokens = moe_inp.shape[0]
    route_probs = local_x_global_routed_expert_weights.view(-1, top_k)

    peer_group_rejection_reason = _standard_ep_peer_group_forward_rejection_reason(
        self,
        moe_inp,
        top_k,
    )
    if peer_group_rejection_reason is None:
        if self.shared_experts is not None:
            with torch.cuda.stream(self.get_dense_stream()):
                shared_out_up, shared_out_gate = self.shared_experts.forward1(
                    moe_inp.view(B, S, D)
                )
        else:
            shared_out_up, shared_out_gate = None, None

        local_x = _standard_ep_peer_group_forward(
            self,
            moe_inp,
            local_x_global_routed_expert_indices.view(-1, top_k),
            route_probs,
        )

        if self.shared_experts is not None:
            assert shared_out_up is not None
            assert shared_out_gate is not None
            with torch.cuda.stream(self.get_dense_stream()):
                shared_out = self.shared_experts.forward2(
                    shared_out_up,
                    shared_out_gate,
                    attn_res_out.shape,
                )
                mixed_shared_out = self._mix_shared_out(
                    shared_out,
                    local_x_global_shared_expert_weights,
                    attn_res_out.shape,
                )
        else:
            mixed_shared_out = None

        local_x = local_x.view(in_shape)
        wait_stream_no_compile(torch.cuda.current_stream(), self.get_dense_stream())

        mlp_out = self._merge_routed_and_shared(local_x, mixed_shared_out)
        final_out = self._res_norm_mlp(attn_res_out, mlp_out)
        return self._attach_routed_aux_loss(
            final_out,
            routed_expert_router_aux_loss_info,
            accumulate_metrics=accumulate_routed_aux_loss_metrics,
        )

    if not _env_flag("OLMO_EP_WAVE_ALLOW_ROWWISE_FALLBACK"):
        raise RuntimeError(
            "OLMo wave EP requires the standard EP peer-group BF16 megakernel "
            f"path. It is unavailable here because {peer_group_rejection_reason}. "
            "Set OLMO_EP_WAVE_ALLOW_ROWWISE_FALLBACK=1 only for legacy "
            "rowwise-shell debugging."
        )

    with torch.no_grad():
        requested_splits = local_batch_size_per_global_routed_expert.to(dtype=torch.long)
        rank_capacity = compute_ep_no_sync_rank_capacity(self, num_out_tokens)
        (
            allowed_splits,
            recv_splits_by_src_local,
            _drop_token_cnt,
            keep_from_src_dest_local,
        ) = cast(
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            sync_tail_drop_allowed_splits_single_a2a(
                self,
                requested_splits,
                rank_capacity=rank_capacity,
                return_keep_matrix=True,
            ),
        )
        dispatch_in_cap = num_out_tokens
        dispatch_out_cap = rank_capacity
        combine_in_cap = rank_capacity
        combine_out_cap = num_input_tokens
        accumulate_ep_no_sync_rowwise_metrics(
            self,
            drop_token_cnt=_drop_token_cnt,
            num_out_tokens=num_out_tokens,
            recv_splits_by_src_local=recv_splits_by_src_local,
            rank_capacity=rank_capacity,
        )

    buffers = get_cached_ep_no_sync_buffers(
        self,
        dispatch_in_cap=dispatch_in_cap,
        dispatch_out_cap=dispatch_out_cap,
        combine_in_cap=combine_in_cap,
        combine_out_cap=combine_out_cap,
        d_model=moe_inp.shape[-1],
        dtype=moe_inp.dtype,
        device=moe_inp.device,
        need_dispatch_in=False,
        need_dispatch_meta=False,
        need_dispatch_out=True,
        need_combine_in=True,
        need_combine_meta=False,
        need_combine_out=False,
        need_combine_gather=False,
    )
    if buffers is None:
        buffers = get_ep_no_sync_buffers(
            self,
            dispatch_in_cap=dispatch_in_cap,
            dispatch_out_cap=dispatch_out_cap,
            combine_in_cap=combine_in_cap,
            combine_out_cap=combine_out_cap,
            d_model=moe_inp.shape[-1],
            dtype=moe_inp.dtype,
            device=moe_inp.device,
            need_dispatch_in=False,
            need_dispatch_meta=False,
            need_dispatch_out=True,
            need_combine_in=True,
            need_combine_meta=False,
            need_combine_out=False,
            need_combine_gather=False,
        )

    routing_map = local_x_global_routed_expert_indices.view(-1, top_k).int()
    with torch.no_grad():
        batch_size_per_local_expert = recv_splits_by_src_local.sum(dim=0, dtype=torch.long)
        # Keep this calculation here even though the UMMA wrapper uses the
        # unpadded counts; it validates the same capacity invariant as rowwise
        # EP and keeps the forward shell aligned with the production path.
        padded_local_expert_splits_for_capacity(
            recv_splits_by_src_local,
            rank_capacity=rank_capacity,
        )
        expert_batch_size_per_local_expert = batch_size_per_local_expert
        dst_ranks, dst_rows = build_rowwise_route_maps(
            self,
            routing_map=routing_map,
            allowed_splits=allowed_splits,
            keep_from_src_dest_local=keep_from_src_dest_local,
        )
        rowwise_nblocks = self.ep_no_sync_rowwise_nblocks

    if self.shared_experts is not None:
        with torch.cuda.stream(self.get_dense_stream()):
            shared_out_up, shared_out_gate = self.shared_experts.forward1(
                moe_inp.view(B, S, D)
            )
    else:
        shared_out_up, shared_out_gate = None, None

    dispatch_rank_major = _DispatchRowwiseAutograd.apply(
        moe_inp,
        None,
        dst_ranks,
        dst_rows,
        buffers.dispatch_out,
        None,
        group_name,
        self.ep_pg,
        rowwise_nblocks,
        False,
        True,
        True,
        False,
    )

    dispatch_rank_major = rowwise_bf16_mega_moe_local_umma_compute(
        dispatch_rank_major,
        expert_batch_size_per_local_expert.to(dtype=torch.long).contiguous(),
        self.routed_experts.w_up_gate,
        self.routed_experts.w_down,
    )

    expert_out_aliases_symm_expert_out = (
        dispatch_rank_major.data_ptr() == buffers.combine_in.data_ptr()
        and dispatch_rank_major.storage_offset() == buffers.combine_in.storage_offset()
        and tuple(dispatch_rank_major.shape) == tuple(buffers.combine_in.shape)
    )
    local_x = _RowwiseCombineWeightedAutograd.apply(
        dispatch_rank_major,
        buffers.combine_in,
        None,
        None,
        None,
        None,
        dst_ranks,
        dst_rows,
        route_probs,
        group_name,
        self.ep_pg,
        self.ep_no_sync_rowwise_nblocks,
        expert_out_aliases_symm_expert_out,
        True,
        False,
    )

    if self.shared_experts is not None:
        assert shared_out_up is not None
        assert shared_out_gate is not None
        with torch.cuda.stream(self.get_dense_stream()):
            shared_out = self.shared_experts.forward2(
                shared_out_up,
                shared_out_gate,
                attn_res_out.shape,
            )
            mixed_shared_out = self._mix_shared_out(
                shared_out,
                local_x_global_shared_expert_weights,
                attn_res_out.shape,
            )
    else:
        mixed_shared_out = None

    local_x = local_x.view(in_shape)
    wait_stream_no_compile(torch.cuda.current_stream(), self.get_dense_stream())

    mlp_out = self._merge_routed_and_shared(local_x, mixed_shared_out)
    final_out = self._res_norm_mlp(attn_res_out, mlp_out)
    return self._attach_routed_aux_loss(
        final_out,
        routed_expert_router_aux_loss_info,
        accumulate_metrics=accumulate_routed_aux_loss_metrics,
    )
