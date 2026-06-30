from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import nvtx
import torch
import torch.distributed as dist

from olmo_core.distributed.utils import get_rank

from ...moe.utils import wait_stream_no_compile
from .ep_config import ExpertParallelPath
from .ep_no_sync_buffers import compute_ep_no_sync_rank_capacity
from .ep_no_sync_common import sync_tail_drop_allowed_splits_single_a2a
from .ep_no_sync_rowwise_helpers import (
    accumulate_ep_no_sync_rowwise_metrics,
    build_rowwise_route_maps,
    should_accumulate_ep_no_sync_rowwise_metrics,
)
from .routed_experts import requires_host_side_split_sizes, use_torch_grouped_mm

if TYPE_CHECKING:
    from olmo_core.nn.ddp.block import OLMoDDPTransformerBlock

try:
    _torch_compile_disable = torch.compiler.disable
except AttributeError:

    def _torch_compile_disable(fn):
        return fn


@dataclass
class _DeepEpV2Runtime:
    deep_ep: object
    buffer: object
    num_max_tokens_per_rank: int
    hidden: int
    num_topk: int
    num_experts: int
    num_local_experts: int
    expert_alignment: int
    num_sms: int
    num_qps: int
    async_with_compute_stream: bool


@_torch_compile_disable
def _import_deepep(deepep_path: Optional[str]) -> object:
    resolved_path = deepep_path or os.getenv("OLMO_DEEPEP_PATH", "/workspace/DeepEP")
    if resolved_path:
        resolved_path = os.path.abspath(resolved_path)
        if os.path.isdir(resolved_path) and resolved_path not in sys.path:
            sys.path.insert(0, resolved_path)
    try:
        import deep_ep  # type: ignore[import-not-found]
    except Exception as e:
        raise RuntimeError(
            "Failed to import DeepEP for EP path='deepep_v2'. "
            "Build/install DeepEP first, set OLMO_DEEPEP_PATH, or set "
            "ep.deepep.path. "
            f"Original error: {type(e).__name__}: {e}"
        ) from e
    return deep_ep


def _deep_ep_wait(event: object, *, async_with_compute_stream: bool) -> None:
    if async_with_compute_stream:
        event.current_stream_wait()


def _expanded_expert_counts(handle: object, expert_alignment: int) -> torch.Tensor:
    psum = handle.psum_num_recv_tokens_per_expert
    if psum.ndim != 1:
        raise RuntimeError(
            "DeepEP handle.psum_num_recv_tokens_per_expert must be 1D "
            f"(got shape={tuple(psum.shape)})"
        )
    starts = torch.empty_like(psum)
    starts.fill_(0)
    if psum.numel() > 1:
        previous = psum[:-1]
        if expert_alignment == 1:
            starts[1:] = previous
        else:
            starts[1:] = ((previous + expert_alignment - 1) // expert_alignment) * expert_alignment
    return (psum - starts).to(dtype=torch.int32)


@_torch_compile_disable
def _expanded_weight_grad_to_topk_grad(
    *,
    block: OLMoDDPTransformerBlock,
    runtime: _DeepEpV2Runtime,
    handle: object,
    expanded_weight_grad: torch.Tensor,
    local_num_tokens: int,
    topk_weights_dtype: torch.dtype,
) -> torch.Tensor:
    metadata = handle.recv_src_metadata
    if metadata.ndim != 2 or metadata.shape[1] < 2 + runtime.num_topk:
        raise RuntimeError(
            "DeepEP expanded recv metadata must be [num_recv_tokens, 2 + top_k], "
            f"got shape={tuple(metadata.shape)} for top_k={runtime.num_topk}"
        )
    if block.ep_pg is None:
        raise RuntimeError("deepep_v2 top-k weight backward requires an EP process group")

    grad_flat = expanded_weight_grad.reshape(-1).to(dtype=torch.float32)
    grad_by_src = torch.zeros(
        (block.ep_world_size, runtime.num_max_tokens_per_rank, runtime.num_topk),
        device=grad_flat.device,
        dtype=torch.float32,
    )
    src_global = metadata[:, 0].to(dtype=torch.long)
    metadata_row = torch.arange(metadata.shape[0], device=metadata.device, dtype=torch.long)
    actual_recv_tokens = handle.psum_num_recv_tokens_per_scaleup_rank[-1].to(dtype=torch.long)
    src_rank = torch.div(
        src_global,
        runtime.num_max_tokens_per_rank,
        rounding_mode="floor",
    )
    src_token = src_global - src_rank * runtime.num_max_tokens_per_rank
    expanded_slots_by_lane = metadata[:, 2 : 2 + runtime.num_topk].to(dtype=torch.long)
    flat_grad_by_src = grad_by_src.reshape(-1)
    for lane in range(runtime.num_topk):
        expanded_slots = expanded_slots_by_lane[:, lane]
        valid = (
            (metadata_row < actual_recv_tokens)
            & (src_rank >= 0)
            & (src_rank < block.ep_world_size)
            & (src_token >= 0)
            & (src_token < runtime.num_max_tokens_per_rank)
            & (expanded_slots >= 0)
            & (expanded_slots < grad_flat.numel())
        )
        valid_slots = expanded_slots[valid]
        flat_dst = (
            (src_rank[valid] * runtime.num_max_tokens_per_rank + src_token[valid])
            * runtime.num_topk
            + lane
        )
        flat_grad_by_src.scatter_add_(0, flat_dst, grad_flat.index_select(0, valid_slots))

    # Every receiving rank owns a different subset of expanded rows. Sum those
    # per-source contributions so the original token owner can return a normal
    # [local_tokens, top_k] gradient to its router.
    dist.all_reduce(grad_by_src, group=block.ep_pg)
    local_ep_rank = get_rank(block.ep_pg)
    grad_topk = grad_by_src[local_ep_rank, :local_num_tokens, :]
    if grad_topk.dtype != topk_weights_dtype:
        grad_topk = grad_topk.to(dtype=topk_weights_dtype)
    return grad_topk


def _validate_deepep_v2_block(
    block: OLMoDDPTransformerBlock,
    x: torch.Tensor,
) -> None:
    if block.ep.path != ExpertParallelPath.deepep_v2:
        raise RuntimeError(
            "combined_forward_ep_deepep_v2 requires "
            f"path={ExpertParallelPath.deepep_v2!r}"
        )
    if not x.is_cuda:
        raise RuntimeError("deepep_v2 EP requires CUDA input")
    if x.dtype != torch.bfloat16:
        raise RuntimeError(f"deepep_v2 EP currently supports bf16 only, got {x.dtype}")
    if x.shape[-1] % 256 != 0:
        raise RuntimeError(
            "deepep_v2 BF16 combine requires d_model divisible by 256 "
            f"(got {x.shape[-1]})"
        )
    if block.ep_pg is None:
        raise RuntimeError("deepep_v2 EP requires block.ep_pg to be initialized")
    if block.routed_experts is None or block.routed_experts_router is None:
        raise RuntimeError("deepep_v2 EP requires routed experts and a routed router")
    if block.num_local_routed_experts is None:
        raise RuntimeError("deepep_v2 EP requires local routed expert count")
    deepep_cfg = block.ep.deepep
    if deepep_cfg.expert_alignment != 1:
        raise RuntimeError(
            "deepep_v2 model path currently requires deepep.expert_alignment=1. "
            "The expanded dispatch layout may contain aligned padding, while "
            "RoutedExperts.forward consumes packed per-expert rows."
        )
    if block.routed_experts.b_down is not None:
        raise RuntimeError(
            "deepep_v2 EP with deepep.weighting='swiglu' requires bias-free "
            "routed expert down projections."
        )
    if (
        block.rowwise_fp8 is not None
        and block.rowwise_fp8.enabled
        and x.device.type == "cuda"
    ):
        raise RuntimeError("deepep_v2 EP does not support rowwise FP8 experts yet")
    if not use_torch_grouped_mm():
        raise RuntimeError("deepep_v2 EP requires torch.grouped_mm support")
    if requires_host_side_split_sizes():
        raise RuntimeError("deepep_v2 EP does not support host-side split sizes")


@_torch_compile_disable
def _global_num_max_tokens_per_rank(
    block: OLMoDDPTransformerBlock,
    local_tokens: int,
    device: torch.device,
) -> int:
    requested = local_tokens
    requested_tensor = torch.tensor([requested], device=device, dtype=torch.long)
    dist.all_reduce(requested_tensor, op=dist.ReduceOp.MAX, group=block.ep_pg)
    return int(requested_tensor.item())


@_torch_compile_disable
def _get_deepep_v2_runtime(
    block: OLMoDDPTransformerBlock,
    *,
    local_tokens: int,
    hidden: int,
    top_k: int,
    device: torch.device,
) -> _DeepEpV2Runtime:
    assert block.ep_pg is not None
    assert block.routed_experts_router is not None
    assert block.num_local_routed_experts is not None

    num_max_tokens_per_rank = _global_num_max_tokens_per_rank(
        block,
        local_tokens,
        device,
    )
    runtime = getattr(block, "_deepep_v2_runtime", None)
    if (
        runtime is not None
        and runtime.hidden == hidden
        and runtime.num_topk == top_k
        and runtime.num_experts == block.routed_experts_router.num_experts
        and runtime.num_local_experts == block.num_local_routed_experts
        and runtime.expert_alignment == block.ep.deepep.expert_alignment
    ):
        if num_max_tokens_per_rank <= runtime.num_max_tokens_per_rank:
            return runtime
        raise RuntimeError(
            "deepep_v2 EP saw more tokens than its existing ElasticBuffer "
            f"capacity: requested={num_max_tokens_per_rank}, "
            f"capacity={runtime.num_max_tokens_per_rank}. Initialize "
            "deepep_v2 with the largest local token shape first; "
            "ep.capacity_factor only controls expanded receive tail-drop "
            "capacity."
        )

    if runtime is not None:
        raise RuntimeError(
            "deepep_v2 EP runtime shape changed after initialization: "
            f"old=(hidden={runtime.hidden}, top_k={runtime.num_topk}, "
            f"experts={runtime.num_experts}, local_experts={runtime.num_local_experts}) "
            f"new=(hidden={hidden}, top_k={top_k}, "
            f"experts={block.routed_experts_router.num_experts}, "
            f"local_experts={block.num_local_routed_experts})"
        )

    deepep_cfg = block.ep.deepep
    deep_ep = _import_deepep(deepep_cfg.path)
    num_allocated_qps = max(deepep_cfg.num_allocated_qps, deepep_cfg.num_qps)
    buffer = deep_ep.ElasticBuffer(
        block.ep_pg,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=top_k,
        deterministic=False,
        allow_hybrid_mode=deepep_cfg.allow_hybrid_mode,
        allow_multiple_reduction=deepep_cfg.allow_multiple_reduction,
        prefer_overlap_with_compute=deepep_cfg.prefer_overlap_with_compute,
        num_allocated_qps=num_allocated_qps,
        explicitly_destroy=False,
    )
    for required_method in ("dispatch_expanded_into", "dispatch_cached_expanded_into"):
        if not hasattr(buffer, required_method):
            raise RuntimeError(
                "deepep_v2 model path requires the modified DeepEP working copy "
                f"with ElasticBuffer.{required_method}. Use ep.deepep.path or "
                "OLMO_DEEPEP_PATH to point at /workspace/DeepEP."
            )
    num_sms = (
        int(deepep_cfg.num_sms)
        if deepep_cfg.num_sms != 0
        else int(buffer.get_theoretical_num_sms(block.routed_experts_router.num_experts, top_k))
    )
    num_qps = (
        int(deepep_cfg.num_qps)
        if deepep_cfg.num_qps != 0
        else int(buffer.get_theoretical_num_qps(num_sms))
    )
    runtime = _DeepEpV2Runtime(
        deep_ep=deep_ep,
        buffer=buffer,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=top_k,
        num_experts=block.routed_experts_router.num_experts,
        num_local_experts=block.num_local_routed_experts,
        expert_alignment=deepep_cfg.expert_alignment,
        num_sms=num_sms,
        num_qps=num_qps,
        async_with_compute_stream=deepep_cfg.async_mode,
    )
    block._deepep_v2_runtime = runtime
    return runtime


class _DeepEpV2Autograd(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        source_input: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        block: OLMoDDPTransformerBlock,
        runtime: _DeepEpV2Runtime,
        rank_capacity: int,
    ) -> torch.Tensor:
        recv_x_out = torch.empty(
            (int(rank_capacity), runtime.hidden),
            device=source_input.device,
            dtype=source_input.dtype,
        )
        recv_topk_weights_out = torch.empty(
            (int(rank_capacity),),
            device=topk_weights.device,
            dtype=topk_weights.dtype,
        )
        recv_x, _recv_topk_idx, expanded_topk_weights, handle, event = (
            runtime.buffer.dispatch_expanded_into(
                source_input,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                recv_x_out=recv_x_out,
                recv_topk_weights_out=recv_topk_weights_out,
                num_experts=runtime.num_experts,
                num_max_tokens_per_rank=runtime.num_max_tokens_per_rank,
                expert_alignment=runtime.expert_alignment,
                num_sms=runtime.num_sms,
                num_qps=runtime.num_qps,
                async_with_compute_stream=runtime.async_with_compute_stream,
                do_cpu_sync=False,
            )
        )
        _deep_ep_wait(event, async_with_compute_stream=runtime.async_with_compute_stream)
        if expanded_topk_weights is None:
            raise RuntimeError("deepep_v2 expanded dispatch did not return top-k weights")

        batch_size_per_expert = _expanded_expert_counts(handle, runtime.expert_alignment)
        need_grad_topk_weights = ctx.needs_input_grad[2]
        expanded_weights = expanded_topk_weights.reshape(-1, 1)
        if need_grad_topk_weights:
            expanded_weights = expanded_weights.detach().requires_grad_(True)
        recv_x_for_experts = recv_x.detach().requires_grad_(True)
        assert block.routed_experts is not None
        with torch.enable_grad():
            expert_out = block.routed_experts(
                recv_x_for_experts,
                batch_size_per_expert,
                row_weights=expanded_weights,
            )

        combined_x, _combined_topk_weights, event = runtime.buffer.combine(
            expert_out,
            handle=handle,
            num_sms=runtime.num_sms,
            num_qps=runtime.num_qps,
            async_with_compute_stream=runtime.async_with_compute_stream,
        )
        _deep_ep_wait(event, async_with_compute_stream=runtime.async_with_compute_stream)

        ctx.block = block
        ctx.runtime = runtime
        ctx.handle = handle
        ctx.local_num_tokens = int(topk_weights.shape[0])
        ctx.topk_weights_dtype = topk_weights.dtype
        ctx.need_grad_topk_weights = need_grad_topk_weights
        ctx.save_for_backward(recv_x_for_experts, expert_out, expanded_weights)
        return combined_x

    @staticmethod
    def backward(ctx, grad_combined_x: torch.Tensor):  # type: ignore[override]
        runtime: _DeepEpV2Runtime = ctx.runtime
        block: OLMoDDPTransformerBlock = ctx.block
        handle = ctx.handle
        recv_x, expert_out, expanded_weights = ctx.saved_tensors

        grad_weighted_expert_out = torch.empty_like(recv_x)
        _grad_weighted_expert_out, _grad_topk_idx, _grad_topk_weights, _handle, event = (
            runtime.buffer.dispatch_cached_expanded_into(
                grad_combined_x.contiguous(),
                handle=handle,
                recv_x_out=grad_weighted_expert_out,
                num_sms=runtime.num_sms,
                num_qps=runtime.num_qps,
                async_with_compute_stream=runtime.async_with_compute_stream,
            )
        )
        _deep_ep_wait(event, async_with_compute_stream=runtime.async_with_compute_stream)

        torch.autograd.backward(expert_out, grad_weighted_expert_out)
        if recv_x.grad is None:
            raise RuntimeError("deepep_v2 expert backward did not produce grad for recv_x")
        grad_topk_weights = None
        if ctx.need_grad_topk_weights:
            if expanded_weights.grad is None:
                raise RuntimeError(
                    "deepep_v2 expert backward did not produce grad for expanded top-k weights"
                )
            grad_topk_weights = _expanded_weight_grad_to_topk_grad(
                block=block,
                runtime=runtime,
                handle=handle,
                expanded_weight_grad=expanded_weights.grad,
                local_num_tokens=ctx.local_num_tokens,
                topk_weights_dtype=ctx.topk_weights_dtype,
            )

        combined_grad_x, _combined_grad_topk_weights, event = runtime.buffer.combine(
            recv_x.grad,
            handle=handle,
            num_sms=runtime.num_sms,
            num_qps=runtime.num_qps,
            async_with_compute_stream=runtime.async_with_compute_stream,
        )
        _deep_ep_wait(event, async_with_compute_stream=runtime.async_with_compute_stream)

        return combined_grad_x, None, grad_topk_weights, None, None, None


def combined_forward_ep_deepep_v2(
    block: OLMoDDPTransformerBlock,
    x: torch.Tensor,
    *,
    accumulate_routed_aux_loss_metrics: Optional[bool] = None,
    loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
    **kwargs,
) -> torch.Tensor:
    """Forward with DeepEP V2 expanded dispatch/combine.

    This is intentionally a narrow first model-path integration of the
    standalone benchmark backend. It skips OLMo symmetric-memory systems and
    uses DeepEP's own ElasticBuffer on the EP process group.
    """
    self = block
    _validate_deepep_v2_block(self, x)
    assert self.routed_experts is not None
    assert self.routed_experts_router is not None

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

    mixed_shared_out = None
    if self.shared_experts is not None:
        wait_stream_no_compile(
            this_stream=self.get_dense_stream(),
            other_stream=torch.cuda.current_stream(),
        )
        with torch.cuda.stream(self.get_dense_stream()):
            shared_out = self.shared_experts(moe_inp)
            mixed_shared_out = self._mix_shared_out(
                shared_out,
                local_x_global_shared_expert_weights,
                attn_res_out.shape,
            )

    moe_inp = moe_inp.view(-1, in_shape[-1])
    top_k = self.routed_experts_router.top_k
    routing_map = local_x_global_routed_expert_indices.view(-1, top_k)
    route_weights = local_x_global_routed_expert_weights.view(-1, top_k)
    num_out_tokens = routing_map.numel()
    with torch.no_grad():
        rank_capacity = compute_ep_no_sync_rank_capacity(self, num_out_tokens)
        (
            allowed_splits,
            recv_splits_by_src_local,
            _drop_token_cnt,
            keep_from_src_dest_local,
        ) = sync_tail_drop_allowed_splits_single_a2a(
            self,
            local_batch_size_per_global_routed_expert.to(dtype=torch.long),
            rank_capacity=rank_capacity,
            return_keep_matrix=True,
        )
        _dst_ranks, dst_rows = build_rowwise_route_maps(
            self,
            routing_map=routing_map,
            allowed_splits=allowed_splits,
            keep_from_src_dest_local=keep_from_src_dest_local,
        )
        route_keep_mask = dst_rows >= 0
        if should_accumulate_ep_no_sync_rowwise_metrics(accumulate_routed_aux_loss_metrics):
            accumulate_ep_no_sync_rowwise_metrics(
                self,
                drop_token_cnt=_drop_token_cnt,
                num_out_tokens=num_out_tokens,
                recv_splits_by_src_local=recv_splits_by_src_local,
                rank_capacity=rank_capacity,
            )

    runtime = _get_deepep_v2_runtime(
        self,
        local_tokens=moe_inp.shape[0],
        hidden=moe_inp.shape[-1],
        top_k=top_k,
        device=moe_inp.device,
    )
    # Reuse rowwise's deterministic tail-drop policy, then present dropped
    # routes to DeepEP as invalid top-k slots. DeepEP ignores negative expert
    # ids during dispatch counting/copy and combine reduction.
    topk_idx = torch.where(
        route_keep_mask,
        routing_map,
        routing_map.new_full(routing_map.shape, -1),
    ).to(
        dtype=runtime.deep_ep.topk_idx_t,
    ).contiguous()
    topk_weights = torch.where(
        route_keep_mask,
        route_weights,
        torch.zeros_like(route_weights),
    ).to(
        dtype=torch.float32,
    ).contiguous()

    with nvtx.annotate("deepep_v2/routed", color="green"):
        routed_out = _DeepEpV2Autograd.apply(
            moe_inp,
            topk_idx,
            topk_weights,
            self,
            runtime,
            rank_capacity,
        )

    x_moe = routed_out.view(in_shape)
    wait_stream_no_compile(torch.cuda.current_stream(), self.get_dense_stream())

    mlp_out = self._merge_routed_and_shared(x_moe, mixed_shared_out)
    final_out = self._res_norm_mlp(attn_res_out, mlp_out)
    return self._attach_routed_aux_loss(
        final_out,
        routed_expert_router_aux_loss_info,
    )
