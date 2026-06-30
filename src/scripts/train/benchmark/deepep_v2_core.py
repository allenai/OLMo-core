from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - Triton may be unavailable in some test envs.
    triton = None
    tl = None

from olmo_core.config import DType
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig

from .common import (
    _build_block,
    _compile_hot_modules,
    _install_deepep_balanced_router,
    _patch_moe_only,
)
from .expert_probe import (
    _init_probe_routed_expert_weights,
    _resolve_probe_weight_init,
)


if triton is not None:

    @triton.jit
    def _deepep_rowwise_scale_kernel(
        x_ptr,
        weights_ptr,
        out_ptr,
        rows: tl.constexpr,
        hidden: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        row_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        col_idx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        mask = (row_idx < rows) & (col_idx < hidden)
        offsets = row_idx * hidden + col_idx
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        weights = tl.load(weights_ptr + row_idx, mask=row_idx < rows, other=0.0).to(tl.float32)
        tl.store(out_ptr + offsets, x * weights, mask=mask)


def _import_deepep(deepep_path: str):
    if deepep_path:
        deepep_path = os.path.abspath(deepep_path)
        if os.path.isdir(deepep_path) and deepep_path not in sys.path:
            sys.path.insert(0, deepep_path)
    try:
        import deep_ep  # type: ignore[import-not-found]
    except Exception as e:
        raise RuntimeError(
            "Failed to import DeepEP for --modes deepep_v2. "
            "Build/install DeepEP first, or pass --deepep-path /path/to/DeepEP. "
            f"Original error: {type(e).__name__}: {e}"
        ) from e
    return deep_ep


@dataclass
class DeepEpV2State:
    deep_ep: object
    buffer: object
    routed_experts: torch.nn.Module
    source_input: torch.Tensor
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    rank: int
    world_size: int
    num_experts: int
    num_local_experts: int
    num_max_tokens_per_rank: int
    expert_alignment: int
    num_sms: int
    num_qps: int
    expert_buffer_mode: str
    weighting_mode: str
    async_with_compute_stream: bool
    do_cpu_sync: bool
    wave_dispatch_stream: torch.cuda.Stream | None = None
    wave_compute_stream: torch.cuda.Stream | None = None
    wave_combine_stream: torch.cuda.Stream | None = None


@dataclass
class DeepEpV2ForwardResult:
    recv_x: torch.Tensor
    expanded_topk_weights: torch.Tensor
    expert_out: torch.Tensor
    combined_x: torch.Tensor
    handle: object
    expert_out_is_weighted: bool = False
    grad_combined_x: torch.Tensor | None = None
    static_wave: DeepEpV2WaveInput | None = None
    static_recv_x_global: torch.Tensor | None = None


@dataclass
class DeepEpV2WaveForwardResult:
    wave_results: list[DeepEpV2ForwardResult]
    combined_x: torch.Tensor
    grad_combined_x: torch.Tensor | None = None


@dataclass
class DeepEpV2WaveDispatchResult:
    wave: "DeepEpV2WaveInput"
    recv_x: torch.Tensor
    expanded_topk_weights: torch.Tensor | None
    recv_topk_idx: torch.Tensor | None
    recv_topk_weights: torch.Tensor | None
    handle: object
    event: object


@dataclass
class DeepEpV2WaveComputeResult:
    wave: "DeepEpV2WaveInput"
    recv_x_for_experts: torch.Tensor
    expert_out: torch.Tensor
    expanded_weights: torch.Tensor
    expert_out_is_weighted: bool
    weighted_expert_out: torch.Tensor
    handle: object
    recv_x_global: torch.Tensor | None
    compute_done: torch.cuda.Event


@dataclass(frozen=True)
class DeepEpV2WaveInput:
    wave_idx: int
    expert_start: int
    expert_end: int
    wave_base: int
    wave_end: int
    batch_size_per_expert: torch.Tensor
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor

    @property
    def wave_rows(self) -> int:
        return self.wave_end - self.wave_base


def _make_balanced_topk_idx(
    *,
    tokens: int,
    top_k: int,
    num_experts: int,
    world_size: int,
    rank: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if num_experts % world_size != 0:
        raise RuntimeError(
            f"deepep_v2 requires --num-experts divisible by ranks "
            f"({num_experts} vs {world_size})"
        )
    local_experts = num_experts // world_size
    token_idx = torch.arange(tokens, device="cuda", dtype=torch.long).view(tokens, 1)
    slot_idx = torch.arange(top_k, device="cuda", dtype=torch.long).view(1, top_k)
    expert_rank = (token_idx + rank + slot_idx) % world_size
    local_expert = (token_idx * top_k + slot_idx) % local_experts
    return (expert_rank * local_experts + local_expert).to(dtype=dtype).contiguous()


def _align_int(value: int, alignment: int) -> int:
    if alignment <= 1:
        return int(value)
    return ((int(value) + alignment - 1) // alignment) * alignment


def _deepep_v2_local_expert_counts(state: DeepEpV2State) -> list[int]:
    topk_idx_long = state.topk_idx.to(dtype=torch.long)
    valid_idx = topk_idx_long[topk_idx_long >= 0]
    global_counts = torch.bincount(
        valid_idx,
        minlength=state.num_experts,
    ).to(dtype=torch.long)
    dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)

    local_start = state.rank * state.num_local_experts
    local_end = local_start + state.num_local_experts
    return [
        int(v)
        for v in global_counts[local_start:local_end].detach().cpu().tolist()
    ]


def _deepep_v2_expanded_offsets(
    counts: Sequence[int],
    *,
    expert_alignment: int,
) -> list[int]:
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + _align_int(int(count), expert_alignment))
    return offsets


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


def _num_recv_tokens(handle: object, *, device: torch.device) -> torch.Tensor:
    psum = handle.psum_num_recv_tokens_per_scaleup_rank
    if psum.ndim != 1:
        raise RuntimeError(
            "DeepEP handle.psum_num_recv_tokens_per_scaleup_rank must be 1D "
            f"(got shape={tuple(psum.shape)})"
        )
    return psum[-1].to(device=device, dtype=torch.long)


def _deep_ep_wait(event: object, *, async_with_compute_stream: bool) -> None:
    if async_with_compute_stream:
        event.current_stream_wait()


def _reshape_expanded_weights(
    expanded_topk_weights: torch.Tensor | None,
    *,
    num_rows: int,
    dtype: torch.dtype,
    default_weight: float = 1.0,
) -> torch.Tensor:
    if expanded_topk_weights is None:
        return torch.full((num_rows, 1), default_weight, device="cuda", dtype=dtype)
    if expanded_topk_weights.numel() != num_rows:
        raise RuntimeError(
            "DeepEP expanded top-k weights do not match expanded rows: "
            f"weights={tuple(expanded_topk_weights.shape)} rows={num_rows}"
        )
    return expanded_topk_weights.reshape(num_rows, 1).to(dtype=dtype)


def _deepep_v2_rowwise_scale_triton(
    x: torch.Tensor,
    weights: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    block_m: int = 16,
    block_n: int = 256,
) -> torch.Tensor:
    if triton is None or tl is None:
        raise RuntimeError("Triton is required for --deepep-weighting post_triton")
    if x.ndim != 2:
        raise ValueError(f"expected x rank-2 [rows, hidden], got {tuple(x.shape)}")
    if weights.numel() != x.shape[0]:
        raise ValueError(
            "rowwise scale weights must have one value per row: "
            f"weights={tuple(weights.shape)} rows={x.shape[0]}"
        )
    if not x.is_cuda:
        raise ValueError("rowwise scale Triton path requires CUDA input")
    if weights.device != x.device:
        raise ValueError(f"weights device must match x device: {weights.device} vs {x.device}")
    if not x.is_contiguous():
        raise ValueError("rowwise scale Triton path requires contiguous x")
    weights_flat = weights.reshape(-1)
    if not weights_flat.is_contiguous():
        weights_flat = weights_flat.contiguous()
    if out is None:
        out = torch.empty_like(x)
    else:
        if tuple(out.shape) != tuple(x.shape):
            raise ValueError(f"out shape mismatch: expected {tuple(x.shape)}, got {tuple(out.shape)}")
        if out.device != x.device or out.dtype != x.dtype:
            raise ValueError("out device/dtype must match x")
        if not out.is_contiguous():
            raise ValueError("rowwise scale Triton path requires contiguous out")
    rows, hidden = x.shape
    if rows == 0 or hidden == 0:
        return out
    grid = (triton.cdiv(rows, int(block_m)), triton.cdiv(hidden, int(block_n)))
    _deepep_rowwise_scale_kernel[grid](
        x,
        weights_flat,
        out,
        rows,
        hidden,
        BLOCK_M=int(block_m),
        BLOCK_N=int(block_n),
        num_warps=4,
        num_stages=4,
    )
    return out


def _validate_deepep_v2_args(args: argparse.Namespace, *, world_size: int) -> None:
    if args.dtype != "bf16":
        raise RuntimeError("deepep_v2 currently supports --dtype bf16 only")
    if os.getenv("EP_REUSE_NCCL_COMM", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }:
        raise RuntimeError(
            "deepep_v2 currently requires EP_REUSE_NCCL_COMM=0 in this "
            "environment. Reusing Torch's NCCL communicator can load a mixed "
            "NCCL runtime and segfault during DeepEP ElasticBuffer setup."
        )
    if args.d_model % 256 != 0:
        raise RuntimeError(
            "deepep_v2 BF16 combine requires --d-model divisible by 256 "
            f"(got {args.d_model})."
        )
    if args.num_experts % world_size != 0:
        raise RuntimeError(
            f"deepep_v2 requires --num-experts divisible by ranks "
            f"({args.num_experts} vs {world_size})"
        )
    if args.deepep_expert_alignment < 1:
        raise RuntimeError("--deepep-expert-alignment must be >= 1")
    if args.deepep_max_tokens_factor < 1.0:
        raise RuntimeError("--deepep-max-tokens-factor must be >= 1.0")
    if args.deepep_weighting == "post_triton" and triton is None:
        raise RuntimeError("--deepep-weighting post_triton requires Triton")


def _build_deepep_v2_probe_routed_experts(
    args: argparse.Namespace,
    *,
    rank: int,
    world_size: int,
    config_dtype: DType,
    reset_seed: bool = True,
) -> torch.nn.Module:
    if reset_seed:
        torch.manual_seed(20260625 + rank)
    num_local_experts = args.num_experts // world_size
    routed_experts = RoutedExpertsConfig(
        d_model=args.d_model,
        hidden_size=args.hidden_size,
        num_experts=num_local_experts,
        bias=False,
        dtype=config_dtype,
    ).build(init_device="cuda")
    _init_probe_routed_expert_weights(
        routed_experts,
        weight_init=_resolve_probe_weight_init(args, source_default="normal"),
    )
    routed_experts.train()
    if args.compile and not args.no_compile:
        routed_experts.forward = torch.compile(  # type: ignore[method-assign]
            routed_experts.forward,
            fullgraph=False,
            dynamic=False,
        )
    return routed_experts


def _build_rowwise_apply_ep_probe_routed_experts(
    args: argparse.Namespace,
    *,
    world_size: int,
    ep_mesh: DeviceMesh,
    config_dtype: DType,
) -> torch.nn.Module:
    block = _build_block(
        d_model=args.d_model,
        hidden_size=args.hidden_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
        capacity_factor=args.capacity_factor,
        rowwise_nblocks=args.rowwise_nblocks,
        rowwise_wave=False,
        rowwise_wave_num_waves=args.rowwise_wave_num_waves,
        rowwise_wave_recompute_linear1=args.rowwise_wave_recompute_linear1,
        rowwise_wave_recompute_act=args.rowwise_wave_recompute_act,
        include_shared_expert=not args.no_shared_expert,
        shared_hidden_size=args.shared_hidden_size,
        uniform_routing=not args.random_routing,
        random_routing=args.random_routing,
        config_dtype=config_dtype,
    )
    if not args.full_block:
        _patch_moe_only(block)
    block.apply_ep(ep_mesh)
    assert block.routed_experts is not None
    _init_probe_routed_expert_weights(
        block.routed_experts,
        weight_init=_resolve_probe_weight_init(args, source_default="empty"),
    )
    if args.balanced_routing == "deepep":
        if args.random_routing:
            raise RuntimeError("--balanced-routing deepep conflicts with --random-routing")
        _install_deepep_balanced_router(block, world_size=world_size)
    block.train()
    if args.compile and not args.no_compile:
        if args.compile_block:
            block = torch.compile(block, fullgraph=False, dynamic=False)
        else:
            _compile_hot_modules(block)
    if block.routed_experts is None:
        raise RuntimeError("rowwise_apply_ep probe failed to build routed experts")
    return block.routed_experts


def _validate_deepep_v2_weighting_module(
    args: argparse.Namespace,
    routed_experts: torch.nn.Module,
) -> None:
    if args.deepep_weighting != "swiglu":
        return
    if getattr(routed_experts, "b_down", None) is not None:
        raise RuntimeError(
            "--deepep-weighting swiglu requires bias-free routed experts. "
            "The pre-down row-weight multiply is only equivalent to post-Linear2 "
            "weighting when the down projection has no bias. Use bias=False "
            "when building RoutedExperts, or use --deepep-weighting post/post_triton."
        )


def _build_deepep_v2_state(
    args: argparse.Namespace,
    *,
    tokens: int,
    rank: int,
    world_size: int,
    config_dtype: DType,
    input_dtype: torch.dtype,
) -> DeepEpV2State:
    _validate_deepep_v2_args(args, world_size=world_size)

    deep_ep = _import_deepep(args.deepep_path)
    num_max_tokens_per_rank = int(math.ceil(tokens * args.deepep_max_tokens_factor))
    num_allocated_qps = max(args.deepep_num_allocated_qps, args.deepep_num_qps)
    buffer = deep_ep.ElasticBuffer(
        dist.group.WORLD,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=args.d_model,
        num_topk=args.top_k,
        deterministic=False,
        allow_hybrid_mode=args.deepep_allow_hybrid_mode,
        allow_multiple_reduction=args.deepep_allow_multiple_reduction,
        prefer_overlap_with_compute=args.deepep_prefer_overlap_with_compute,
        num_allocated_qps=num_allocated_qps,
        explicitly_destroy=True,
    )
    num_sms = (
        int(args.deepep_num_sms)
        if args.deepep_num_sms != 0
        else int(buffer.get_theoretical_num_sms(args.num_experts, args.top_k))
    )
    num_qps = (
        int(args.deepep_num_qps)
        if args.deepep_num_qps != 0
        else int(buffer.get_theoretical_num_qps(num_sms))
    )

    torch.manual_seed(20260625 + rank)
    source_input = (0.2 * torch.randn(tokens, args.d_model, device="cuda")).to(input_dtype)
    topk_idx = _make_balanced_topk_idx(
        tokens=tokens,
        top_k=args.top_k,
        num_experts=args.num_experts,
        world_size=world_size,
        rank=rank,
        dtype=deep_ep.topk_idx_t,
    )
    topk_weights = torch.full(
        (tokens, args.top_k),
        1.0 / float(args.top_k),
        device="cuda",
        dtype=torch.float32,
    )

    routed_experts = _build_deepep_v2_probe_routed_experts(
        args,
        rank=rank,
        world_size=world_size,
        config_dtype=config_dtype,
        reset_seed=False,
    )
    _validate_deepep_v2_weighting_module(args, routed_experts)

    return DeepEpV2State(
        deep_ep=deep_ep,
        buffer=buffer,
        routed_experts=routed_experts,
        source_input=source_input,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        rank=rank,
        world_size=world_size,
        num_experts=args.num_experts,
        num_local_experts=args.num_experts // world_size,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        expert_alignment=args.deepep_expert_alignment,
        num_sms=num_sms,
        num_qps=num_qps,
        expert_buffer_mode=str(args.deepep_expert_buffer_mode),
        weighting_mode=str(args.deepep_weighting),
        async_with_compute_stream=bool(args.deepep_async),
        do_cpu_sync=bool(args.deepep_do_cpu_sync),
        wave_dispatch_stream=torch.cuda.Stream(),
        wave_compute_stream=torch.cuda.Stream(),
        wave_combine_stream=torch.cuda.Stream(),
    )


def _deepep_v2_dispatch(
    state: DeepEpV2State,
    *,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    label: str,
    async_with_compute_stream: bool,
    do_cpu_sync: bool,
    wait_for_completion: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, object, object]:
    torch.cuda.nvtx.range_push(label)
    try:
        recv_x, _recv_topk_idx, expanded_topk_weights, handle, event = state.buffer.dispatch(
            state.source_input,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_experts=state.num_experts,
            num_max_tokens_per_rank=state.num_max_tokens_per_rank,
            expert_alignment=state.expert_alignment,
            num_sms=state.num_sms,
            num_qps=state.num_qps,
            async_with_compute_stream=async_with_compute_stream,
            do_cpu_sync=do_cpu_sync,
            do_expand=True,
            use_tma_aligned_col_major_sf=True,
        )
        if wait_for_completion:
            _deep_ep_wait(event, async_with_compute_stream=async_with_compute_stream)
    finally:
        torch.cuda.nvtx.range_pop()
    return recv_x, expanded_topk_weights, handle, event


def _deepep_v2_dispatch_static_expanded(
    state: DeepEpV2State,
    *,
    wave: DeepEpV2WaveInput,
    recv_x_out: torch.Tensor,
    recv_topk_weights_out: torch.Tensor,
    label: str,
    async_with_compute_stream: bool,
    do_cpu_sync: bool,
    wait_for_completion: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, object, object]:
    if not hasattr(state.buffer, "dispatch_expanded_into"):
        raise RuntimeError(
            "deepep_v2_wave layout 'expand_static' requires the modified "
            "DeepEP working copy with ElasticBuffer.dispatch_expanded_into. "
            "Use --deepep-path /workspace/DeepEP."
        )

    torch.cuda.nvtx.range_push(label)
    try:
        recv_x, _recv_topk_idx, expanded_topk_weights, handle, event = (
            state.buffer.dispatch_expanded_into(
                state.source_input,
                topk_idx=wave.topk_idx,
                topk_weights=wave.topk_weights,
                recv_x_out=recv_x_out,
                recv_topk_weights_out=recv_topk_weights_out,
                expanded_row_offset=wave.wave_base,
                num_experts=state.num_experts,
                num_max_tokens_per_rank=state.num_max_tokens_per_rank,
                expert_alignment=state.expert_alignment,
                num_sms=state.num_sms,
                num_qps=state.num_qps,
                async_with_compute_stream=async_with_compute_stream,
                do_cpu_sync=do_cpu_sync,
            )
        )
        if wait_for_completion:
            _deep_ep_wait(event, async_with_compute_stream=async_with_compute_stream)
    finally:
        torch.cuda.nvtx.range_pop()
    return recv_x, expanded_topk_weights, handle, event


def _deepep_v2_dispatch_nonexpanded(
    state: DeepEpV2State,
    *,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    label: str,
    async_with_compute_stream: bool,
    do_cpu_sync: bool,
    wait_for_completion: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, object, object]:
    torch.cuda.nvtx.range_push(label)
    try:
        recv_x, recv_topk_idx, recv_topk_weights, handle, event = state.buffer.dispatch(
            state.source_input,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_experts=state.num_experts,
            num_max_tokens_per_rank=state.num_max_tokens_per_rank,
            expert_alignment=state.expert_alignment,
            num_sms=state.num_sms,
            num_qps=state.num_qps,
            async_with_compute_stream=async_with_compute_stream,
            do_cpu_sync=do_cpu_sync,
            do_expand=False,
            use_tma_aligned_col_major_sf=True,
        )
        if wait_for_completion:
            _deep_ep_wait(event, async_with_compute_stream=async_with_compute_stream)
    finally:
        torch.cuda.nvtx.range_pop()
    return recv_x, recv_topk_idx, recv_topk_weights, handle, event


def _deepep_v2_compute_experts(
    state: DeepEpV2State,
    *,
    recv_x: torch.Tensor,
    expanded_topk_weights: torch.Tensor | None,
    handle: object,
    label: str,
    track_expert_grad: bool,
    weight_in_swiglu: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    batch_size_per_expert = _expanded_expert_counts(handle, state.expert_alignment)
    recv_x_for_experts = recv_x.detach().requires_grad_(True) if track_expert_grad else recv_x
    expanded_weights = (
        _reshape_expanded_weights(
            expanded_topk_weights,
            num_rows=recv_x.shape[0],
            dtype=recv_x.dtype,
        )
        if weight_in_swiglu
        else None
    )
    down_proj_out = None
    up_proj_input_grad_out = None
    if state.expert_buffer_mode in {"down", "all"}:
        down_proj_out = torch.empty_like(recv_x)
    if state.expert_buffer_mode == "all":
        up_proj_input_grad_out = recv_x_for_experts.detach()

    torch.cuda.nvtx.range_push(label)
    try:
        expert_out = state.routed_experts(
            recv_x_for_experts,
            batch_size_per_expert,
            down_proj_out=down_proj_out,
            up_proj_input_grad_out=up_proj_input_grad_out,
            row_weights=expanded_weights,
        )
    finally:
        torch.cuda.nvtx.range_pop()

    if expanded_weights is None:
        expanded_weights = _reshape_expanded_weights(
            expanded_topk_weights,
            num_rows=expert_out.shape[0],
            dtype=expert_out.dtype,
        )
    return recv_x_for_experts, expert_out, expanded_weights, weight_in_swiglu


def _deepep_v2_compute_experts_static_expanded(
    state: DeepEpV2State,
    *,
    recv_x: torch.Tensor,
    expanded_topk_weights: torch.Tensor,
    weighted_expert_out: torch.Tensor,
    wave: DeepEpV2WaveInput,
    label: str,
    track_expert_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    if state.expert_alignment != 1:
        raise RuntimeError(
            "deepep_v2_wave layout 'expand_static' currently requires "
            "--deepep-expert-alignment 1. The current autograd RoutedExperts "
            "path consumes packed expert rows, while aligned static rows can "
            "contain padding between experts."
        )

    recv_x_slice = recv_x.narrow(0, wave.wave_base, wave.wave_rows)
    recv_x_for_experts = (
        recv_x_slice.detach().requires_grad_(True)
        if track_expert_grad
        else recv_x_slice
    )
    expanded_topk_weights_for_experts = expanded_topk_weights.narrow(
        0,
        wave.wave_base,
        wave.wave_rows,
    )
    expanded_weights = _reshape_expanded_weights(
        expanded_topk_weights_for_experts,
        num_rows=wave.wave_rows,
        dtype=recv_x_for_experts.dtype,
    )
    weight_in_swiglu = state.weighting_mode == "swiglu"
    down_proj_out = None
    if state.expert_buffer_mode in {"down", "all"}:
        down_proj_out = torch.empty_like(recv_x_for_experts)

    torch.cuda.nvtx.range_push(label)
    try:
        expert_out = state.routed_experts(
            recv_x_for_experts,
            wave.batch_size_per_expert,
            down_proj_out=down_proj_out,
            row_weights=expanded_weights if weight_in_swiglu else None,
        )
    finally:
        torch.cuda.nvtx.range_pop()

    weighted_slice = (
        expert_out
        if weight_in_swiglu
        else _deepep_v2_weight_expert_output(
            expert_out,
            expanded_weights,
            label=label,
            mode=state.weighting_mode,
        )
    )
    torch.cuda.nvtx.range_push(f"{label}/store_global_weighted")
    try:
        weighted_expert_out.narrow(0, wave.wave_base, wave.wave_rows).copy_(weighted_slice)
    finally:
        torch.cuda.nvtx.range_pop()

    return recv_x_for_experts, expert_out, expanded_weights, weight_in_swiglu


def _deepep_v2_compute_experts_nonexpanded(
    state: DeepEpV2State,
    *,
    recv_x: torch.Tensor,
    recv_topk_idx: torch.Tensor,
    recv_topk_weights: torch.Tensor,
    handle: object,
    label: str,
    track_expert_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recv_x_for_experts = recv_x.detach().requires_grad_(True) if track_expert_grad else recv_x

    torch.cuda.nvtx.range_push(f"{label}/pack_routes")
    try:
        recv_token_idx = torch.arange(
            recv_topk_idx.shape[0],
            device=recv_topk_idx.device,
            dtype=torch.long,
        ).view(-1, 1)
        valid_route_mask = (
            (recv_token_idx < _num_recv_tokens(handle, device=recv_topk_idx.device))
            & (recv_topk_idx >= 0)
            & (recv_topk_idx < state.num_local_experts)
        )
        route_token_idx, route_slot_idx = torch.nonzero(
            valid_route_mask,
            as_tuple=True,
        )
        route_expert_idx = recv_topk_idx[route_token_idx, route_slot_idx].to(torch.long)
        route_order = torch.argsort(route_expert_idx, stable=True)
        route_token_idx = route_token_idx[route_order]
        route_slot_idx = route_slot_idx[route_order]
        route_expert_idx = route_expert_idx[route_order]
        batch_size_per_expert = torch.bincount(
            route_expert_idx,
            minlength=state.num_local_experts,
        ).to(dtype=torch.int32)
        packed_x = recv_x_for_experts.index_select(0, route_token_idx)
    finally:
        torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push(label)
    try:
        route_weights = recv_topk_weights[route_token_idx, route_slot_idx].to(
            dtype=packed_x.dtype,
        ).view(-1, 1)
        expert_out = state.routed_experts(
            packed_x,
            batch_size_per_expert,
            row_weights=route_weights if state.weighting_mode == "swiglu" else None,
        )
    finally:
        torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push(f"{label}/local_reduce")
    try:
        weighted_route_out = (
            expert_out
            if state.weighting_mode == "swiglu"
            else _deepep_v2_weight_expert_output(
                expert_out,
                route_weights.to(dtype=expert_out.dtype),
                label=f"{label}/local_reduce",
                mode=state.weighting_mode,
            )
        )
        local_accum = torch.zeros_like(recv_x_for_experts)
        local_accum.index_add_(0, route_token_idx, weighted_route_out)
        local_accum_weights = torch.ones(
            (local_accum.shape[0], 1),
            device=local_accum.device,
            dtype=local_accum.dtype,
        )
    finally:
        torch.cuda.nvtx.range_pop()

    return recv_x_for_experts, local_accum, local_accum_weights


def _deepep_v2_weight_expert_output(
    expert_out: torch.Tensor,
    expanded_weights: torch.Tensor,
    *,
    label: str,
    mode: str = "post",
) -> torch.Tensor:
    torch.cuda.nvtx.range_push(f"{label}/weight_{mode}")
    try:
        if mode == "post_triton":
            weighted_expert_out = _deepep_v2_rowwise_scale_triton(
                expert_out,
                expanded_weights,
            )
        elif mode == "post":
            weighted_expert_out = expert_out * expanded_weights
        else:
            raise RuntimeError(f"post-Linear2 weighting does not support mode {mode!r}")
    finally:
        torch.cuda.nvtx.range_pop()
    return weighted_expert_out


def _deepep_v2_prepare_expert_grad(
    state: DeepEpV2State,
    *,
    grad_weighted_expert_out: torch.Tensor,
    expanded_weights: torch.Tensor,
    expert_out_is_weighted: bool,
    label: str,
) -> torch.Tensor:
    torch.cuda.nvtx.range_push(label)
    try:
        if expert_out_is_weighted:
            return grad_weighted_expert_out
        if state.weighting_mode == "post_triton":
            return _deepep_v2_rowwise_scale_triton(
                grad_weighted_expert_out,
                expanded_weights,
            )
        if state.weighting_mode == "post":
            return grad_weighted_expert_out * expanded_weights
        raise RuntimeError(
            f"--deepep-weighting {state.weighting_mode!r} expected weighted expert outputs"
        )
    finally:
        torch.cuda.nvtx.range_pop()


def _deepep_v2_combine(
    state: DeepEpV2State,
    *,
    weighted_expert_out: torch.Tensor,
    handle: object,
    label: str,
    async_with_compute_stream: bool,
    wait_for_completion: bool = True,
) -> tuple[torch.Tensor, object]:
    torch.cuda.nvtx.range_push(label)
    try:
        combined_x, _combined_topk_weights, event = state.buffer.combine(
            weighted_expert_out,
            handle=handle,
            num_sms=state.num_sms,
            num_qps=state.num_qps,
            async_with_compute_stream=async_with_compute_stream,
        )
        if wait_for_completion:
            _deep_ep_wait(event, async_with_compute_stream=async_with_compute_stream)
    finally:
        torch.cuda.nvtx.range_pop()
    return combined_x, event


def _deepep_v2_forward_from_topk(
    state: DeepEpV2State,
    *,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    label: str,
    track_expert_grad: bool,
    async_with_compute_stream: bool,
    do_cpu_sync: bool,
) -> DeepEpV2ForwardResult:
    recv_x, expanded_topk_weights, handle, _dispatch_event = _deepep_v2_dispatch(
        state,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        label=f"{label}/dispatch",
        async_with_compute_stream=async_with_compute_stream,
        do_cpu_sync=do_cpu_sync,
    )
    recv_x_for_experts, expert_out, expanded_weights, expert_out_is_weighted = _deepep_v2_compute_experts(
        state,
        recv_x=recv_x,
        expanded_topk_weights=expanded_topk_weights,
        handle=handle,
        label=f"{label}/experts",
        track_expert_grad=track_expert_grad,
        weight_in_swiglu=state.weighting_mode == "swiglu",
    )
    weighted_expert_out = (
        expert_out
        if expert_out_is_weighted
        else _deepep_v2_weight_expert_output(
            expert_out,
            expanded_weights,
            label=label,
            mode=state.weighting_mode,
        )
    )
    combined_x, _combine_event = _deepep_v2_combine(
        state,
        weighted_expert_out=weighted_expert_out,
        handle=handle,
        label=f"{label}/combine",
        async_with_compute_stream=async_with_compute_stream,
    )

    return DeepEpV2ForwardResult(
        recv_x=recv_x_for_experts,
        expanded_topk_weights=expanded_weights,
        expert_out=expert_out,
        combined_x=combined_x,
        handle=handle,
        expert_out_is_weighted=expert_out_is_weighted,
    )


def _deepep_v2_forward_from_topk_nonexpanded(
    state: DeepEpV2State,
    *,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    label: str,
    track_expert_grad: bool,
    async_with_compute_stream: bool,
    do_cpu_sync: bool,
) -> DeepEpV2ForwardResult:
    recv_x, recv_topk_idx, recv_topk_weights, handle, _dispatch_event = (
        _deepep_v2_dispatch_nonexpanded(
            state,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            label=f"{label}/dispatch_nonexpanded",
            async_with_compute_stream=async_with_compute_stream,
            do_cpu_sync=do_cpu_sync,
        )
    )
    recv_x_for_experts, local_accum, local_accum_weights = (
        _deepep_v2_compute_experts_nonexpanded(
            state,
            recv_x=recv_x,
            recv_topk_idx=recv_topk_idx,
            recv_topk_weights=recv_topk_weights,
            handle=handle,
            label=f"{label}/experts_nonexpanded",
            track_expert_grad=track_expert_grad,
        )
    )
    combined_x, _combine_event = _deepep_v2_combine(
        state,
        weighted_expert_out=local_accum,
        handle=handle,
        label=f"{label}/combine_nonexpanded",
        async_with_compute_stream=async_with_compute_stream,
    )

    return DeepEpV2ForwardResult(
        recv_x=recv_x_for_experts,
        expanded_topk_weights=local_accum_weights,
        expert_out=local_accum,
        combined_x=combined_x,
        handle=handle,
        expert_out_is_weighted=True,
    )


def _deepep_v2_forward(
    state: DeepEpV2State,
    *,
    label: str,
    track_expert_grad: bool,
) -> DeepEpV2ForwardResult:
    return _deepep_v2_forward_from_topk(
        state,
        topk_idx=state.topk_idx,
        topk_weights=state.topk_weights,
        label=label,
        track_expert_grad=track_expert_grad,
        async_with_compute_stream=state.async_with_compute_stream,
        do_cpu_sync=state.do_cpu_sync,
    )


def _prepare_deepep_v2_backward(
    state: DeepEpV2State,
    *,
    label: str,
) -> DeepEpV2ForwardResult:
    result = _deepep_v2_forward(
        state,
        label=f"{label}/forward_prep",
        track_expert_grad=True,
    )
    torch.cuda.nvtx.range_push(f"{label}/grad_prep")
    try:
        result.grad_combined_x = torch.ones_like(result.combined_x)
    finally:
        torch.cuda.nvtx.range_pop()
    return result


def _run_deepep_v2_backward_from_result(
    state: DeepEpV2State,
    result: DeepEpV2ForwardResult,
    *,
    label: str,
    zero_expert_grads: bool = True,
) -> torch.Tensor:
    if result.grad_combined_x is None:
        result.grad_combined_x = torch.ones_like(result.combined_x)

    torch.cuda.nvtx.range_push(f"{label}/combine_backward_dispatch")
    try:
        if getattr(result.handle, "do_expand", False) and hasattr(state.buffer, "dispatch_cached_expanded_into"):
            grad_weighted_expert_out = torch.empty_like(result.recv_x)
            _grad_weighted_expert_out, _grad_topk_idx, _grad_topk_weights, _handle, event = (
                state.buffer.dispatch_cached_expanded_into(
                    result.grad_combined_x,
                    handle=result.handle,
                    recv_x_out=grad_weighted_expert_out,
                    num_sms=state.num_sms,
                    num_qps=state.num_qps,
                    async_with_compute_stream=state.async_with_compute_stream,
                )
            )
        else:
            grad_weighted_expert_out, _grad_topk_idx, _grad_topk_weights, _handle, event = state.buffer.dispatch(
                result.grad_combined_x,
                handle=result.handle,
                num_sms=state.num_sms,
                num_qps=state.num_qps,
                async_with_compute_stream=state.async_with_compute_stream,
            )
        _deep_ep_wait(event, async_with_compute_stream=state.async_with_compute_stream)
    finally:
        torch.cuda.nvtx.range_pop()

    grad_expert_out = _deepep_v2_prepare_expert_grad(
        state,
        grad_weighted_expert_out=grad_weighted_expert_out,
        expanded_weights=result.expanded_topk_weights,
        expert_out_is_weighted=result.expert_out_is_weighted,
        label=f"{label}/prepare_expert_grad",
    )

    torch.cuda.nvtx.range_push(f"{label}/experts_backward")
    try:
        torch.autograd.backward(result.expert_out, grad_expert_out)
    finally:
        torch.cuda.nvtx.range_pop()

    if result.recv_x.grad is None:
        raise RuntimeError("deepep_v2 expert backward did not produce grad for recv_x")

    torch.cuda.nvtx.range_push(f"{label}/dispatch_backward_combine")
    try:
        combined_grad_x, _combined_grad_topk_weights, event = state.buffer.combine(
            result.recv_x.grad,
            handle=result.handle,
            num_sms=state.num_sms,
            num_qps=state.num_qps,
            async_with_compute_stream=state.async_with_compute_stream,
        )
        _deep_ep_wait(event, async_with_compute_stream=state.async_with_compute_stream)
    finally:
        torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push(f"{label}/zero_grad")
    try:
        result.recv_x.grad = None
        if zero_expert_grads:
            state.routed_experts.zero_grad(set_to_none=True)
    finally:
        torch.cuda.nvtx.range_pop()
    return combined_grad_x


def _run_one_deepep_v2_iter(
    state: DeepEpV2State,
    *,
    label: str,
    pass_type: str,
) -> None:
    if pass_type == "forward":
        with torch.no_grad():
            _deepep_v2_forward(
                state,
                label=label,
                track_expert_grad=False,
            )
        return

    result = _deepep_v2_forward(
        state,
        label=f"{label}/forward",
        track_expert_grad=True,
    )
    _run_deepep_v2_backward_from_result(
        state,
        result,
        label=f"{label}/backward",
    )
