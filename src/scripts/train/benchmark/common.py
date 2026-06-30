from __future__ import annotations

import os
import time
import types
from typing import Iterable

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from olmo_core.config import DType
from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlock
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.moe import MoERouterGatingFunction
from olmo_core.nn.moe.v2.ep_config import ExpertParallelConfig, ExpertParallelPath
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig


def _dtype_config(name: str) -> tuple[DType, torch.dtype]:
    if name == "bf16":
        return DType.bfloat16, torch.bfloat16
    if name == "fp32":
        return DType.float32, torch.float32
    raise ValueError(name)


def _init_dist() -> tuple[int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    _setup_debug_print("init_dist:enter", local_rank=local_rank)
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        _setup_debug_print("init_process_group:enter", local_rank=local_rank)
        dist.init_process_group("nccl")
        _setup_debug_print("init_process_group:exit", local_rank=local_rank)
    _setup_debug_print(
        "init_dist:exit",
        rank=dist.get_rank(),
        local_rank=local_rank,
        world_size=dist.get_world_size(),
    )
    return dist.get_rank(), local_rank, dist.get_world_size()


def _setup_debug_enabled() -> bool:
    raw = os.getenv("OLMO_BENCH_SETUP_DEBUG", os.getenv("IBGDA_BENCH_DEBUG", "0"))
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _setup_debug_rank_enabled() -> bool:
    ranks = os.getenv("OLMO_BENCH_SETUP_DEBUG_RANKS")
    if not ranks:
        return True
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = int(os.getenv("RANK", "0") or "0")
    allowed = {int(part) for part in ranks.replace(",", " ").split() if part}
    return rank in allowed


def _setup_debug_print(label: str, **fields: object) -> None:
    if not _setup_debug_enabled() or not _setup_debug_rank_enabled():
        return
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else os.getenv("RANK", "?")
    local_rank = os.getenv("LOCAL_RANK", "?")
    parts = [
        "[BENCH_SETUP]",
        f"t={time.perf_counter():.6f}",
        f"rank={rank}",
        f"local_rank={local_rank}",
        label,
    ]
    parts.extend(f"{key}={value}" for key, value in fields.items())
    print(" ".join(parts), flush=True)


def _build_ep_mesh(world_size: int) -> DeviceMesh:
    mesh = torch.arange(world_size, dtype=torch.int).view(1, world_size)
    return DeviceMesh(
        device_type="cuda",
        mesh=mesh,
        mesh_dim_names=("ep_dp", "ep_mp"),
    )


def _build_block(
    *,
    d_model: int,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    capacity_factor: float,
    rowwise_nblocks: int,
    rowwise_wave: bool,
    rowwise_wave_num_waves: int,
    rowwise_wave_recompute_linear1: bool,
    rowwise_wave_recompute_act: bool,
    include_shared_expert: bool,
    shared_hidden_size: int,
    uniform_routing: bool,
    random_routing: bool,
    config_dtype: DType,
) -> OLMoDDPTransformerBlock:
    _setup_debug_print(
        "build_block:enter",
        d_model=d_model,
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k,
        rowwise_wave=rowwise_wave,
    )
    layer_norm = LayerNormConfig(
        name=LayerNormType.rms,
        eps=1e-6,
        bias=False,
        dtype=config_dtype,
    )
    block = OLMoDDPTransformerBlock(
        d_model=d_model,
        block_idx=0,
        n_layers=1,
        sequence_mixer=AttentionConfig(
            name=AttentionType.default,
            n_heads=2,
            n_kv_heads=2,
            bias=False,
            use_flash=False,
            dtype=config_dtype,
        ),
        attention_norm=layer_norm,
        routed_experts_router=MoERouterConfigV2(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            gating_function=MoERouterGatingFunction.softmax,
            uniform_expert_assignment=uniform_routing and not random_routing,
            random_expert_assignment=random_routing,
            lb_loss_weight=None,
            z_loss_weight=None,
            dtype=config_dtype,
        ),
        routed_experts=RoutedExpertsConfig(
            d_model=d_model,
            hidden_size=hidden_size,
            num_experts=num_experts,
            bias=False,
            dtype=config_dtype,
        ),
        shared_experts=(
            SharedExpertsConfig(
                d_model=d_model,
                hidden_size=shared_hidden_size,
                num_experts=1,
                bias=False,
                dtype=config_dtype,
            )
            if include_shared_expert
            else None
        ),
        shared_experts_router=None,
        feed_forward_norm=layer_norm,
        ep=ExpertParallelConfig(
            path=(
                ExpertParallelPath.rowwise_wave
                if rowwise_wave
                else ExpertParallelPath.rowwise_nvshmem
            ),
            capacity_factor=capacity_factor,
            major_align=1,
            rowwise_nblocks=rowwise_nblocks,
            rowwise_wave_num_waves=(
                rowwise_wave_num_waves if rowwise_wave else 1
            ),
            rowwise_wave_recompute_linear1=(
                rowwise_wave_recompute_linear1 if rowwise_wave else False
            ),
            rowwise_wave_recompute_act=(
                rowwise_wave_recompute_act if rowwise_wave else False
            ),
        ),
        init_device="cuda",
    )
    _setup_debug_print(
        "build_block:exit",
        cuda_allocated_gib=f"{torch.cuda.memory_allocated() / 1024**3:.2f}",
        cuda_reserved_gib=f"{torch.cuda.memory_reserved() / 1024**3:.2f}",
    )
    return block


def _patch_moe_only(block: OLMoDDPTransformerBlock) -> None:
    def ident_attn(self, block_inp, **kwargs):
        return block_inp

    def mlp_only(self, residual, mlp_out):
        return mlp_out

    block._checkpointed_res_norm_attn = types.MethodType(ident_attn, block)
    block._res_norm_attn = types.MethodType(ident_attn, block)
    block._res_norm_mlp = types.MethodType(mlp_only, block)


def _compile_hot_modules(block: OLMoDDPTransformerBlock) -> None:
    if block.routed_experts is not None:
        block.routed_experts.forward = torch.compile(  # type: ignore[method-assign]
            block.routed_experts.forward,
            fullgraph=False,
            dynamic=False,
        )
        block.routed_experts.forward_row_offset = torch.compile(  # type: ignore[method-assign]
            block.routed_experts.forward_row_offset,
            fullgraph=False,
            dynamic=False,
        )
    if block.shared_experts is not None:
        block.shared_experts.forward1 = torch.compile(  # type: ignore[method-assign]
            block.shared_experts.forward1,
            fullgraph=False,
            dynamic=False,
        )
        block.shared_experts.forward2 = torch.compile(  # type: ignore[method-assign]
            block.shared_experts.forward2,
            fullgraph=False,
            dynamic=False,
        )


def _install_deepep_balanced_router(
    block: OLMoDDPTransformerBlock,
    *,
    world_size: int,
) -> None:
    """Install the exact deterministic top-k formula used by deepep_v2."""

    def _make_forward(router):
        def _forward(local_x, scores_only, loss_div_factor=None):
            del loss_div_factor
            B, S, _ = local_x.shape
            if scores_only:
                return torch.ones(
                    B,
                    S,
                    router.num_experts,
                    device=local_x.device,
                    dtype=local_x.dtype,
                ), None, None, None

            if router.num_experts % world_size != 0:
                raise RuntimeError(
                    "deepep balanced routing requires num_experts divisible by "
                    f"world_size ({router.num_experts} vs {world_size})"
                )

            tokens = B * S
            local_experts = router.num_experts // world_size
            token_idx = torch.arange(
                tokens,
                device=local_x.device,
                dtype=torch.long,
            ).view(tokens, 1)
            slot_idx = torch.arange(
                router.top_k,
                device=local_x.device,
                dtype=torch.long,
            ).view(1, router.top_k)
            expert_rank = (token_idx + dist.get_rank() + slot_idx) % world_size
            local_expert = (token_idx * router.top_k + slot_idx) % local_experts
            expert_indices = (
                expert_rank * local_experts + local_expert
            ).view(B, S, router.top_k).contiguous()

            expert_weights = torch.full(
                (B, S, router.top_k),
                1.0 / float(router.top_k),
                device=local_x.device,
                dtype=local_x.dtype,
            )
            batch_size_per_expert = torch.bincount(
                expert_indices.reshape(-1),
                minlength=router.num_experts,
            ).to(dtype=torch.long)
            return expert_weights, expert_indices, batch_size_per_expert, None

        return _forward

    assert block.routed_experts_router is not None
    block.routed_experts_router.forward = _make_forward(block.routed_experts_router)


def _cuda_profiler_start() -> None:
    torch.cuda.cudart().cudaProfilerStart()


def _cuda_profiler_stop() -> None:
    torch.cuda.cudart().cudaProfilerStop()


def _run_one_iter(
    block: OLMoDDPTransformerBlock,
    *,
    tokens: int,
    d_model: int,
    input_dtype: torch.dtype,
    label: str,
    pass_type: str,
    static_input: torch.Tensor | None = None,
) -> None:
    if static_input is None:
        torch.cuda.nvtx.range_push(f"{label}/input")
        try:
            x = torch.randn(
                1,
                tokens,
                d_model,
                device="cuda",
                dtype=input_dtype,
                requires_grad=(pass_type in ("backward", "forward_backward")),
            )
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        if pass_type != "forward":
            raise RuntimeError("static_input is only supported for forward profiling")
        x = static_input

    torch.cuda.nvtx.range_push(f"{label}/forward")
    try:
        if pass_type == "forward":
            with torch.no_grad():
                y = block(x)
        else:
            y = block(x)
    finally:
        torch.cuda.nvtx.range_pop()

    if pass_type == "forward":
        return

    torch.cuda.nvtx.range_push(f"{label}/loss")
    try:
        loss = y.square().mean()
    finally:
        torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push(f"{label}/backward")
    try:
        loss.backward()
    finally:
        torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push(f"{label}/zero_grad")
    try:
        block.zero_grad(set_to_none=True)
    finally:
        torch.cuda.nvtx.range_pop()


def _prepare_backward_loss(
    block: OLMoDDPTransformerBlock,
    *,
    tokens: int,
    d_model: int,
    input_dtype: torch.dtype,
    label: str,
) -> torch.Tensor:
    torch.cuda.nvtx.range_push(f"{label}/input")
    try:
        x = torch.randn(
            1,
            tokens,
            d_model,
            device="cuda",
            dtype=input_dtype,
            requires_grad=True,
        )
    finally:
        torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push(f"{label}/forward_prep")
    try:
        y = block(x)
    finally:
        torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push(f"{label}/loss_prep")
    try:
        return y.square().mean()
    finally:
        torch.cuda.nvtx.range_pop()


def _run_backward_from_loss(
    block: OLMoDDPTransformerBlock,
    loss: torch.Tensor,
    *,
    label: str,
) -> None:
    torch.cuda.nvtx.range_push(f"{label}/backward")
    try:
        loss.backward()
    finally:
        torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push(f"{label}/zero_grad")
    try:
        block.zero_grad(set_to_none=True)
    finally:
        torch.cuda.nvtx.range_pop()


def _median_rank_ms(values: Iterable[torch.Tensor]) -> float:
    return max(float(v[0].item()) for v in values)
