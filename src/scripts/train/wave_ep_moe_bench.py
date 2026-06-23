"""
Benchmark OLMo rowwise EP against the OLMo-owned fused BF16 MegaMoE target.

Example:
    PYTHONPATH=src torchrun --standalone --nproc-per-node=4 \
        src/scripts/train/wave_ep_moe_bench.py

Nsight Systems example:
    PYTHONPATH=src nsys profile -o wave_ep_moe_ep4 \
        --capture-range=cudaProfilerApi --capture-range-end=stop \
        torchrun --standalone --nproc-per-node=4 \
        src/scripts/train/wave_ep_moe_bench.py --modes rowwise --profile

Modes:
    rowwise: baseline rowwise NVSHMEM EP.
    wave: model-facing forward-only OLMo wave bring-up path.
    standard_ep_mega: standalone standard-shape EP4/32-expert fused BF16
        MegaMoE megakernel path. This is a kernel benchmark, not model wiring.
    standard_ep_mega_peer_group: standalone EP4 rank-local BF16 MegaMoE path
        with real symmetric peer workspaces and normal per-rank kernel launch.
    standard_ep_mega_collective: standalone EP4/world4 rank-local BF16
        MegaMoE NVSHMEM world collective-launch diagnostic path.
"""

from __future__ import annotations

import argparse
import os
import statistics
import types
from dataclasses import dataclass
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


@dataclass(frozen=True)
class BenchCase:
    name: str
    use_wave: bool
    use_bf16_persistent_mega: bool = False
    kernel_standard_ep_mega: bool = False
    kernel_standard_ep_mega_peer_group: bool = False
    kernel_standard_ep_mega_collective: bool = False
    kernel_standard_ep_mega_umma: bool = False


def _parse_modes(raw: str) -> list[BenchCase]:
    cases: list[BenchCase] = []
    for part in raw.split(","):
        mode = part.strip().lower()
        if not mode:
            continue
        if mode == "rowwise":
            cases.append(BenchCase("rowwise", False))
        elif mode in {
            "wave",
            "olmo_wave",
            "bf16_persistent_mega",
            "persistent_mega",
            "wave_bf16_persistent",
            "wave_bf16_persistent_mega",
        }:
            cases.append(BenchCase("bf16_persistent_mega", True, True))
        elif mode in {
            "standard_ep_mega",
            "standard_ep_full_mega",
            "kernel_standard_ep_mega",
            "kernel_mega",
        }:
            cases.append(BenchCase("standard_ep_mega", False, False, True))
        elif mode in {
            "standard_ep_mega_umma",
            "standard_ep_full_mega_umma",
            "kernel_standard_ep_mega_umma",
            "kernel_mega_umma",
        }:
            cases.append(BenchCase("standard_ep_mega_umma", False, False, True, False, False, True))
        elif mode in {
            "standard_ep_mega_peer_group",
            "standard_ep_peer_group",
            "kernel_standard_ep_peer_group",
            "kernel_mega_peer_group",
        }:
            cases.append(
                BenchCase(
                    "standard_ep_mega_peer_group",
                    False,
                    False,
                    False,
                    True,
                    False,
                )
            )
        elif mode in {
            "standard_ep_mega_peer_group_umma",
            "standard_ep_peer_group_umma",
            "kernel_standard_ep_peer_group_umma",
            "kernel_mega_peer_group_umma",
        }:
            cases.append(
                BenchCase(
                    "standard_ep_mega_peer_group_umma",
                    False,
                    False,
                    False,
                    True,
                    False,
                    True,
                )
            )
        elif mode in {
            "standard_ep_mega_collective",
            "standard_ep_collective",
            "kernel_standard_ep_collective",
            "kernel_mega_collective",
        }:
            cases.append(
                BenchCase(
                    "standard_ep_mega_collective",
                    False,
                    False,
                    False,
                    False,
                    True,
                )
            )
        elif mode in {
            "standard_ep_mega_collective_umma",
            "standard_ep_collective_umma",
            "kernel_standard_ep_collective_umma",
            "kernel_mega_collective_umma",
        }:
            cases.append(
                BenchCase(
                    "standard_ep_mega_collective_umma",
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                )
            )
        else:
            raise ValueError(
                f"Unknown mode {mode!r}. Expected rowwise,wave,bf16_persistent_mega,"
                "standard_ep_mega,standard_ep_mega_umma,"
                "standard_ep_mega_peer_group,standard_ep_mega_peer_group_umma,"
                "standard_ep_mega_collective,standard_ep_mega_collective_umma"
            )
    if not cases:
        raise ValueError("At least one benchmark mode is required")
    return cases


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "MoE-only OLMo EP benchmark. By default it patches dense attention "
            "to identity so large token counts remain an EP/MoE measurement."
        )
    )
    parser.add_argument("--tokens", type=int, nargs="+", default=[16384])
    parser.add_argument(
        "--modes",
        type=str,
        default="rowwise",
        help=(
            "Comma-separated modes. 'rowwise' is the current baseline. "
            "'wave'/'bf16_persistent_mega' select the model-facing forward-only "
            "wave path. 'standard_ep_mega' runs the standalone standard-shape "
            "fused BF16 megakernel. Suffix standard_ep_mega modes with '_umma' "
            "to use the 128x128x64 BF16 TMA/UMMA compute branch. "
            "'standard_ep_mega_peer_group' runs the rank-local peer-workspace "
            "kernel and requires 4 ranks. 'standard_ep_mega_collective' runs "
            "the rank-local NVSHMEM collective-launch diagnostic path and "
            "requires 4 ranks."
        ),
    )
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--d-model", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--num-experts", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--capacity-factor", type=float, default=1.25)
    parser.add_argument("--rowwise-nblocks", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument(
        "--pass-type",
        choices=("forward", "forward_backward"),
        default="forward_backward",
        help="Measure forward only or training-style forward+backward.",
    )
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile on expert modules.")
    parser.add_argument("--no-compile", action="store_true", help="Deprecated no-op; compile is disabled by default.")
    parser.add_argument(
        "--compile-block",
        action="store_true",
        help="With --compile, compile the entire block instead of only expert modules.",
    )
    parser.add_argument("--no-shared-expert", action="store_true")
    parser.add_argument("--shared-hidden-size", type=int, default=4096)
    parser.add_argument(
        "--full-block",
        action="store_true",
        help="Do not patch attention/residual helpers to identity.",
    )
    parser.add_argument(
        "--random-routing",
        action="store_true",
        help="Use router random_expert_assignment=True instead of uniform assignment.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Wrap measured iterations in cudaProfilerStart/Stop for nsys capture.",
    )
    parser.add_argument(
        "--check-standard-ep-umma-parity",
        action="store_true",
        help=(
            "For standard_ep_mega*_umma modes, run one untimed WMMA baseline "
            "iteration and assert UMMA output parity before timing."
        ),
    )
    parser.add_argument(
        "--standard-ep-umma-parity-atol",
        type=float,
        default=2e-2,
        help="Absolute tolerance for --check-standard-ep-umma-parity.",
    )
    return parser.parse_args()


def _dtype_config(name: str) -> tuple[DType, torch.dtype]:
    if name == "bf16":
        return DType.bfloat16, torch.bfloat16
    if name == "fp32":
        return DType.float32, torch.float32
    raise ValueError(name)


def _init_dist() -> tuple[int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    return dist.get_rank(), local_rank, dist.get_world_size()


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
    use_wave: bool,
    use_bf16_persistent_mega: bool,
    include_shared_expert: bool,
    shared_hidden_size: int,
    uniform_routing: bool,
    random_routing: bool,
    config_dtype: DType,
) -> OLMoDDPTransformerBlock:
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
            path=ExpertParallelPath.wave_mega
            if use_wave
            else ExpertParallelPath.rowwise_nvshmem,
            capacity_factor=capacity_factor,
            major_align=1,
            rowwise_nblocks=rowwise_nblocks,
            wave_use_bf16_persistent_mega_forward=(
                use_bf16_persistent_mega and use_wave
            ),
        ),
        init_device="cuda",
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
) -> None:
    torch.cuda.nvtx.range_push(f"{label}/input")
    try:
        x = torch.randn(
            1,
            tokens,
            d_model,
            device="cuda",
            dtype=input_dtype,
            requires_grad=(pass_type == "forward_backward"),
        )
    finally:
        torch.cuda.nvtx.range_pop()

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


@dataclass
class StandardEpMegaKernelState:
    source_input: torch.Tensor
    route_expert_indices: torch.Tensor
    probs: torch.Tensor
    up_gate_weight: torch.Tensor
    down_weight: torch.Tensor
    gathered_out: torch.Tensor
    out: torch.Tensor
    workspace_config: dict[str, int]
    workspace: torch.Tensor
    rank_workspace_bases: torch.Tensor
    global_counts: torch.Tensor
    global_offsets: torch.Tensor
    expert_cursors: torch.Tensor
    packed_route: torch.Tensor
    route_to_slot: torch.Tensor
    packed_input: torch.Tensor
    h: torch.Tensor
    packed_expert_out: torch.Tensor
    barrier_state: torch.Tensor
    w1_up: torch.Tensor | None = None
    w1_gate: torch.Tensor | None = None


def _build_standard_ep_mega_kernel_state(
    args: argparse.Namespace,
    *,
    tokens: int,
    rank: int,
    world_size: int,
    peer_group: bool,
    collective: bool,
    umma: bool = False,
) -> StandardEpMegaKernelState:
    if args.dtype != "bf16":
        raise RuntimeError("standard_ep_mega currently supports --dtype bf16 only")
    if args.pass_type != "forward":
        raise RuntimeError("standard_ep_mega currently supports --pass-type forward only")
    if args.num_experts != 32 or args.top_k != 4:
        raise RuntimeError("standard_ep_mega requires --num-experts 32 --top-k 4")
    if tokens > 16384:
        raise RuntimeError("standard_ep_mega supports at most --tokens 16384")
    if (peer_group or collective) and world_size != 4:
        raise RuntimeError(
            "standard_ep_mega_peer_group/collective requires torchrun --nproc-per-node=4"
        )

    from olmo_core.kernels.symm_mem_vdev2d import (
        rowwise_bf16_mega_moe_standard_ep_workspace_config,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(20260623 + rank)
    source_input = (0.2 * torch.randn(tokens, args.d_model, device=device)).to(torch.bfloat16)
    token_idx = torch.arange(tokens, device=device, dtype=torch.long).view(tokens, 1)
    topk_idx = torch.arange(args.top_k, device=device, dtype=torch.long).view(1, args.top_k)
    if peer_group or collective:
        route_expert_indices = (
            topk_idx * 8 + (token_idx + rank + topk_idx) % 8
        ).to(dtype=torch.long)
    else:
        route_expert_indices = (token_idx * args.top_k + topk_idx + rank) % args.num_experts
    probs = torch.full(
        (tokens, args.top_k),
        1.0 / float(args.top_k),
        device=device,
        dtype=torch.float32,
    )
    torch.manual_seed(20260623)
    up_gate_weight = (
        0.15
        * torch.randn(
            args.num_experts,
            2 * args.hidden_size,
            args.d_model,
            device=device,
        )
    ).to(torch.bfloat16)
    down_weight = (
        0.15
        * torch.randn(
            args.num_experts,
            args.hidden_size,
            args.d_model,
            device=device,
        )
    ).to(torch.bfloat16)
    gathered_out = torch.empty(
        (tokens, args.top_k, args.d_model),
        device=device,
        dtype=torch.bfloat16,
    )
    out = torch.empty((tokens, args.d_model), device=device, dtype=torch.bfloat16)
    workspace_config = rowwise_bf16_mega_moe_standard_ep_workspace_config(
        num_tokens=tokens,
        hidden=args.d_model,
        intermediate=args.hidden_size,
    )
    if workspace_config["top_k"] != args.top_k:
        raise RuntimeError(
            f"standard_ep_mega workspace top_k={workspace_config['top_k']} "
            f"does not match --top-k {args.top_k}"
        )
    if workspace_config["num_total_experts"] != args.num_experts:
        raise RuntimeError(
            "standard_ep_mega workspace num_total_experts="
            f"{workspace_config['num_total_experts']} does not match "
            f"--num-experts {args.num_experts}"
        )
    num_route_slots = workspace_config["num_route_slots"]
    if num_route_slots != tokens * args.top_k:
        raise RuntimeError(
            f"standard_ep_mega workspace num_route_slots={num_route_slots} "
            f"does not match tokens*top_k={tokens * args.top_k}"
        )
    if workspace_config["packed_values"] % num_route_slots != 0:
        raise RuntimeError("standard_ep_mega workspace packed_values is not route-aligned")
    if workspace_config["h_values"] % num_route_slots != 0:
        raise RuntimeError("standard_ep_mega workspace h_values is not route-aligned")
    packed_hidden = workspace_config["packed_values"] // num_route_slots
    h_hidden = workspace_config["h_values"] // num_route_slots
    if packed_hidden != args.d_model or h_hidden != args.hidden_size:
        raise RuntimeError(
            "standard_ep_mega workspace tensor widths do not match "
            f"d_model/hidden_size ({packed_hidden}, {h_hidden}) vs "
            f"({args.d_model}, {args.hidden_size})"
        )
    if peer_group or collective:
        from olmo_core.kernels import olmo_symm_mem

        workspace = olmo_symm_mem.empty(
            (workspace_config["workspace_stride_bytes"],),
            dtype=torch.uint8,
            device=device,
            group=dist.group.WORLD,
        )
        rank_workspace_bases = olmo_symm_mem.peer_base_ptrs(
            workspace,
            group=dist.group.WORLD,
        )
        packed_capacity = workspace_config["local_packed_capacity"]
    else:
        workspace = torch.empty(
            (workspace_config["workspace_bytes"],),
            device=device,
            dtype=torch.uint8,
        )
        rank_workspace_bases = torch.empty(
            (workspace_config["num_ranks"],),
            device=device,
            dtype=torch.long,
        )
        packed_capacity = num_route_slots
        if umma:
            packed_capacity += workspace_config["num_total_experts"] * (128 - 1)
    global_counts = torch.empty(
        (workspace_config["num_total_experts"],),
        device=device,
        dtype=torch.long,
    )
    global_offsets = torch.empty(
        (workspace_config["num_total_experts"] + 1,),
        device=device,
        dtype=torch.long,
    )
    expert_cursors = torch.empty_like(global_counts)
    packed_route = torch.empty((packed_capacity,), device=device, dtype=torch.long)
    route_to_slot = torch.empty((num_route_slots,), device=device, dtype=torch.long)
    packed_input = torch.empty(
        (packed_capacity, packed_hidden),
        device=device,
        dtype=torch.bfloat16,
    )
    h = torch.empty((packed_capacity, h_hidden), device=device, dtype=torch.bfloat16)
    packed_expert_out = torch.empty_like(packed_input)
    w1_up = torch.empty_like(h) if umma else None
    w1_gate = torch.empty_like(h) if umma else None
    barrier_state = torch.empty(
        (workspace_config["barrier_state_len"],),
        device=device,
        dtype=torch.int32,
    )
    return StandardEpMegaKernelState(
        source_input=source_input,
        route_expert_indices=route_expert_indices.contiguous(),
        probs=probs.contiguous(),
        up_gate_weight=up_gate_weight.contiguous(),
        down_weight=down_weight.contiguous(),
        gathered_out=gathered_out,
        out=out,
        workspace_config=workspace_config,
        workspace=workspace,
        rank_workspace_bases=rank_workspace_bases,
        global_counts=global_counts,
        global_offsets=global_offsets,
        expert_cursors=expert_cursors,
        packed_route=packed_route,
        route_to_slot=route_to_slot,
        packed_input=packed_input,
        h=h,
        packed_expert_out=packed_expert_out,
        barrier_state=barrier_state,
        w1_up=w1_up,
        w1_gate=w1_gate,
    )


def _run_one_standard_ep_mega_kernel_iter(
    state: StandardEpMegaKernelState,
    *,
    label: str,
    rank: int,
    peer_group: bool,
    collective: bool,
    umma: bool = False,
) -> None:
    from olmo_core.kernels.symm_mem_vdev2d import (
        rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace,
        rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_collective_world_umma,
        rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_collective_world,
        rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_peer_group_umma,
        rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_peer_group,
        rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_umma,
    )

    if collective:
        nvtx_name = "standard_ep_forward_collective_world_umma" if umma else "standard_ep_forward_collective_world"
    elif peer_group:
        nvtx_name = "standard_ep_forward_peer_group_umma" if umma else "standard_ep_forward_peer_group"
    else:
        nvtx_name = "standard_ep_forward_persistent_workspace_umma" if umma else "standard_ep_forward_persistent_workspace"
    torch.cuda.nvtx.range_push(f"{label}/{nvtx_name}")
    try:
        if umma:
            if state.w1_up is None or state.w1_gate is None:
                raise RuntimeError("standard_ep UMMA state is missing W1 scratch tensors")
            if collective:
                rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_collective_world_umma(
                    state.source_input,
                    state.gathered_out,
                    state.out,
                    state.route_expert_indices,
                    state.probs,
                    state.up_gate_weight,
                    state.down_weight,
                    state.workspace,
                    state.rank_workspace_bases,
                    state.global_counts,
                    state.global_offsets,
                    state.expert_cursors,
                    state.packed_route,
                    state.route_to_slot,
                    state.packed_input,
                    state.h,
                    state.packed_expert_out,
                    state.barrier_state,
                    state.w1_up,
                    state.w1_gate,
                    caller_rank_idx=rank,
                )
            elif peer_group:
                rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_peer_group_umma(
                    state.source_input,
                    state.gathered_out,
                    state.out,
                    state.route_expert_indices,
                    state.probs,
                    state.up_gate_weight,
                    state.down_weight,
                    state.workspace,
                    state.rank_workspace_bases,
                    state.global_counts,
                    state.global_offsets,
                    state.expert_cursors,
                    state.packed_route,
                    state.route_to_slot,
                    state.packed_input,
                    state.h,
                    state.packed_expert_out,
                    state.barrier_state,
                    state.w1_up,
                    state.w1_gate,
                    caller_rank_idx=rank,
                )
            else:
                rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_umma(
                    state.source_input,
                    state.gathered_out,
                    state.out,
                    state.route_expert_indices,
                    state.probs,
                    state.up_gate_weight,
                    state.down_weight,
                    state.workspace,
                    state.rank_workspace_bases,
                    state.global_counts,
                    state.global_offsets,
                    state.expert_cursors,
                    state.packed_route,
                    state.route_to_slot,
                    state.packed_input,
                    state.h,
                    state.packed_expert_out,
                    state.barrier_state,
                    state.w1_up,
                    state.w1_gate,
                )
        elif collective:
            rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_collective_world(
                state.source_input,
                state.gathered_out,
                state.out,
                state.route_expert_indices,
                state.probs,
                state.up_gate_weight,
                state.down_weight,
                state.workspace,
                state.rank_workspace_bases,
                state.global_counts,
                state.global_offsets,
                state.expert_cursors,
                state.packed_route,
                state.route_to_slot,
                state.packed_input,
                state.h,
                state.packed_expert_out,
                state.barrier_state,
                caller_rank_idx=rank,
            )
        elif peer_group:
            rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_peer_group(
                state.source_input,
                state.gathered_out,
                state.out,
                state.route_expert_indices,
                state.probs,
                state.up_gate_weight,
                state.down_weight,
                state.workspace,
                state.rank_workspace_bases,
                state.global_counts,
                state.global_offsets,
                state.expert_cursors,
                state.packed_route,
                state.route_to_slot,
                state.packed_input,
                state.h,
                state.packed_expert_out,
                state.barrier_state,
                caller_rank_idx=rank,
            )
        else:
            rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace(
                state.source_input,
                state.gathered_out,
                state.out,
                state.route_expert_indices,
                state.probs,
                state.up_gate_weight,
                state.down_weight,
                state.workspace,
                state.rank_workspace_bases,
                state.global_counts,
                state.global_offsets,
                state.expert_cursors,
                state.packed_route,
                state.route_to_slot,
                state.packed_input,
                state.h,
                state.packed_expert_out,
                state.barrier_state,
            )
    finally:
        torch.cuda.nvtx.range_pop()

def _median_rank_ms(values: Iterable[torch.Tensor]) -> float:
    return max(float(v[0].item()) for v in values)


def _bench_standard_ep_mega_kernel_case(
    args: argparse.Namespace,
    *,
    tokens: int,
    rank: int,
    world_size: int,
    peer_group: bool = False,
    collective: bool = False,
    umma: bool = False,
) -> None:
    if collective:
        mode_name = "standard_ep_mega_collective_umma" if umma else "standard_ep_mega_collective"
    elif peer_group:
        mode_name = "standard_ep_mega_peer_group_umma" if umma else "standard_ep_mega_peer_group"
    else:
        mode_name = "standard_ep_mega_umma" if umma else "standard_ep_mega"
    if rank == 0:
        if collective:
            print(
                f"[bench] mode={mode_name} runs the standalone "
                "standard-shape EP4/world4 OLMo-owned BF16 rank-local "
                "NVSHMEM world collective-launch diagnostic path"
                f"{' with the 128x128x64 TMA/UMMA compute branch.' if umma else '.'}",
                flush=True,
            )
        elif peer_group:
            print(
                f"[bench] mode={mode_name} runs the standalone "
                "standard-shape EP4 OLMo-owned BF16 rank-local megakernel with "
                "real symmetric peer workspaces and normal per-rank launch"
                f"{' plus the 128x128x64 TMA/UMMA compute branch.' if umma else '.'}",
                flush=True,
            )
        else:
            print(
                f"[bench] mode={mode_name} runs the standalone standard-shape "
                "OLMo-owned BF16 fused MegaMoE megakernel. It is not model wiring "
                "or a distributed peer-window transport benchmark"
                f"{' and uses the 128x128x64 TMA/UMMA compute branch.' if umma else '.'}",
                flush=True,
            )

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.nvtx.range_push(f"BENCH/{mode_name}/tokens_{tokens}/build")
    try:
        state = _build_standard_ep_mega_kernel_state(
            args,
            tokens=tokens,
            rank=rank,
            world_size=world_size,
            peer_group=peer_group,
            collective=collective,
            umma=umma,
        )
    finally:
        torch.cuda.nvtx.range_pop()

    if umma and args.check_standard_ep_umma_parity:
        torch.cuda.nvtx.range_push(f"BENCH/{mode_name}/tokens_{tokens}/parity")
        try:
            baseline_state = _build_standard_ep_mega_kernel_state(
                args,
                tokens=tokens,
                rank=rank,
                world_size=world_size,
                peer_group=peer_group,
                collective=collective,
                umma=False,
            )
            _run_one_standard_ep_mega_kernel_iter(
                baseline_state,
                label=f"BENCH/{mode_name}/tokens_{tokens}/parity_wmma",
                rank=rank,
                peer_group=peer_group,
                collective=collective,
                umma=False,
            )
            _run_one_standard_ep_mega_kernel_iter(
                state,
                label=f"BENCH/{mode_name}/tokens_{tokens}/parity_umma",
                rank=rank,
                peer_group=peer_group,
                collective=collective,
                umma=True,
            )
            torch.cuda.synchronize()
            diff = (baseline_state.out.float() - state.out.float()).abs()
            local = torch.tensor(
                [float(diff.max().item()), float(diff.mean().item())],
                device="cuda",
                dtype=torch.float32,
            )
            global_diff = local.clone()
            dist.all_reduce(global_diff, op=dist.ReduceOp.MAX)
            if rank == 0:
                print(
                    "CHECK "
                    f"{mode_name}: tokens/rank={tokens} "
                    f"max_abs={float(global_diff[0].item()):.6g} "
                    f"max_mean_abs={float(global_diff[1].item()):.6g} "
                    f"atol={args.standard_ep_umma_parity_atol:.6g}",
                    flush=True,
                )
            if float(global_diff[0].item()) > args.standard_ep_umma_parity_atol:
                raise RuntimeError(
                    f"{mode_name} parity check failed: "
                    f"max_abs={float(global_diff[0].item()):.6g} "
                    f"> atol={args.standard_ep_umma_parity_atol:.6g}"
                )
            dist.barrier()
        finally:
            torch.cuda.nvtx.range_pop()

    for idx in range(args.warmup):
        label = f"BENCH/{mode_name}/tokens_{tokens}/warmup_{idx}"
        torch.cuda.nvtx.range_push(f"{label}/total")
        try:
            _run_one_standard_ep_mega_kernel_iter(
                state,
                label=label,
                rank=rank,
                peer_group=peer_group,
                collective=collective,
                umma=umma,
            )
        finally:
            torch.cuda.nvtx.range_pop()

    warmup_done = torch.cuda.Event(enable_timing=False)
    warmup_done.record()
    warmup_done.synchronize()

    if args.profile:
        dist.barrier()
        _cuda_profiler_start()

    events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    for idx in range(args.iters):
        label = f"BENCH/{mode_name}/tokens_{tokens}/iter_{idx}"
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.cuda.nvtx.range_push(f"{label}/total")
        try:
            _run_one_standard_ep_mega_kernel_iter(
                state,
                label=label,
                rank=rank,
                peer_group=peer_group,
                collective=collective,
                umma=umma,
            )
        finally:
            torch.cuda.nvtx.range_pop()
        end.record()
        events.append((start, end))

    if args.profile:
        _cuda_profiler_stop()
        dist.barrier()

    if events:
        events[-1][1].synchronize()
    times = [start.elapsed_time(end) for start, end in events]
    local_ms = statistics.median(times)
    local_mem_gib = torch.cuda.max_memory_allocated() / 1024**3
    local = torch.tensor([local_ms, local_mem_gib], device="cuda")
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)

    if rank == 0:
        max_ms = _median_rank_ms(gathered)
        max_mem_gib = max(float(v[1].item()) for v in gathered)
        print(
            "BENCH "
            f"{mode_name}: ranks={world_size} tokens/rank={tokens} "
            f"pass=forward kernel_only=True peer_group={peer_group} "
            f"collective={collective} dtype=bf16 "
            f"d={args.d_model} hidden={args.hidden_size} experts={args.num_experts} "
            f"top_k={args.top_k} "
            f"ms/iter(max_rank)={max_ms:.3f} "
            f"local_tokens/s={tokens / (max_ms / 1000.0):.1f} "
            f"max_mem_GiB={max_mem_gib:.2f}",
            flush=True,
        )


def _bench_case(
    args: argparse.Namespace,
    case: BenchCase,
    *,
    tokens: int,
    rank: int,
    world_size: int,
    ep_mesh: DeviceMesh,
) -> None:
    if case.kernel_standard_ep_mega:
        _bench_standard_ep_mega_kernel_case(
            args,
            tokens=tokens,
            rank=rank,
            world_size=world_size,
            umma=case.kernel_standard_ep_mega_umma,
        )
        return
    if case.kernel_standard_ep_mega_peer_group:
        _bench_standard_ep_mega_kernel_case(
            args,
            tokens=tokens,
            rank=rank,
            world_size=world_size,
            peer_group=True,
            umma=case.kernel_standard_ep_mega_umma,
        )
        return
    if case.kernel_standard_ep_mega_collective:
        _bench_standard_ep_mega_kernel_case(
            args,
            tokens=tokens,
            rank=rank,
            world_size=world_size,
            collective=True,
            umma=case.kernel_standard_ep_mega_umma,
        )
        return

    capacity_factor = args.capacity_factor
    if rank == 0 and case.use_wave:
        print(
            "[bench] mode=wave/bf16_persistent_mega selects the production "
            "OLMo-owned fused BF16 MegaMoE target through the model-facing "
            "forward-only bring-up path.",
            flush=True,
        )

    torch.manual_seed(20260619 + rank)
    torch.cuda.reset_peak_memory_stats()
    config_dtype, input_dtype = _dtype_config(args.dtype)

    torch.cuda.nvtx.range_push(f"BENCH/{case.name}/tokens_{tokens}/build")
    try:
        block = _build_block(
            d_model=args.d_model,
            hidden_size=args.hidden_size,
            num_experts=args.num_experts,
            top_k=args.top_k,
            capacity_factor=capacity_factor,
            rowwise_nblocks=args.rowwise_nblocks,
            use_wave=case.use_wave,
            use_bf16_persistent_mega=case.use_bf16_persistent_mega,
            include_shared_expert=not args.no_shared_expert,
            shared_hidden_size=args.shared_hidden_size,
            uniform_routing=not args.random_routing,
            random_routing=args.random_routing,
            config_dtype=config_dtype,
        )
        if not args.full_block:
            _patch_moe_only(block)
        block.apply_ep(ep_mesh)
        block.train()
        compile_enabled = bool(args.compile and not args.no_compile)
        if compile_enabled:
            if args.compile_block:
                block = torch.compile(block, fullgraph=False, dynamic=False)
            else:
                _compile_hot_modules(block)
    finally:
        torch.cuda.nvtx.range_pop()

    for idx in range(args.warmup):
        label = f"BENCH/{case.name}/tokens_{tokens}/warmup_{idx}"
        torch.cuda.nvtx.range_push(f"{label}/total")
        try:
            _run_one_iter(
                block,
                tokens=tokens,
                d_model=args.d_model,
                input_dtype=input_dtype,
                label=label,
                pass_type=args.pass_type,
            )
        finally:
            torch.cuda.nvtx.range_pop()
    warmup_done = torch.cuda.Event(enable_timing=False)
    warmup_done.record()
    warmup_done.synchronize()

    if args.profile:
        dist.barrier()
        _cuda_profiler_start()

    events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    for idx in range(args.iters):
        label = f"BENCH/{case.name}/tokens_{tokens}/iter_{idx}"
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.cuda.nvtx.range_push(f"{label}/total")
        try:
            _run_one_iter(
                block,
                tokens=tokens,
                d_model=args.d_model,
                input_dtype=input_dtype,
                label=label,
                pass_type=args.pass_type,
            )
        finally:
            torch.cuda.nvtx.range_pop()
        end.record()
        events.append((start, end))

    if args.profile:
        _cuda_profiler_stop()
        dist.barrier()

    if events:
        events[-1][1].synchronize()
    times = [start.elapsed_time(end) for start, end in events]
    local_ms = statistics.median(times)
    local_mem_gib = torch.cuda.max_memory_allocated() / 1024**3
    local = torch.tensor([local_ms, local_mem_gib], device="cuda")
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)

    if rank == 0:
        max_ms = _median_rank_ms(gathered)
        max_mem_gib = max(float(v[1].item()) for v in gathered)
        print(
            "BENCH "
            f"{case.name}: ranks={world_size} tokens/rank={tokens} "
            f"pass={args.pass_type} moe_only={not args.full_block} "
            f"shared={not args.no_shared_expert} dtype={args.dtype} "
            f"compile={'none' if not args.compile or args.no_compile else ('block' if args.compile_block else 'experts')} "
            f"d={args.d_model} hidden={args.hidden_size} experts={args.num_experts} "
            f"top_k={args.top_k} cap={capacity_factor} "
            f"ms/iter(max_rank)={max_ms:.3f} "
            f"local_tokens/s={tokens / (max_ms / 1000.0):.1f} "
            f"global_tokens/s={tokens * world_size / (max_ms / 1000.0):.1f} "
            f"max_mem_GiB={max_mem_gib:.2f}",
            flush=True,
        )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    args = _parse_args()
    rank, _, world_size = _init_dist()
    cases = _parse_modes(args.modes)
    ep_mesh = _build_ep_mesh(world_size)

    try:
        for tokens in args.tokens:
            for case in cases:
                _bench_case(
                    args,
                    case,
                    tokens=tokens,
                    rank=rank,
                    world_size=world_size,
                    ep_mesh=ep_mesh,
                )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
