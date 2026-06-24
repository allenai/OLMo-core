"""
Benchmark OLMo rowwise NVSHMEM EP against the OLMo-owned TMA/IBGDA backend.

Example:
    CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src \
      torchrun --standalone --nproc-per-node=4 \
        src/scripts/train/tma_ibgda_rowwise_bench.py \
        --modes rowwise,tma_ibgda --tokens 4096 --iters 5

This is a model-facing MoE block benchmark. By default it patches attention and
residual work to identity so measured time is dominated by routing, EP
transport, grouped expert MLPs, combine, and backward.
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
class BenchResult:
    mode: str
    local_ms: list[float]
    rank_mean_ms: float
    rank_p50_ms: float
    rank_max_ms: float
    world_mean_ms: float
    world_max_ms: float


def _parse_modes(raw: str) -> list[str]:
    modes: list[str] = []
    for part in raw.split(","):
        mode = part.strip().lower()
        if not mode:
            continue
        if mode not in {"rowwise", "nvshmem", "tma_ibgda", "tma"}:
            raise ValueError(
                f"Unknown mode {mode!r}. Expected comma-separated rowwise,tma_ibgda"
            )
        if mode == "nvshmem":
            mode = "rowwise"
        if mode == "tma":
            mode = "tma_ibgda"
        modes.append(mode)
    if not modes:
        raise ValueError("At least one mode is required")
    return modes


def _tma_ibgda_contract_for_log(modes: list[str]) -> str | None:
    if "tma_ibgda" not in modes:
        return None
    try:
        from olmo_core.kernels import tma_ibgda_ep

        if not tma_ibgda_ep.is_available():
            return "unavailable"
        contract = tma_ibgda_ep.extension_contract()
    except Exception as e:
        return f"unavailable:{type(e).__name__}:{e}"

    keys = (
        "route_record_bytes",
        "route_record_words",
        "workspace_alignment",
        "doorbell_bytes",
        "completion_bytes",
        "peer_window_layout_bytes",
        "kernel_launch_config_bytes",
        "bf16_only",
        "has_gpu_route_preprocess",
        "has_ibgda_dispatch",
        "has_tma_load_dispatch",
        "has_ibgda_combine",
        "has_route_dot_backward",
        "has_peer_window_layout_planner",
    )
    return ",".join(f"{key}={contract.get(key)!r}" for key in keys)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Model-facing rowwise NVSHMEM vs TMA/IBGDA MoE benchmark"
    )
    parser.add_argument("--modes", default="rowwise,tma_ibgda")
    parser.add_argument("--tokens", type=int, nargs="+", default=[4096])
    parser.add_argument("--d-model", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--capacity-factor", type=float, default=2.0)
    parser.add_argument("--rowwise-nblocks", type=int, default=256)
    parser.add_argument(
        "--tma-ibgda-num-sms",
        type=int,
        default=None,
        help=(
            "Common TMA/IBGDA launch block count fallback. Dispatch/combine "
            "default to --rowwise-nblocks when neither this nor their "
            "stage-specific knobs are set."
        ),
    )
    parser.add_argument(
        "--tma-ibgda-dispatch-num-sms",
        type=int,
        default=None,
        help="TMA/IBGDA dispatch launch block count.",
    )
    parser.add_argument(
        "--tma-ibgda-combine-num-sms",
        type=int,
        default=None,
        help="TMA/IBGDA combine launch block count.",
    )
    parser.add_argument(
        "--tma-ibgda-preprocess-num-sms",
        type=int,
        default=None,
        help="TMA/IBGDA route-preprocess launch block count.",
    )
    parser.add_argument(
        "--tma-ibgda-symmetric-expert-out",
        action="store_true",
        help="Write routed expert output directly into a TMA/IBGDA symmetric buffer.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument(
        "--pass-type",
        choices=("forward", "forward_backward"),
        default="forward_backward",
    )
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--no-shared-expert", action="store_true")
    parser.add_argument("--shared-hidden-size", type=int, default=2048)
    parser.add_argument("--full-block", action="store_true")
    parser.add_argument(
        "--random-routing",
        action="store_true",
        help="Use router random_expert_assignment instead of uniform routing.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Wrap measured iterations in cudaProfilerStart/Stop for nsys.",
    )
    args = parser.parse_args()
    if args.tma_ibgda_symmetric_expert_out and args.pass_type != "forward":
        parser.error(
            "--tma-ibgda-symmetric-expert-out is currently forward-only; "
            "use --pass-type forward"
        )
    return args


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
    mode: str,
    d_model: int,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    capacity_factor: float,
    rowwise_nblocks: int,
    tma_ibgda_num_sms: int | None,
    tma_ibgda_dispatch_num_sms: int | None,
    tma_ibgda_combine_num_sms: int | None,
    tma_ibgda_preprocess_num_sms: int | None,
    tma_ibgda_symmetric_expert_out: bool,
    include_shared_expert: bool,
    shared_hidden_size: int,
    random_routing: bool,
) -> OLMoDDPTransformerBlock:
    rowwise_backend = "nvshmem" if mode == "rowwise" else mode
    ep_path = (
        ExpertParallelPath.rowwise_tma_ibgda
        if rowwise_backend == "tma_ibgda"
        else ExpertParallelPath.rowwise_nvshmem
    )
    layer_norm = LayerNormConfig(
        name=LayerNormType.rms,
        eps=1e-6,
        bias=False,
        dtype=DType.float32,
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
            dtype=DType.float32,
        ),
        attention_norm=layer_norm,
        routed_experts_router=MoERouterConfigV2(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            gating_function=MoERouterGatingFunction.softmax,
            uniform_expert_assignment=not random_routing,
            random_expert_assignment=random_routing,
            lb_loss_weight=None,
            z_loss_weight=None,
            dtype=DType.float32,
        ),
        routed_experts=RoutedExpertsConfig(
            d_model=d_model,
            hidden_size=hidden_size,
            num_experts=num_experts,
            bias=False,
            dtype=DType.float32,
        ),
        shared_experts=(
            SharedExpertsConfig(
                d_model=d_model,
                hidden_size=shared_hidden_size,
                num_experts=1,
                bias=False,
                dtype=DType.float32,
            )
            if include_shared_expert
            else None
        ),
        shared_experts_router=None,
        feed_forward_norm=layer_norm,
        ep=ExpertParallelConfig(
            path=ep_path,
            rowwise_nblocks=rowwise_nblocks,
            tma_ibgda_num_sms=(
                tma_ibgda_num_sms if rowwise_backend == "tma_ibgda" else None
            ),
            tma_ibgda_dispatch_num_sms=(
                tma_ibgda_dispatch_num_sms if rowwise_backend == "tma_ibgda" else None
            ),
            tma_ibgda_combine_num_sms=(
                tma_ibgda_combine_num_sms if rowwise_backend == "tma_ibgda" else None
            ),
            tma_ibgda_preprocess_num_sms=(
                tma_ibgda_preprocess_num_sms if rowwise_backend == "tma_ibgda" else None
            ),
            tma_ibgda_symmetric_expert_out=(
                tma_ibgda_symmetric_expert_out and rowwise_backend == "tma_ibgda"
            ),
            capacity_factor=capacity_factor,
            major_align=1,
        ),
        init_device="cuda",
    )
    return block.to(dtype=torch.bfloat16)


def _patch_moe_only(block: OLMoDDPTransformerBlock) -> None:
    def ident_attn(self, block_inp, **kwargs):
        del kwargs
        return block_inp

    def mlp_only(self, residual, mlp_out):
        del residual
        return mlp_out

    block._checkpointed_res_norm_attn = types.MethodType(ident_attn, block)
    block._res_norm_attn = types.MethodType(ident_attn, block)
    block._res_norm_mlp = types.MethodType(mlp_only, block)


def _cuda_profiler_start() -> None:
    torch.cuda.cudart().cudaProfilerStart()


def _cuda_profiler_stop() -> None:
    torch.cuda.cudart().cudaProfilerStop()


def _run_one_iter(
    block: OLMoDDPTransformerBlock,
    *,
    x: torch.Tensor,
    pass_type: str,
    label: str,
) -> None:
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

    torch.cuda.nvtx.range_push(f"{label}/backward")
    try:
        loss = y.float().square().mean()
        if os.getenv("OLMO_TMA_IBGDA_BENCH_DETECT_ANOMALY", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            with torch.autograd.detect_anomaly():
                loss.backward()
        else:
            loss.backward()
    finally:
        torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push(f"{label}/zero_grad")
    try:
        block.zero_grad(set_to_none=True)
    finally:
        torch.cuda.nvtx.range_pop()


def _make_inputs(
    *,
    count: int,
    tokens: int,
    d_model: int,
    pass_type: str,
) -> list[torch.Tensor]:
    return [
        torch.randn(
            1,
            tokens,
            d_model,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=(pass_type == "forward_backward"),
        )
        for _ in range(count)
    ]


def _reduce_metric(value: float, *, op: dist.ReduceOp) -> float:
    tensor = torch.tensor([value], dtype=torch.float64, device="cuda")
    dist.all_reduce(tensor, op=op)
    return float(tensor.item())


def _bench_mode(
    args: argparse.Namespace,
    *,
    mode: str,
    tokens: int,
    rank: int,
    world_size: int,
    ep_mesh: DeviceMesh,
) -> BenchResult:
    torch.manual_seed(args.seed)
    block = _build_block(
        mode=mode,
        d_model=args.d_model,
        hidden_size=args.hidden_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
        capacity_factor=args.capacity_factor,
        rowwise_nblocks=args.rowwise_nblocks,
        tma_ibgda_num_sms=args.tma_ibgda_num_sms,
        tma_ibgda_dispatch_num_sms=args.tma_ibgda_dispatch_num_sms,
        tma_ibgda_combine_num_sms=args.tma_ibgda_combine_num_sms,
        tma_ibgda_preprocess_num_sms=args.tma_ibgda_preprocess_num_sms,
        tma_ibgda_symmetric_expert_out=args.tma_ibgda_symmetric_expert_out,
        include_shared_expert=not args.no_shared_expert,
        shared_hidden_size=args.shared_hidden_size,
        random_routing=args.random_routing,
    )
    block.apply_ep(ep_mesh)
    if not args.full_block:
        _patch_moe_only(block)
    block.train()

    inputs = _make_inputs(
        count=args.warmup + args.iters,
        tokens=tokens,
        d_model=args.d_model,
        pass_type=args.pass_type,
    )
    torch.cuda.synchronize()
    dist.barrier()

    for i in range(args.warmup):
        _run_one_iter(
            block,
            x=inputs[i],
            pass_type=args.pass_type,
            label=f"{mode}/warmup_{i}",
        )
    torch.cuda.synchronize()
    dist.barrier()

    if args.profile:
        _cuda_profiler_start()

    events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    for i in range(args.iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _run_one_iter(
            block,
            x=inputs[args.warmup + i],
            pass_type=args.pass_type,
            label=f"{mode}/iter_{i}",
        )
        end.record()
        events.append((start, end))
    torch.cuda.synchronize()
    dist.barrier()

    if args.profile:
        _cuda_profiler_stop()

    local_ms = [start.elapsed_time(end) for start, end in events]
    rank_mean = statistics.fmean(local_ms)
    rank_p50 = statistics.median(local_ms)
    rank_max = max(local_ms)
    world_sum = _reduce_metric(rank_mean, op=dist.ReduceOp.SUM)
    world_max = _reduce_metric(rank_max, op=dist.ReduceOp.MAX)
    return BenchResult(
        mode=mode,
        local_ms=local_ms,
        rank_mean_ms=rank_mean,
        rank_p50_ms=rank_p50,
        rank_max_ms=rank_max,
        world_mean_ms=world_sum / float(world_size),
        world_max_ms=world_max,
    )


def _print_results(
    *,
    tokens: int,
    results: Iterable[BenchResult],
    rank: int,
) -> None:
    if rank != 0:
        return
    print(f"\n# tokens={tokens}")
    print("mode,rank_mean_ms,rank_p50_ms,rank_max_ms,world_mean_ms,world_max_ms")
    for result in results:
        print(
            f"{result.mode},"
            f"{result.rank_mean_ms:.3f},"
            f"{result.rank_p50_ms:.3f},"
            f"{result.rank_max_ms:.3f},"
            f"{result.world_mean_ms:.3f},"
            f"{result.world_max_ms:.3f}",
            flush=True,
        )


def main() -> None:
    args = _parse_args()
    modes = _parse_modes(args.modes)
    rank, local_rank, world_size = _init_dist()
    ep_mesh = _build_ep_mesh(world_size)

    if rank == 0:
        tma_contract = _tma_ibgda_contract_for_log(modes)
        print(
            "TMA/IBGDA rowwise benchmark "
            f"world_size={world_size} local_rank0_device={torch.cuda.get_device_name(local_rank)} "
            f"modes={','.join(modes)} pass_type={args.pass_type} "
            f"d_model={args.d_model} hidden_size={args.hidden_size} "
            f"num_experts={args.num_experts} top_k={args.top_k} "
            f"rowwise_nblocks={args.rowwise_nblocks} "
            f"tma_ibgda_num_sms={args.tma_ibgda_num_sms} "
            f"tma_ibgda_dispatch_num_sms={args.tma_ibgda_dispatch_num_sms} "
            f"tma_ibgda_combine_num_sms={args.tma_ibgda_combine_num_sms} "
            f"tma_ibgda_preprocess_num_sms={args.tma_ibgda_preprocess_num_sms} "
            f"tma_ibgda_symmetric_expert_out={args.tma_ibgda_symmetric_expert_out} "
            f"tma_ibgda_extension_contract={tma_contract}",
            flush=True,
        )

    for tokens in args.tokens:
        results = [
            _bench_mode(
                args,
                mode=mode,
                tokens=tokens,
                rank=rank,
                world_size=world_size,
                ep_mesh=ep_mesh,
            )
            for mode in modes
        ]
        _print_results(tokens=tokens, results=results, rank=rank)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
