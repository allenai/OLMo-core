"""MoE EP benchmark entrypoint implementation."""

from __future__ import annotations

import argparse
import math
import os
import statistics
import time

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from .args import BenchCase, _parse_args, _parse_modes
from .common import (
    _build_block,
    _build_ep_mesh,
    _compile_hot_modules,
    _cuda_profiler_start,
    _cuda_profiler_stop,
    _dtype_config,
    _init_dist,
    _install_deepep_balanced_router,
    _median_rank_ms,
    _patch_moe_only,
    _prepare_backward_loss,
    _run_backward_from_loss,
    _run_one_iter,
    _setup_debug_print,
)
from .deepep_v2 import _bench_deepep_v2_case
from .expert_probe import (
    _init_probe_routed_expert_weights,
    _resolve_weight_init_value,
    _run_pre_dispatch_expert_probe,
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
    if case.deepep_v2 or case.deepep_v2_wave:
        _bench_deepep_v2_case(
            args,
            tokens=tokens,
            rank=rank,
            world_size=world_size,
            ep_mesh=ep_mesh,
            use_wave=case.deepep_v2_wave,
        )
        return

    capacity_factor = args.capacity_factor
    if rank == 0 and case.rowwise_wave:
        print(
            "[bench] mode=rowwise_wave selects the experimental expert-major "
            "rowwise backend.",
            flush=True,
        )
    torch.manual_seed(20260619 + rank)
    torch.cuda.reset_peak_memory_stats()
    config_dtype, input_dtype = _dtype_config(args.dtype)

    torch.cuda.nvtx.range_push(f"BENCH/{case.name}/tokens_{tokens}/build")
    try:
        _setup_debug_print("case_build:enter", case=case.name, tokens=tokens)
        block = _build_block(
            d_model=args.d_model,
            hidden_size=args.hidden_size,
            num_experts=args.num_experts,
            top_k=args.top_k,
            capacity_factor=capacity_factor,
            rowwise_nblocks=args.rowwise_nblocks,
            rowwise_wave=case.rowwise_wave,
            rowwise_wave_num_waves=args.rowwise_wave_num_waves,
            rowwise_wave_recompute_linear1=args.rowwise_wave_recompute_linear1,
            rowwise_wave_recompute_act=args.rowwise_wave_recompute_act,
            include_shared_expert=not args.no_shared_expert,
            shared_hidden_size=args.shared_hidden_size,
            uniform_routing=not args.random_routing,
            random_routing=args.random_routing,
            config_dtype=config_dtype,
        )
        _setup_debug_print("patch_moe_only:enter", case=case.name)
        if not args.full_block:
            _patch_moe_only(block)
        _setup_debug_print("patch_moe_only:exit", case=case.name)
        _setup_debug_print(
            "block_apply_ep:enter",
            case=case.name,
            ep_world_size=world_size,
            cuda_allocated_gib=f"{torch.cuda.memory_allocated() / 1024**3:.2f}",
            cuda_reserved_gib=f"{torch.cuda.memory_reserved() / 1024**3:.2f}",
        )
        block.apply_ep(ep_mesh)
        _setup_debug_print(
            "block_apply_ep:exit",
            case=case.name,
            cuda_allocated_gib=f"{torch.cuda.memory_allocated() / 1024**3:.2f}",
            cuda_reserved_gib=f"{torch.cuda.memory_reserved() / 1024**3:.2f}",
        )
        if block.routed_experts is not None:
            _setup_debug_print("init_probe_weights:enter", case=case.name)
            _init_probe_routed_expert_weights(
                block.routed_experts,
                weight_init=_resolve_weight_init_value(
                    str(args.model_local_expert_weight_init),
                    source_default="empty",
                ),
            )
            _setup_debug_print("init_probe_weights:exit", case=case.name)
        if args.balanced_routing == "deepep":
            if args.random_routing:
                raise RuntimeError("--balanced-routing deepep conflicts with --random-routing")
            _setup_debug_print("install_deepep_balanced_router:enter", case=case.name)
            _install_deepep_balanced_router(block, world_size=world_size)
            _setup_debug_print("install_deepep_balanced_router:exit", case=case.name)
        block.train()
        compile_enabled = bool(args.compile and not args.no_compile)
        if compile_enabled:
            _setup_debug_print("compile:enter", case=case.name, compile_block=args.compile_block)
            if args.compile_block:
                block = torch.compile(block, fullgraph=False, dynamic=False)
            else:
                _compile_hot_modules(block)
            _setup_debug_print("compile:exit", case=case.name)
        _setup_debug_print("case_build:exit", case=case.name)
    finally:
        torch.cuda.nvtx.range_pop()

    static_input = None
    if args.profile and args.pass_type == "forward":
        static_input = torch.randn(
            1,
            tokens,
            args.d_model,
            device="cuda",
            dtype=input_dtype,
        )

    profile_started = False
    if args.profile and args.pre_dispatch_expert_iters > 0:
        dist.barrier()
        _cuda_profiler_start()
        profile_started = True

    if args.pre_dispatch_expert_iters > 0:
        if block.routed_experts is None:
            raise RuntimeError(f"{case.name} pre-dispatch probe requires routed experts")
        _run_pre_dispatch_expert_probe(
            block.routed_experts,
            mode_name=case.name,
            num_iters=int(args.pre_dispatch_expert_iters),
            tokens=tokens,
            top_k=args.top_k,
            d_model=args.d_model,
            input_dtype=input_dtype,
            pass_type=args.pass_type,
            rank=rank,
            world_size=world_size,
        )

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
                pass_type=(
                    "forward_backward"
                    if args.pass_type == "backward"
                    else args.pass_type
                ),
                static_input=static_input,
            )
        finally:
            torch.cuda.nvtx.range_pop()
    warmup_done = torch.cuda.Event(enable_timing=False)
    warmup_done.record()
    warmup_done.synchronize()

    if args.profile and not profile_started:
        dist.barrier()
        _cuda_profiler_start()

    host_sync_timing = os.getenv("OLMO_BENCH_HOST_SYNC_TIMING", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    host_times: list[float] = []
    for idx in range(args.iters):
        label = f"BENCH/{case.name}/tokens_{tokens}/iter_{idx}"
        backward_loss = None
        if args.pass_type == "backward":
            torch.cuda.nvtx.range_push(f"{label}/prep")
            try:
                backward_loss = _prepare_backward_loss(
                    block,
                    tokens=tokens,
                    d_model=args.d_model,
                    input_dtype=input_dtype,
                    label=label,
                )
            finally:
                torch.cuda.nvtx.range_pop()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        if host_sync_timing:
            torch.cuda.synchronize()
        host_start = time.perf_counter() if host_sync_timing else 0.0
        start.record()
        torch.cuda.nvtx.range_push(f"{label}/total")
        try:
            if args.pass_type == "backward":
                assert backward_loss is not None
                _run_backward_from_loss(
                    block,
                    backward_loss,
                    label=label,
                )
            else:
                _run_one_iter(
                    block,
                    tokens=tokens,
                    d_model=args.d_model,
                    input_dtype=input_dtype,
                    label=label,
                    pass_type=args.pass_type,
                    static_input=static_input,
                )
        finally:
            torch.cuda.nvtx.range_pop()
        end.record()
        events.append((start, end))
        if host_sync_timing:
            torch.cuda.synchronize()
            host_times.append((time.perf_counter() - host_start) * 1000.0)
        if case.rowwise_wave and not args.profile:
            # These experimental modes use rowwise/NVSHMEM stream barriers.
            # Keep benchmark iterations host-ordered. During nsys profiling,
            # leave iterations queued and synchronize after cudaProfilerStop()
            # so the profile does not contain inter-iteration host syncs.
            end.synchronize()

    if args.profile:
        _cuda_profiler_stop()
        dist.barrier()

    if events:
        events[-1][1].synchronize()
    times = [start.elapsed_time(end) for start, end in events]
    local_ms = statistics.median(times)
    local_host_ms = statistics.median(host_times) if host_times else float("nan")
    local_mem_gib = torch.cuda.max_memory_allocated() / 1024**3
    local = torch.tensor([local_ms, local_host_ms, local_mem_gib], device="cuda")
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)

    if rank == 0:
        max_ms = _median_rank_ms(gathered)
        max_host_ms = max(float(v[1].item()) for v in gathered)
        max_mem_gib = max(float(v[2].item()) for v in gathered)
        host_timing_part = (
            f"host_ms/iter(max_rank)={max_host_ms:.3f} "
            if math.isfinite(max_host_ms)
            else ""
        )
        throughput_ms = max_host_ms if math.isfinite(max_host_ms) else max_ms
        print(
            "BENCH "
            f"{case.name}: ranks={world_size} tokens/rank={tokens} "
            f"pass={args.pass_type} moe_only={not args.full_block} "
            f"shared={not args.no_shared_expert} dtype={args.dtype} "
            f"compile={'none' if not args.compile or args.no_compile else ('block' if args.compile_block else 'experts')} "
            f"d={args.d_model} hidden={args.hidden_size} experts={args.num_experts} "
            f"top_k={args.top_k} cap={capacity_factor} "
            f"balanced_routing={args.balanced_routing} "
            f"rowwise_wave_num_waves={args.rowwise_wave_num_waves if case.rowwise_wave else 0} "
            f"rowwise_wave_recompute_linear1={args.rowwise_wave_recompute_linear1 if case.rowwise_wave else False} "
            f"rowwise_wave_recompute_act={args.rowwise_wave_recompute_act if case.rowwise_wave else False} "
            f"ms/iter(max_rank)={max_ms:.3f} "
            f"{host_timing_part}"
            f"local_tokens/s={tokens / (throughput_ms / 1000.0):.1f} "
            f"global_tokens/s={tokens * world_size / (throughput_ms / 1000.0):.1f} "
            f"max_mem_GiB={max_mem_gib:.2f}",
            flush=True,
        )

    if os.getenv("OLMO_BENCH_OS_EXIT_AFTER_BENCH", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        torch.cuda.synchronize()
        exit_sleep_s = float(os.getenv("OLMO_BENCH_EXIT_SLEEP_S", "0"))
        if exit_sleep_s > 0:
            time.sleep(exit_sleep_s)
        os._exit(0)

    if os.getenv("OLMO_BENCH_HARD_EXIT_AFTER_BENCH", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        torch.cuda.synchronize()
        exit_sleep_s = float(os.getenv("OLMO_BENCH_EXIT_SLEEP_S", "0"))
        if exit_sleep_s > 0:
            time.sleep(exit_sleep_s)
        return


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
        if (
            dist.is_initialized()
            and os.getenv("OLMO_BENCH_HARD_EXIT_AFTER_BENCH", "0").strip().lower()
            not in {"1", "true", "yes", "on"}
        ):
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
