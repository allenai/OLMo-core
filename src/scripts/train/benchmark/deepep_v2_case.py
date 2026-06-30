from __future__ import annotations

import argparse
import math
import os
import statistics
import time

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from .common import (
    _cuda_profiler_start,
    _cuda_profiler_stop,
    _dtype_config,
    _median_rank_ms,
)
from .deepep_v2_core import (
    DeepEpV2ForwardResult,
    _build_deepep_v2_probe_routed_experts,
    _build_deepep_v2_state,
    _build_rowwise_apply_ep_probe_routed_experts,
    _make_balanced_topk_idx,
    _prepare_deepep_v2_backward,
    _run_deepep_v2_backward_from_result,
    _run_one_deepep_v2_iter,
    _validate_deepep_v2_args,
)
from .deepep_v2_wave import (
    DeepEpV2WaveForwardResult,
    _build_deepep_v2_wave_inputs,
    _prepare_deepep_v2_wave_backward,
    _run_deepep_v2_wave_backward_from_result,
    _run_one_deepep_v2_wave_iter,
    _validate_deepep_v2_wave_backward,
    _validate_deepep_v2_wave_forward,
)
from .expert_probe import (
    _deepep_pre_dispatch_expert_iters,
    _run_pre_dispatch_expert_probe,
)


def _bench_deepep_v2_case(
    args: argparse.Namespace,
    *,
    tokens: int,
    rank: int,
    world_size: int,
    ep_mesh: DeviceMesh,
    use_wave: bool = False,
) -> None:
    mode_name = "deepep_v2_wave" if use_wave else "deepep_v2"
    if rank == 0:
        if use_wave:
            print(
                "[bench] mode=deepep_v2_wave runs standalone DeepEP V2 wave "
                "expanded dispatch + OLMo grouped expert MLP + DeepEP V2 "
                "combine. It is benchmark bring-up, not model-path wiring yet.",
                flush=True,
            )
        else:
            print(
                "[bench] mode=deepep_v2 runs standalone DeepEP V2 expanded dispatch "
                "+ OLMo grouped expert MLP + DeepEP V2 combine. It is benchmark "
                "bring-up, not model-path wiring yet.",
                flush=True,
            )

    torch.manual_seed(20260625 + rank)
    torch.cuda.reset_peak_memory_stats()
    config_dtype, input_dtype = _dtype_config(args.dtype)
    deepep_probe_iters = _deepep_pre_dispatch_expert_iters(args)

    if args.deepep_skip_import_buffer_for_pre_dispatch_probe:
        if deepep_probe_iters <= 0:
            raise RuntimeError(
                "--deepep-skip-import-buffer-for-pre-dispatch-probe requires "
                "--pre-dispatch-expert-iters or --deepep-pre-dispatch-expert-iters"
            )
        _validate_deepep_v2_args(args, world_size=world_size)
        torch.cuda.nvtx.range_push(f"BENCH/{mode_name}/tokens_{tokens}/build_probe_no_import_buffer")
        try:
            torch.manual_seed(20260625 + rank)
            source_input = (0.2 * torch.randn(tokens, args.d_model, device="cuda")).to(input_dtype)
            topk_idx = _make_balanced_topk_idx(
                tokens=tokens,
                top_k=args.top_k,
                num_experts=args.num_experts,
                world_size=world_size,
                rank=rank,
                dtype=torch.int64,
            )
            topk_weights = torch.full(
                (tokens, args.top_k),
                1.0 / float(args.top_k),
                device="cuda",
                dtype=torch.float32,
            )
            if args.deepep_probe_routed_experts_source == "standalone":
                routed_experts = _build_deepep_v2_probe_routed_experts(
                    args,
                    rank=rank,
                    world_size=world_size,
                    config_dtype=config_dtype,
                    reset_seed=False,
                )
            elif args.deepep_probe_routed_experts_source == "rowwise_apply_ep":
                routed_experts = _build_rowwise_apply_ep_probe_routed_experts(
                    args,
                    world_size=world_size,
                    ep_mesh=ep_mesh,
                    config_dtype=config_dtype,
                )
            else:
                raise ValueError(args.deepep_probe_routed_experts_source)
        finally:
            torch.cuda.nvtx.range_pop()

        if rank == 0:
            print(
                "[bench] deepep_v2 diagnostic: skipped deep_ep import and "
                "ElasticBuffer construction; running fake pre-dispatch expert "
                "probe only. "
                f"routed_experts_source={args.deepep_probe_routed_experts_source} "
                f"weight_init={args.deepep_probe_weight_init}",
                flush=True,
            )

        if args.profile:
            dist.barrier()
            _cuda_profiler_start()
        _run_pre_dispatch_expert_probe(
            routed_experts,
            mode_name=f"{mode_name}_no_import_buffer",
            num_iters=deepep_probe_iters,
            tokens=tokens,
            top_k=args.top_k,
            d_model=args.d_model,
            input_dtype=input_dtype,
            pass_type=args.pass_type,
            rank=rank,
            world_size=world_size,
        )
        if args.profile:
            _cuda_profiler_stop()
            dist.barrier()

        del routed_experts, source_input, topk_idx, topk_weights
        torch.cuda.synchronize()
        return

    torch.cuda.nvtx.range_push(f"BENCH/{mode_name}/tokens_{tokens}/build")
    try:
        state = _build_deepep_v2_state(
            args,
            tokens=tokens,
            rank=rank,
            world_size=world_size,
            config_dtype=config_dtype,
            input_dtype=input_dtype,
        )
    finally:
        torch.cuda.nvtx.range_pop()

    try:
        if rank == 0:
            wave_do_cpu_sync = (
                bool(args.deepep_wave_do_cpu_sync)
                if args.deepep_wave_do_cpu_sync is not None
                else state.do_cpu_sync
            )
            print(
                f"[bench] {mode_name} config: num_sms={state.num_sms} "
                f"num_qps={state.num_qps} allocated_qps={state.buffer.num_allocated_qps} "
                f"local_experts={state.num_local_experts} "
                f"num_max_tokens_per_rank={state.num_max_tokens_per_rank} "
                f"deepep_max_tokens_factor={args.deepep_max_tokens_factor} "
                f"expert_buffer_mode={state.expert_buffer_mode} "
                f"weighting={state.weighting_mode} "
                f"expert_alignment={state.expert_alignment} "
                f"async={state.async_with_compute_stream} "
                f"do_cpu_sync={state.do_cpu_sync} "
                f"wave_num_waves={args.deepep_wave_num_waves if use_wave else 0} "
                f"wave_overlap={bool(args.deepep_wave_overlap) if use_wave else False} "
                f"wave_layout={args.deepep_wave_layout if use_wave else 'none'} "
                f"wave_do_cpu_sync={wave_do_cpu_sync if use_wave else False}",
                flush=True,
            )
            if use_wave and args.deepep_wave_overlap and wave_do_cpu_sync:
                print(
                    "[bench] warning: deepep_v2_wave overlap is enabled while "
                    "wave dispatch CPU sync is enabled; launch-side CPU waits "
                    "can limit communication/compute pipelining. Use "
                    "--no-deepep-wave-do-cpu-sync to test the no-CPU-sync path.",
                    flush=True,
                )

        wave_inputs: list[DeepEpV2WaveInput] | None = None
        wave_overlap = bool(args.deepep_wave_overlap)
        wave_do_cpu_sync = (
            bool(args.deepep_wave_do_cpu_sync)
            if args.deepep_wave_do_cpu_sync is not None
            else state.do_cpu_sync
        )
        if use_wave:
            torch.cuda.nvtx.range_push(f"BENCH/{mode_name}/tokens_{tokens}/build_wave_inputs")
            try:
                wave_inputs = _build_deepep_v2_wave_inputs(
                    state,
                    num_waves=int(args.deepep_wave_num_waves),
                )
            finally:
                torch.cuda.nvtx.range_pop()
            if rank == 0:
                wave_ranges = [
                    f"{w.expert_start}:{w.expert_end}@rows{w.wave_base}:{w.wave_end}"
                    for w in wave_inputs
                ]
                print(
                    f"[bench] deepep_v2_wave local expert/row ranges={wave_ranges}",
                    flush=True,
                )
            if args.deepep_validate_wave_forward:
                _validate_deepep_v2_wave_forward(
                    state,
                    wave_inputs=wave_inputs,
                    wave_layout=str(args.deepep_wave_layout),
                    overlap=wave_overlap,
                    wave_do_cpu_sync=wave_do_cpu_sync,
                    atol=float(args.deepep_validate_wave_forward_atol),
                    rank=rank,
                )
            if args.deepep_validate_wave_backward:
                _validate_deepep_v2_wave_backward(
                    state,
                    wave_inputs=wave_inputs,
                    wave_layout=str(args.deepep_wave_layout),
                    overlap=wave_overlap,
                    wave_do_cpu_sync=wave_do_cpu_sync,
                    atol=float(args.deepep_validate_wave_backward_atol),
                    rank=rank,
                )

        profile_started = False
        if args.profile and deepep_probe_iters > 0:
            dist.barrier()
            _cuda_profiler_start()
            dist.barrier()
            profile_started = True

        _run_pre_dispatch_expert_probe(
            state.routed_experts,
            mode_name=mode_name,
            num_iters=deepep_probe_iters,
            tokens=tokens,
            top_k=args.top_k,
            d_model=args.d_model,
            input_dtype=input_dtype,
            pass_type=args.pass_type,
            rank=rank,
            world_size=world_size,
        )

        for idx in range(args.warmup):
            label = f"BENCH/{mode_name}/tokens_{tokens}/warmup_{idx}"
            torch.cuda.nvtx.range_push(f"{label}/total")
            try:
                warmup_pass_type = (
                    "forward_backward"
                    if args.pass_type == "backward"
                    else args.pass_type
                )
                if use_wave:
                    assert wave_inputs is not None
                    _run_one_deepep_v2_wave_iter(
                        state,
                        wave_inputs=wave_inputs,
                        wave_layout=str(args.deepep_wave_layout),
                        label=label,
                        pass_type=warmup_pass_type,
                        overlap=wave_overlap,
                        do_cpu_sync=wave_do_cpu_sync,
                    )
                else:
                    _run_one_deepep_v2_iter(
                        state,
                        label=label,
                        pass_type=warmup_pass_type,
                    )
            finally:
                torch.cuda.nvtx.range_pop()

        warmup_done = torch.cuda.Event(enable_timing=False)
        warmup_done.record()
        warmup_done.synchronize()

        if args.profile and not profile_started:
            dist.barrier()
            _cuda_profiler_start()
            dist.barrier()

        host_sync_timing = os.getenv("OLMO_BENCH_HOST_SYNC_TIMING", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        host_times: list[float] = []
        for idx in range(args.iters):
            label = f"BENCH/{mode_name}/tokens_{tokens}/iter_{idx}"
            backward_result = None
            if args.pass_type == "backward":
                torch.cuda.nvtx.range_push(f"{label}/prep")
                try:
                    if use_wave:
                        assert wave_inputs is not None
                        backward_result = _prepare_deepep_v2_wave_backward(
                            state,
                            wave_inputs=wave_inputs,
                            wave_layout=str(args.deepep_wave_layout),
                            label=label,
                            overlap=wave_overlap,
                            do_cpu_sync=wave_do_cpu_sync,
                        )
                    else:
                        backward_result = _prepare_deepep_v2_backward(
                            state,
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
                    assert backward_result is not None
                    if use_wave:
                        assert isinstance(backward_result, DeepEpV2WaveForwardResult)
                        _run_deepep_v2_wave_backward_from_result(
                            state,
                            backward_result,
                            label=label,
                            overlap=wave_overlap,
                        )
                    else:
                        assert isinstance(backward_result, DeepEpV2ForwardResult)
                        _run_deepep_v2_backward_from_result(
                            state,
                            backward_result,
                            label=label,
                        )
                else:
                    if use_wave:
                        assert wave_inputs is not None
                        _run_one_deepep_v2_wave_iter(
                            state,
                            wave_inputs=wave_inputs,
                            wave_layout=str(args.deepep_wave_layout),
                            label=label,
                            pass_type=args.pass_type,
                            overlap=wave_overlap,
                            do_cpu_sync=wave_do_cpu_sync,
                        )
                    else:
                        _run_one_deepep_v2_iter(
                            state,
                            label=label,
                            pass_type=args.pass_type,
                        )
            finally:
                torch.cuda.nvtx.range_pop()
            end.record()
            if args.sync_between_iters:
                torch.cuda.synchronize()
            events.append((start, end))
            if host_sync_timing:
                torch.cuda.synchronize()
                host_times.append((time.perf_counter() - host_start) * 1000.0)

        if args.profile:
            _cuda_profiler_stop()
            dist.barrier()

        if not events:
            return
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
                f"{mode_name}: ranks={world_size} tokens/rank={tokens} "
                f"pass={args.pass_type} standalone=True moe_only=True shared=False "
                f"dtype={args.dtype} "
                f"compile={'experts' if args.compile and not args.no_compile else 'none'} "
                f"d={args.d_model} hidden={args.hidden_size} experts={args.num_experts} "
                f"local_experts={state.num_local_experts} top_k={args.top_k} "
                "balanced_routing=deepep "
                f"num_sms={state.num_sms} num_qps={state.num_qps} "
                f"num_max_tokens_per_rank={state.num_max_tokens_per_rank} "
                f"deepep_max_tokens_factor={args.deepep_max_tokens_factor} "
                f"expert_buffer_mode={state.expert_buffer_mode} "
                f"expert_alignment={state.expert_alignment} "
                f"async={state.async_with_compute_stream} "
                f"do_cpu_sync={state.do_cpu_sync} "
                f"wave_num_waves={args.deepep_wave_num_waves if use_wave else 0} "
                f"wave_overlap={wave_overlap if use_wave else False} "
                f"wave_layout={args.deepep_wave_layout if use_wave else 'none'} "
                f"wave_do_cpu_sync={wave_do_cpu_sync if use_wave else False} "
                f"sync_between_iters={bool(args.sync_between_iters)} "
                f"ms/iter(max_rank)={max_ms:.3f} "
                f"{host_timing_part}"
                f"local_tokens/s={tokens / (throughput_ms / 1000.0):.1f} "
                f"global_tokens/s={tokens * world_size / (throughput_ms / 1000.0):.1f} "
                f"max_mem_GiB={max_mem_gib:.2f}",
                flush=True,
            )
    finally:
        if hasattr(state.buffer, "destroy"):
            torch.cuda.synchronize()
            dist.barrier()
            state.buffer.destroy()
            dist.barrier()
