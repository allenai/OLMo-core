"""Grouped-MM wave chunking microbenchmark.

This intentionally excludes router, dispatch, combine, and repacking work. It
answers one question: how much slower do the expert grouped GEMMs get when the
same balanced expert-major workload is split into waves?
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F


@dataclass(frozen=True)
class WaveSpec:
    name: str
    xs: tuple[torch.Tensor, ...]
    up_weights: tuple[torch.Tensor, ...]
    down_weights: tuple[torch.Tensor, ...]
    offs: tuple[torch.Tensor, ...]
    rows_per_group: int
    groups_per_wave: int

    @property
    def waves(self) -> int:
        return len(self.xs)

    @property
    def total_rows(self) -> int:
        return sum(int(x.shape[0]) for x in self.xs)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare single grouped_mm, source-token-major waves, and "
            "expert-major waves for the local expert GEMM workload."
        )
    )
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=6144)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument(
        "--ep-size",
        type=int,
        default=0,
        help=(
            "Expert-parallel size used to derive local experts. 0 uses the "
            "distributed world size when initialized, otherwise 32."
        ),
    )
    parser.add_argument("--waves", type=int, default=4)
    parser.add_argument(
        "--cases",
        type=str,
        default="single,source_token_major,expert_major",
        help=(
            "Comma-separated subset of: single, source_token_major, "
            "expert_major."
        ),
    )
    parser.add_argument(
        "--op",
        choices=("up", "down", "mlp"),
        default="mlp",
        help=(
            "'up' times D->2H grouped_mm, 'down' times H->D grouped_mm, "
            "and 'mlp' times up grouped_mm + SwiGLU + down grouped_mm."
        ),
    )
    parser.add_argument(
        "--pass-type",
        choices=("forward", "backward", "forward_backward"),
        default="forward",
    )
    parser.add_argument(
        "--grad-targets",
        choices=("both", "input", "weight"),
        default="both",
        help=(
            "For backward modes, choose whether to require gradients for both "
            "inputs and weights, only inputs, or only weights."
        ),
    )
    parser.add_argument(
        "--validate-correctness",
        action="store_true",
        help=(
            "Before timing, compare single-wave and expert-major outputs and "
            "gradients on a small same-structure problem."
        ),
    )
    parser.add_argument("--validate-tokens", type=int, default=128)
    parser.add_argument("--validate-d-model", type=int, default=128)
    parser.add_argument("--validate-hidden-size", type=int, default=192)
    parser.add_argument("--validate-atol", type=float, default=5e-2)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the per-case loss/forward function with torch.compile.",
    )
    parser.add_argument(
        "--compile-fullgraph",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forwarded to torch.compile(fullgraph=...).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Wrap measured iterations in cudaProfilerStart/Stop.",
    )
    parser.add_argument(
        "--profile-include-warmup",
        action="store_true",
        help="Start CUDA profiler before warmup so warmup NVTX ranges are captured.",
    )
    parser.add_argument(
        "--dist",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Initialize torch.distributed. By default this is enabled only "
            "when WORLD_SIZE > 1."
        ),
    )
    return parser.parse_args()


def _init_runtime(args: argparse.Namespace) -> tuple[int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    should_init_dist = (
        int(os.environ.get("WORLD_SIZE", "1")) > 1
        if args.dist is None
        else bool(args.dist)
    )
    if should_init_dist and not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    return rank, local_rank, world_size


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(name)


def _parse_cases(raw: str) -> tuple[str, ...]:
    aliases = {
        "single": "single",
        "single_wave": "single",
        "source": "source_token_major",
        "source_token": "source_token_major",
        "source_token_major": "source_token_major",
        "token": "source_token_major",
        "token_major": "source_token_major",
        "expert": "expert_major",
        "expert_major": "expert_major",
    }
    out: list[str] = []
    for part in raw.split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise ValueError(
                f"unknown case {key!r}; expected single,source_token_major,expert_major"
            )
        case = aliases[key]
        if case not in out:
            out.append(case)
    if not out:
        raise ValueError("at least one case is required")
    return tuple(out)


def _make_equal_offs(
    *,
    groups: int,
    rows_per_group: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.arange(
        rows_per_group,
        rows_per_group * groups + 1,
        rows_per_group,
        device=device,
        dtype=torch.int32,
    )


def _make_specs(
    args: argparse.Namespace,
    *,
    rank: int,
    world_size: int,
) -> dict[str, WaveSpec]:
    device = torch.device("cuda")
    dtype = _dtype(args.dtype)
    ep_size = int(args.ep_size)
    if ep_size == 0:
        ep_size = world_size if dist.is_initialized() else 32
    if ep_size <= 0:
        raise ValueError(f"--ep-size must be positive, got {ep_size}")
    if args.num_experts % ep_size != 0:
        raise ValueError(
            f"--num-experts ({args.num_experts}) must be divisible by ep_size ({ep_size})"
        )
    local_experts = args.num_experts // ep_size
    if args.waves <= 0:
        raise ValueError(f"--waves must be positive, got {args.waves}")
    if args.waves > local_experts:
        raise ValueError(
            f"--waves ({args.waves}) cannot exceed local_experts ({local_experts})"
        )
    if local_experts % args.waves != 0:
        raise ValueError(
            f"local_experts ({local_experts}) must be divisible by waves ({args.waves})"
        )

    total_rows = args.tokens * args.top_k
    if total_rows % local_experts != 0:
        raise ValueError(
            "balanced rows require tokens * top_k divisible by local_experts: "
            f"{args.tokens} * {args.top_k} vs {local_experts}"
        )
    rows_per_expert = total_rows // local_experts
    if rows_per_expert % args.waves != 0:
        raise ValueError(
            f"rows_per_expert ({rows_per_expert}) must be divisible by waves ({args.waves})"
        )
    source_rows_per_expert = rows_per_expert // args.waves
    experts_per_wave = local_experts // args.waves
    requires_grad = args.pass_type in {"backward", "forward_backward"}
    input_requires_grad = requires_grad and args.grad_targets in {"both", "input"}
    weight_requires_grad = requires_grad and args.grad_targets in {"both", "weight"}

    torch.manual_seed(args.seed + rank)
    x_full = torch.empty(total_rows, args.d_model, device=device, dtype=dtype).normal_(
        mean=0.0,
        std=1.0,
    )
    x_full.requires_grad_(input_requires_grad)
    x_source_wave = tuple(
        torch.empty(
            local_experts * source_rows_per_expert,
            args.d_model,
            device=device,
            dtype=dtype,
        )
        .normal_(mean=0.0, std=1.0)
        .requires_grad_(input_requires_grad)
        for _ in range(args.waves)
    )
    x_down_full = torch.empty(
        total_rows,
        args.hidden_size,
        device=device,
        dtype=dtype,
    ).normal_(mean=0.0, std=1.0)
    x_down_full.requires_grad_(input_requires_grad)
    x_down_source_wave = tuple(
        torch.empty(
            local_experts * source_rows_per_expert,
            args.hidden_size,
            device=device,
            dtype=dtype,
        )
        .normal_(mean=0.0, std=1.0)
        .requires_grad_(input_requires_grad)
        for _ in range(args.waves)
    )

    w_up_gate_storage = torch.empty(
        local_experts,
        2 * args.hidden_size,
        args.d_model,
        device=device,
        dtype=dtype,
    ).normal_(mean=0.0, std=0.02)
    w_up_gate_storage.requires_grad_(weight_requires_grad)
    w_up_gate = w_up_gate_storage.transpose(1, 2)
    w_down = torch.empty(
        local_experts,
        args.hidden_size,
        args.d_model,
        device=device,
        dtype=dtype,
    ).normal_(mean=0.0, std=0.02)
    w_down.requires_grad_(weight_requires_grad)

    full_offs = _make_equal_offs(
        groups=local_experts,
        rows_per_group=rows_per_expert,
        device=device,
    )
    source_offs = _make_equal_offs(
        groups=local_experts,
        rows_per_group=source_rows_per_expert,
        device=device,
    )
    expert_offs = _make_equal_offs(
        groups=experts_per_wave,
        rows_per_group=rows_per_expert,
        device=device,
    )

    expert_xs = tuple(
        x_full[
            wave * experts_per_wave * rows_per_expert : (wave + 1)
            * experts_per_wave
            * rows_per_expert
        ]
        for wave in range(args.waves)
    )
    expert_down_xs = tuple(
        x_down_full[
            wave * experts_per_wave * rows_per_expert : (wave + 1)
            * experts_per_wave
            * rows_per_expert
        ]
        for wave in range(args.waves)
    )
    expert_up_weights = tuple(
        w_up_gate[wave * experts_per_wave : (wave + 1) * experts_per_wave]
        for wave in range(args.waves)
    )
    expert_down_weights = tuple(
        w_down[wave * experts_per_wave : (wave + 1) * experts_per_wave]
        for wave in range(args.waves)
    )

    if args.op == "down":
        single_xs = (x_down_full,)
        source_xs = x_down_source_wave
    else:
        single_xs = (x_full,)
        source_xs = x_source_wave

    specs = {
        "single": WaveSpec(
            name="single",
            xs=single_xs,
            up_weights=(w_up_gate,),
            down_weights=(w_down,),
            offs=(full_offs,),
            rows_per_group=rows_per_expert,
            groups_per_wave=local_experts,
        ),
        "source_token_major": WaveSpec(
            name="source_token_major",
            xs=source_xs,
            up_weights=(w_up_gate,) * args.waves,
            down_weights=(w_down,) * args.waves,
            offs=(source_offs,) * args.waves,
            rows_per_group=source_rows_per_expert,
            groups_per_wave=local_experts,
        ),
        "expert_major": WaveSpec(
            name="expert_major",
            xs=expert_down_xs if args.op == "down" else expert_xs,
            up_weights=expert_up_weights,
            down_weights=expert_down_weights,
            offs=(expert_offs,) * args.waves,
            rows_per_group=rows_per_expert,
            groups_per_wave=experts_per_wave,
        ),
    }

    return specs


def _swiglu(up_gate: torch.Tensor) -> torch.Tensor:
    up, gate = up_gate.chunk(2, dim=-1)
    return up * F.silu(gate)


def _forward_wave_raw(
    x: torch.Tensor,
    w_up_gate: torch.Tensor,
    w_down: torch.Tensor,
    offs: torch.Tensor,
    *,
    op: str,
) -> torch.Tensor:
    if op == "up":
        return F.grouped_mm(x, w_up_gate, offs=offs)
    if op == "down":
        return F.grouped_mm(x, w_down, offs=offs)
    up_gate = F.grouped_mm(x, w_up_gate, offs=offs)
    h = _swiglu(up_gate)
    return F.grouped_mm(h, w_down, offs=offs)


def _forward_wave(
    x: torch.Tensor,
    w_up_gate: torch.Tensor,
    w_down: torch.Tensor,
    offs: torch.Tensor,
    *,
    op: str,
    label: str,
) -> torch.Tensor:
    torch.cuda.nvtx.range_push(label)
    try:
        return _forward_wave_raw(x, w_up_gate, w_down, offs, op=op)
    finally:
        torch.cuda.nvtx.range_pop()


def _forward_spec_raw(spec: WaveSpec, *, op: str) -> tuple[torch.Tensor, ...]:
    return tuple(
        _forward_wave_raw(x, w_up_gate, w_down, offs, op=op)
        for x, w_up_gate, w_down, offs in zip(
            spec.xs,
            spec.up_weights,
            spec.down_weights,
            spec.offs,
        )
    )


def _forward_spec(spec: WaveSpec, *, op: str, case: str) -> tuple[torch.Tensor, ...]:
    outs: list[torch.Tensor] = []
    for idx, (x, w_up_gate, w_down, offs) in enumerate(
        zip(spec.xs, spec.up_weights, spec.down_weights, spec.offs)
    ):
        outs.append(
            _forward_wave(
                x,
                w_up_gate,
                w_down,
                offs,
                op=op,
                label=f"grouped_mm_wave/{case}/wave_{idx}",
            )
        )
    return tuple(outs)


def _global_mean_square_loss(outs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    total_numel = sum(out.numel() for out in outs)
    if total_numel == 0:
        raise RuntimeError("cannot build loss from empty outputs")
    return sum(out.float().square().sum() for out in outs) / float(total_numel)


def _max_abs_or_zero(a: torch.Tensor | None, b: torch.Tensor | None) -> float:
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return float("inf")
    return float((a - b).abs().max().item())


def _autograd_base(tensor: torch.Tensor) -> torch.Tensor:
    base = tensor
    while getattr(base, "_base", None) is not None:
        base = base._base  # type: ignore[assignment]
    return base


def _zero_unique_grads(tensors: tuple[torch.Tensor, ...]) -> None:
    seen: set[int] = set()
    for tensor in tensors:
        base = _autograd_base(tensor)
        ptr = int(base.untyped_storage().data_ptr())
        if ptr in seen:
            continue
        seen.add(ptr)
        if base.grad is not None:
            base.grad = None


def _validate_expert_major_correctness(
    args: argparse.Namespace,
    *,
    rank: int,
    world_size: int,
) -> None:
    device = torch.device("cuda")
    dtype = _dtype(args.dtype)
    ep_size = int(args.ep_size)
    if ep_size == 0:
        ep_size = world_size if dist.is_initialized() else 32
    if args.num_experts % ep_size != 0:
        raise ValueError(
            f"--num-experts ({args.num_experts}) must be divisible by ep_size ({ep_size})"
        )
    local_experts = args.num_experts // ep_size
    if args.waves > local_experts:
        raise ValueError(
            f"--waves ({args.waves}) cannot exceed local_experts ({local_experts})"
        )
    if local_experts % args.waves != 0:
        raise ValueError(
            f"local_experts ({local_experts}) must be divisible by waves ({args.waves})"
        )
    total_rows = int(args.validate_tokens) * int(args.top_k)
    if total_rows % local_experts != 0:
        raise ValueError(
            "--validate-tokens * --top-k must be divisible by local_experts: "
            f"{args.validate_tokens} * {args.top_k} vs {local_experts}"
        )
    rows_per_expert = total_rows // local_experts
    experts_per_wave = local_experts // int(args.waves)

    torch.manual_seed(args.seed + 17 + rank)
    if args.op == "down":
        x_cols = int(args.validate_hidden_size)
    else:
        x_cols = int(args.validate_d_model)
    x_single = torch.randn(total_rows, x_cols, device=device, dtype=dtype).requires_grad_()
    x_expert = x_single.detach().clone().requires_grad_()

    w_up_single_base = torch.randn(
        local_experts,
        2 * int(args.validate_hidden_size),
        int(args.validate_d_model),
        device=device,
        dtype=dtype,
    ).mul_(0.02).requires_grad_()
    w_up_expert_base = w_up_single_base.detach().clone().requires_grad_()
    w_down_single = torch.randn(
        local_experts,
        int(args.validate_hidden_size),
        int(args.validate_d_model),
        device=device,
        dtype=dtype,
    ).mul_(0.02).requires_grad_()
    w_down_expert = w_down_single.detach().clone().requires_grad_()
    w_up_single = w_up_single_base.transpose(1, 2)
    w_up_expert = w_up_expert_base.transpose(1, 2)

    full_offs = _make_equal_offs(
        groups=local_experts,
        rows_per_group=rows_per_expert,
        device=device,
    )
    expert_offs = _make_equal_offs(
        groups=experts_per_wave,
        rows_per_group=rows_per_expert,
        device=device,
    )

    single_out = _forward_wave(
        x_single,
        w_up_single,
        w_down_single,
        full_offs,
        op=args.op,
        label="grouped_mm_wave/validate/single",
    )
    expert_outs = []
    for wave in range(int(args.waves)):
        row_start = wave * experts_per_wave * rows_per_expert
        row_end = (wave + 1) * experts_per_wave * rows_per_expert
        expert_start = wave * experts_per_wave
        expert_end = (wave + 1) * experts_per_wave
        expert_outs.append(
            _forward_wave(
                x_expert[row_start:row_end],
                w_up_expert[expert_start:expert_end],
                w_down_expert[expert_start:expert_end],
                expert_offs,
                op=args.op,
                label=f"grouped_mm_wave/validate/expert_major_wave_{wave}",
            )
        )
    expert_out = torch.cat(expert_outs, dim=0)

    out_max_abs = float((single_out - expert_out).abs().max().item())
    single_loss = _global_mean_square_loss((single_out,))
    expert_loss = _global_mean_square_loss(tuple(expert_outs))
    single_loss.backward()
    expert_loss.backward()

    input_grad_max_abs = _max_abs_or_zero(x_single.grad, x_expert.grad)
    up_grad_max_abs = _max_abs_or_zero(w_up_single_base.grad, w_up_expert_base.grad)
    down_grad_max_abs = _max_abs_or_zero(w_down_single.grad, w_down_expert.grad)
    max_abs = max(out_max_abs, input_grad_max_abs, up_grad_max_abs, down_grad_max_abs)

    if rank == 0:
        print(
            "VALIDATE grouped_mm_wave expert_major "
            f"op={args.op} tokens={args.validate_tokens} "
            f"d={args.validate_d_model} hidden={args.validate_hidden_size} "
            f"out_max_abs={out_max_abs:.6g} "
            f"input_grad_max_abs={input_grad_max_abs:.6g} "
            f"up_grad_max_abs={up_grad_max_abs:.6g} "
            f"down_grad_max_abs={down_grad_max_abs:.6g} "
            f"atol={args.validate_atol}",
            flush=True,
        )
    if max_abs > float(args.validate_atol):
        raise RuntimeError(
            "expert-major correctness validation failed: "
            f"max_abs={max_abs} atol={args.validate_atol}"
        )


def _make_forward_backward_fn(
    spec: WaveSpec,
    *,
    op: str,
    case: str,
) -> Callable[[], None]:
    def _fn() -> None:
        _zero_unique_grads(spec.xs)
        _zero_unique_grads(spec.up_weights + spec.down_weights)

        outs = _forward_spec(spec, op=op, case=case)
        loss = _global_mean_square_loss(outs)
        loss.backward()

    return _fn


def _make_loss_fn(
    spec: WaveSpec,
    *,
    op: str,
    compile_enabled: bool,
    compile_fullgraph: bool,
) -> Callable[[], torch.Tensor]:
    def _loss_fn() -> torch.Tensor:
        return _global_mean_square_loss(_forward_spec_raw(spec, op=op))

    if not compile_enabled:
        return _loss_fn
    compiled = torch.compile(
        _loss_fn,
        fullgraph=bool(compile_fullgraph),
        dynamic=False,
    )
    return compiled


def _make_backward_only_prep(
    spec: WaveSpec,
    *,
    op: str,
    case: str,
    loss_fn: Callable[[], torch.Tensor] | None = None,
) -> Callable[[], Callable[[], None]]:
    if loss_fn is None:
        loss_fn = _make_loss_fn(
            spec,
            op=op,
            compile_enabled=False,
            compile_fullgraph=False,
        )

    def _prep() -> Callable[[], None]:
        _zero_unique_grads(spec.xs)
        _zero_unique_grads(spec.up_weights + spec.down_weights)

        torch.cuda.nvtx.range_push(f"grouped_mm_wave/{case}/loss")
        try:
            loss = loss_fn()
        finally:
            torch.cuda.nvtx.range_pop()

        def _backward() -> None:
            loss.backward()

        return _backward

    return _prep


def _set_requires_grad(
    spec: WaveSpec,
    requires_grad: bool,
    *,
    grad_targets: str,
) -> None:
    input_requires_grad = requires_grad and grad_targets in {"both", "input"}
    weight_requires_grad = requires_grad and grad_targets in {"both", "weight"}
    for x in spec.xs:
        _autograd_base(x).requires_grad_(input_requires_grad)
    seen_params: set[int] = set()
    for weights in (spec.up_weights, spec.down_weights):
        for weight in weights:
            base = _autograd_base(weight)
            ptr = int(base.untyped_storage().data_ptr())
            if ptr in seen_params:
                continue
            seen_params.add(ptr)
            base.requires_grad_(weight_requires_grad)


def _cuda_profiler_start() -> None:
    torch.cuda.cudart().cudaProfilerStart()


def _cuda_profiler_stop() -> None:
    torch.cuda.cudart().cudaProfilerStop()


def _time_case(
    args: argparse.Namespace,
    spec: WaveSpec,
    *,
    rank: int,
    world_size: int,
) -> None:
    requires_grad = args.pass_type in {"backward", "forward_backward"}
    _set_requires_grad(spec, requires_grad, grad_targets=args.grad_targets)
    loss_fn = _make_loss_fn(
        spec,
        op=args.op,
        compile_enabled=bool(args.compile),
        compile_fullgraph=bool(args.compile_fullgraph),
    )

    if args.pass_type == "forward":
        def warmup_fn() -> None:
            with torch.no_grad():
                loss_fn()

        timed_prep = None
        timed_fn = warmup_fn
    elif args.pass_type == "forward_backward":
        timed_prep = None
        def timed_fn() -> None:
            _zero_unique_grads(spec.xs)
            _zero_unique_grads(spec.up_weights + spec.down_weights)
            loss = loss_fn()
            loss.backward()

        warmup_fn = timed_fn
    else:
        prep = _make_backward_only_prep(
            spec,
            op=args.op,
            case=spec.name,
            loss_fn=loss_fn,
        )

        def warmup_fn() -> None:
            prep()()

        timed_prep = prep
        timed_fn = None

    profile_started = False
    if args.profile and args.compile:
        # Trigger TorchDynamo/AOTAutograd/Inductor before the profiler range so
        # captured warmup reflects compiled execution rather than codegen.
        warmup_fn()
        torch.cuda.synchronize()

    if args.profile and args.profile_include_warmup:
        if dist.is_initialized():
            dist.barrier()
        _cuda_profiler_start()
        profile_started = True

    for idx in range(args.warmup):
        torch.cuda.nvtx.range_push(f"grouped_mm_wave/{spec.name}/warmup_{idx}")
        try:
            warmup_fn()
        finally:
            torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()

    if args.profile and not profile_started:
        if dist.is_initialized():
            dist.barrier()
        _cuda_profiler_start()
        profile_started = True

    events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    for idx in range(args.iters):
        label = f"grouped_mm_wave/{spec.name}/iter_{idx}"
        fn: Callable[[], None]
        if timed_prep is not None:
            torch.cuda.nvtx.range_push(f"{label}/prep")
            try:
                fn = timed_prep()
            finally:
                torch.cuda.nvtx.range_pop()
        else:
            assert timed_fn is not None
            fn = timed_fn

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.cuda.nvtx.range_push(f"{label}/timed")
        try:
            fn()
        finally:
            torch.cuda.nvtx.range_pop()
        end.record()
        events.append((start, end))

    if profile_started:
        _cuda_profiler_stop()
        if dist.is_initialized():
            dist.barrier()

    if events:
        events[-1][1].synchronize()
    times = [start.elapsed_time(end) for start, end in events]
    local_ms = statistics.median(times)
    local_mem_gib = torch.cuda.max_memory_allocated() / 1024**3
    local = torch.tensor([local_ms, local_mem_gib], device="cuda")

    if dist.is_initialized():
        gathered = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(gathered, local)
    else:
        gathered = [local]

    if rank == 0:
        max_ms = max(float(item[0].item()) for item in gathered)
        max_mem_gib = max(float(item[1].item()) for item in gathered)
        print(
            "BENCH grouped_mm_wave "
            f"case={spec.name} pass={args.pass_type} op={args.op} "
            f"ranks={world_size} tokens/rank={args.tokens} top_k={args.top_k} "
            f"total_rows/rank={spec.total_rows} waves={spec.waves} "
            f"groups_per_wave={spec.groups_per_wave} "
            f"rows_per_group={spec.rows_per_group} "
            f"d={args.d_model} hidden={args.hidden_size} "
            f"experts={args.num_experts} dtype={args.dtype} "
            f"grad_targets={args.grad_targets} "
            f"compile={bool(args.compile)} "
            f"warmup={args.warmup} iters={args.iters} "
            f"ms/iter(max_rank)={max_ms:.3f} "
            f"max_mem_GiB={max_mem_gib:.2f}",
            flush=True,
        )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if not hasattr(F, "grouped_mm"):
        raise RuntimeError("torch.nn.functional.grouped_mm is required")

    args = _parse_args()
    rank, _, world_size = _init_runtime(args)
    cases = _parse_cases(args.cases)
    try:
        if rank == 0:
            print(
                "grouped_mm_wave_bench "
                f"time={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                f"cases={','.join(cases)}",
                flush=True,
            )
        if args.validate_correctness:
            _validate_expert_major_correctness(args, rank=rank, world_size=world_size)
        specs = _make_specs(args, rank=rank, world_size=world_size)
        for case in cases:
            torch.cuda.reset_peak_memory_stats()
            _time_case(args, specs[case], rank=rank, world_size=world_size)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
