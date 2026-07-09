import argparse
import os
import statistics
import time
from typing import Optional

import torch
import torch.distributed as dist


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Small distributed smoke for rowwise NVSHMEM MXFP8 dispatch/combine. "
            "This is intended to isolate hangs from full MoE training."
        )
    )
    parser.add_argument(
        "--mode",
        choices=(
            "dispatch_scaled",
            "combine_scaled",
            "roundtrip_scaled",
            "raw_dispatch",
            "raw_dispatch_weighted",
            "raw_gather",
        ),
        default="roundtrip_scaled",
    )
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument(
        "--capacity",
        type=int,
        default=None,
        help=(
            "Allocated expert-buffer rows. Defaults to rows * top_k. "
            "Use this to model EP capacity-factor padding separately from valid routed rows."
        ),
    )
    parser.add_argument("--nblocks", type=int, default=128)
    parser.add_argument(
        "--dispatch-impl",
        choices=("two_put", "paired_put", "packed_put", "fused_weighted"),
        default="two_put",
    )
    parser.add_argument(
        "--combine-impl",
        choices=("two_get", "paired_get"),
        default="two_get",
    )
    parser.add_argument(
        "--peer-pattern",
        choices=("self", "ring", "remote_node"),
        default="remote_node",
    )
    parser.add_argument(
        "--raw-dtype",
        choices=("bf16", "fp16", "fp32", "uint8", "int8", "fp8_e4m3", "fp8_e5m2", "fp8_e8m0"),
        default="bf16",
        help="Tensor dtype for raw_dispatch/raw_gather modes.",
    )
    parser.add_argument(
        "--raw-symmetric-source",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use an OLMo symmetric-memory local source tensor for raw dispatch modes.",
    )
    parser.add_argument(
        "--raw-copy-source-each-iter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When --raw-symmetric-source is set, copy the regular source tensor into "
            "the symmetric source inside every timed iteration. By default the source "
            "is prefilled once so raw_dispatch measures the rowwise PUT path."
        ),
    )
    parser.add_argument(
        "--zero-raw-dispatch-out",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Zero the raw dispatch destination before each timed dispatch. "
            "Disabled by default because valid route rows are fully overwritten."
        ),
    )
    parser.add_argument(
        "--raw-pre-barrier",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use rowwise raw dispatch/gather pre_barrier.",
    )
    parser.add_argument(
        "--raw-post-barrier",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use rowwise raw dispatch/gather post_barrier.",
    )
    parser.add_argument(
        "--raw-symmetric-dest",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use an OLMo symmetric-memory local destination tensor for raw gather mode.",
    )
    parser.add_argument(
        "--scaled-symmetric-source",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use OLMo symmetric-memory local q/scales tensors for scaled dispatch source staging.",
    )
    parser.add_argument(
        "--scaled-symmetric-dest",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use OLMo symmetric-memory local q/scales tensors for scaled combine-get destination staging.",
    )
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--atol", type=float, default=0.25)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument(
        "--strict-comm-compare",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For alternate dispatch/combine modes, also run the existing two-op path and "
            "byte-compare the communicated q/scales or final combine output. This is "
            "not included in the reported timing."
        ),
    )
    return parser.parse_args()


def _init_dist() -> tuple[int, int, int, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def _setup_olmo_symm_mem(group: dist.ProcessGroup, device: torch.device) -> str:
    from olmo_core.kernels import olmo_symm_mem

    os.environ["OLMO_USE_OWN_SYMM_MEM"] = "1"
    olmo_symm_mem.register_group(group, device=device)
    return group.group_name


def _alloc_symm(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    from olmo_core.kernels import olmo_symm_mem

    tensor = olmo_symm_mem.empty(shape, dtype=dtype, device=device, group=group)
    olmo_symm_mem.rendezvous(tensor, group=group)
    return tensor


def _destination_rank(
    *,
    rank: int,
    world_size: int,
    local_world_size: int,
    pattern: str,
) -> int:
    if pattern == "self":
        return rank
    if pattern == "ring":
        return (rank + 1) % world_size
    if pattern == "remote_node":
        if world_size <= local_world_size:
            raise RuntimeError("remote_node peer pattern needs at least two nodes")
        if world_size % local_world_size != 0:
            raise RuntimeError(
                f"world_size={world_size} must be divisible by local_world_size={local_world_size}"
            )
        return (rank + local_world_size) % world_size
    raise RuntimeError(f"unknown peer pattern {pattern!r}")


def _make_route_maps(
    *,
    rows: int,
    top_k: int,
    rank: int,
    world_size: int,
    local_world_size: int,
    pattern: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    dst_rank = _destination_rank(
        rank=rank,
        world_size=world_size,
        local_world_size=local_world_size,
        pattern=pattern,
    )
    dst_ranks = torch.full((rows, top_k), dst_rank, dtype=torch.long, device=device)
    row_ids = torch.arange(rows * top_k, dtype=torch.long, device=device).view(rows, top_k)
    return dst_ranks.contiguous(), row_ids.contiguous()


def _expected_source_rank(
    *,
    rank: int,
    world_size: int,
    local_world_size: int,
    pattern: str,
) -> int:
    if pattern == "self":
        return rank
    if pattern == "ring":
        return (rank - 1) % world_size
    if pattern == "remote_node":
        return (rank - local_world_size) % world_size
    raise RuntimeError(f"unknown peer pattern {pattern!r}")


def _make_input(rows: int, dim: int, rank: int, device: torch.device) -> torch.Tensor:
    # Keep values simple so MXFP8 round-trip error is small.
    base = float(rank + 1)
    row_offsets = torch.arange(rows, device=device, dtype=torch.float32).view(rows, 1) * 0.001
    col_offsets = torch.arange(dim, device=device, dtype=torch.float32).view(1, dim) * 0.00001
    return (base + row_offsets + col_offsets).to(dtype=torch.bfloat16)


def _raw_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    if name == "uint8":
        return torch.uint8
    if name == "int8":
        return torch.int8
    if name == "fp8_e4m3":
        return torch.float8_e4m3fn
    if name == "fp8_e5m2":
        return torch.float8_e5m2
    if name == "fp8_e8m0":
        return torch.float8_e8m0fnu
    raise RuntimeError(f"unknown raw dtype {name!r}")


def _align_up(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _packed_row_bytes(dim: int, block_size: int = 32, alignment: int = 128) -> int:
    q_bytes = int(dim)
    scale_bytes = int(dim) // int(block_size)
    scale_offset = _align_up(q_bytes, alignment)
    return _align_up(scale_offset + scale_bytes, alignment)


def _make_raw_input(
    rows: int,
    dim: int,
    rank: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    row_ids = torch.arange(rows, dtype=torch.int64, device=device).view(rows, 1)
    col_ids = torch.arange(dim, dtype=torch.int64, device=device).view(1, dim)
    pattern = rank * 17 + row_ids * 3 + col_ids
    if dtype == torch.uint8:
        return (pattern % 251).to(dtype=torch.uint8)
    if dtype == torch.int8:
        return ((pattern % 127) - 63).to(dtype=torch.int8)
    if dtype == torch.float8_e8m0fnu:
        # E8M0 has no sign or mantissa. Use exact powers-of-two-like values.
        exponents = (pattern % 8).to(dtype=torch.float32) - 4.0
        return torch.pow(torch.full_like(exponents, 2.0), exponents).to(dtype=dtype)
    values = (
        float(rank + 1)
        + row_ids.to(dtype=torch.float32) * 0.01
        + col_ids.to(dtype=torch.float32) * 0.001
    )
    return values.to(dtype=dtype)


def _assert_raw_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual_bytes = actual.contiguous().view(torch.uint8)
    expected_bytes = expected.contiguous().view(torch.uint8)
    torch.testing.assert_close(actual_bytes, expected_bytes, atol=0, rtol=0)


def _raw_max_byte_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_bytes = actual.contiguous().view(torch.uint8).to(torch.int16)
    expected_bytes = expected.contiguous().view(torch.uint8).to(torch.int16)
    return float((actual_bytes - expected_bytes).abs().max().item())


def _run_raw_dispatch(
    *,
    args: argparse.Namespace,
    source: torch.Tensor,
    dst_ranks: torch.Tensor,
    dst_rows: torch.Tensor,
    out: torch.Tensor,
    probs: Optional[torch.Tensor],
    group_name: str,
    nblocks: int,
) -> None:
    from olmo_core.kernels.symm_mem_vdev2d import rowwise_dispatch_put

    if args.zero_raw_dispatch_out:
        out.fill_(0)
    rowwise_dispatch_put(
        source,
        out,
        dst_ranks,
        dst_rows,
        group_name,
        probs=probs,
        nblocks=nblocks,
        pre_barrier=args.raw_pre_barrier,
        post_barrier=args.raw_post_barrier,
    )


def _run_raw_gather(
    *,
    args: argparse.Namespace,
    expert: torch.Tensor,
    out: torch.Tensor,
    src_ranks: torch.Tensor,
    src_rows: torch.Tensor,
    group_name: str,
    nblocks: int,
) -> None:
    from olmo_core.kernels.symm_mem_vdev2d import rowwise_gather_get

    rowwise_gather_get(
        expert,
        out,
        src_ranks,
        src_rows,
        group_name,
        nblocks=nblocks,
        pre_barrier=args.raw_pre_barrier,
        post_barrier=args.raw_post_barrier,
    )


def _run_dispatch_scaled(
    *,
    args: argparse.Namespace,
    source: torch.Tensor,
    dst_ranks: torch.Tensor,
    dst_rows: torch.Tensor,
    out_q: torch.Tensor,
    out_scales: torch.Tensor,
    input_q: Optional[torch.Tensor],
    input_scales: Optional[torch.Tensor],
    packed_input: Optional[torch.Tensor],
    packed_out: Optional[torch.Tensor],
    group_name: str,
    nblocks: int,
) -> None:
    if args.dispatch_impl == "fused_weighted":
        if input_q is not None or input_scales is not None:
            raise RuntimeError("--dispatch-impl fused_weighted does not support --scaled-symmetric-source")
        from olmo_core.kernels.symm_mem_vdev2d import rowwise_dispatch_put_scaled_weighted

        probs = torch.ones(
            tuple(dst_ranks.shape),
            dtype=torch.float32,
            device=source.device,
        )
        rowwise_dispatch_put_scaled_weighted(
            source,
            probs,
            out_q,
            out_scales,
            dst_ranks,
            dst_rows,
            group_name,
            block_size=32,
            nblocks=nblocks,
        )
        return

    if args.dispatch_impl == "paired_put":
        from olmo_core.kernels.symm_mem_vdev2d import rowwise_dispatch_put_scaled_pair

        rowwise_dispatch_put_scaled_pair(
            source.contiguous(),
            out_q,
            out_scales,
            dst_ranks,
            dst_rows,
            group_name,
            block_size=32,
            nblocks=nblocks,
            input_q=input_q,
            input_scales=input_scales,
        )
        return

    if args.dispatch_impl == "packed_put":
        from olmo_core.kernels.symm_mem_vdev2d import rowwise_dispatch_put_scaled_packed

        rowwise_dispatch_put_scaled_packed(
            source.contiguous(),
            out_q,
            out_scales,
            dst_ranks,
            dst_rows,
            group_name,
            block_size=32,
            nblocks=nblocks,
            input_q=input_q,
            input_scales=input_scales,
            packed_input=packed_input,
            packed_out=packed_out,
        )
        return

    from olmo_core.kernels.symm_mem_vdev2d import rowwise_dispatch_put_scaled

    rowwise_dispatch_put_scaled(
        source.contiguous(),
        out_q,
        out_scales,
        dst_ranks,
        dst_rows,
        group_name,
        block_size=32,
        nblocks=nblocks,
        input_q=input_q,
        input_scales=input_scales,
    )


def _run_combine_scaled(
    *,
    args: argparse.Namespace,
    expert_q: torch.Tensor,
    expert_scales: torch.Tensor,
    out: torch.Tensor,
    src_ranks: torch.Tensor,
    src_rows: torch.Tensor,
    probs: Optional[torch.Tensor],
    gathered_q: Optional[torch.Tensor],
    gathered_scales: Optional[torch.Tensor],
    group_name: str,
    nblocks: int,
) -> None:
    if args.combine_impl == "paired_get":
        from olmo_core.kernels.symm_mem_vdev2d import rowwise_combine_get_scaled_pair

        if src_ranks.ndim != 2 or src_rows.ndim != 2:
            raise RuntimeError("src_ranks/src_rows must be rank-2")
        rowwise_combine_get_scaled_pair(
            expert_q,
            expert_scales,
            out,
            src_ranks,
            src_rows,
            group_name,
            probs=probs,
            block_size=32,
            nblocks=nblocks,
            gathered_q_out=gathered_q,
            gathered_scales_out=gathered_scales,
        )
        return

    from olmo_core.kernels.symm_mem_vdev2d import rowwise_combine_get_scaled

    if src_ranks.ndim != 2 or src_rows.ndim != 2:
        raise RuntimeError("src_ranks/src_rows must be rank-2")
    rowwise_combine_get_scaled(
        expert_q,
        expert_scales,
        out,
        src_ranks,
        src_rows,
        group_name,
        probs=probs,
        block_size=32,
        nblocks=nblocks,
        gathered_q_out=gathered_q,
        gathered_scales_out=gathered_scales,
    )


def _quantize_local_expert(
    x: torch.Tensor,
    q: torch.Tensor,
    scales: torch.Tensor,
) -> None:
    from olmo_core.kernels.mxfp8_utils import quantize_rows_to_mxfp8

    quantize_rows_to_mxfp8(x.contiguous(), block_size=32, out=q, scales_out=scales)


def _dequantize(q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    from olmo_core.kernels.mxfp8_utils import dequantize_rows_from_mxfp8

    return dequantize_rows_from_mxfp8(
        q,
        scales,
        block_size=32,
        out_dtype=torch.bfloat16,
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args = _parse_args()
    scaled_mode = args.mode in {"dispatch_scaled", "combine_scaled", "roundtrip_scaled"}
    raw_mode = args.mode in {"raw_dispatch", "raw_dispatch_weighted", "raw_gather"}
    if scaled_mode and args.dim % 32 != 0:
        raise RuntimeError(f"--dim must be divisible by 32, got {args.dim}")
    if args.rows <= 0 or args.top_k <= 0:
        raise RuntimeError("--rows and --top-k must be positive")
    num_routes = args.rows * args.top_k
    capacity = args.capacity if args.capacity is not None else num_routes
    if capacity < num_routes:
        raise RuntimeError(
            f"--capacity must be >= rows * top_k ({num_routes}), got {capacity}"
        )
    if args.mode in {"raw_dispatch_weighted", "raw_gather"} and args.top_k != 1:
        raise RuntimeError("raw_dispatch_weighted/raw_gather currently require --top-k 1")
    if args.mode == "raw_dispatch_weighted" and args.raw_dtype not in {"bf16", "fp16"}:
        raise RuntimeError("raw_dispatch_weighted only supports --raw-dtype bf16 or fp16")
    if args.strict_comm_compare:
        if args.mode == "dispatch_scaled" and args.dispatch_impl not in {"paired_put", "packed_put"}:
            raise RuntimeError(
                "--strict-comm-compare for dispatch_scaled requires --dispatch-impl paired_put or packed_put"
            )
        if (
            args.mode == "dispatch_scaled"
            and args.peer_pattern == "remote_node"
            and not args.scaled_symmetric_source
        ):
            raise RuntimeError(
                "--strict-comm-compare dispatch_scaled with remote_node compares against two_put, "
                "so it also needs --scaled-symmetric-source for the reference PUT source"
            )
        if args.mode == "combine_scaled" and args.combine_impl != "paired_get":
            raise RuntimeError("--strict-comm-compare for combine_scaled requires --combine-impl paired_get")
        if args.mode not in {"dispatch_scaled", "combine_scaled"}:
            raise RuntimeError("--strict-comm-compare currently supports dispatch_scaled or combine_scaled")

    rank, local_rank, world_size, device = _init_dist()
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(torch.cuda.device_count())))
    group = dist.group.WORLD

    group_name = _setup_olmo_symm_mem(group, device)

    source = _make_input(args.rows, args.dim, rank, device)
    dst_ranks, dst_rows = _make_route_maps(
        rows=args.rows,
        top_k=args.top_k,
        rank=rank,
        world_size=world_size,
        local_world_size=local_world_size,
        pattern=args.peer_pattern,
        device=device,
    )
    dst_rank = _destination_rank(
        rank=rank,
        world_size=world_size,
        local_world_size=local_world_size,
        pattern=args.peer_pattern,
    )
    src_rank = _expected_source_rank(
        rank=rank,
        world_size=world_size,
        local_world_size=local_world_size,
        pattern=args.peer_pattern,
    )
    if raw_mode:
        dtype = _raw_dtype(args.raw_dtype)
        source = _make_raw_input(args.rows, args.dim, rank, dtype, device)
        expected_dispatch = _make_raw_input(args.rows, args.dim, src_rank, dtype, device)
        if args.mode == "raw_dispatch" and args.top_k > 1:
            expected_dispatch = expected_dispatch.repeat_interleave(args.top_k, dim=0)
        expected_gather = _make_raw_input(args.rows, args.dim, dst_rank, dtype, device)
    else:
        source = _make_input(args.rows, args.dim, rank, device)
        expected_dispatch = _make_input(args.rows, args.dim, src_rank, device).repeat_interleave(
            args.top_k,
            dim=0,
        )
        expected_gather = _make_input(args.rows, args.dim, dst_rank, device)

    probs = torch.full(
        (args.rows, args.top_k),
        1.0 / float(args.top_k),
        dtype=torch.float32,
        device=device,
    )

    q: Optional[torch.Tensor] = None
    scales: Optional[torch.Tensor] = None
    raw_symm: Optional[torch.Tensor] = None
    raw_source_symm: Optional[torch.Tensor] = None
    raw_dest_symm: Optional[torch.Tensor] = None
    scaled_source_q: Optional[torch.Tensor] = None
    scaled_source_scales: Optional[torch.Tensor] = None
    scaled_source_packed: Optional[torch.Tensor] = None
    scaled_out_packed: Optional[torch.Tensor] = None
    scaled_gather_q: Optional[torch.Tensor] = None
    scaled_gather_scales: Optional[torch.Tensor] = None
    strict_q_ref: Optional[torch.Tensor] = None
    strict_scales_ref: Optional[torch.Tensor] = None
    strict_out_ref: Optional[torch.Tensor] = None
    strict_gather_q_ref: Optional[torch.Tensor] = None
    strict_gather_scales_ref: Optional[torch.Tensor] = None
    if scaled_mode:
        q = _alloc_symm((capacity, args.dim), dtype=torch.float8_e4m3fn, device=device, group=group)
        scales = _alloc_symm(
            (capacity, args.dim // 32),
            dtype=torch.float8_e8m0fnu,
            device=device,
            group=group,
        )
        out = torch.empty((args.rows, args.dim), device=device, dtype=torch.bfloat16)
        if args.scaled_symmetric_source:
            scaled_source_q = _alloc_symm(
                (args.rows, args.dim),
                dtype=torch.float8_e4m3fn,
                device=device,
                group=group,
            )
            scaled_source_scales = _alloc_symm(
                (args.rows, args.dim // 32),
                dtype=torch.float8_e8m0fnu,
                device=device,
                group=group,
            )
        if args.dispatch_impl == "packed_put":
            packed_row_bytes = _packed_row_bytes(args.dim)
            scaled_source_packed = _alloc_symm(
                (args.rows, packed_row_bytes),
                dtype=torch.uint8,
                device=device,
                group=group,
            )
            scaled_out_packed = _alloc_symm(
                (capacity, packed_row_bytes),
                dtype=torch.uint8,
                device=device,
                group=group,
            )
        if args.strict_comm_compare and args.mode == "dispatch_scaled":
            strict_q_ref = _alloc_symm(
                (capacity, args.dim),
                dtype=torch.float8_e4m3fn,
                device=device,
                group=group,
            )
            strict_scales_ref = _alloc_symm(
                (capacity, args.dim // 32),
                dtype=torch.float8_e8m0fnu,
                device=device,
                group=group,
            )
        if args.strict_comm_compare and args.mode == "combine_scaled":
            strict_out_ref = torch.empty_like(out)
            if args.scaled_symmetric_dest:
                strict_gather_q_ref = _alloc_symm(
                    (args.rows, args.top_k, args.dim),
                    dtype=torch.float8_e4m3fn,
                    device=device,
                    group=group,
                )
                strict_gather_scales_ref = _alloc_symm(
                    (args.rows, args.top_k, args.dim // 32),
                    dtype=torch.float8_e8m0fnu,
                    device=device,
                    group=group,
                )
        if args.scaled_symmetric_dest:
            scaled_gather_q = _alloc_symm(
                (args.rows, args.top_k, args.dim),
                dtype=torch.float8_e4m3fn,
                device=device,
                group=group,
            )
            scaled_gather_scales = _alloc_symm(
                (args.rows, args.top_k, args.dim // 32),
                dtype=torch.float8_e8m0fnu,
                device=device,
                group=group,
            )
    else:
        raw_symm = _alloc_symm((capacity, args.dim), dtype=source.dtype, device=device, group=group)
        out = torch.empty((args.rows, args.dim), device=device, dtype=source.dtype)
        if args.raw_symmetric_source:
            raw_source_symm = _alloc_symm((args.rows, args.dim), dtype=source.dtype, device=device, group=group)
            raw_source_symm.copy_(source)
        if args.raw_symmetric_dest:
            raw_dest_symm = _alloc_symm((args.rows, args.dim), dtype=source.dtype, device=device, group=group)

    def run_once() -> torch.Tensor:
        if args.mode == "dispatch_scaled":
            assert q is not None and scales is not None
            _run_dispatch_scaled(
                args=args,
                source=source,
                dst_ranks=dst_ranks,
                dst_rows=dst_rows,
                out_q=q,
                out_scales=scales,
                input_q=scaled_source_q,
                input_scales=scaled_source_scales,
                packed_input=scaled_source_packed,
                packed_out=scaled_out_packed,
                group_name=group_name,
                nblocks=args.nblocks,
            )
            return _dequantize(q, scales)[:num_routes]

        if args.mode == "combine_scaled":
            assert q is not None and scales is not None
            expert_source = source.repeat_interleave(args.top_k, dim=0)
            _quantize_local_expert(expert_source, q[:num_routes], scales[:num_routes])
            _run_combine_scaled(
                args=args,
                expert_q=q,
                expert_scales=scales,
                out=out,
                src_ranks=dst_ranks,
                src_rows=dst_rows,
                probs=probs,
                gathered_q=scaled_gather_q,
                gathered_scales=scaled_gather_scales,
                group_name=group_name,
                nblocks=args.nblocks,
            )
            return out

        if args.mode in {"raw_dispatch", "raw_dispatch_weighted"}:
            assert raw_symm is not None
            dispatch_source = source
            if raw_source_symm is not None:
                if args.raw_copy_source_each_iter:
                    raw_source_symm.copy_(source)
                dispatch_source = raw_source_symm
            _run_raw_dispatch(
                args=args,
                source=dispatch_source,
                dst_ranks=dst_ranks,
                dst_rows=dst_rows,
                out=raw_symm,
                probs=probs if args.mode == "raw_dispatch_weighted" else None,
                group_name=group_name,
                nblocks=args.nblocks,
            )
            return raw_symm[:num_routes]

        if args.mode == "raw_gather":
            assert raw_symm is not None
            raw_symm.copy_(source)
            gather_out = raw_dest_symm if raw_dest_symm is not None else out
            _run_raw_gather(
                args=args,
                expert=raw_symm,
                out=gather_out,
                src_ranks=dst_ranks,
                src_rows=dst_rows,
                group_name=group_name,
                nblocks=args.nblocks,
            )
            return gather_out

        assert q is not None and scales is not None
        _run_dispatch_scaled(
            args=args,
            source=source,
            dst_ranks=dst_ranks,
            dst_rows=dst_rows,
            out_q=q,
            out_scales=scales,
            input_q=scaled_source_q,
            input_scales=scaled_source_scales,
            packed_input=scaled_source_packed,
            packed_out=scaled_out_packed,
            group_name=group_name,
            nblocks=args.nblocks,
        )
        _run_combine_scaled(
            args=args,
            expert_q=q,
            expert_scales=scales,
            out=out,
            src_ranks=dst_ranks,
            src_rows=dst_rows,
            probs=probs,
            gathered_q=scaled_gather_q,
            gathered_scales=scaled_gather_scales,
            group_name=group_name,
            nblocks=args.nblocks,
        )
        return out

    try:
        for idx in range(args.warmup):
            run_once()
        torch.cuda.synchronize(device)
        dist.barrier(group=group)

        times_ms: list[float] = []
        result = None
        for idx in range(args.iters):
            dist.barrier(group=group)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            result = run_once()
            torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            times_ms.append(elapsed_ms)

        strict_max_byte_diff = 0.0
        if args.strict_comm_compare:
            dist.barrier(group=group)
            torch.cuda.synchronize(device)
            if args.mode == "dispatch_scaled":
                assert q is not None and scales is not None
                assert strict_q_ref is not None and strict_scales_ref is not None
                saved_dispatch_impl = args.dispatch_impl
                try:
                    args.dispatch_impl = "two_put"
                    _run_dispatch_scaled(
                        args=args,
                        source=source,
                        dst_ranks=dst_ranks,
                        dst_rows=dst_rows,
                        out_q=strict_q_ref,
                        out_scales=strict_scales_ref,
                        input_q=scaled_source_q,
                        input_scales=scaled_source_scales,
                        packed_input=scaled_source_packed,
                        packed_out=scaled_out_packed,
                        group_name=group_name,
                        nblocks=args.nblocks,
                    )
                finally:
                    args.dispatch_impl = saved_dispatch_impl
                _run_dispatch_scaled(
                    args=args,
                    source=source,
                    dst_ranks=dst_ranks,
                    dst_rows=dst_rows,
                    out_q=q,
                    out_scales=scales,
                    input_q=scaled_source_q,
                    input_scales=scaled_source_scales,
                    packed_input=scaled_source_packed,
                    packed_out=scaled_out_packed,
                    group_name=group_name,
                    nblocks=args.nblocks,
                )
                torch.cuda.synchronize(device)
                _assert_raw_equal(q[:num_routes], strict_q_ref[:num_routes])
                _assert_raw_equal(scales[:num_routes], strict_scales_ref[:num_routes])
                strict_max_byte_diff = max(
                    _raw_max_byte_diff(q[:num_routes], strict_q_ref[:num_routes]),
                    _raw_max_byte_diff(scales[:num_routes], strict_scales_ref[:num_routes]),
                )
            elif args.mode == "combine_scaled":
                assert q is not None and scales is not None
                assert strict_out_ref is not None
                expert_source = source.repeat_interleave(args.top_k, dim=0)
                _quantize_local_expert(expert_source, q[:num_routes], scales[:num_routes])
                saved_combine_impl = args.combine_impl
                try:
                    args.combine_impl = "two_get"
                    _run_combine_scaled(
                        args=args,
                        expert_q=q,
                        expert_scales=scales,
                        out=strict_out_ref,
                        src_ranks=dst_ranks,
                        src_rows=dst_rows,
                        probs=probs,
                        gathered_q=(
                            strict_gather_q_ref
                            if strict_gather_q_ref is not None
                            else scaled_gather_q
                        ),
                        gathered_scales=(
                            strict_gather_scales_ref
                            if strict_gather_scales_ref is not None
                            else scaled_gather_scales
                        ),
                        group_name=group_name,
                        nblocks=args.nblocks,
                    )
                finally:
                    args.combine_impl = saved_combine_impl
                _run_combine_scaled(
                    args=args,
                    expert_q=q,
                    expert_scales=scales,
                    out=out,
                    src_ranks=dst_ranks,
                    src_rows=dst_rows,
                    probs=probs,
                    gathered_q=scaled_gather_q,
                    gathered_scales=scaled_gather_scales,
                    group_name=group_name,
                    nblocks=args.nblocks,
                )
                torch.cuda.synchronize(device)
                _assert_raw_equal(out, strict_out_ref)
                strict_max_byte_diff = _raw_max_byte_diff(out, strict_out_ref)
                if strict_gather_q_ref is not None and scaled_gather_q is not None:
                    _assert_raw_equal(scaled_gather_q, strict_gather_q_ref)
                    strict_max_byte_diff = max(
                        strict_max_byte_diff,
                        _raw_max_byte_diff(scaled_gather_q, strict_gather_q_ref),
                    )
                if strict_gather_scales_ref is not None and scaled_gather_scales is not None:
                    _assert_raw_equal(scaled_gather_scales, strict_gather_scales_ref)
                    strict_max_byte_diff = max(
                        strict_max_byte_diff,
                        _raw_max_byte_diff(scaled_gather_scales, strict_gather_scales_ref),
                    )
            else:
                raise RuntimeError(f"strict compare does not support mode {args.mode!r}")

        assert result is not None
        if args.mode == "roundtrip_scaled":
            expected = source
        elif args.mode in {"dispatch_scaled", "raw_dispatch", "raw_dispatch_weighted"}:
            expected = expected_dispatch
        elif args.mode in {"combine_scaled", "raw_gather"}:
            expected = expected_gather
        else:
            raise RuntimeError(f"unknown mode {args.mode!r}")
        if raw_mode:
            _assert_raw_equal(result, expected)
            max_err = _raw_max_byte_diff(result, expected)
        else:
            torch.testing.assert_close(result, expected, atol=args.atol, rtol=args.rtol)
            max_err = float((result.float() - expected.float()).abs().max().item())

        local = torch.tensor(
            [statistics.median(times_ms), max_err, strict_max_byte_diff],
            dtype=torch.float64,
            device=device,
        )
        gathered = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(gathered, local, group=group)
        if rank == 0:
            max_ms = max(float(item[0].item()) for item in gathered)
            max_err = max(float(item[1].item()) for item in gathered)
            max_strict_byte_diff = max(float(item[2].item()) for item in gathered)
            print(
                "ROWWISE_FP8_NVSHMEM_SMOKE_OK "
                f"mode={args.mode} ranks={world_size} rows={args.rows} dim={args.dim} "
                f"top_k={args.top_k} capacity={capacity} nblocks={args.nblocks} "
                f"peer_pattern={args.peer_pattern} dispatch_impl={args.dispatch_impl} "
                f"combine_impl={args.combine_impl} "
                f"raw_dtype={args.raw_dtype} "
                f"raw_copy_source_each_iter={args.raw_copy_source_each_iter} "
                f"zero_raw_dispatch_out={args.zero_raw_dispatch_out} "
                f"raw_pre_barrier={args.raw_pre_barrier} "
                f"raw_post_barrier={args.raw_post_barrier} "
                f"strict_comm_compare={args.strict_comm_compare} "
                f"strict_max_byte_diff={max_strict_byte_diff:.0f} "
                f"max_rank_median_ms={max_ms:.3f} "
                f"max_err={max_err:.6f}",
                flush=True,
            )
        dist.barrier(group=group)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
