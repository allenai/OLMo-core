import argparse
from statistics import median
from typing import Callable, Dict, List, Tuple

import torch

from olmo_core.nn.moe.utils import (
    build_chunk_routing_map_no_compile,
    moe_chunk_reorder_no_compile,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark chunk reorder backends (te|triton|cuda) for MoE no-sync 1D path."
    )
    parser.add_argument("--rows", type=int, default=16384, help="Input row count.")
    parser.add_argument("--hidden", type=int, default=4096, help="Input hidden size.")
    parser.add_argument("--ep-world-size", type=int, default=8, help="Number of EP source ranks.")
    parser.add_argument("--num-local-experts", type=int, default=4, help="Local experts per rank.")
    parser.add_argument("--tail-fraction", type=float, default=0.1, help="Tail (unused) row fraction.")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--backends",
        type=str,
        default="te,triton,cuda",
        help="Comma separated backend list.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    return parser.parse_args()


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _event_timed_ms(fn: Callable[[], None], warmup_iters: int, iters: int) -> List[float]:
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    times_ms: List[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times_ms.append(float(start.elapsed_time(end)))
    return times_ms


def _build_splits(
    rows: int,
    ep_world_size: int,
    num_local_experts: int,
    tail_fraction: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if not (0.0 <= tail_fraction < 1.0):
        raise ValueError(f"tail_fraction must be in [0, 1), got {tail_fraction}")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    num_chunks = ep_world_size * num_local_experts
    valid_rows = int(rows * (1.0 - tail_fraction))
    probs = torch.rand(num_chunks, generator=generator, device=device, dtype=torch.float32)
    probs = probs / probs.sum()
    counts = torch.floor(probs * valid_rows).to(dtype=torch.long)
    assigned = int(counts.sum().item())
    rem = valid_rows - assigned
    if rem > 0:
        idx = torch.randint(0, num_chunks, (rem,), generator=generator, device=device)
        counts.index_add_(0, idx, torch.ones_like(idx, dtype=torch.long))
    return counts.view(ep_world_size, num_local_experts)


def _chunk_reorder_roundtrip(
    x: torch.Tensor,
    recv_splits_by_src_local: torch.Tensor,
    backend: str,
    out_buffer: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    routing_map = build_chunk_routing_map_no_compile(
        recv_splits_by_src_local,
        rows=x.shape[0],
    )
    local_major, row_id_map = moe_chunk_reorder_no_compile(
        x,
        routing_map=routing_map,
        num_out_tokens=x.shape[0],
        backend=backend,
    )
    restored = moe_chunk_reorder_no_compile(
        local_major,
        row_id_map=row_id_map,
        out=out_buffer,
        backend=backend,
    )
    return local_major, restored


def _check_parity(
    x: torch.Tensor,
    recv_splits_by_src_local: torch.Tensor,
    backend: str,
    atol: float,
    rtol: float,
) -> Dict[str, bool]:
    result: Dict[str, bool] = {}

    x_ref = x.detach().clone().requires_grad_(True)
    out_ref_buf = torch.empty_like(x_ref)
    _, restored_ref = _chunk_reorder_roundtrip(x_ref, recv_splits_by_src_local, "te", out_ref_buf)
    grad_w = torch.randn_like(restored_ref)
    loss_ref = (restored_ref * grad_w).sum()
    loss_ref.backward()
    grad_ref = x_ref.grad.detach().clone()

    x_test = x.detach().clone().requires_grad_(True)
    out_test_buf = torch.empty_like(x_test)
    _, restored_test = _chunk_reorder_roundtrip(x_test, recv_splits_by_src_local, backend, out_test_buf)
    loss_test = (restored_test * grad_w).sum()
    loss_test.backward()
    grad_test = x_test.grad.detach().clone()

    torch.testing.assert_close(restored_test, restored_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(grad_test, grad_ref, atol=atol, rtol=rtol)
    result["output_match_te"] = True
    result["grad_match_te"] = True
    result["out_alias"] = restored_test.data_ptr() == out_test_buf.data_ptr()
    return result


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args = _parse_args()
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = _dtype_from_name(args.dtype)
    backends = [b.strip().lower() for b in args.backends.split(",") if b.strip()]

    torch.manual_seed(args.seed)
    recv_splits_by_src_local = _build_splits(
        rows=args.rows,
        ep_world_size=args.ep_world_size,
        num_local_experts=args.num_local_experts,
        tail_fraction=args.tail_fraction,
        seed=args.seed + 1,
        device=device,
    )
    x_base = torch.randn(args.rows, args.hidden, device=device, dtype=dtype)

    payload_gb = (x_base.numel() * x_base.element_size()) / 1e9
    print(
        f"rows={args.rows} hidden={args.hidden} dtype={dtype} "
        f"ep_world_size={args.ep_world_size} num_local_experts={args.num_local_experts} "
        f"valid_rows={int(recv_splits_by_src_local.sum().item())} tail_rows={args.rows - int(recv_splits_by_src_local.sum().item())}"
    )

    results = []
    for backend in backends:
        try:
            out_buf = torch.empty_like(x_base)
            # Prime backend (includes CUDA JIT compilation if needed).
            _chunk_reorder_roundtrip(
                x_base.detach(),
                recv_splits_by_src_local,
                backend,
                out_buf,
            )

            parity = _check_parity(
                x_base.detach(),
                recv_splits_by_src_local,
                backend,
                atol=args.atol,
                rtol=args.rtol,
            )

            def run_forward() -> None:
                routing_map = build_chunk_routing_map_no_compile(
                    recv_splits_by_src_local,
                    rows=x_base.shape[0],
                )
                out_local, row_id_map = moe_chunk_reorder_no_compile(
                    x_base,
                    routing_map=routing_map,
                    num_out_tokens=x_base.shape[0],
                    backend=backend,
                )
                _ = moe_chunk_reorder_no_compile(
                    out_local,
                    row_id_map=row_id_map,
                    out=out_buf,
                    backend=backend,
                )

            x_bwd = x_base.detach().clone().requires_grad_(True)
            bwd_out_buf = torch.empty_like(x_bwd)
            out_local_bwd, restored_bwd = _chunk_reorder_roundtrip(
                x_bwd, recv_splits_by_src_local, backend, bwd_out_buf
            )
            del out_local_bwd
            grad_out = torch.randn_like(restored_bwd)

            def run_backward_only() -> None:
                if x_bwd.grad is not None:
                    x_bwd.grad.zero_()
                restored_bwd.backward(grad_out, retain_graph=True)

            def run_fwd_bwd() -> None:
                x_fb = x_base.detach().clone().requires_grad_(True)
                fb_out_buf = torch.empty_like(x_fb)
                _, restored_fb = _chunk_reorder_roundtrip(
                    x_fb, recv_splits_by_src_local, backend, fb_out_buf
                )
                loss = (restored_fb * restored_fb).sum()
                loss.backward()

            fwd_times = _event_timed_ms(run_forward, args.warmup_iters, args.iters)
            bwd_times = _event_timed_ms(run_backward_only, args.warmup_iters, args.iters)
            fwd_bwd_times = _event_timed_ms(run_fwd_bwd, args.warmup_iters, args.iters)

            # Round-trip forward and backward each move one full tensor twice.
            fwd_bw = (2.0 * payload_gb) / (median(fwd_times) / 1e3)
            bwd_bw = (2.0 * payload_gb) / (median(bwd_times) / 1e3)
            fwd_bwd_bw = (4.0 * payload_gb) / (median(fwd_bwd_times) / 1e3)

            results.append(
                {
                    "backend": backend,
                    "fwd_ms": median(fwd_times),
                    "bwd_ms": median(bwd_times),
                    "fwd_bwd_ms": median(fwd_bwd_times),
                    "fwd_bw_gbs": fwd_bw,
                    "bwd_bw_gbs": bwd_bw,
                    "fwd_bwd_bw_gbs": fwd_bwd_bw,
                    **parity,
                }
            )
        except Exception as e:
            results.append({"backend": backend, "error": str(e)})

    print("=== Chunk Reorder Benchmark ===")
    for item in results:
        if "error" in item:
            print(f"{item['backend']}: ERROR: {item['error']}")
            continue
        print(
            f"{item['backend']}: "
            f"fwd={item['fwd_ms']:.3f} ms ({item['fwd_bw_gbs']:.2f} GB/s), "
            f"bwd={item['bwd_ms']:.3f} ms ({item['bwd_bw_gbs']:.2f} GB/s), "
            f"fwd+bwd={item['fwd_bwd_ms']:.3f} ms ({item['fwd_bwd_bw_gbs']:.2f} GB/s), "
            f"parity_out={item['output_match_te']} parity_grad={item['grad_match_te']} "
            f"out_alias={item['out_alias']}"
        )


if __name__ == "__main__":
    main()
