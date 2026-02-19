import argparse
from statistics import median
from typing import Callable

import torch

from olmo_core.nn.moe.utils import (
    moe_permute_no_compile,
    moe_unpermute_1d_fused_drop_no_compile,
    moe_unpermute_no_compile,
)


def _parse_csv_int(arg: str) -> list[int]:
    return [int(v.strip()) for v in arg.split(",") if v.strip()]


def _parse_csv_float(arg: str) -> list[float]:
    return [float(v.strip()) for v in arg.split(",") if v.strip()]


def _dtype_from_name(name: str) -> torch.dtype:
    name = name.lower()
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _build_keep_reorder(
    *,
    num_rows: int,
    keep_fraction: float,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    keep_count = int(round(keep_fraction * num_rows))
    keep_count = max(0, min(keep_count, num_rows))
    keep_mask = torch.zeros(num_rows, device=device, dtype=torch.bool)
    if keep_count > 0:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        keep_rows = torch.randperm(num_rows, generator=g, device=device)[:keep_count]
        keep_mask[keep_rows] = True

    token_ids = torch.arange(num_rows, device=device, dtype=torch.long)
    keep_i64 = keep_mask.to(dtype=torch.long)
    drop_i64 = (~keep_mask).to(dtype=torch.long)
    keep_rank = torch.cumsum(keep_i64, dim=0) - 1
    drop_rank = torch.cumsum(drop_i64, dim=0) - 1
    num_kept = keep_i64.sum(dtype=torch.long)
    packed_pos = torch.where(keep_mask, keep_rank, num_kept + drop_rank)

    reorder_indices = torch.empty_like(token_ids)
    reorder_indices.scatter_(0, packed_pos, token_ids)

    inverse_reorder_indices = torch.empty_like(reorder_indices)
    inverse_reorder_indices.scatter_(0, reorder_indices, token_ids)
    packed_keep_mask = keep_mask.index_select(0, reorder_indices)
    return reorder_indices, inverse_reorder_indices, packed_keep_mask


def _legacy_restore_drop_unpermute(
    *,
    combine_out: torch.Tensor,
    row_id_map: torch.Tensor,
    local_inverse_reorder_indices: torch.Tensor,
    packed_keep_mask: torch.Tensor,
    merging_probs: torch.Tensor,
    restore_shape: torch.Size,
) -> torch.Tensor:
    restored = combine_out.index_select(0, local_inverse_reorder_indices)
    restored_keep_mask = packed_keep_mask.index_select(0, local_inverse_reorder_indices)
    restored = torch.where(
        restored_keep_mask.unsqueeze(-1),
        restored,
        torch.zeros_like(restored),
    )
    return moe_unpermute_no_compile(
        inp=restored,
        row_id_map=row_id_map,
        merging_probs=merging_probs,
        restore_shape=restore_shape,
        map_type="index",
    )


def _event_timed_ms(fn: Callable[[], None], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
    return median(times)


def _build_case(
    *,
    num_tokens: int,
    d_model: int,
    top_k: int,
    keep_fraction: float,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Size]:
    torch.manual_seed(seed)
    num_experts = 32
    x = torch.randn(num_tokens, d_model, device=device, dtype=dtype)
    routing_map = torch.randint(
        low=0,
        high=num_experts,
        size=(num_tokens, top_k),
        device=device,
        dtype=torch.int32,
    )
    permuted, row_id_map = moe_permute_no_compile(
        inp=x,
        routing_map=routing_map,
        num_out_tokens=num_tokens * top_k,
        map_type="index",
    )
    reorder_indices, local_inverse_reorder_indices, packed_keep_mask = _build_keep_reorder(
        num_rows=permuted.shape[0],
        keep_fraction=keep_fraction,
        seed=seed + 1,
        device=device,
    )
    combine_out = permuted.index_select(0, reorder_indices)
    merging_probs = torch.rand(num_tokens, top_k, device=device, dtype=torch.float32)
    num_kept = packed_keep_mask.to(dtype=torch.long).sum(dtype=torch.long)
    return (
        combine_out,
        row_id_map,
        local_inverse_reorder_indices,
        packed_keep_mask,
        merging_probs,
        num_kept,
        x.shape,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark 1D restore+unpermute fused TE path.")
    parser.add_argument("--num-tokens", type=int, default=2048)
    parser.add_argument("--d-models", type=str, default="1024,2048,3072")
    parser.add_argument("--top-k-values", type=str, default="4")
    parser.add_argument("--keep-fractions", type=str, default="1.0,0.7,0.4")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda", torch.cuda.current_device())
    dtype = _dtype_from_name(args.dtype)
    d_models = _parse_csv_int(args.d_models)
    top_k_values = _parse_csv_int(args.top_k_values)
    keep_fractions = _parse_csv_float(args.keep_fractions)

    failures: list[str] = []
    print("=== RestoreDrop+Unpermute Fusion Benchmark ===")
    print(
        f"num_tokens={args.num_tokens} dtype={dtype} d_models={d_models} "
        f"top_k_values={top_k_values} keep_fractions={keep_fractions}"
    )

    case_idx = 0
    for d_model in d_models:
        for top_k in top_k_values:
            for keep_fraction in keep_fractions:
                case_seed = args.seed + case_idx * 17
                case_idx += 1

                (
                    combine_out,
                    row_id_map,
                    local_inverse_reorder_indices,
                    packed_keep_mask,
                    merging_probs,
                    num_kept,
                    restore_shape,
                ) = _build_case(
                    num_tokens=args.num_tokens,
                    d_model=d_model,
                    top_k=top_k,
                    keep_fraction=keep_fraction,
                    dtype=dtype,
                    device=device,
                    seed=case_seed,
                )

                # correctness parity
                out_legacy = _legacy_restore_drop_unpermute(
                    combine_out=combine_out,
                    row_id_map=row_id_map,
                    local_inverse_reorder_indices=local_inverse_reorder_indices,
                    packed_keep_mask=packed_keep_mask,
                    merging_probs=merging_probs,
                    restore_shape=restore_shape,
                )
                out_fused = moe_unpermute_1d_fused_drop_no_compile(
                    inp=combine_out,
                    row_id_map=row_id_map,
                    local_inverse_reorder_indices=local_inverse_reorder_indices,
                    packed_keep_mask=packed_keep_mask,
                    merging_probs=merging_probs,
                    num_kept=num_kept,
                    map_type="index",
                )
                torch.testing.assert_close(out_fused, out_legacy, atol=5e-3, rtol=5e-3)

                def _run_fwd_legacy() -> None:
                    _legacy_restore_drop_unpermute(
                        combine_out=combine_out,
                        row_id_map=row_id_map,
                        local_inverse_reorder_indices=local_inverse_reorder_indices,
                        packed_keep_mask=packed_keep_mask,
                        merging_probs=merging_probs,
                        restore_shape=restore_shape,
                    )

                def _run_fwd_fused() -> None:
                    moe_unpermute_1d_fused_drop_no_compile(
                        inp=combine_out,
                        row_id_map=row_id_map,
                        local_inverse_reorder_indices=local_inverse_reorder_indices,
                        packed_keep_mask=packed_keep_mask,
                        merging_probs=merging_probs,
                        num_kept=num_kept,
                        map_type="index",
                    )

                combine_bwd_legacy = combine_out.detach().clone().requires_grad_(True)
                probs_bwd_legacy = merging_probs.detach().clone().requires_grad_(True)
                out_bwd_legacy = _legacy_restore_drop_unpermute(
                    combine_out=combine_bwd_legacy,
                    row_id_map=row_id_map,
                    local_inverse_reorder_indices=local_inverse_reorder_indices,
                    packed_keep_mask=packed_keep_mask,
                    merging_probs=probs_bwd_legacy,
                    restore_shape=restore_shape,
                )
                grad_out = torch.randn_like(out_bwd_legacy)

                combine_bwd_fused = combine_out.detach().clone().requires_grad_(True)
                probs_bwd_fused = merging_probs.detach().clone().requires_grad_(True)
                out_bwd_fused = moe_unpermute_1d_fused_drop_no_compile(
                    inp=combine_bwd_fused,
                    row_id_map=row_id_map,
                    local_inverse_reorder_indices=local_inverse_reorder_indices,
                    packed_keep_mask=packed_keep_mask,
                    merging_probs=probs_bwd_fused,
                    num_kept=num_kept,
                    map_type="index",
                )

                def _run_bwd_legacy() -> None:
                    if combine_bwd_legacy.grad is not None:
                        combine_bwd_legacy.grad.zero_()
                    if probs_bwd_legacy.grad is not None:
                        probs_bwd_legacy.grad.zero_()
                    out_bwd_legacy.backward(grad_out, retain_graph=True)

                def _run_bwd_fused() -> None:
                    if combine_bwd_fused.grad is not None:
                        combine_bwd_fused.grad.zero_()
                    if probs_bwd_fused.grad is not None:
                        probs_bwd_fused.grad.zero_()
                    out_bwd_fused.backward(grad_out, retain_graph=True)

                def _run_fwd_bwd_legacy() -> None:
                    c = combine_out.detach().clone().requires_grad_(True)
                    p = merging_probs.detach().clone().requires_grad_(True)
                    out = _legacy_restore_drop_unpermute(
                        combine_out=c,
                        row_id_map=row_id_map,
                        local_inverse_reorder_indices=local_inverse_reorder_indices,
                        packed_keep_mask=packed_keep_mask,
                        merging_probs=p,
                        restore_shape=restore_shape,
                    )
                    (out * out).sum().backward()

                def _run_fwd_bwd_fused() -> None:
                    c = combine_out.detach().clone().requires_grad_(True)
                    p = merging_probs.detach().clone().requires_grad_(True)
                    out = moe_unpermute_1d_fused_drop_no_compile(
                        inp=c,
                        row_id_map=row_id_map,
                        local_inverse_reorder_indices=local_inverse_reorder_indices,
                        packed_keep_mask=packed_keep_mask,
                        merging_probs=p,
                        num_kept=num_kept,
                        map_type="index",
                    )
                    (out * out).sum().backward()

                fwd_legacy = _event_timed_ms(_run_fwd_legacy, args.warmup_iters, args.iters)
                fwd_fused = _event_timed_ms(_run_fwd_fused, args.warmup_iters, args.iters)
                bwd_legacy = _event_timed_ms(_run_bwd_legacy, args.warmup_iters, args.iters)
                bwd_fused = _event_timed_ms(_run_bwd_fused, args.warmup_iters, args.iters)
                fwd_bwd_legacy = _event_timed_ms(_run_fwd_bwd_legacy, args.warmup_iters // 2, max(1, args.iters // 2))
                fwd_bwd_fused = _event_timed_ms(_run_fwd_bwd_fused, args.warmup_iters // 2, max(1, args.iters // 2))

                print(
                    f"d_model={d_model:4d} top_k={top_k} keep={keep_fraction:0.1f} | "
                    f"fwd {fwd_legacy:.4f}->{fwd_fused:.4f} ms | "
                    f"bwd {bwd_legacy:.4f}->{bwd_fused:.4f} ms | "
                    f"fwd+bwd {fwd_bwd_legacy:.4f}->{fwd_bwd_fused:.4f} ms"
                )

                if fwd_fused > fwd_legacy:
                    failures.append(
                        f"fwd regression at d_model={d_model}, top_k={top_k}, keep_fraction={keep_fraction}: "
                        f"{fwd_fused:.4f} > {fwd_legacy:.4f} ms"
                    )
                if bwd_fused > bwd_legacy:
                    failures.append(
                        f"bwd regression at d_model={d_model}, top_k={top_k}, keep_fraction={keep_fraction}: "
                        f"{bwd_fused:.4f} > {bwd_legacy:.4f} ms"
                    )
                if fwd_bwd_fused > fwd_bwd_legacy:
                    failures.append(
                        f"fwd+bwd regression at d_model={d_model}, top_k={top_k}, keep_fraction={keep_fraction}: "
                        f"{fwd_bwd_fused:.4f} > {fwd_bwd_legacy:.4f} ms"
                    )

    if failures:
        print("=== FAILURES ===")
        for failure in failures:
            print(failure)
        raise SystemExit(1)
    print("All benchmark cases passed non-regression gate.")


if __name__ == "__main__":
    main()
