"""
Benchmark FA4 full attention with different packed-document layouts.

Example:
    PYTHONPATH=src torchrun --standalone --nproc-per-node=8 \
        src/scripts/train/fa4_intra_doc_attention_bench.py

Small smoke test:
    PYTHONPATH=src python src/scripts/train/fa4_intra_doc_attention_bench.py \
        --seq-len 128 --d-model 256 --d-attn 256 --n-heads 4 --n-kv-heads 1 \
        --head-dim 64 --warmup-iters 1 --bench-iters 1 --mode forward --no-qk-norm
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import warnings
from dataclasses import dataclass
from typing import Optional

warnings.filterwarnings(
    "ignore",
    message=r"Use explicit .*scalar\.ptr.*",
    category=Warning,
)
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"nvidia_cutlass_dsl.*")

import torch
import torch.distributed as dist
from torch import nn

from olmo_core.config import DType
from olmo_core.nn.attention import AttentionBackendName, FusedAttentionV2
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.rope import RoPEConfig, RoPEType


SEQ_LEN = 64 * 1024


@dataclass(frozen=True)
class BenchCase:
    name: str
    target_doc_len: Optional[int]
    intra_doc_masking: bool


CASES = (
    BenchCase("short_4k_intradoc", 4 * 1024, True),
    BenchCase("mid_16k_intradoc", 16 * 1024, True),
    BenchCase("long_32k_intradoc", 32 * 1024, True),
    BenchCase("one_64k_intradoc", SEQ_LEN, True),
    BenchCase("full_64k_no_intradoc", None, False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--bench-iters", type=int, default=5)
    parser.add_argument("--mode", choices=("fwd-bwd", "forward"), default="fwd-bwd")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--same-docs-across-ranks", action="store_true")
    parser.add_argument("--jitter-frac", type=float, default=0.25)
    parser.add_argument("--vocab-size", type=int, default=100_000)

    parser.add_argument("--d-model", type=int, default=2048)
    parser.add_argument("--d-attn", type=int, default=4096)
    parser.add_argument("--n-heads", type=int, default=32)
    parser.add_argument("--n-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--rope-theta", type=float, default=500_000.0)
    parser.add_argument("--qk-norm", dest="qk_norm", action="store_true", default=True)
    parser.add_argument("--no-qk-norm", dest="qk_norm", action="store_false")
    parser.add_argument("--qk-norm-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--tf32", action="store_true")
    args = parser.parse_args()

    if args.seq_len <= 0:
        parser.error("--seq-len must be positive")
    if args.warmup_iters < 0:
        parser.error("--warmup-iters must be non-negative")
    if args.bench_iters <= 0:
        parser.error("--bench-iters must be positive")
    if args.d_attn != args.n_heads * args.head_dim:
        parser.error("--d-attn must equal --n-heads * --head-dim")
    if args.n_heads % args.n_kv_heads != 0:
        parser.error("--n-heads must be divisible by --n-kv-heads")
    if not (0.0 <= args.jitter_frac < 1.0):
        parser.error("--jitter-frac must be in [0, 1)")
    return args


def init_distributed() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA and the FA4 attention backend")

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if world_size > 1 and not dist.is_initialized():
        try:
            dist.init_process_group(backend="nccl", device_id=device)
        except TypeError:
            dist.init_process_group(backend="nccl")

    return rank, local_rank, world_size, device


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def dtype_from_name(name: str) -> torch.dtype:
    return DType(name).as_pt()


def make_doc_lens(total: int, target: int, *, jitter_frac: float, seed: int) -> list[int]:
    if target >= total:
        return [total]

    rng = random.Random(seed)
    low = max(1, int(round(target * (1.0 - jitter_frac))))
    high = max(low, int(round(target * (1.0 + jitter_frac))))
    doc_lens: list[int] = []
    remaining = total

    while remaining > 0:
        if remaining <= high:
            doc_lens.append(remaining)
            break
        length = rng.randint(low, high)
        doc_lens.append(length)
        remaining -= length

    if len(doc_lens) > 1 and len(set(doc_lens)) == 1:
        doc_lens[-2] += 1
        doc_lens[-1] -= 1

    assert all(length > 0 for length in doc_lens)
    assert sum(doc_lens) == total
    return doc_lens


def case_doc_lens(
    case: BenchCase,
    *,
    seq_len: int,
    jitter_frac: float,
    seed: int,
) -> Optional[list[int]]:
    if not case.intra_doc_masking:
        return None
    assert case.target_doc_len is not None
    return make_doc_lens(seq_len, case.target_doc_len, jitter_frac=jitter_frac, seed=seed)


def cu_doc_lens_from_doc_lens(doc_lens: list[int], device: torch.device) -> torch.Tensor:
    cu_doc_lens = [0]
    for length in doc_lens:
        cu_doc_lens.append(cu_doc_lens[-1] + length)
    return torch.tensor(cu_doc_lens, dtype=torch.int32, device=device)


def attention_pairs(doc_lens: Optional[list[int]], seq_len: int) -> int:
    lens = doc_lens if doc_lens is not None else [seq_len]
    return sum(length * (length + 1) // 2 for length in lens)


def preview_lens(doc_lens: Optional[list[int]], max_items: int = 8) -> str:
    if doc_lens is None:
        return "none"
    if len(doc_lens) <= max_items:
        return ",".join(str(length) for length in doc_lens)
    head_count = max_items // 2
    tail_count = max_items - head_count
    head = ",".join(str(length) for length in doc_lens[:head_count])
    tail = ",".join(str(length) for length in doc_lens[-tail_count:])
    return f"{head},...,{tail}"


def build_attention(args: argparse.Namespace, device: torch.device) -> FusedAttentionV2:
    dtype = dtype_from_name(args.dtype)
    qk_norm: Optional[LayerNormConfig] = None
    if args.qk_norm:
        qk_norm = LayerNormConfig(
            name=LayerNormType.rms,
            eps=1e-6,
            bias=False,
            dtype=DType(args.qk_norm_dtype),
        )

    attention = FusedAttentionV2(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        head_dim=args.head_dim,
        d_attn=args.d_attn,
        bias=False,
        rope=RoPEConfig(
            name=RoPEType.default,
            theta=args.rope_theta,
            scaling=None,
            full_precision=True,
        ),
        qk_norm=qk_norm,
        backend=AttentionBackendName.flash_4,
        use_head_qk_norm=bool(qk_norm),
        dtype=dtype,
        init_device=str(device),
    )
    attention.train()
    return attention


@torch.no_grad()
def make_input(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    dtype = dtype_from_name(args.dtype)
    input_ids = torch.randint(args.vocab_size, (1, args.seq_len), device=device)
    embedding = nn.Embedding(args.vocab_size, args.d_model, dtype=dtype, device=device)
    x = embedding(input_ids).detach()
    del embedding, input_ids
    torch.cuda.empty_cache()
    return x


def run_one_iter(
    attention: FusedAttentionV2,
    x_base: torch.Tensor,
    *,
    mode: str,
    cu_doc_lens: Optional[torch.Tensor],
    max_doc_len: Optional[int],
) -> None:
    if mode == "forward":
        with torch.no_grad():
            attention(x_base, cu_doc_lens=cu_doc_lens, max_doc_len=max_doc_len)
        return

    attention.zero_grad(set_to_none=True)
    x = x_base.detach().requires_grad_(True)
    out = attention(x, cu_doc_lens=cu_doc_lens, max_doc_len=max_doc_len)
    loss = out.sum() * (1.0 / out.numel())
    loss.backward()


def event_time_ms(fn, *, warmup_iters: int, bench_iters: int) -> list[float]:
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(bench_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(float(start.elapsed_time(end)))
    return times_ms


def run_case(
    case: BenchCase,
    *,
    args: argparse.Namespace,
    attention: FusedAttentionV2,
    x_base: torch.Tensor,
    rank: int,
    device: torch.device,
) -> dict[str, object]:
    doc_seed_rank = 0 if args.same_docs_across_ranks else rank
    doc_seed = args.seed + 1009 * (CASES.index(case) + 1) + doc_seed_rank
    doc_lens = case_doc_lens(
        case,
        seq_len=args.seq_len,
        jitter_frac=args.jitter_frac,
        seed=doc_seed,
    )
    cu_doc_lens = None
    max_doc_len = None
    if doc_lens is not None:
        cu_doc_lens = cu_doc_lens_from_doc_lens(doc_lens, device)
        max_doc_len = max(doc_lens)

    torch.cuda.reset_peak_memory_stats(device)
    barrier()
    times_ms = event_time_ms(
        lambda: run_one_iter(
            attention,
            x_base,
            mode=args.mode,
            cu_doc_lens=cu_doc_lens,
            max_doc_len=max_doc_len,
        ),
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
    )
    barrier()

    work = attention_pairs(doc_lens, args.seq_len)
    full_work = attention_pairs(None, args.seq_len)
    return {
        "rank": rank,
        "case": case.name,
        "doc_count": len(doc_lens) if doc_lens is not None else 1,
        "max_doc_len": max_doc_len if max_doc_len is not None else args.seq_len,
        "work": work,
        "work_ratio": work / full_work,
        "mean_ms": statistics.fmean(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "times_ms": times_ms,
        "peak_mem_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "doc_lens": preview_lens(doc_lens),
    }


def gather_result(result: dict[str, object]) -> list[dict[str, object]]:
    if not is_distributed():
        return [result]
    gathered: list[Optional[dict[str, object]]] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, result)
    return [item for item in gathered if item is not None]


def print_case_results(results: list[dict[str, object]], *, world_size: int) -> None:
    results = sorted(results, key=lambda item: int(item["rank"]))
    for result in results:
        print(
            "rank={rank:02d} case={case} docs={doc_count} max_doc={max_doc_len} "
            "work_ratio={work_ratio:.4f} mean_ms={mean_ms:.3f} "
            "min_ms={min_ms:.3f} max_ms={max_ms:.3f} peak_mem_gib={peak_mem_gib:.2f} "
            "lens={doc_lens}".format(**result),
            flush=True,
        )

    if world_size <= 1:
        return

    mean_ms = [float(result["mean_ms"]) for result in results]
    work_ratios = [float(result["work_ratio"]) for result in results]
    fastest = min(results, key=lambda item: float(item["mean_ms"]))
    slowest = max(results, key=lambda item: float(item["mean_ms"]))
    speed_ratio = float(slowest["mean_ms"]) / float(fastest["mean_ms"])
    print(
        "summary case={case} avg_ms={avg_ms:.3f} fastest_rank={fastest_rank} "
        "fastest_ms={fastest_ms:.3f} slowest_rank={slowest_rank} slowest_ms={slowest_ms:.3f} "
        "rank_time_ratio={speed_ratio:.3f} work_ratio_range=[{work_min:.4f},{work_max:.4f}]".format(
            case=results[0]["case"],
            avg_ms=statistics.fmean(mean_ms),
            fastest_rank=fastest["rank"],
            fastest_ms=float(fastest["mean_ms"]),
            slowest_rank=slowest["rank"],
            slowest_ms=float(slowest["mean_ms"]),
            speed_ratio=speed_ratio,
            work_min=min(work_ratios),
            work_max=max(work_ratios),
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    rank, _local_rank, world_size, device = init_distributed()

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.set_float32_matmul_precision("high")
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if rank == 0:
        print(
            f"FA4 attention benchmark: seq_len={args.seq_len} mode={args.mode} "
            f"warmup={args.warmup_iters} iters={args.bench_iters} world_size={world_size} "
            f"d_model={args.d_model} d_attn={args.d_attn} n_heads={args.n_heads} "
            f"n_kv_heads={args.n_kv_heads} head_dim={args.head_dim} dtype={args.dtype} "
            f"qk_norm={args.qk_norm} same_docs_across_ranks={args.same_docs_across_ranks}",
            flush=True,
        )

    attention = build_attention(args, device)
    x_base = make_input(args, device)
    barrier()

    for case in CASES:
        result = run_case(
            case,
            args=args,
            attention=attention,
            x_base=x_base,
            rank=rank,
            device=device,
        )
        results = gather_result(result)
        if rank == 0:
            print_case_results(results, world_size=world_size)
            print("", flush=True)

    barrier()
    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
