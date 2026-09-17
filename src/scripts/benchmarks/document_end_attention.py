"""Compare exact document-end backends on CPU or CUDA; does not launch jobs.

PYTHONPATH=src python src/scripts/benchmarks/document_end_attention.py --length 256
Use --backend tiled for long sequences where the eager oracle is too expensive.
"""

import argparse
import json
import statistics
import time

import torch

from olmo_core.nn.attention.landmark_document_end import document_end_compressive_attention
from olmo_core.nn.attention.landmark_document_end_tiled import (
    tiled_document_end_compressive_attention,
)
from olmo_core.nn.attention.summary_mask import build_summary_roles


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    p.add_argument("--backend", choices=["both", "tiled", "reference"], default="both")
    p.add_argument("--length", type=int, default=256)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--kv-heads", type=int, default=2)
    p.add_argument("--head-dim", type=int, default=32)
    p.add_argument("--query-tile-size", type=int, default=64)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args()
    if args.length < 8 or args.repeats < 1 or args.heads % args.kv_heads:
        p.error("Need length >= 8, repeats >= 1, and heads divisible by kv-heads")
    torch.set_num_threads(args.threads)
    torch.manual_seed(17)
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    # Intentionally unequal documents, followed by a nonempty query/answer region.
    tokens = [1]
    for n in (7, 29, 13, 61) * (args.length // 64 + 1):
        if len(tokens) + n + 3 > args.length - 3:
            break
        tokens += [10] + [2] * n + [11, 12]
    tokens += [3] * (args.length - len(tokens))
    roles = build_summary_roles(
        torch.tensor([tokens], device=device),
        doc_start_id=10,
        doc_end_id=11,
        summary_token_id=12,
        eos_id=13,
        pad_id=14,
    )
    q = torch.randn(
        1, args.heads, args.length, args.head_dim, device=device, dtype=dtype, requires_grad=True
    )
    k, v = [
        torch.randn(
            1,
            args.kv_heads,
            args.length,
            args.head_dim,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        for _ in range(2)
    ]
    grad = torch.randn_like(q)

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    def forward(name):
        if name == "tiled":
            return tiled_document_end_compressive_attention(
                q, k, v, roles, query_tile_size=args.query_tile_size
            )
        return document_end_compressive_attention(
            q,
            k.repeat_interleave(args.heads // args.kv_heads, 1),
            v.repeat_interleave(args.heads // args.kv_heads, 1),
            roles,
        )

    def measure(fn):
        fn()  # warmup
        times = []
        for _ in range(args.repeats):
            sync()
            begin = time.perf_counter()
            fn()
            sync()
            times.append((time.perf_counter() - begin) * 1000)
        return statistics.median(times)

    results = {}
    for name in ["reference", "tiled"] if args.backend == "both" else [args.backend]:

        def inference():
            with torch.no_grad():
                forward(name)

        def training():
            out = forward(name)
            torch.autograd.grad(out, (q, k, v), grad)

        # Storage deduplication measures tensors retained for backward, including
        # inputs, rather than counting the same storage repeatedly through views.
        # This is NOT allocator peak memory; CUDA allocator peak is reported separately.
        storages = {}

        def pack(t):
            storage = t.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            out = forward(name)
        torch.autograd.grad(out, (q, k, v), grad)
        del out
        peak = None
        if args.device == "cuda":
            sync()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            training()
            sync()
            peak = torch.cuda.max_memory_allocated() - baseline
        results[name] = {
            "forward_ms": measure(inference),
            "forward_backward_ms": measure(training),
            "saved_tensor_storage_bytes_including_inputs": sum(storages.values()),
            "cuda_peak_extra_bytes": peak,
        }
    print(
        json.dumps({"config": vars(args), "torch": torch.__version__, "results": results}, indent=2)
    )


if __name__ == "__main__":
    main()
