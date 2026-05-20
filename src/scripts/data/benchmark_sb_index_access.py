"""Benchmark stupid-backoff index access backends.

This compares the existing mmap path with the explicit-read path for
``StupidBackoffNgramLM.compute_overrides_for_sequence`` on the same sampled
token sequences. It is meant to answer whether SB override construction is
sensitive to mmap page-fault overhead before launching full Beaker jobs.

Example::

    python analysis/scripts/benchmark_sb_index_access.py \
        --table-dir /weka/oe-training-default/ai2-llm/ngram-tables/pilots/pilot-2026-04-23-fraction1e-4-n5-counts \
        --index-access mmap --index-access pread \
        --cap 2=128 --cap 3=128 --cap 4=128 --cap 5=128 \
        --num-sequences 32 --sequence-length 8192 --mirror-to-shm \
        --workers 1 --workers 2 --workers 4 --workers 8
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from olmo_core.data.stupid_backoff_ngram import StupidBackoffNgramLM


_WORKER_READER: StupidBackoffNgramLM | None = None
_WORKER_SEQUENCES: np.ndarray | None = None


def _parse_caps(values: list[str] | None) -> dict[int, int] | None:
    if not values:
        return None
    out: dict[int, int] = {}
    for value in values:
        order_s, cap_s = value.split("=", 1)
        out[int(order_s)] = int(cap_s)
    return out


def _sample_sequences(
    reader: StupidBackoffNgramLM,
    *,
    num_sequences: int,
    sequence_length: int,
    seed: int,
) -> np.ndarray:
    order1 = reader.orders[1]
    cont_kenlm = np.asarray(order1["continuations"], dtype=np.uint32)
    counts = np.asarray(order1["counts"], dtype=np.float64)
    cont_dolma2 = reader.kenlm_to_dolma2[cont_kenlm].astype(np.int64, copy=False)
    valid = cont_dolma2 < reader.vocab_size
    cont_dolma2 = cont_dolma2[valid]
    counts = counts[valid]
    probs = counts / counts.sum()
    rng = np.random.default_rng(seed)
    return rng.choice(
        cont_dolma2,
        size=(num_sequences, sequence_length),
        replace=True,
        p=probs,
    ).astype(np.int64, copy=False)


def _load_or_sample_sequences(args, reader: StupidBackoffNgramLM) -> np.ndarray:
    if args.input_ids_npy is not None:
        arr = np.load(args.input_ids_npy)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f"--input-ids-npy must have shape (S,) or (B,S); got {arr.shape}")
        return np.asarray(arr[: args.num_sequences, : args.sequence_length], dtype=np.int64)
    return _sample_sequences(
        reader,
        num_sequences=args.num_sequences,
        sequence_length=args.sequence_length,
        seed=args.seed,
    )


def _summarize(values: list[float]) -> str:
    if not values:
        return "n=0"
    return (
        f"n={len(values)} mean={statistics.fmean(values):.4f}s "
        f"median={statistics.median(values):.4f}s "
        f"min={min(values):.4f}s max={max(values):.4f}s"
    )


def _worker_init(reader_kwargs: dict[str, Any], sequences: np.ndarray) -> None:
    global _WORKER_READER, _WORKER_SEQUENCES
    _WORKER_READER = StupidBackoffNgramLM(**reader_kwargs)
    _WORKER_SEQUENCES = sequences


def _worker_compute(idx: int) -> tuple[float, int]:
    assert _WORKER_READER is not None
    assert _WORKER_SEQUENCES is not None
    seq = _WORKER_SEQUENCES[idx]
    t0 = time.perf_counter()
    pos, _tok, _score = _WORKER_READER.compute_overrides_for_sequence(seq)
    return time.perf_counter() - t0, int(pos.shape[0])


def _run_backend(
    *,
    backend: str,
    workers: int,
    args,
    caps: dict[int, int] | None,
    sequences: np.ndarray,
) -> None:
    reader_kwargs = dict(
        table_dir=args.table_dir,
        dolma2_vocab_size=args.dolma2_vocab_size,
        N_max=args.N_max,
        alpha=args.alpha,
        max_order_continuations=caps,
        mirror_to_shm=args.mirror_to_shm,
        index_access=backend,
    )
    t0 = time.perf_counter()
    if workers == 1:
        reader = StupidBackoffNgramLM(**reader_kwargs)
        init_s = time.perf_counter() - t0
        times: list[float] = []
        rows: list[int] = []
        for seq in sequences:
            t_seq = time.perf_counter()
            pos, _tok, _score = reader.compute_overrides_for_sequence(seq)
            times.append(time.perf_counter() - t_seq)
            rows.append(int(pos.shape[0]))
    else:
        # Beaker runs Linux, so fork keeps the sampled sequences copy-on-write
        # instead of pickling multi-MB arrays into every task.
        ctx = mp.get_context("fork")
        with ctx.Pool(
            processes=workers,
            initializer=_worker_init,
            initargs=(reader_kwargs, sequences),
        ) as pool:
            results = list(pool.imap(_worker_compute, range(sequences.shape[0]), chunksize=1))
        init_s = float("nan")
        times = [t for t, _ in results]
        rows = [r for _, r in results]

    wall_s = time.perf_counter() - t0
    throughput = sequences.shape[0] / wall_s if wall_s > 0 else float("inf")
    print(
        f"backend={backend} workers={workers} init={init_s:.3f}s wall={wall_s:.3f}s "
        f"throughput={throughput:.3f} seq/s "
        f"times=({_summarize(times)}) "
        f"overrides_mean={statistics.fmean(rows):,.0f} "
        f"overrides_min={min(rows):,} overrides_max={max(rows):,}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-dir", required=True)
    parser.add_argument(
        "--index-access",
        action="append",
        choices=("mmap", "pread"),
        default=None,
        help="Backend(s) to test. Repeat to compare both.",
    )
    parser.add_argument("--cap", action="append", default=None, metavar="ORDER=CAP")
    parser.add_argument("--num-sequences", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--dolma2-vocab-size", type=int, default=100352)
    parser.add_argument("--N-max", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--input-ids-npy", type=Path, default=None)
    parser.add_argument("--mirror-to-shm", action="store_true")
    parser.add_argument(
        "--workers",
        action="append",
        type=int,
        default=None,
        help=(
            "Worker process counts to test. Repeat to run a scaling sweep. "
            "workers=1 runs in-process; workers>1 uses multiprocessing fork."
        ),
    )
    args = parser.parse_args()

    backends = args.index_access or ["mmap", "pread"]
    worker_counts = args.workers or [1]
    caps = _parse_caps(args.cap)
    sequences = None
    for backend in backends:
        if sequences is None:
            reader = StupidBackoffNgramLM(
                args.table_dir,
                dolma2_vocab_size=args.dolma2_vocab_size,
                N_max=args.N_max,
                alpha=args.alpha,
                max_order_continuations=caps,
                mirror_to_shm=args.mirror_to_shm,
                index_access=backend,
            )
            sequences = _load_or_sample_sequences(args, reader)
        for workers in worker_counts:
            if workers <= 0:
                raise ValueError(f"--workers must be positive; got {workers}")
            print(
                f"run_start backend={backend} workers={workers} "
                f"pid={os.getpid()} sequences={sequences.shape[0]}",
                flush=True,
            )
            _run_backend(
                backend=backend,
                workers=workers,
                args=args,
                caps=caps,
                sequences=sequences,
            )


if __name__ == "__main__":
    main()
