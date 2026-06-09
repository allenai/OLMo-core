"""
Training-time Kneser-Ney top-k lookup for ngram product-of-experts training.

Reads a precomputed top-K forward index (FXTK v=1) produced by
``data_gen/build_topk_forward_index.py``. For each query position with
context h, we binary-search the longest matching order's prefix table,
read the row's K (token, log-prob) pairs, and renormalize to a
distribution over those K tokens.

No kenlm or C++ adapter at runtime — all cross-order P_KN combination is
done at build time, and the lookup is a constant-time row read after a
binary search.

Inputs (per position): a context of up to ``N_max - 1`` tokens (the caller
walks left from the target position; doc-boundary handling is the caller's
responsibility — high-order probes that cross a boundary just miss and the
lookup degrades through the order ladder).

Outputs (per position): ``(topk_ids: int32[K], topk_probs: float32[K])``
with ``topk_probs`` summing to 1.
"""

from __future__ import annotations

import fcntl
import hashlib
import mmap
import os
import shutil
import struct
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np


# ----------------------------------------------------------------------
# /dev/shm staging — same idea as before. mmap'd reads from weka with 64+
# concurrent dataloader workers thrash the OS page cache; one sequential
# copy into tmpfs avoids that.
# ----------------------------------------------------------------------

_REMOTE_PREFIXES = ("/weka/",)
_SHM_CACHE_ROOT = Path(
    os.environ.get("OLMO_NGRAM_SHM_ROOT", "/dev/shm/olmo_core_ngram_cache")
)


def _mirror_to_shm(path: str) -> str:
    """Copy a remote-ish file to /dev/shm once per node (flock-protected)."""
    if not any(path.startswith(p) for p in _REMOTE_PREFIXES):
        return path
    src_size = os.path.getsize(path)
    path_hash = hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]
    cache_dir = _SHM_CACHE_ROOT / path_hash
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / Path(path).name
    try:
        if cache_path.is_file() and cache_path.stat().st_size == src_size:
            return str(cache_path)
    except OSError:
        pass
    lock_path = str(cache_path) + ".lock"
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        if cache_path.is_file() and cache_path.stat().st_size == src_size:
            return str(cache_path)
        tmp_path = Path(str(cache_path) + ".partial")
        try:
            print(
                f"[ngram_topk] mirroring {path} ({src_size // 2**20} MB) "
                f"→ {cache_path} ...",
                flush=True,
            )
            with open(path, "rb") as src, open(tmp_path, "wb") as dst:
                shutil.copyfileobj(src, dst, length=32 * 1024 * 1024)
            tmp_path.replace(cache_path)
            print(
                f"[ngram_topk] mirrored {Path(path).name} "
                f"({src_size // 2**20} MB) into /dev/shm",
                flush=True,
            )
        except BaseException:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
    return str(cache_path)


# ----------------------------------------------------------------------
# FXTK v=1 file format — must match data_gen/build_topk_forward_index.py.
# ----------------------------------------------------------------------

_MAGIC = b"FXTK"
_VERSION = 1
_HEADER_SIZE = 64
_SECTION_HEADER_SIZE = 64
_RAW_MAGIC = b"FIX1"
_RAW_VERSION = 2
_RAW_HEADER_SIZE = 64
_RAW_SECTION_HEADER_SIZE = 48


class _OrderSection:
    """Per-order tables, mmap-backed numpy views."""

    def __init__(
        self,
        order: int,
        prefixes_2d: np.ndarray,
        topk_tokens: np.ndarray,
        topk_logprobs: np.ndarray,
    ):
        self.order = order
        self.plen = order - 1
        self.prefixes_2d = prefixes_2d  # (n, plen) uint32, lex-sorted
        self.topk_tokens = topk_tokens  # (n, K) uint32
        self.topk_logprobs = topk_logprobs  # (n, K) float32
        # Structured 1-D view for lex-comparison binary search. uint32
        # cells in the 2-D array reinterpret as a struct of ``plen`` u4
        # fields, which numpy compares lex-first when sorting/searching.
        if self.plen > 0:
            struct_dtype = np.dtype([(f"f{j}", "<u4") for j in range(self.plen)])
            self.prefixes_struct = prefixes_2d.view(struct_dtype).reshape(-1)
        else:
            # Order 1: single empty-prefix row. No search needed.
            self.prefixes_struct = None


class TopKForwardIndex:
    """Mmap-backed reader for FXTK v=1 precomputed top-K forward index."""

    def __init__(self, path: str):
        self.path = path
        self._mm: Optional[mmap.mmap] = None
        self._fd: Optional[int] = None
        self.K: int = 0
        self.vocab_size: int = 0
        self.sections_by_order: dict[int, _OrderSection] = {}

    def _ensure_open(self):
        if self._mm is not None:
            return
        fd = os.open(self.path, os.O_RDONLY)
        try:
            mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        self._mm = mm

        if mm[:4] != _MAGIC:
            raise ValueError(
                f"unexpected magic {bytes(mm[:4])!r} in {self.path} (expected {_MAGIC!r})"
            )
        version, K, n_orders, vocab_size = struct.unpack("<IIII", mm[4:20])
        if version != _VERSION:
            raise ValueError(
                f"FXTK version {version} != {_VERSION}; rebuild with current "
                "data_gen/build_topk_forward_index.py"
            )
        self.K = int(K)
        self.vocab_size = int(vocab_size)

        sh_off = _HEADER_SIZE
        for _ in range(n_orders):
            sh = mm[sh_off : sh_off + _SECTION_HEADER_SIZE]
            order, _pad, n_prefixes, p_off, tt_off, tl_off = struct.unpack(
                "<II QQQQ", sh[:40]
            )
            sh_off += _SECTION_HEADER_SIZE

            plen = order - 1
            if plen > 0:
                prefixes = np.frombuffer(
                    mm, dtype=np.uint32, count=n_prefixes * plen, offset=p_off
                ).reshape(n_prefixes, plen)
            else:
                # Order 1: zero-byte prefix block.
                prefixes = np.zeros((1, 0), dtype=np.uint32)

            topk_tokens = np.frombuffer(
                mm, dtype=np.uint32, count=n_prefixes * self.K, offset=tt_off
            ).reshape(n_prefixes, self.K)
            topk_logprobs = np.frombuffer(
                mm, dtype=np.float32, count=n_prefixes * self.K, offset=tl_off
            ).reshape(n_prefixes, self.K)

            self.sections_by_order[order] = _OrderSection(
                order=order,
                prefixes_2d=prefixes,
                topk_tokens=topk_tokens,
                topk_logprobs=topk_logprobs,
            )

    def lookup_one(self, context: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """Return (tokens, logprobs) for the longest matching order.

        ``context`` is the up-to-(N_max-1)-token window ending at the target
        position's left boundary, oldest first. We try longest-match first
        and fall through; order 1 (empty prefix) is always present.
        """
        self._ensure_open()
        ctx_len = len(context)
        # Try the longest order first. We have sections for orders that
        # exist in the file (typically 1..N_max).
        max_order_in_file = max(self.sections_by_order)
        upper = min(max_order_in_file, ctx_len + 1)
        for n in range(upper, 0, -1):
            sect = self.sections_by_order.get(n)
            if sect is None:
                continue
            plen = sect.plen
            if plen == 0:
                # Order 1 always matches with the empty prefix.
                return sect.topk_tokens[0], sect.topk_logprobs[0]
            if ctx_len < plen:
                continue
            query = np.asarray(context[-plen:], dtype=np.uint32).reshape(1, plen)
            struct_dtype = sect.prefixes_struct.dtype
            query_view = query.view(struct_dtype).reshape(-1)
            idx = int(np.searchsorted(sect.prefixes_struct, query_view[0]))
            if idx < sect.prefixes_struct.shape[0] and (
                sect.prefixes_struct[idx] == query_view[0]
            ):
                return sect.topk_tokens[idx], sect.topk_logprobs[idx]

        # Should be unreachable: order 1 (empty prefix) is always present.
        raise RuntimeError(
            f"top-K forward index lookup fell through all orders for context "
            f"of length {ctx_len}; index file may be missing order 1"
        )

    def close(self):
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        self.sections_by_order.clear()


class _ContextOrderSection:
    """Per-order prefix table for context-ID lookup only."""

    def __init__(
        self,
        order: int,
        prefixes_2d: np.ndarray,
        global_offset: int,
    ):
        self.order = order
        self.plen = order - 1
        self.prefixes_2d = prefixes_2d
        self.global_offset = int(global_offset)
        if self.plen > 0:
            struct_dtype = np.dtype([(f"f{j}", "<u4") for j in range(self.plen)])
            self.prefixes_struct = prefixes_2d.view(struct_dtype).reshape(-1)
        else:
            self.prefixes_struct = None


class ContextForwardIndex:
    """Mmap-backed context-key reader for FXTK v=1 indexes.

    This reader intentionally maps only the prefix blocks from
    ``forward_index_topk.bin``. It does not expose or construct views over the
    top-k token IDs or log-probability arrays, which keeps Engram-style
    learned-memory baselines from accidentally using corpus-derived
    continuation statistics.
    """

    def __init__(self, path: str, *, N_max: int = 5):
        self.path = path
        self.N_max = int(N_max)
        self._mm: Optional[mmap.mmap] = None
        self._fd: Optional[int] = None
        self.K: int = 0
        self.vocab_size: int = 0
        self.num_contexts: int = 0
        self.sections_by_order: dict[int, _ContextOrderSection] = {}

    def _ensure_open(self):
        if self._mm is not None:
            return
        fd = os.open(self.path, os.O_RDONLY)
        try:
            mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        self._mm = mm

        if mm[:4] != _MAGIC:
            raise ValueError(
                f"unexpected magic {bytes(mm[:4])!r} in {self.path} (expected {_MAGIC!r})"
            )
        version, K, n_orders, vocab_size = struct.unpack("<IIII", mm[4:20])
        if version != _VERSION:
            raise ValueError(
                f"FXTK version {version} != {_VERSION}; rebuild with current "
                "data_gen/build_topk_forward_index.py"
            )
        self.K = int(K)
        self.vocab_size = int(vocab_size)

        section_headers = []
        sh_off = _HEADER_SIZE
        for _ in range(n_orders):
            sh = mm[sh_off : sh_off + _SECTION_HEADER_SIZE]
            order, _pad, n_prefixes, p_off, _tt_off, _tl_off = struct.unpack(
                "<II QQQQ", sh[:40]
            )
            sh_off += _SECTION_HEADER_SIZE
            section_headers.append((int(order), int(n_prefixes), int(p_off)))

        max_order = max((order for order, _, _ in section_headers), default=0)
        if max_order < self.N_max:
            raise ValueError(
                f"context forward index has max order {max_order} but caller "
                f"passed N_max={self.N_max}"
            )

        global_offset = 0
        for order, n_prefixes, p_off in sorted(section_headers):
            if order > self.N_max:
                continue
            plen = order - 1
            if plen > 0:
                prefixes = np.frombuffer(
                    mm, dtype=np.uint32, count=n_prefixes * plen, offset=p_off
                ).reshape(n_prefixes, plen)
            else:
                prefixes = np.zeros((1, 0), dtype=np.uint32)
            self.sections_by_order[order] = _ContextOrderSection(
                order=order,
                prefixes_2d=prefixes,
                global_offset=global_offset,
            )
            global_offset += n_prefixes

        if 1 not in self.sections_by_order:
            raise ValueError(f"context forward index {self.path} is missing order 1")
        self.num_contexts = int(global_offset)

    def lookup_one(self, context: Sequence[int]) -> int:
        """Return the learned-memory row ID for the longest matching context."""
        self._ensure_open()
        ctx_len = len(context)
        upper = min(max(self.sections_by_order), ctx_len + 1, self.N_max)
        for n in range(upper, 0, -1):
            sect = self.sections_by_order.get(n)
            if sect is None:
                continue
            plen = sect.plen
            if plen == 0:
                return sect.global_offset
            if ctx_len < plen:
                continue
            query = np.asarray(context[-plen:], dtype=np.uint32).reshape(1, plen)
            assert sect.prefixes_struct is not None
            struct_dtype = sect.prefixes_struct.dtype
            query_view = query.view(struct_dtype).reshape(-1)
            idx = int(np.searchsorted(sect.prefixes_struct, query_view[0]))
            if idx < sect.prefixes_struct.shape[0] and (
                sect.prefixes_struct[idx] == query_view[0]
            ):
                return sect.global_offset + idx

        raise RuntimeError(
            f"context forward index lookup fell through all orders for context "
            f"of length {ctx_len}; index file may be missing order 1"
        )

    def close(self):
        self.sections_by_order.clear()
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        self.num_contexts = 0


class RawContextForwardIndex:
    """Mmap-backed context-key reader for FIX1 v=2 raw forward indexes."""

    def __init__(self, path: str, *, N_max: int = 5):
        self.path = path
        self.N_max = int(N_max)
        self._mm: Optional[mmap.mmap] = None
        self._fd: Optional[int] = None
        self.vocab_size: int = 0
        self.num_contexts: int = 0
        self.sections_by_order: dict[int, _ContextOrderSection] = {}

    def _ensure_open(self):
        if self._mm is not None:
            return
        fd = os.open(self.path, os.O_RDONLY)
        try:
            mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        self._mm = mm

        if mm[:4] != _RAW_MAGIC:
            raise ValueError(
                f"unexpected magic {bytes(mm[:4])!r} in {self.path} "
                f"(expected {_RAW_MAGIC!r})"
            )
        version, n_orders, vocab_size = struct.unpack("<III", mm[4:16])
        if version != _RAW_VERSION:
            raise ValueError(
                f"FIX1 version {version} != {_RAW_VERSION}; rebuild with current "
                "data_gen/build_forward_index.py"
            )
        self.vocab_size = int(vocab_size)

        section_headers = []
        sh_off = _RAW_HEADER_SIZE
        for _ in range(n_orders):
            sh = mm[sh_off : sh_off + _RAW_SECTION_HEADER_SIZE]
            order, _pad, n_prefixes, _n_continuations, pw_off, _co_off, _c_off = (
                struct.unpack("<IIQQQQQ", sh)
            )
            sh_off += _RAW_SECTION_HEADER_SIZE
            section_headers.append((int(order), int(n_prefixes), int(pw_off)))

        max_order = max((order for order, _, _ in section_headers), default=1)
        if max_order < self.N_max:
            raise ValueError(
                f"raw context forward index has max order {max_order} but caller "
                f"passed N_max={self.N_max}"
            )

        # Raw FIX1 indexes do not contain order 1, so synthesize the empty
        # context row as global row 0.
        self.sections_by_order[1] = _ContextOrderSection(
            order=1,
            prefixes_2d=np.zeros((1, 0), dtype=np.uint32),
            global_offset=0,
        )
        global_offset = 1
        for order, n_prefixes, p_off in sorted(section_headers):
            if order > self.N_max:
                continue
            plen = order - 1
            prefixes = np.frombuffer(
                mm, dtype=np.uint32, count=n_prefixes * plen, offset=p_off
            ).reshape(n_prefixes, plen)
            self.sections_by_order[order] = _ContextOrderSection(
                order=order,
                prefixes_2d=prefixes,
                global_offset=global_offset,
            )
            global_offset += n_prefixes
        self.num_contexts = int(global_offset)

    def lookup_one(self, context: Sequence[int]) -> int:
        self._ensure_open()
        ctx_len = len(context)
        upper = min(max(self.sections_by_order), ctx_len + 1, self.N_max)
        for n in range(upper, 0, -1):
            sect = self.sections_by_order.get(n)
            if sect is None:
                continue
            plen = sect.plen
            if plen == 0:
                return sect.global_offset
            if ctx_len < plen:
                continue
            query = np.asarray(context[-plen:], dtype=np.uint32).reshape(1, plen)
            assert sect.prefixes_struct is not None
            struct_dtype = sect.prefixes_struct.dtype
            query_view = query.view(struct_dtype).reshape(-1)
            idx = int(np.searchsorted(sect.prefixes_struct, query_view[0]))
            if idx < sect.prefixes_struct.shape[0] and (
                sect.prefixes_struct[idx] == query_view[0]
            ):
                return sect.global_offset + idx
        raise RuntimeError(
            f"raw context forward index lookup fell through all orders for context "
            f"of length {ctx_len}; index file may be missing order 1 fallback"
        )

    def close(self):
        self.sections_by_order.clear()
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        self.num_contexts = 0


# ----------------------------------------------------------------------
# Public class used by the composable data source and eval-time PoE wrapper.
# ----------------------------------------------------------------------


class NgramTopKSource:
    """Lookup over a precomputed top-k Kneser-Ney forward index.

    :param table_dir: Directory containing ``forward_index_topk.bin``.
        Or a path directly to that file.
    :param K: Top-K size per position. Must equal the K the index was built
        with.
    :param N_max: Highest ngram order. Must be ≤ the highest order present
        in the index.
    :param output_log_probs: Deprecated compatibility argument. Values are
        always returned as natural-log full-vocabulary ngram probabilities.
    """

    def __init__(
        self,
        table_dir: Union[str, os.PathLike],
        K: int = 16,
        N_max: int = 5,
        output_log_probs: bool = True,
        # Accepted for backwards compatibility with older configs but ignored;
        # the precompute already folded in the unigram path exactly.
        unigram_shortlist: Optional[int] = None,
    ):
        self.K = int(K)
        self.N_max = int(N_max)
        self.output_log_probs = True

        p = Path(table_dir)
        if p.is_dir():
            fwd_path = p / "forward_index_topk.bin"
        elif p.name == "forward_index_topk.bin":
            fwd_path = p
        else:
            raise ValueError(
                f"table_dir must be a directory containing forward_index_topk.bin "
                f"or that file directly. Got: {p}"
            )
        if not fwd_path.is_file():
            raise FileNotFoundError(f"missing top-K forward index: {fwd_path}")
        self.forward_index_topk_path = str(fwd_path)

        # Lazy: don't open mmap in __init__ so the source pickles cleanly
        # to spawn-mode dataloader workers. First lookup_batch triggers
        # /dev/shm mirror + mmap.
        self._index: Optional[TopKForwardIndex] = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_index"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _ensure_open(self):
        if self._index is not None:
            return
        tag = f"[ngram_topk pid={os.getpid()}]"
        import time as _t

        t0 = _t.time()
        print(f"{tag} _ensure_open: mirroring forward_index_topk to /dev/shm", flush=True)
        local_path = _mirror_to_shm(self.forward_index_topk_path)
        print(f"{tag} mirror done in {_t.time() - t0:.2f}s; opening", flush=True)
        idx = TopKForwardIndex(local_path)
        idx._ensure_open()
        if idx.K != self.K:
            raise ValueError(
                f"top-K forward index has K={idx.K} but caller passed K={self.K}"
            )
        max_order = max(idx.sections_by_order)
        if max_order < self.N_max:
            raise ValueError(
                f"top-K forward index has max order {max_order} but caller "
                f"passed N_max={self.N_max}"
            )
        self._index = idx
        print(f"{tag} _ensure_open: total {_t.time() - t0:.2f}s; ready", flush=True)

    def __del__(self):
        idx = getattr(self, "_index", None)
        if idx is not None:
            try:
                idx.close()
            except Exception:
                pass

    # Per-process counter: log only the first few lookup_batch calls.
    _lookup_batch_call_count: dict = {}
    _lookup_batch_log_first_n: int = 8

    def lookup_batch(
        self, contexts: Sequence[Sequence[int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """For each context, return top-k continuations and their natural-log
        full-vocabulary Kneser-Ney probabilities.

        Sentinel slots are returned as ``-inf`` so downstream exponentiation
        gives them zero weight.

        :param contexts: One sequence of ints per query position. Each
            sequence is the (up to N_max-1)-token left context, oldest first.
        :returns: ``(ids[N, K] int32, log_probs[N, K] float32)``.
        """
        self._ensure_open()
        my_pid = os.getpid()
        cnt = NgramTopKSource._lookup_batch_call_count.get(my_pid, 0)
        log_this = cnt < NgramTopKSource._lookup_batch_log_first_n
        NgramTopKSource._lookup_batch_call_count[my_pid] = cnt + 1
        if log_this:
            import time as _t

            t_start = _t.time()
            print(
                f"[ngram_topk pid={my_pid}] lookup_batch call #{cnt}: "
                f"{len(contexts)} contexts",
                flush=True,
            )

        N = len(contexts)
        K = self.K
        ids_out = np.zeros((N, K), dtype=np.int32)
        # Initialize to -inf so untouched rows / unused slots have no weight
        # when downstream softmax exponentiates them.
        values_out = np.full((N, K), -np.inf, dtype=np.float32)
        ln10 = np.float32(np.log(10.0))

        idx = self._index
        for i, ctx in enumerate(contexts):
            tokens, logprobs = idx.lookup_one(ctx)
            mx = float(logprobs.max())
            if mx == float("-inf"):
                continue
            ids_out[i] = tokens.astype(np.int32, copy=False)
            # Raw kenlm log-probabilities, converted log10 -> natural log.
            # Sentinel slots stay at -inf.
            values_out[i] = logprobs.astype(np.float32) * ln10

        if log_this:
            print(
                f"[ngram_topk pid={my_pid}] lookup_batch call #{cnt} done "
                f"in {_t.time() - t_start:.3f}s",
                flush=True,
            )
        return ids_out, values_out


class NgramContextSource:
    """Lookup observed ngram contexts as learned-memory row IDs.

    This uses the prefix blocks from ``forward_index_topk.bin`` and deliberately
    ignores the top-k continuation token IDs and log probabilities.
    """

    def __init__(
        self,
        table_dir: Union[str, Path],
        *,
        N_max: int = 5,
    ):
        self.table_dir = str(table_dir)
        self.N_max = int(N_max)
        p = Path(table_dir)
        if p.is_dir():
            raw_path = p / "forward_index.bin"
            topk_path = p / "forward_index_topk.bin"
            if raw_path.exists():
                fwd_path = raw_path
            elif topk_path.exists():
                fwd_path = topk_path
            else:
                fwd_path = topk_path
        elif p.name in {"forward_index.bin", "forward_index_topk.bin"}:
            fwd_path = p
        else:
            raise ValueError(
                f"table_dir must be a directory containing forward_index.bin or "
                f"forward_index_topk.bin, or one of those files directly, got {table_dir}"
            )
        if not fwd_path.exists():
            raise FileNotFoundError(f"context index file not found: {fwd_path}")
        self.forward_index_path = str(fwd_path)
        self._idx: Optional[Union[ContextForwardIndex, RawContextForwardIndex]] = None

    def _ensure_open(self) -> Union[ContextForwardIndex, RawContextForwardIndex]:
        if self._idx is not None:
            return self._idx
        tag = (
            f"[ngram_context pid={os.getpid()} table={self.forward_index_path} "
            f"N_max={self.N_max}]"
        )
        print(f"{tag} _ensure_open: mirroring context index to /dev/shm", flush=True)
        local_path = _mirror_to_shm(self.forward_index_path)
        if Path(local_path).name == "forward_index.bin":
            idx = RawContextForwardIndex(local_path, N_max=self.N_max)
        else:
            idx = ContextForwardIndex(local_path, N_max=self.N_max)
        idx._ensure_open()
        self._idx = idx
        print(
            f"{tag} opened context index: {idx.num_contexts:,} contexts, "
            f"vocab={idx.vocab_size}",
            flush=True,
        )
        return idx

    @property
    def num_contexts(self) -> int:
        return self._ensure_open().num_contexts

    @property
    def vocab_size(self) -> int:
        return self._ensure_open().vocab_size

    def lookup_batch(self, contexts: Sequence[Sequence[int]]) -> np.ndarray:
        """Return one learned-memory row ID per context."""
        idx = self._ensure_open()
        out = np.empty(len(contexts), dtype=np.int64)
        for i, ctx in enumerate(contexts):
            out[i] = idx.lookup_one(ctx)
        return out

    def close(self):
        if self._idx is not None:
            self._idx.close()
            self._idx = None
