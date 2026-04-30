"""
Training-time soft-target lookup for ngram-train.

Backed by:
  - A KenLM trie binary (``pilot.binary``) built with
    ``build_binary trie -q 8 -a 22`` — provides KN-smoothed conditional
    probabilities via ``BaseScore``.
  - A forward-indexed continuation file (``forward_index.bin``) built by
    ``data_gen/build_forward_index.py`` — for each prefix h of length 1
    through ``N_max - 1``, lists the dolma2 token IDs that complete it at
    order ``|h|+1``.

For each query position with context h:
  1. We enumerate candidate next-tokens by binary-searching the forward
     index at every relevant order, plus a precomputed top-N unigram
     shortlist.
  2. For each candidate w, we compute log P_KN(w | h) via KenLM's
     ``BaseScore`` (full modified-KN cross-order combine).
  3. Top-K by log-prob; renormalize to sum to 1.

The C++ adapter (``data_gen/_ngram_lookup/lookup.cc``) does steps 1-2;
this wrapper handles ctypes plumbing and step 3's renormalization.

Inputs (per position): a context of up to ``N_max - 1`` tokens (the
caller — an InstanceSource — is responsible for walking left from the
target position and stopping at the preceding EOS so the window doesn't
cross document boundaries).

Outputs (per position): ``(topk_ids: int32[K], topk_probs: float32[K])``
with ``topk_probs`` summing to 1.
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence, Tuple, Union

import numpy as np


# File-system prefixes whose contents we treat as "remote" and copy into a
# RAM-backed local cache before opening. Weka-mounted training data lives
# at /weka/...; reading the kenlm trie binary + forward index directly from
# Weka via mmap with 64+ concurrent dataloader workers thrashes the OS page
# cache and stalls training at step 0. /dev/shm is tmpfs (RAM-backed) on
# all our beaker training nodes; one sequential copy populates it.
_REMOTE_PREFIXES = ("/weka/",)
_SHM_CACHE_ROOT = Path(os.environ.get("OLMO_NGRAM_SHM_ROOT", "/dev/shm/olmo_core_ngram_cache"))


def _mirror_to_shm(path: str) -> str:
    """If ``path`` lives on a remote-ish filesystem (weka), copy it once
    into ``/dev/shm`` (tmpfs / RAM-backed) and return the cache path. The
    copy is guarded by an exclusive flock so concurrent dataloader
    processes don't race; whichever gets the lock first does the copy and
    the rest attach to the already-cached file.

    Local-disk paths (e.g. /tmp/...) are returned unchanged.
    """
    if not any(path.startswith(p) for p in _REMOTE_PREFIXES):
        return path
    src_size = os.path.getsize(path)
    path_hash = hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]
    cache_dir = _SHM_CACHE_ROOT / path_hash
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / Path(path).name
    # Fast path: cache already populated by a sibling process.
    try:
        if cache_path.is_file() and cache_path.stat().st_size == src_size:
            return str(cache_path)
    except OSError:
        pass
    lock_path = str(cache_path) + ".lock"
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        # Re-check inside the lock — another process may have done it.
        if cache_path.is_file() and cache_path.stat().st_size == src_size:
            return str(cache_path)
        tmp_path = Path(str(cache_path) + ".partial")
        try:
            print(
                f"[ngram_soft_target] mirroring {path} ({src_size // 2**20} MB) "
                f"→ {cache_path} ...",
                flush=True,
            )
            with open(path, "rb") as src, open(tmp_path, "wb") as dst:
                shutil.copyfileobj(src, dst, length=32 * 1024 * 1024)
            tmp_path.replace(cache_path)
            print(
                f"[ngram_soft_target] mirrored {Path(path).name} "
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


# Default OUT_DIR matches what _ngram_lookup/build.sh uses; both must agree.
_DEFAULT_DYLIB_OUT = Path("/tmp/olmo_core_ngram_lookup")
_NGRAM_LOOKUP_SRC = Path(__file__).parent / "_ngram_lookup"


def _build_dylib_with_lock(out_dir: Path) -> Path:
    """Run build.sh under an exclusive flock so concurrent dataloader processes
    don't race on the build. Returns the path to the resulting .so/.dylib.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "dylib" if sys.platform == "darwin" else "so"
    so_path = out_dir / f"libngram_lookup.{suffix}"
    lock_path = out_dir / "build.lock"

    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        if so_path.is_file():
            # Another process built it while we were waiting on the lock.
            return so_path
        env = os.environ.copy()
        env["OUT_DIR"] = str(out_dir)
        result = subprocess.run(
            ["bash", str(_NGRAM_LOOKUP_SRC / "build.sh")],
            env=env,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if result.returncode != 0 or not so_path.is_file():
            raise RuntimeError(
                f"_ngram_lookup/build.sh failed (returncode={result.returncode})\n"
                f"--- output ---\n{result.stdout.decode(errors='replace')}"
            )
    return so_path


def _resolve_dylib_path() -> Path:
    """Find ``libngram_lookup.{dylib,so}``.

    Resolution order:
      1. ``$OLMO_NGRAM_LOOKUP_DYLIB`` env var, if set.
      2. The default training-job build dir (``/tmp/olmo_core_ngram_lookup``);
         if missing, build it via the bundled ``_ngram_lookup/build.sh``.
    """
    env = os.environ.get("OLMO_NGRAM_LOOKUP_DYLIB")
    if env:
        p = Path(env)
        if p.is_file():
            return p
        raise FileNotFoundError(f"OLMO_NGRAM_LOOKUP_DYLIB={env} does not exist")

    suffix = "dylib" if sys.platform == "darwin" else "so"
    candidate = _DEFAULT_DYLIB_OUT / f"libngram_lookup.{suffix}"
    if candidate.is_file():
        return candidate
    return _build_dylib_with_lock(_DEFAULT_DYLIB_OUT)


_LIB = None


def _get_lib() -> ctypes.CDLL:
    """Lazy-load the dylib once per process and bind ctypes signatures."""
    global _LIB
    if _LIB is not None:
        return _LIB
    lib = ctypes.CDLL(str(_resolve_dylib_path()))

    lib.ngram_lookup_open.argtypes = [
        ctypes.c_char_p,  # trie_path
        ctypes.c_char_p,  # forward_index_path
        ctypes.c_uint,    # unigram_shortlist_size
        ctypes.c_uint,    # max_continuations_per_prefix
    ]
    lib.ngram_lookup_open.restype = ctypes.c_void_p

    lib.ngram_lookup_close.argtypes = [ctypes.c_void_p]
    lib.ngram_lookup_close.restype = None

    lib.ngram_lookup_order.argtypes = [ctypes.c_void_p]
    lib.ngram_lookup_order.restype = ctypes.c_uint

    lib.ngram_lookup_vocab_size.argtypes = [ctypes.c_void_p]
    lib.ngram_lookup_vocab_size.restype = ctypes.c_uint64

    lib.ngram_lookup_n_forward_orders.argtypes = [ctypes.c_void_p]
    lib.ngram_lookup_n_forward_orders.restype = ctypes.c_uint

    lib.ngram_lookup_enumerate_top_k.argtypes = [
        ctypes.c_void_p,                                 # handle
        ctypes.POINTER(ctypes.c_uint32),                 # prefix
        ctypes.c_uint,                                   # prefix_len
        ctypes.c_uint,                                   # k
        ctypes.POINTER(ctypes.c_uint32),                 # out_token_ids
        ctypes.POINTER(ctypes.c_float),                  # out_log_probs
    ]
    lib.ngram_lookup_enumerate_top_k.restype = ctypes.c_int

    _LIB = lib
    return lib


class NgramTableSoftTargetSource:
    """Soft-target lookup over a KenLM trie binary + a forward continuation index.

    :param table_dir: Directory containing ``pilot.binary`` and
        ``forward_index.bin``. Or a path directly to either file (we'll
        find the sibling).
    :param K: Top-K size per position.
    :param N_max: Highest ngram order — must equal the order the trie
        binary was built with (e.g., 5 for our pilot-1e-3-n5 build).
    :param unigram_shortlist: Size of the precomputed unigram fallback
        candidate set used at the order-1 level.
    """

    def __init__(
        self,
        table_dir: Union[str, os.PathLike],
        K: int = 16,
        N_max: int = 5,
        unigram_shortlist: int = 100,
        max_continuations_per_prefix: int = 64,
    ):
        self.K = int(K)
        self.N_max = int(N_max)
        self.unigram_shortlist_size = int(unigram_shortlist)
        self.max_continuations_per_prefix = int(max_continuations_per_prefix)
        if self.unigram_shortlist_size < self.K:
            raise ValueError(
                f"unigram_shortlist ({self.unigram_shortlist_size}) must be >= K ({self.K})"
            )
        if self.max_continuations_per_prefix < self.K:
            raise ValueError(
                f"max_continuations_per_prefix ({self.max_continuations_per_prefix}) "
                f"must be >= K ({self.K}) — otherwise we can't fill K candidates "
                f"from a single observed prefix"
            )

        p = Path(table_dir)
        if p.is_dir():
            trie_path = p / "pilot.binary"
            fwd_path = p / "forward_index.bin"
        elif p.name == "pilot.binary":
            trie_path = p
            fwd_path = p.parent / "forward_index.bin"
        elif p.name == "forward_index.bin":
            trie_path = p.parent / "pilot.binary"
            fwd_path = p
        else:
            raise ValueError(
                f"table_dir must be a directory containing pilot.binary + "
                f"forward_index.bin, or one of those files directly. Got: {p}"
            )
        if not trie_path.is_file():
            raise FileNotFoundError(f"missing trie binary: {trie_path}")
        if not fwd_path.is_file():
            raise FileNotFoundError(f"missing forward index: {fwd_path}")
        self.trie_path = trie_path
        self.forward_index_path = fwd_path

        # Lazy: don't open the dylib or kenlm in __init__. The wrapper has
        # to be picklable so torch DataLoader can spawn worker processes
        # with it — and ctypes.CDLL's internal _FuncPtr type can't pickle.
        # First call to lookup_batch triggers the actual open, which
        # mirrors weka files to /dev/shm (one per node) and constructs
        # the C++ ModelWrapper. Each worker process pays this once.
        self._lib = None
        self._handle = None

    def __getstate__(self):
        # Drop the unpicklable lib + handle; everything else is plain data.
        state = self.__dict__.copy()
        state["_lib"] = None
        state["_handle"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _ensure_open(self):
        """Mirror tables to /dev/shm and open the C++ ModelWrapper. Idempotent.

        Heavily instrumented because this runs once per dataloader worker
        across N×8 worker processes; if any phase silently hangs, the
        per-process logs are how we'd find the broken phase.
        """
        if self._handle is not None:
            return
        # Tag logs with pid so per-worker progress is distinguishable.
        tag = f"[ngram_soft_target pid={os.getpid()}]"
        import time as _t
        t0 = _t.time()
        print(f"{tag} _ensure_open: mirror trie + forward_index to /dev/shm", flush=True)
        trie_local = _mirror_to_shm(str(self.trie_path))
        fwd_local = _mirror_to_shm(str(self.forward_index_path))
        print(f"{tag} mirror done in {_t.time() - t0:.2f}s; resolving dylib", flush=True)
        t1 = _t.time()
        lib = _get_lib()
        print(f"{tag} dylib loaded in {_t.time() - t1:.2f}s; opening kenlm + forward index", flush=True)
        t2 = _t.time()
        handle = lib.ngram_lookup_open(
            trie_local.encode("utf-8"),
            fwd_local.encode("utf-8"),
            self.unigram_shortlist_size,
            self.max_continuations_per_prefix,
        )
        print(
            f"{tag} ngram_lookup_open returned in {_t.time() - t2:.2f}s "
            f"(handle={'ok' if handle else 'NULL'})",
            flush=True,
        )
        if not handle:
            raise RuntimeError(
                f"ngram_lookup_open failed; check stderr. trie={self.trie_path}, "
                f"forward_index={self.forward_index_path}"
            )
        actual_order = int(lib.ngram_lookup_order(handle))
        if actual_order != self.N_max:
            lib.ngram_lookup_close(handle)
            raise ValueError(
                f"trie binary has order={actual_order} but caller passed N_max={self.N_max}"
            )
        # Bind only after both checks pass.
        self._lib = lib
        self._handle = handle
        print(f"{tag} _ensure_open: total {_t.time() - t0:.2f}s; ready", flush=True)

    def __del__(self):
        h = getattr(self, "_handle", None)
        lib = getattr(self, "_lib", None)
        if h and lib is not None:
            lib.ngram_lookup_close(h)
            self._handle = None

    # Per-process counter: log the first N lookup_batch calls so we can see
    # warm-up behavior without spamming every training step.
    _lookup_batch_call_count: dict = {}
    _lookup_batch_log_first_n: int = 16

    def lookup_batch(
        self, contexts: Sequence[Sequence[int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """For each context, return top-K continuations and renormalized probs.

        :param contexts: One sequence of ints per query position. Each
            sequence is the (up to N_max-1)-token context, oldest first.
        :returns: ``(ids[N, K] int32, probs[N, K] float32)``. Each row of
            ``probs`` sums to 1.
        """
        self._ensure_open()
        my_pid = os.getpid()
        cnt = NgramTableSoftTargetSource._lookup_batch_call_count.get(my_pid, 0)
        log_this = cnt < NgramTableSoftTargetSource._lookup_batch_log_first_n
        NgramTableSoftTargetSource._lookup_batch_call_count[my_pid] = cnt + 1
        if log_this:
            import time as _t
            t_start = _t.time()
            print(
                f"[ngram_soft_target pid={my_pid}] lookup_batch call #{cnt}: "
                f"{len(contexts)} contexts",
                flush=True,
            )
        N = len(contexts)
        K = self.K
        all_ids = np.zeros((N, K), dtype=np.int32)
        all_probs = np.zeros((N, K), dtype=np.float32)

        ids_buf = np.zeros(K, dtype=np.uint32)
        logp_buf = np.zeros(K, dtype=np.float32)
        ids_ptr = ids_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        logp_ptr = logp_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        enumerate_fn = self._lib.ngram_lookup_enumerate_top_k

        for i, ctx in enumerate(contexts):
            ctx_arr = np.ascontiguousarray(ctx, dtype=np.uint32)
            ctx_ptr = ctx_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
            n_out = enumerate_fn(self._handle, ctx_ptr, ctx_arr.size, K, ids_ptr, logp_ptr)
            if n_out <= 0:
                continue

            logp = logp_buf[:n_out]
            shifted = logp - logp.max()
            linear = np.power(10.0, shifted)
            total = linear.sum()
            if total > 0:
                linear /= total

            all_ids[i, :n_out] = ids_buf[:n_out].astype(np.int32)
            all_probs[i, :n_out] = linear

        if log_this:
            print(
                f"[ngram_soft_target pid={my_pid}] lookup_batch call #{cnt} done "
                f"in {_t.time() - t_start:.2f}s",
                flush=True,
            )
        return all_ids, all_probs
