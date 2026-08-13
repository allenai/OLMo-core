"""CPU parity + speed test for the CHUNK_CACHE_IDS incremental chunk-id cache.

Proves the cached path is ELEMENTWISE IDENTICAL to the uncached path over a simulated
prefill-then-decode trajectory, including the adversarial case the cache's correctness argument
turns on: a doc_start / doc_end appearing in GENERATED tokens (which must force a full recompute).

No GPU, no vLLM. Run with the corpus-reasoning-olmo python.
"""

import importlib
import os
import sys
import time
import types

import numpy as np
import torch

REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, REPO_SRC)

DOC_START, DOC_END = 151648, 151649


class FakeBatch:
    def __init__(self, token_ids_cpu, req_ids):
        self.token_ids_cpu = token_ids_cpu
        self.req_ids = req_ids


class FakeMeta:
    def __init__(self, seq_lens):
        self.num_reqs = len(seq_lens)
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int32)


def load_patch(cache_enabled: bool):
    os.environ["CHUNK_CACHE_IDS"] = "1" if cache_enabled else "0"
    for name in list(sys.modules):
        if name.endswith("vllm_chunked_patch"):
            del sys.modules[name]
    mod = importlib.import_module("corpus_reasoning.lib.vllm_chunked_patch")
    mod.set_doc_token_ids(DOC_START, DOC_END)
    return mod


def make_corpus(num_reqs, prompt_len, n_chunks, seed=0):
    buf = np.zeros((num_reqs, prompt_len + 600), dtype=np.int32)
    rng = np.random.default_rng(seed)
    per = prompt_len // (n_chunks + 1)
    for ri in range(num_reqs):
        buf[ri, :] = rng.integers(1000, 50000, size=buf.shape[1])
        for c in range(n_chunks):
            s = c * per
            e = min(s + per - 2, prompt_len - 1)
            buf[ri, s] = DOC_START
            buf[ri, e] = DOC_END
        # leave one request with a trailing UNMATCHED doc_start (partial-prefill shape)
        if ri == 1:
            buf[ri, prompt_len - 3] = DOC_START
    return buf


def trajectory(num_reqs, prompt_len, n_gen, evil_marker_at=None):
    """Yield (seq_lens,) for a prefill step followed by n_gen decode steps."""
    yield [prompt_len] * num_reqs
    for t in range(1, n_gen + 1):
        yield [prompt_len + t] * num_reqs
    if evil_marker_at is not None:
        pass


def run(mod, buf, num_reqs, prompt_len, n_gen, req_ids):
    out = []
    ib = FakeBatch(buf, req_ids)
    for seq_lens in trajectory(num_reqs, prompt_len, n_gen):
        cm = FakeMeta(seq_lens)
        t = mod._build_chunk_ids_for_batch(ib, cm, torch.device("cpu"))
        out.append(t.numpy().copy())
    return out


def main():
    num_reqs, prompt_len, n_chunks, n_gen = 4, 4000, 90, 40
    buf = make_corpus(num_reqs, prompt_len, n_chunks)
    req_ids = [f"r{i}" for i in range(num_reqs)]

    # --- case 1: ordinary generation (no markers in generated tokens) -----------------
    plain = load_patch(False)
    ref = run(plain, buf, num_reqs, prompt_len, n_gen, req_ids)
    cached = load_patch(True)
    got = run(cached, buf, num_reqs, prompt_len, n_gen, req_ids)
    assert len(ref) == len(got)
    for i, (a, b) in enumerate(zip(ref, got)):
        assert a.shape == b.shape, f"step {i}: shape {a.shape} vs {b.shape}"
        assert np.array_equal(a, b), f"step {i}: MISMATCH at {np.flatnonzero(a != b)[:10]}"
    print(f"case 1 (plain generation): {len(ref)} steps IDENTICAL")

    # --- case 2: the model emits doc_start/doc_end DURING generation ------------------
    buf2 = buf.copy()
    buf2[:, prompt_len + 5] = DOC_START
    buf2[:, prompt_len + 12] = DOC_END
    buf2[:, prompt_len + 20] = DOC_START
    plain = load_patch(False)
    ref2 = run(plain, buf2, num_reqs, prompt_len, n_gen, req_ids)
    cached = load_patch(True)
    got2 = run(cached, buf2, num_reqs, prompt_len, n_gen, req_ids)
    for i, (a, b) in enumerate(zip(ref2, got2)):
        assert np.array_equal(a, b), f"case2 step {i}: MISMATCH"
    print(f"case 2 (markers inside generated tokens): {len(ref2)} steps IDENTICAL")

    # --- case 3: slot reuse -- same slot index, DIFFERENT request id ------------------
    plain = load_patch(False)
    a1 = run(plain, buf, num_reqs, prompt_len, 5, req_ids)
    a2 = run(plain, buf2, num_reqs, prompt_len, 5, [f"s{i}" for i in range(num_reqs)])
    cached = load_patch(True)
    b1 = run(cached, buf, num_reqs, prompt_len, 5, req_ids)
    b2 = run(cached, buf2, num_reqs, prompt_len, 5, [f"s{i}" for i in range(num_reqs)])
    for i, (x, y) in enumerate(list(zip(a1, b1)) + list(zip(a2, b2))):
        assert np.array_equal(x, y), f"case3 step {i}: MISMATCH"
    print("case 3 (slot reuse by a new request id): IDENTICAL")

    # --- speed -----------------------------------------------------------------------
    big = make_corpus(8, 8875, 187, seed=7)
    ids8 = [f"b{i}" for i in range(8)]
    for label, enabled in (("uncached", False), ("cached", True)):
        m = load_patch(enabled)
        ib = FakeBatch(big, ids8)
        cm0 = FakeMeta([8875] * 8)
        m._build_chunk_ids_for_batch(ib, cm0, torch.device("cpu"))  # prefill/warm
        t0 = time.perf_counter()
        n = 200
        for t in range(n):
            cm = FakeMeta([8875 + 1 + t] * 8)
            m._build_chunk_ids_for_batch(ib, cm, torch.device("cpu"))
        dt = (time.perf_counter() - t0) / n * 1000.0
        print(f"decode-step chunk_ids, 8 reqs x 8875 tok: {label:9s} {dt:7.3f} ms/call")

    print("\nALL PARITY CHECKS PASSED")


if __name__ == "__main__":
    main()
