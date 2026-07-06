"""
Random-baseline Jaccard similarity for landmark gate selection.

Context for the landmark gate similarity analysis (see
``src/olmo_core/nn/attention/landmark_gate_analysis.py``): we measure how similar
the *opened-gate sets* are across layers, heads, models, or decode steps, using
Jaccard similarity. To read those numbers we need a chance baseline.

Setup:
  * The sequence is divided into blocks of ``block_size = mem_freq + 1 = 64``
    tokens; each block ends in one landmark ("memory") token = one gate. So there
    is exactly one landmark gate per 64 tokens, and at context length ``L`` the
    number of gates available for retrieval at decode is ``N = L / 64``.
  * Hard top-k retrieval keeps exactly ``top_k`` gates open per (layer, head,
    step). The gate-log runs (``launch_long_context_evals.sh``) use a FIXED,
    length-independent ``top_k = 64``.

Baseline (symmetric comparison): if two heads/layers/steps/models each opened
``k`` gates chosen uniformly at random and independently from the ``N`` available
gates, the expected Jaccard ``|A n B| / |A u B|`` is what we'd see by chance.
Because ``k`` is fixed while ``N = L/64`` grows, the baseline FALLS with length --
roughly halving each time the context doubles.

Exact value: the overlap ``c = |A n B|`` is hypergeometric -- fixing set A (k of
N), set B draws k of N, and ``c`` of them land in A -- so

    P(c) = C(k, c) * C(N - k, k - c) / C(N, k),   c ~ Hypergeom(N, k, k)

and, since ``|A u B| = 2k - c``,

    E[Jaccard] = sum_c  P(c) * c / (2k - c).

This is exact and instant (no simulation). Note ``k / (2N - k)`` is only an
*approximation* of this (it is the ratio of expectations, ``E[|AnB|] / E[|AuB|]``,
which differs from ``E[|AnB| / |AuB|]`` because Jaccard is a ratio); it is shown
alongside for reference and is off by ~1e-3 here.
"""

from math import comb

BLOCK_SIZE = 64
TOP_K = 64  # fixed, length-independent (gate-log setup)
LENGTHS = [8192, 16384, 32768, 65536]  # 8k, 16k, 32k, 64k


def exact_jaccard(N: int, k: int) -> float:
    """Exact E[Jaccard] of two independent uniform ``k``-subsets of ``{0..N-1}``.

    Sums the hypergeometric overlap distribution ``c ~ Hypergeom(N, k, k)``.
    """
    if k >= N:  # every gate open on both sides -> identical sets
        return 1.0
    denom = comb(N, k)
    total = 0.0
    # c can range over max(0, 2k - N) .. k; terms with c == 0 contribute 0.
    for c in range(max(1, 2 * k - N), k + 1):
        p = comb(k, c) * comb(N - k, k - c) / denom
        total += p * c / (2 * k - c)
    return total


def approx_jaccard(N: int, k: int) -> float:
    """Approximation ``k / (2N - k)`` (ratio of expectations)."""
    return 1.0 if k >= N else k / (2 * N - k)


def main() -> None:
    print("Landmark gate random-baseline Jaccard")
    print(f"  block_size = {BLOCK_SIZE} tokens/gate   ->   N = L / {BLOCK_SIZE} gates")
    print(f"  top_k = {TOP_K} (fixed)   symmetric comparison (both sides open top_k)\n")
    print(f"{'context':>8} {'N=L/64':>8} {'top_k':>6} {'k/N':>7} {'baseline Jaccard':>18} {'k/(2N-k)':>10}")
    for L in LENGTHS:
        N = L // BLOCK_SIZE
        j = exact_jaccard(N, TOP_K)
        print(
            f"{L // 1024:>6}k {N:>8} {TOP_K:>6} {TOP_K / N:>7.3f} "
            f"{j:>18.4f} {approx_jaccard(N, TOP_K):>10.4f}"
        )


if __name__ == "__main__":
    main()
