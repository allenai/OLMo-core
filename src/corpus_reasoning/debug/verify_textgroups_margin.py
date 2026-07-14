"""Verify the isolation-margin lever: for M in {0,5,10}, generate examples and
count how many NON-GOLD triples land near the target. M>0 should clear the band
[T-M, T+M] of all decoys, leaving only the K gold triples at exactly T."""
import random
import sys
from itertools import combinations
# sys.path hack removed: corpus_reasoning is a package on PYTHONPATH=src
from corpus_reasoning.data import generate_textgroups_data as G


class A:
    group_size = 3
    tolerance = 0
    feature = "mixed"
    cmin = 4
    cmax = 40
    target = 70
    filler_max = None   # let build_example pick (auto-bumps when separation>0)
    num_groups = 2


def near_miss_profile(ex):
    """Return {band: count of NON-gold triples whose sum is within band of T}."""
    c = ex["_meta"]["counts"]
    T = ex["_meta"]["target"]
    gold = {tuple(sorted(j - 1 for j in g)) for g in ex["gold_doc_indices"]}
    prof = {}
    for band in (0, 1, 2, 5, 10):
        prof[band] = sum(
            1 for t in combinations(range(len(c)), 3)
            if tuple(sorted(t)) not in gold and abs(c[t[0]] + c[t[1]] + c[t[2]] - T) <= band)
    return prof


for M, ndocs in [(M, nd) for M in (0, 3, 5, 8, 10) for nd in (20, 100)]:
    a = A(); a.separation = M; a.num_docs = ndocs
    rng = random.Random(0)
    agg = {b: 0 for b in (0, 1, 2, 5, 10)}
    n = 100
    max_count = 0
    sum_count = 0
    fmax = None
    fails = 0
    for _ in range(n):
        try:
            ex = G.build_example(a, rng)
        except RuntimeError:
            fails += 1
            continue
        p = near_miss_profile(ex)
        for b in agg:
            agg[b] += p[b]
        cs = ex["_meta"]["counts"]
        max_count = max(max_count, max(cs))
        sum_count += sum(cs) / len(cs)
        fmax = ex["_meta"]["filler_max"]
    done = n - fails
    if done == 0:
        print(f"\n### M={M} n_docs={ndocs}: INFEASIBLE (0/{n} assembled)")
        continue
    n = done  # average over successes only
    print(f"\n### separation M={M}  (n_docs={ndocs}, {done}/{done+fails} assembled, "
          f"filler_max={fmax})")
    print(f"   passage feature-count: avg {sum_count/n:.1f}, max {max_count}")
    print("   non-gold triples within band of T=70:")
    for b in (0, 1, 2, 5, 10):
        print(f"     |sum-70| <= {b:2d}:  {agg[b]/n:6.2f}")

# scaling sanity at large N with a margin
print("\n### large-N with margin M=5")
for N in (100, 500, 1000):
    a = A(); a.separation = 5; a.num_docs = N
    rng = random.Random(0)
    ok = 0
    try:
        for _ in range(3):
            ex = G.build_example(a, rng)
            p = near_miss_profile(ex)
            assert p[5] == 0, f"near-miss leak at N={N}: {p}"
            ok += 1
    except Exception as e:
        print(f"  N={N:5d}: {ok}/3 ok  FAIL: {str(e)[:70]}")
        continue
    print(f"  N={N:5d}: {ok}/3 ok  margin clean (0 non-gold triples within 5 of T)")

print("\nDONE")
