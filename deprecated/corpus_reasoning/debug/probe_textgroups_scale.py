"""Probe textgroups scaling with the filler-max trick: confirm sample_values
converges at large N and that exact-K still holds (full C(N,3) brute force)."""
import random
import sys
import time
from itertools import combinations
# sys.path hack removed: corpus_reasoning is a package on PYTHONPATH=src
from corpus_reasoning.data import generate_textgroups_data as G


class A:
    group_size = 3
    tolerance = 0
    feature = "mixed"


def brute_k(ex):
    c = ex["_meta"]["counts"]
    T, W = ex["_meta"]["target"], ex["_meta"]["tolerance"]
    return sum(1 for t in combinations(range(len(c)), 3)
               if abs(c[t[0]] + c[t[1]] + c[t[2]] - T) <= W)


settings = [
    dict(cmin=4, cmax=40, target=70, filler_max=50),
    dict(cmin=4, cmax=40, target=70, filler_max=70),
]
for s in settings:
    print(f"\n### {s}")
    for N in (100, 500, 1000):
        a = A(); a.num_docs = N; a.num_groups = 2
        for k, v in s.items():
            setattr(a, k, v)
        rng = random.Random(0)
        t0 = time.time()
        ok = 0
        last = None
        try:
            for _ in range(3):
                last = G.build_example(a, rng)
                ok += 1
        except Exception as e:
            print(f"  N={N:5d}: {ok}/3 ok   FAIL: {str(e)[:60]}")
            continue
        dt = time.time() - t0
        k_brute = brute_k(last)
        verdict = "OK" if k_brute == a.num_groups else f"BAD(K={k_brute})"
        # rough context size: total feature tokens across passages
        ncounts = last["_meta"]["counts"]
        print(f"  N={N:5d}: {ok}/3 ok  ({dt:.2f}s)  brute T-triples={k_brute} "
              f"[{verdict}]  count range [{min(ncounts)},{max(ncounts)}]")
