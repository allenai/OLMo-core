"""Generate groups-of-4 data (O(N^3+) toy task).

Each example is N arithmetic expressions (or bare numbers with --numbers-only).
The task: find every group of G expressions (default G=4) whose answers are all
within X of each other — a tight cluster. Naively O(N^G) to find a G-cluster.

This is the G-tuple generalization of mathmatch (which finds PAIRS within x).
The PDF's example property is "4 expressions that add up to within 100"; a
literal subset-SUM property cannot be made provably unique without astronomical
numbers or brute-forcing all C(N,G) subsets, so we use the all-pairwise-within-X
("tight cluster") property, which mirrors mathmatch and is provably unique with
small numbers (see below). `--target-sum` is a documented alternative.

Construction guarantees exactly K gold groups:
  - K cluster centers, pairwise spaced > 2X apart.
  - Each gold group is G distinct values inside a span of X around its center.
  - Every distractor is > X from EVERY other value (gold or distractor).
  => any G values that are all-pairwise-within-X must lie in one span of width
     X; by the separation, that can only be one planted group. (Same argument
     as mathmatch's pair construction, lifted to G-tuples.)

`gold_doc_indices` stores each group as a list of G 1-indexed IDs:
`[[3, 8, 12, 19], ...]`. Output schema/metric are shared with the `cycle`
task (set-of-groups), so evaluate with `--task groups4`.

Usage:
    python scripts/data/generate_groups4_data.py \\
        --num-docs 100 --num-groups 1 --group-size 4 --tolerance 5 \\
        --ans-min -500 --ans-max 500 --num-train 2000 --num-eval 300
"""

import argparse
import json
import random


def build_expression(target, rng, term_min=1, term_max=99):
    """Small arithmetic expression evaluating exactly to `target`."""
    n_terms = rng.randint(4, 5)
    terms = [rng.randint(term_min, term_max) for _ in range(n_terms - 1)]
    signs = [rng.choice([1, -1]) for _ in range(n_terms - 1)]
    partial = sum(s * t for s, t in zip(signs, terms))
    last = target - partial
    if last == 0:
        terms[0] += 1
        last = target - sum(s * t for s, t in zip(signs, terms))
    terms.append(abs(last))
    signs.append(1 if last >= 0 else -1)
    parts = []
    for i, (s, t) in enumerate(zip(signs, terms)):
        if i == 0:
            parts.append(str(s * t))
        else:
            parts.append("+" if s == 1 else "-")
            parts.append(str(t))
    return " ".join(parts)


def sample_values(n_docs, n_groups, group_size, tol, ans_min, ans_max, rng,
                  max_attempts=400):
    """Return (values, gold_index_groups) with exactly n_groups all-within-tol
    clusters of `group_size`, every other value > tol from all others."""
    G, K = group_size, n_groups
    for _ in range(max_attempts):
        # 1) K centers spaced > 2*tol apart so clusters can't merge or touch.
        spacing = 2 * tol + 1
        cand = list(range(ans_min + tol, ans_max - tol + 1))
        rng.shuffle(cand)
        centers = []
        for c in cand:
            if all(abs(c - o) > spacing for o in centers):
                centers.append(c)
                if len(centers) == K:
                    break
        if len(centers) < K:
            continue

        # 2) G distinct values within a span of tol around each center.
        group_vals = []
        ok = True
        for c in centers:
            window = list(range(c, c + tol + 1))
            if len(window) < G:
                ok = False
                break
            group_vals.append(rng.sample(window, G))
        if not ok:
            continue
        flat_group = [v for g in group_vals for v in g]
        if len(set(flat_group)) != len(flat_group):
            continue  # collision across windows (rare); retry

        # 3) distractors, each > tol from every chosen value.
        n_distract = n_docs - K * G
        pool = list(range(ans_min, ans_max + 1))
        rng.shuffle(pool)
        chosen = list(flat_group)
        distractors = []
        for v in pool:
            if all(abs(v - o) > tol for o in chosen):
                distractors.append(v)
                chosen.append(v)
                if len(distractors) == n_distract:
                    break
        if len(distractors) < n_distract:
            continue

        values = flat_group + distractors
        gold = [list(range(i * G, i * G + G)) for i in range(K)]
        return values, gold

    raise RuntimeError(
        f"Could not assemble values after {max_attempts} attempts. Widen "
        f"[ans_min, ans_max] or lower num_docs/tolerance/group_size.")


def build_example(args, rng):
    G = args.group_size
    values, gold_idx = sample_values(
        args.num_docs, args.num_groups, G, args.tolerance,
        args.ans_min, args.ans_max, rng)

    if args.numbers_only:
        exprs = [str(v) for v in values]
    else:
        exprs = [build_expression(v, rng) for v in values]

    order = list(range(len(values)))
    rng.shuffle(order)
    perm = {old: new + 1 for new, old in enumerate(order)}  # 1-indexed
    shuffled = [exprs[i] for i in order]

    gold = sorted(sorted(perm[i] for i in group) for group in gold_idx)
    query = (f"Find every group of {G} expressions whose answers are all within "
             f"{args.tolerance} of each other (every pair in the group differs "
             f"by at most {args.tolerance}).")

    return {
        "documents": [{"text": s} for s in shuffled],
        "queries": [query],
        "answers": [],
        "gold_doc_indices": gold,   # list of groups, each G 1-indexed IDs
        "source": "groups4",
        "_meta": {"group_size": G, "num_groups": args.num_groups,
                  "tolerance": args.tolerance, "numbers_only": args.numbers_only},
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-docs", type=int, default=100)
    ap.add_argument("--num-groups", type=int, default=1,
                    help="exact number of gold groups (K)")
    ap.add_argument("--group-size", type=int, default=4,
                    help="expressions per group (G); larger = harder (N^G)")
    ap.add_argument("--tolerance", type=int, default=5,
                    help="max pairwise difference within a group (X)")
    ap.add_argument("--ans-min", type=int, default=-500)
    ap.add_argument("--ans-max", type=int, default=500)
    ap.add_argument("--numbers-only", action="store_true",
                    help="documents are bare numbers, no arithmetic to evaluate")
    ap.add_argument("--num-train", type=int, default=2000)
    ap.add_argument("--num-eval", type=int, default=300)
    ap.add_argument("--output-dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    no = "_numsonly" if args.numbers_only else ""
    for split_label, count in [("train", args.num_train), ("eval", args.num_eval)]:
        if count <= 0:
            continue
        examples = [build_example(args, rng) for _ in range(count)]
        tag = (f"n{args.num_docs}_g{args.group_size}_k{args.num_groups}"
               f"_x{args.tolerance}{no}")
        path = f"{args.output_dir}/groups4_{split_label}_{tag}.jsonl"
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"{split_label}: {len(examples)} examples -> {path}")


if __name__ == "__main__":
    main()
