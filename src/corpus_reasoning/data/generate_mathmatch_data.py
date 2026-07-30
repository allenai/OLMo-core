"""Generate mathmatch data.

Each example is a list of N arithmetic expressions. The task is:
"Find all pairs of expressions whose answers differ by at most x."
The generator guarantees exactly K such pairs per example, and every other
pair of expression-answers is strictly farther than x apart.

Mirrors the contradiction / matching-n-gram tasks
(`gold_doc_indices: [[a, b], ...]` with 1-indexed positions) so the same
prompt builder, parser, and metrics apply.

With --numbers-only the documents are the bare answer values instead of
arithmetic expressions (the query text is unchanged), isolating the matching
task from the per-document arithmetic.

Usage:
    python scripts/data/generate_mathmatch_data.py \\
        --num-docs 20 --num-pairs 3 --tolerance 2 \\
        --num-train 1000 --num-eval 200 --output-dir data
"""

import argparse
import random
from tqdm import tqdm

from corpus_reasoning.lib.io import save_jsonl, print_dataset_stats


def sample_answer_set(n_docs, n_pairs, tol, ans_min, ans_max, rng,
                      max_attempts=200):
    """Return a list of N answers with exactly n_pairs within-tol pairs.

    Construction: pick K base values spaced > 2*tol+tol apart so their
    neighborhoods don't bleed into each other or into distractors; for each,
    pick a partner within tol; then fill distractors all > tol from every
    other answer.
    """
    for _ in range(max_attempts):
        # 1) K base values spaced > 3*tol apart (a "safe halo" around each pair)
        base_min_spacing = max(1, 3 * tol + 1)
        candidates = list(range(ans_min, ans_max + 1))
        rng.shuffle(candidates)
        bases = []
        for v in candidates:
            if all(abs(v - b) >= base_min_spacing for b in bases):
                bases.append(v)
                if len(bases) == n_pairs:
                    break
        if len(bases) < n_pairs:
            continue

        # 2) partner for each base, within tol but not equal to base
        partners = []
        ok = True
        for b in bases:
            offsets = [d for d in range(-tol, tol + 1) if d != 0]
            rng.shuffle(offsets)
            partner = None
            for d in offsets:
                cand = b + d
                if ans_min <= cand <= ans_max:
                    partner = cand
                    break
            if partner is None:
                ok = False; break
            partners.append(partner)
        if not ok:
            continue

        pair_values = bases + partners

        # 3) distractors, each > tol from every existing answer
        distractors = []
        n_distract = n_docs - 2 * n_pairs
        pool = list(range(ans_min, ans_max + 1))
        rng.shuffle(pool)
        for v in pool:
            if all(abs(v - other) > tol for other in pair_values + distractors):
                distractors.append(v)
                if len(distractors) == n_distract:
                    break
        if len(distractors) < n_distract:
            continue

        answers = pair_values + distractors
        # gold pairs are (bases[i], partners[i]) — record by their indices in `answers`
        gold_idx_pairs = [(i, n_pairs + i) for i in range(n_pairs)]
        return answers, gold_idx_pairs

    raise RuntimeError(
        f"Could not assemble answer set after {max_attempts} attempts. "
        f"Try widening [ans_min, ans_max] or lowering n_docs/n_pairs."
    )


def build_expression(target, rng, n_ops_min=3, n_ops_max=4,
                     term_min=1, term_max=99):
    """Build a small arithmetic expression evaluating exactly to `target`."""
    n_ops = rng.randint(n_ops_min, n_ops_max)
    n_terms = n_ops + 1
    # pick n_terms-1 terms freely, then choose the last so the sum hits target
    terms = [rng.randint(term_min, term_max) for _ in range(n_terms - 1)]
    signs = [rng.choice([1, -1]) for _ in range(n_terms - 1)]
    partial = sum(s * t for s, t in zip(signs, terms))
    last = target - partial
    # last sign chosen to make magnitude plausible-looking (avoid huge ±)
    if last == 0:
        # avoid +0 / -0; nudge one term and recompute
        terms[0] += 1
        last = target - sum(s * t for s, t in zip(signs, terms))
    sign_last = 1 if last >= 0 else -1
    terms.append(abs(last))
    signs.append(sign_last)

    # render: leading term has implicit + sign omitted; '-N' is fine
    parts = []
    for i, (s, t) in enumerate(zip(signs, terms)):
        if i == 0:
            parts.append(str(s * t))
        else:
            parts.append("+" if s == 1 else "-")
            parts.append(str(t))
    return " ".join(parts)


def build_example(n_docs, n_pairs, tol, ans_min, ans_max, rng,
                  term_min=1, term_max=99, n_ops_min=3, n_ops_max=4,
                  numbers_only=False):
    answers, gold_idx_pairs = sample_answer_set(
        n_docs, n_pairs, tol, ans_min, ans_max, rng)
    if numbers_only:
        # documents are the bare answer values, no arithmetic to evaluate
        exprs = [str(a) for a in answers]
    else:
        exprs = [build_expression(a, rng, n_ops_min=n_ops_min, n_ops_max=n_ops_max,
                                  term_min=term_min, term_max=term_max)
                 for a in answers]

    # shuffle positions, remap gold indices to the new positions (1-indexed)
    order = list(range(n_docs))
    rng.shuffle(order)
    perm = {old: new for new, old in enumerate(order)}
    shuffled_exprs = [exprs[i] for i in order]
    shuffled_answers = [answers[i] for i in order]

    pairs = []
    for a, b in gold_idx_pairs:
        na, nb = perm[a] + 1, perm[b] + 1
        pairs.append(sorted([na, nb]))
    pairs.sort()

    query = (f"Find all pairs of expressions whose answers differ by at most "
             f"{tol}.")

    ex = {
        "documents": [{"text": s} for s in shuffled_exprs],
        "queries": [query],
        "answers": [],
        "gold_doc_indices": pairs,
        "source": "synth_mathmatch",
        "_meta": {
            "tolerance": tol,
            "answer_values": shuffled_answers,
        },
    }
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-docs", type=int, default=20)
    ap.add_argument("--num-pairs", type=int, default=3,
                    help="exact number of within-tolerance gold pairs per example")
    ap.add_argument("--tolerance", type=int, default=2,
                    help="x: gold pairs satisfy |val(a)-val(b)| <= x; every "
                         "non-gold pair is strictly > x apart")
    ap.add_argument("--ans-min", type=int, default=-50)
    ap.add_argument("--ans-max", type=int, default=50)
    ap.add_argument("--term-min", type=int, default=1)
    ap.add_argument("--term-max", type=int, default=99)
    ap.add_argument("--n-ops-min", type=int, default=3)
    ap.add_argument("--n-ops-max", type=int, default=4)
    ap.add_argument("--numbers-only", action="store_true",
                    help="documents are bare answer values (no arithmetic "
                         "expressions); the query text is unchanged")
    ap.add_argument("--num-train", type=int, default=1000)
    ap.add_argument("--num-eval", type=int, default=200)
    ap.add_argument("--output-dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    tag = (f"mathmatch_n{args.num_docs}_k{args.num_pairs}_x{args.tolerance}"
           f"_r{args.ans_min}to{args.ans_max}")
    if args.numbers_only:
        tag += "_numsonly"

    for split, n in [("train", args.num_train), ("eval", args.num_eval)]:
        if n <= 0:
            continue
        examples = [build_example(args.num_docs, args.num_pairs, args.tolerance,
                                  args.ans_min, args.ans_max, rng,
                                  term_min=args.term_min, term_max=args.term_max,
                                  n_ops_min=args.n_ops_min, n_ops_max=args.n_ops_max,
                                  numbers_only=args.numbers_only)
                    for _ in tqdm(range(n), desc=f"Building {split}")]
        path = f"{args.output_dir}/{tag}_{split}.jsonl"
        save_jsonl(path, examples)
        print_dataset_stats(examples, label=f"{split} ({tag})", path=path)


if __name__ == "__main__":
    main()
