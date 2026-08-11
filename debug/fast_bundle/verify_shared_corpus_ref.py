#!/usr/bin/env python3
"""
Verify a shared-corpus eval file. EVAL ONLY.

Hard-fails on anything that would make either half of the claim false — the sharing claim (a
cache-reusing runner would produce wrong KV) or the fidelity claim (the task changed meaning).

Checks:

1. **Prefix invariant** — every row carrying a given ``corpus_id`` has a byte-identical
   ``documents[:shared_prefix_len]``, and ``shared_prefix_sha1`` matches it.
2. **Gold reachability** — every ``gold_doc_indices`` entry is in range. A silent out-of-range gold
   scores 0 at ``parse_rate`` 1.0, which reads as a modeling result.
3. **Gold placement** — reports how many gold documents fall inside the shared prefix vs. the
   per-query tail. For contradiction this asserts the user directive: exactly one member of each
   pair in the prefix, its partner in the tail.
4. **Recency-shortcut control** — the analytic score of "answer with random documents drawn from
   the tail". Quoted next to every Family-B result so a shared score can be read against the
   shortcut floor, not just against the baseline.
5. **No duplicate documents** inside a corpus (a duplicate changes outlier topic counts and gives
   contradiction a spurious pair).

Usage::

    python -m corpus_reasoning.data.verify_shared_corpus <file.jsonl> [--task contradiction]
"""
import argparse
import collections
import hashlib
import json
import sys


def doc_key(d):
    return json.dumps([d.get("title"), d.get("text")], sort_keys=True)


def prefix_sha1(docs):
    h = hashlib.sha1()
    for d in docs:
        h.update(doc_key(d).encode())
        h.update(b"\x00")
    return h.hexdigest()


def flat_gold(row, index_base):
    g = row.get("gold_doc_indices") or []
    if g and isinstance(g[0], list):
        return [i - index_base for pair in g for i in pair]
    return [i - index_base for i in g]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--task", default=None)
    p.add_argument(
        "--index-base",
        type=int,
        default=None,
        help="contradiction gold is 1-INDEXED; every other task in this suite is "
        "0-indexed (see [[gold-doc-indices-per-task-base]]). Inferred from --task "
        "when not given.",
    )
    args = p.parse_args()

    index_base = args.index_base
    if index_base is None:
        index_base = 1 if args.task == "contradiction" else 0

    rows = [json.loads(l) for l in open(args.path) if l.strip()]
    if not rows:
        sys.exit(f"FAIL: {args.path} is empty")

    errors = []
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if "corpus_id" not in r:
            errors.append(f"row {i}: missing corpus_id")
            continue
        groups[r["corpus_id"]].append((i, r))

    # 1. prefix invariant
    for cid, members in groups.items():
        shas = set()
        for i, r in members:
            n = r["shared_prefix_len"]
            actual = prefix_sha1(r["documents"][:n])
            if actual != r["shared_prefix_sha1"]:
                errors.append(f"row {i} ({cid}): shared_prefix_sha1 does not match its own prefix")
            shas.add((n, actual))
        if len(shas) > 1:
            errors.append(
                f"corpus {cid}: {len(shas)} distinct prefixes across {len(members)} rows "
                f"-> KV reuse would be WRONG"
            )

    # 2/3. gold reachability + placement, 5. duplicates
    in_prefix = in_tail = 0
    pair_split_ok = pair_total = 0
    tail_sizes, dup_rows = [], 0
    for i, r in enumerate(rows):
        nd, n = len(r["documents"]), r["shared_prefix_len"]
        tail_sizes.append(nd - n)
        if len({doc_key(d) for d in r["documents"]}) != nd:
            dup_rows += 1
        for gi in flat_gold(r, index_base):
            if not (0 <= gi < nd):
                errors.append(f"row {i}: gold index {gi} out of range (ndocs={nd})")
                continue
            if gi < n:
                in_prefix += 1
            else:
                in_tail += 1
        g = r.get("gold_doc_indices") or []
        if g and isinstance(g[0], list):
            for pair in g:
                pair_total += 1
                sides = [(i0 - index_base) < n for i0 in pair]
                if sorted(sides) == [False, True]:
                    pair_split_ok += 1

    if dup_rows:
        errors.append(f"{dup_rows}/{len(rows)} rows contain duplicate documents inside one corpus")

    total_gold = in_prefix + in_tail
    print(f"file            : {args.path}")
    print(
        f"rows            : {len(rows)}  (eval_size={len(rows)}"
        f"{'  ⚠ <500, quote its SE inline' if len(rows) < 500 else ''})"
    )
    print(
        f"corpora         : {len(groups)}  queries/corpus="
        f"{sorted({len(v) for v in groups.values()})}"
    )
    print(
        f"ndocs           : {len(rows[0]['documents'])}  shared_prefix_len="
        f"{rows[0]['shared_prefix_len']}  tail={sorted(set(tail_sizes))}"
    )
    shared_frac = rows[0]["shared_prefix_len"] / len(rows[0]["documents"])
    print(
        f"shared doc frac : {shared_frac:.3f}   (~{1/max(1e-9, 1-shared_frac):.1f}x prefill "
        f"saving if the tail dominates cost)"
        if shared_frac < 1
        else "shared doc frac : 1.000   (fully shared corpus)"
    )
    if total_gold:
        print(
            f"gold placement  : {in_prefix}/{total_gold} in shared prefix, "
            f"{in_tail}/{total_gold} in per-query tail"
        )
    if pair_total:
        print(
            f"pair split      : {pair_split_ok}/{pair_total} pairs have exactly one member in "
            f"the prefix and its partner in the tail"
        )
        if pair_split_ok != pair_total:
            errors.append(
                f"pair split violated on {pair_total - pair_split_ok} pairs "
                f"-> recency could hand the model a WHOLE answer, not half"
            )

    # 4. recency-shortcut control
    if in_tail and tail_sizes:
        avg_tail = sum(tail_sizes) / len(tail_sizes)
        k = in_tail / len(rows)  # gold docs per query sitting in the tail
        # "answer with k random tail docs": precision = recall = k/avg_tail -> f1 equals it too.
        print(
            f"guess-tail ctrl : f1≈{k/avg_tail:.3f}  (score of blindly answering with {k:.1f} "
            f"random docs from the {avg_tail:.0f}-doc tail)"
        )

    if errors:
        print("\nFAIL:")
        for e in errors[:40]:
            print("  -", e)
        if len(errors) > 40:
            print(f"  ... and {len(errors)-40} more")
        sys.exit(1)
    print("\nOK: all invariants hold")


if __name__ == "__main__":
    main()
