"""Subsample a CTC-suite unified-JSONL training pool to a target size and a token-length ceiling.

The banked CTC-suite runs train on a joint 2k-32k ladder at ``--seq-len 40960``. A run that only
needs the 2k/4k/8k/16k rungs should not pay for that: with the default (non-packing)
``PadToLengthInstanceSource`` every instance is padded to ``--seq-len``, so a 40960 seq-len makes
short examples ~4x more expensive than they need to be, and the trainer hard-fails anyway if any
example exceeds ``--seq-len``.

This script therefore does two things before tokenization:

* **Length ceiling.** Drops examples whose estimated token length exceeds ``--max-est-tokens``, so
  the resulting shard's ``max_example_len`` fits a modest ``--seq-len``. The estimate is
  characters / ``--chars-per-token`` over the rendered context (documents + queries + answers),
  which is tokenizer-agnostic and deliberately conservative -- the real ``max_example_len`` written
  by the converter is the number that gates training, and it is checked there.
* **Uniform subsample.** Samples ``--target`` examples uniformly at random (seeded), preserving the
  pool's joint distribution over corpus size ``n`` within the retained range.

It prints the retained/dropped counts and the ``n`` distribution so the shard is auditable.
"""

import argparse
import json
import random


def est_tokens(ex: dict, chars_per_token: float) -> int:
    """Estimate an example's token length from its rendered character count.

    :param ex: A unified-format example (``documents`` / ``queries`` / ``answers``).
    :param chars_per_token: Characters per token for the target tokenizer (~4.0 for BPE English).

    :returns: The estimated token count.
    """
    chars = 0
    for d in ex.get("documents", []):
        chars += len(d["text"] if isinstance(d, dict) else str(d))
    for k in ("queries", "answers"):
        for q in ex.get(k, []) or []:
            chars += len(q if isinstance(q, str) else json.dumps(q))
    return int(chars / chars_per_token)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--target", type=int, default=10000, help="examples to keep (upper bound)")
    ap.add_argument(
        "--max-est-tokens",
        type=int,
        default=18000,
        help="drop examples estimated longer than this (keeps max_example_len under --seq-len)",
    )
    ap.add_argument("--chars-per-token", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    kept, dropped, ns = [], 0, []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if est_tokens(ex, args.chars_per_token) > args.max_est_tokens:
                dropped += 1
                continue
            kept.append(line)
            ns.append(len(ex.get("documents", [])))

    print(f"pool={args.input}\n  eligible={len(kept)} dropped_too_long={dropped}")
    if ns:
        ns_sorted = sorted(ns)
        q = lambda p: ns_sorted[min(len(ns_sorted) - 1, int(p * len(ns_sorted)))]  # noqa: E731
        print(f"  n(docs) over eligible: min={ns_sorted[0]} p25={q(.25)} p50={q(.5)} "
              f"p75={q(.75)} max={ns_sorted[-1]}")

    rng = random.Random(args.seed)
    if len(kept) > args.target:
        kept = rng.sample(kept, args.target)
    else:
        print(f"  WARNING: pool has only {len(kept)} eligible examples < target {args.target}; "
              "keeping all of them (report the ACTUAL count, never the target)")
    rng.shuffle(kept)
    with open(args.output, "w") as f:
        for line in kept:
            f.write(line + "\n")
    print(f"wrote {len(kept)} examples -> {args.output}")


if __name__ == "__main__":
    main()
