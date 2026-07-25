"""
Verify the built longmino-512k dataset on weka.

Checks, in order of what would actually bite:

1. **Byte integrity** -- every ``part-*.npy`` is a whole number of ``uint32`` tokens, and the bytes
   on disk agree with the token counts the tokenizer job recorded in ``token_counts.json``.
2. **Cross-tokenizer identity** -- the ``qwen3`` and ``qwen35`` trees cover the same documents.
   They read the same staged text and share the routing/filter rules, so a per-stratum document
   count mismatch means a filter or routing bug.
3. **Mix sizing** -- the realized composition, from the measured counts.

Run (via gantry, from the repo root)::

    gantry run --workspace ai2/flex2 --budget ai2/oe-other \\
        --cluster ai2/jupiter-cirrascale-2 \\
        --weka oe-training-default:/weka/oe-training-default \\
        --cpus 4 --gpus 0 --priority urgent --yes \\
        -- python src/scripts/data/verify_longmino_512k.py
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from longmino_512k_common import WEKA_ROOT  # noqa: E402

BYTES_PER_TOKEN = 4  # uint32


def check_tree(root: str, tree: str) -> dict:
    """Sum on-disk bytes per stratum and compare against the recorded token counts."""
    base = os.path.join(root, tree)
    with open(os.path.join(base, "token_counts.json")) as f:
        recorded = json.load(f)["strata"]

    print(f"\n=== {tree}: byte integrity ===")
    ok = True
    for stratum in sorted(recorded):
        parts = sorted(glob.glob(os.path.join(base, stratum, "part-*.npy")))
        total_bytes = sum(os.path.getsize(p) for p in parts)
        ragged = [p for p in parts if os.path.getsize(p) % BYTES_PER_TOKEN]
        from_bytes = total_bytes // BYTES_PER_TOKEN
        claimed = recorded[stratum]["tokens"]
        match = from_bytes == claimed
        ok &= match and not ragged
        flag = "" if match else f"   <-- MISMATCH (recorded {claimed:,})"
        if ragged:
            flag += f"   <-- {len(ragged)} file(s) not a whole number of tokens"
        print(f"  {stratum:42s} {len(parts):4d} parts {from_bytes:15,d} tok{flag}")
    print(f"  byte integrity: {'OK' if ok else 'FAILED'}")
    return recorded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=WEKA_ROOT)
    parser.add_argument("--trees", nargs="+", default=["qwen3", "qwen35"])
    args = parser.parse_args()

    recorded = {t: check_tree(args.root, t) for t in args.trees}

    if len(args.trees) == 2:
        a, b = args.trees
        print(f"\n=== cross-tokenizer identity: {a} vs {b} ===")
        strata = sorted(set(recorded[a]) | set(recorded[b]))
        mismatches = []
        for s in strata:
            da = recorded[a].get(s, {}).get("docs")
            db = recorded[b].get(s, {}).get("docs")
            if da != db:
                mismatches.append((s, da, db))
        if mismatches:
            for s, da, db in mismatches:
                print(f"  {s:42s} {da} vs {db}   <-- MISMATCH")
            print("  identity: FAILED")
        else:
            total = sum(v["docs"] for v in recorded[a].values())
            print(f"  all {len(strata)} strata agree; {total:,} documents in both trees")
            print("  identity: OK")

    for t in args.trees:
        ta = sum(v["tokens"] for v in recorded[t].values())
        print(f"\n{t}: {ta:,} tokens, {sum(v['docs'] for v in recorded[t].values()):,} docs")


if __name__ == "__main__":
    main()
