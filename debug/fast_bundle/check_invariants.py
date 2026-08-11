"""Check a built fast rung against the invariants its speedup and its fairness both rest on.

A fast file is only cheaper if the shared prefix really is byte-identical across a corpus group,
and only *gradeable* if the gold structure survived the rebuild. Both are properties of the file,
so they are checkable without a GPU -- and worth checking, because a broken one of either produces
a plausible score rather than an error.

Checked, per corpus group:

1. ``documents[:shared_prefix_len]`` is byte-identical across every row in the group, and matches
   the recorded ``shared_prefix_sha1``. This is what the KV cache reuses.
2. contradiction: every gold pair straddles the boundary -- one member in the shared prefix, its
   partner in the per-query tail. If both landed in the tail, recency hands over a whole answer and
   an O(N^2) all-pairs search collapses to O(tail^2).
3. outlier: every gold sits in the tail (structural for that task), and no gold text appears in the
   shared prefix -- a trio member in the prefix silently stops being an outlier.
4. Row count and question set are unchanged from the source, so a fast rung grades the same
   questions as its reliable twin.

    python debug/fast_bundle/check_invariants.py <built.jsonl> [--source <independent.jsonl>]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from typing import Dict, List


def load(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def doc_key(d: dict) -> str:
    return json.dumps([d.get("title"), d.get("text")], sort_keys=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("built")
    ap.add_argument("--source", default=None, help="the independent rung it was built from")
    ap.add_argument("--task", default="", help="contradiction / outlier; default: infer from path")
    args = ap.parse_args()

    task = args.task or (
        "contradiction"
        if "contradiction" in args.built
        else "outlier" if "outlier" in args.built else ""
    )
    rows = load(args.built)
    failures: List[str] = []

    groups: Dict[str, List[dict]] = collections.OrderedDict()
    for r in rows:
        groups.setdefault(r["corpus_id"], []).append(r)

    # 1. the shared prefix is actually shared
    prefix_lens = set()
    for cid, grp in groups.items():
        plen = grp[0]["shared_prefix_len"]
        prefix_lens.add(plen)
        want = [doc_key(d) for d in grp[0]["documents"][:plen]]
        for r in grp[1:]:
            if r["shared_prefix_len"] != plen:
                failures.append(f"{cid}: shared_prefix_len differs inside the group")
                break
            got = [doc_key(d) for d in r["documents"][:plen]]
            if got != want:
                first = next(i for i, (a, b) in enumerate(zip(got, want)) if a != b)
                failures.append(f"{cid}: prefix diverges at document {first}")
                break
        shas = {r["shared_prefix_sha1"] for r in grp}
        if len(shas) != 1:
            failures.append(f"{cid}: {len(shas)} distinct shared_prefix_sha1 in one group")

    # 2/3. gold structure
    straddling = both_in_tail = 0
    for r in rows:
        plen = r["shared_prefix_len"]
        if task == "contradiction":
            for pair in r["gold_doc_indices"]:
                a, b = (i - 1 for i in pair)  # this task is 1-indexed on disk
                in_prefix = sum(1 for i in (a, b) if i < plen)
                if in_prefix == 1:
                    straddling += 1
                else:
                    both_in_tail += 1
                    if both_in_tail <= 3:
                        failures.append(
                            f"gold pair {pair} does not straddle the prefix boundary ({plen})"
                        )
        elif task == "outlier":
            prefix_keys = {doc_key(d) for d in r["documents"][:plen]}
            for i in r["gold_doc_indices"]:
                if i < plen:
                    failures.append(f"outlier gold at {i} sits inside the shared prefix")
                if doc_key(r["documents"][i]) in prefix_keys:
                    failures.append("an outlier trio member also appears in the shared prefix")

    # 4. same questions as the independent rung
    if args.source:
        src = load(args.source)
        if len(src) != len(rows):
            failures.append(f"row count changed: {len(src)} -> {len(rows)}")
        else:
            sq = [json.dumps(r.get("queries"), sort_keys=True) for r in src]
            bq = [json.dumps(r.get("queries"), sort_keys=True) for r in rows]
            if collections.Counter(sq) != collections.Counter(bq):
                failures.append("the question set changed -- not comparable to the reliable rung")

    ndocs = len(rows[0]["documents"])
    plen = sorted(prefix_lens)[0]
    print(
        f"rows={len(rows)}  corpora={len(groups)}  ndocs={ndocs}  "
        f"shared_prefix_len={sorted(prefix_lens)}  shared={plen / ndocs:.1%}"
    )
    print(f"queries/corpus={sorted({len(g) for g in groups.values()})}")
    if task == "contradiction":
        print(f"gold pairs straddling the boundary: {straddling}  both-in-tail: {both_in_tail}")

    if failures:
        print(f"\nFAIL ({len(failures)} problems)")
        for f in failures[:10]:
            print("  " + f)
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
