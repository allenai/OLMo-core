"""
Per-row / per-token breakdown of an eval_side_slot_probe JSON that carries ``per_row`` dumps
(probe commit f71d2a9f5+). Answers WHERE a construction loses to full attention:

* contradiction: answer-token CE split by token role -- first claim id of a pair, second claim id
  (the partner), structure ([[ ], ]] <|im_end|>) -- so "can't find the partner" and "wrong claim
  set" separate.
* oolong: per-row CE next to the row's question type (from the shard's head.jsonl).
Also prints the paired standard error of (config - full) over rows, so a delta can be called real.

    python debug/pooled_kv/analyze_slot_rows.py /net/sneetches/data/prasann/slot_probe/v2_contradiction_32768.json [--shard DIR] [--configs REGEX]
"""

import argparse
import json
import re

import numpy as np

DIGITS = set(range(15, 25))  # Qwen3.5: "0".."9" are ids 15..24
COMMA, SPACE = 11, 220


def token_roles(ids):
    """Role per answer token of a '[[a, b], [c, d]]<|im_end|>\\n' answer."""
    roles, seen_comma = [], False
    for t in ids:
        if t in DIGITS:
            roles.append("2nd id" if seen_comma else "1st id")
        elif t == COMMA:
            seen_comma = True; roles.append("struct")
        elif t == SPACE:
            roles.append("struct")
        else:
            seen_comma = False; roles.append("struct")
    return roles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--shard", default=None, help="shard dir with head.jsonl (question types for oolong)")
    ap.add_argument("--configs", default=".", help="regex filter on config names")
    a = ap.parse_args()
    d = json.load(open(a.json))
    task, per = d["task"], d.get("per_row")
    if not per:
        raise SystemExit("no per_row dump in this JSON (older probe)")
    names = [n for n in per if re.search(a.configs, n)]
    full = np.array(per["full"]["ce"])
    print(f"{task} rung {d['rung']} rows {d['rows']}  full CE {full.mean():.3f}")
    if task == "contradiction":
        roles_by_row = [token_roles(ids) for ids in per["full"]["tok_ids"]]
        role_names = ["1st id", "2nd id", "struct"]
        print(f"{'config':52} {'CE':>6} {'Δ':>7} {'±SE':>6} | " + " ".join(f"{r:>8}" for r in role_names) + "   (mean CE by token role; full in first line)")
        for n in ["full"] + [x for x in names if x != "full"]:
            ce = np.array(per[n]["ce"]); diff = ce - full
            by = {r: [] for r in role_names}
            for row_ce, row_roles in zip(per[n]["tok_ce"], roles_by_row):
                for c, r in zip(row_ce, row_roles):
                    by[r].append(c)
            se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else 0.0
            print(f"{n:52} {ce.mean():6.3f} {diff.mean():+7.3f} {se:6.3f} | " + " ".join(f"{np.mean(by[r]):8.3f}" for r in role_names))
    else:
        qtypes = None
        if a.shard:
            qtypes = []
            for line in open(f"{a.shard}/head.jsonl"):
                ex = json.loads(line)
                m = ex.get("_meta", {})
                qtypes.append(f"{m.get('task_group', '?')}: {ex['queries'][0][:60] if ex.get('queries') else ''} -> {ex.get('answers')}")
        print(f"{'config':52} {'CE':>6} {'Δ':>7} {'±SE':>6}")
        for n in ["full"] + [x for x in names if x != "full"]:
            ce = np.array(per[n]["ce"]); diff = ce - full
            se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else 0.0
            print(f"{n:52} {ce.mean():6.3f} {diff.mean():+7.3f} {se:6.3f}")
        print("\nper-row CE (rows x configs):")
        cols = ["full"] + [x for x in names if x != "full"]
        print(f"{'row':>3} " + " ".join(f"{i:>6}" for i in range(len(cols))) + "  question")
        for i, _ in enumerate(full):
            print(f"{i:3d} " + " ".join(f"{per[c]['ce'][i]:6.2f}" for c in cols) + "  " + (qtypes[i] if qtypes else ""))
        for j, c in enumerate(cols):
            print(f"  [{j}] {c}")


if __name__ == "__main__":
    main()
