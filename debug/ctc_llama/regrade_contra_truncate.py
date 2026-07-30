"""Re-grade a contradiction generations dump, truncating each prediction at the FIRST ``]]``.

Why: with ``--cot-mode none`` the target is exactly one JSON pair list, but a model that has not
learned to emit EOS keeps going after closing the list. The evaluator then parses every bracket
pair it can find -- 15-20 of them instead of 3 -- and precision collapses. This is the same class of
harness artifact recorded in ``eval-lc-native-nocot-fullpath-bug`` (MODE=full contradiction read as
f1 0.58 when the real number was 0.947), and it deflates BOTH arms, so the raw ladder is a lower
bound rather than a comparison.

This script does not re-run the model. It re-parses the SAME generations with the first-``]]``
truncation the record prescribes, and reports the harness's own pooled (micro) precision/recall/f1
next to the corrected ones so the size of the artifact is visible.

Usage::

    python debug/ctc_llama/regrade_contra_truncate.py results/ctc_suite_llama/contradiction
"""

import ast
import glob
import json
import os
import re
import sys
from typing import List, Set, Tuple

PAIR_RE = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]")


def parse_pairs(text: str) -> Set[Tuple[int, int]]:
    """Parse ``[[a, b], ...]`` into a set of UNORDERED index pairs.

    :param text: The raw model output (or a truncation of it).

    :returns: Set of ``(lo, hi)`` tuples.
    """
    out = set()
    for a, b in PAIR_RE.findall(text):
        i, j = int(a), int(b)
        if i != j:
            out.add((min(i, j), max(i, j)))
    return out


def truncate_at_first_close(text: str) -> str:
    """Cut the generation at the end of the first ``]]`` -- the no-cot answer's real terminator."""
    k = text.find("]]")
    return text[: k + 2] if k != -1 else text


def gold_pairs(rec: dict) -> Set[Tuple[int, int]]:
    """Read the gold pair set out of a per-example record (stored as a stringified list)."""
    g = rec.get("gold_pairs")
    if isinstance(g, str):
        try:
            g = ast.literal_eval(g)
        except Exception:
            return parse_pairs(g)
    return {(min(int(a), int(b)), max(int(a), int(b))) for a, b in (g or [])}


def micro_f1(preds: List[Set], golds: List[Set]) -> dict:
    """Pooled (micro) precision/recall/f1 -- the aggregation the evaluator itself reports."""
    tp = sum(len(p & g) for p, g in zip(preds, golds))
    np_ = sum(len(p) for p in preds)
    ng = sum(len(g) for g in golds)
    p = tp / np_ if np_ else 0.0
    r = tp / ng if ng else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f, "pred_pairs": np_, "gold_pairs": ng}


def main() -> None:
    root = sys.argv[1]
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "*", "rung_*.generations.json"))):
        arm = os.path.basename(os.path.dirname(path)).split("_", 1)[1]
        rung = int(re.search(r"rung_(\d+)", path).group(1))
        recs = json.load(open(path))
        raw_p, trunc_p, golds = [], [], []
        n_rambled = 0
        for rec in recs:
            gen = rec.get("prediction", "")
            golds.append(gold_pairs(rec))
            rp = parse_pairs(gen)
            tp_ = parse_pairs(truncate_at_first_close(gen))
            raw_p.append(rp)
            trunc_p.append(tp_)
            if len(rp) > len(tp_):
                n_rambled += 1
        rows.append(
            {
                "arm": arm,
                "rung": rung,
                "eval_size": len(recs),
                "rambled_frac": round(n_rambled / max(1, len(recs)), 3),
                "raw": micro_f1(raw_p, golds),
                "truncated": micro_f1(trunc_p, golds),
            }
        )
    rows.sort(key=lambda r: (r["arm"], r["rung"]))
    print(f"{'arm':14s} {'rung':>6s} {'size':>5s} {'ramble':>7s} {'raw_f1':>8s} {'trunc_f1':>9s} {'trunc_P':>8s} {'trunc_R':>8s}")
    for r in rows:
        print(
            f"{r['arm']:14s} {r['rung']:6d} {r['eval_size']:5d} {r['rambled_frac']:7.3f} "
            f"{r['raw']['f1']:8.4f} {r['truncated']['f1']:9.4f} "
            f"{r['truncated']['precision']:8.4f} {r['truncated']['recall']:8.4f}"
        )
    out = os.path.join(root, "regrade_truncated.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
