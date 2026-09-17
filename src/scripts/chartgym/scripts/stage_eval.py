#!/usr/bin/env python3
"""Raw ChartGym shards -> a flat one-row-per-question HF dataset for olmo-eval.

Training rows pack ~16 questions per image (one image encode amortized over many branches);
the eval is flat instead, because the harness scores one Instance per question and the
per-family breakdown is the whole point of the instrument.

Usage::

    python scripts/stage_eval.py --raw raw/eval-v1 --out $MOLMO_EXPERIMENT_DATA_DIR/chartgym/eval-v1 --n 500
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from datasets import Dataset, Features, Image, Value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=None, help="cap, balanced across families")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    shards = sorted(args.raw.glob("shard-*")) or [args.raw]
    rows = []
    for shard in shards:
        qa = shard / "qa.jsonl"
        if not qa.exists():
            continue
        for line in qa.read_text().splitlines():
            if not line.strip():
                continue
            q = json.loads(line)
            img = shard / "figures" / f"{q['figure_id']}.png"
            if not img.exists():
                continue
            rows.append({
                "image": str(img), "question": q["question"], "answer": q["answer"],
                "answer_type": q["answer_type"],
                "tol": float(q["tol"]) if q.get("tol") is not None else -1.0,
                "family": q["family"], "capability": q["capability"],
                "held_out": bool(q["held_out"]), "is_na": bool(q["is_na"]),
                "difficulty": q["difficulty"], "figure_id": q["figure_id"],
                "question_id": q["question_id"],
            })

    if args.n and len(rows) > args.n:
        # Round-robin over families so a prolific family cannot crowd out a rare one --
        # per-family n is what the readout needs, not overall n.
        import random

        rnd = random.Random(args.seed)
        by_family = collections.defaultdict(list)
        for r in rows:
            by_family[r["family"]].append(r)
        for v in by_family.values():
            rnd.shuffle(v)
        picked, families = [], sorted(by_family)
        while len(picked) < args.n and any(by_family[f] for f in families):
            for f in families:
                if by_family[f] and len(picked) < args.n:
                    picked.append(by_family[f].pop())
        rows = picked

    feats = Features({
        "image": Image(decode=False), "question": Value("string"),
        "answer": Value("string"), "answer_type": Value("string"),
        "tol": Value("float32"), "family": Value("string"),
        "capability": Value("string"), "held_out": Value("bool"),
        "is_na": Value("bool"), "difficulty": Value("string"),
        "figure_id": Value("string"), "question_id": Value("string"),
    })
    ds = Dataset.from_list(rows, features=feats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(args.out))

    fam = collections.Counter(r["family"] for r in rows)
    cap = collections.Counter(r["capability"] for r in rows)
    print(f"wrote {len(rows)} rows to {args.out}")
    print(f"  held_out: {sum(r['held_out'] for r in rows)}   NA: {sum(r['is_na'] for r in rows)}")
    print(f"  capabilities: {dict(cap)}")
    print(f"  families: {len(fam)} -> {dict(fam.most_common())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
