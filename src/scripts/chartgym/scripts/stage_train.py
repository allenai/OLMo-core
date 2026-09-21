#!/usr/bin/env python3
"""Raw ChartGym shards -> the training dataset `FineVisionDatasetConfig` already reads.

Schema is `texts: list<struct<user, assistant>>` + `images: list<image>`, so **no new
OLMo-core loader module is needed**: `FineVisionDatasetConfig(dataset_path=...)` loads it
as-is. `FineVisionDataset._build` emits a *flat* turn list, which `encode_sft_example`
splits into independent branches sharing ONE image prefix -- so ~16 questions per chart
cost one image encode, and `root_subsegments_root_tokens` weights each branch 1/sqrt(16)
rather than letting a 16-question figure carry 16x the gradient.

Two things this asserts, because both fail silently otherwise:

* **Row length.** `FineVisionDatasetConfig` has no `skip_overlong`; an over-budget row is
  right-truncated by `truncate_example`, which drops the LAST branches without complaint.
* **Supervised-token count.** `response_logits_only=True` materializes logits only where
  `loss_mask > 0`, and the CoT arm died on a 15.65 GiB allocation at ~27k supervised
  positions x 151,936 vocab x 4 bytes. Short answers keep this far below that, but a
  runaway `list` answer would not.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from datasets import Dataset, Features, Image, Value

MAX_CHARS_PER_ROW = 6000       # ~1.5k tokens of text against a 16,384 budget
# Bare answers stay short; procedure traces (enumeration / search) legitimately run to a
# few hundred characters -- a 14-tick axis enumerated twice is ~350. The cap exists to
# catch runaway generation bugs, not to trim traces.
MAX_TARGET_CHARS = 600


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--num-shards", type=int, default=60)
    args = ap.parse_args()

    shards = sorted(args.raw.glob("shard-*")) or [args.raw]
    by_figure: dict[str, dict] = {}
    dropped = collections.Counter()
    for shard in shards:
        qa = shard / "qa.jsonl"
        if not qa.exists():
            continue
        for line in qa.read_text().splitlines():
            if not line.strip():
                continue
            q = json.loads(line)
            if q["held_out"]:
                dropped["held_out_primitive"] += 1
                continue
            target = q.get("target") or q["answer"]
            if len(target) > MAX_TARGET_CHARS:
                dropped["target_too_long"] += 1
                continue
            img = shard / "figures" / f"{q['figure_id']}.png"
            if not img.exists():
                dropped["missing_image"] += 1
                continue
            row = by_figure.setdefault(q["figure_id"], {
                "images": [str(img)], "texts": [], "figure_id": q["figure_id"],
                "difficulty": q["difficulty"], "families": [], "capabilities": [],
            })
            # Training supervises the procedure (`target`); the eval scores the bare answer.
            row["texts"].append({"user": q["question"], "assistant": target})
            row["families"].append(q["family"])
            row["capabilities"].append(q["capability"])

    rows = []
    for row in by_figure.values():
        chars = sum(len(t["user"]) + len(t["assistant"]) for t in row["texts"])
        if chars > MAX_CHARS_PER_ROW:
            # Trim from the end -- which is exactly what truncate_example would silently
            # do downstream, except here it is counted.
            while row["texts"] and chars > MAX_CHARS_PER_ROW:
                t = row["texts"].pop()
                row["families"].pop()
                row["capabilities"].pop()
                chars -= len(t["user"]) + len(t["assistant"])
                dropped["row_over_budget"] += 1
        if row["texts"]:
            rows.append(row)

    # NB: list literals, not `Sequence(...)`. `Sequence` of a dict feature transposes to a
    # dict-of-lists ({"user": [...], "assistant": [...]}), but FineVisionDataset._build
    # iterates `for turn in row[texts_column]` and calls `turn.get("user")` -- it needs a
    # list of structs. The transposed form raises "'list' object has no attribute 'get'"
    # at staging time if you are lucky, and silently yields zero usable turns if you are not.
    feats = Features({
        "images": [Image(decode=False)],
        "texts": [{"user": Value("string"), "assistant": Value("string")}],
        "figure_id": Value("string"), "difficulty": Value("string"),
        "families": [Value("string")], "capabilities": [Value("string")],
    })
    ds = Dataset.from_list(rows, features=feats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(args.out), num_shards=min(args.num_shards, max(1, len(rows) // 100)))

    n_turns = sum(len(r["texts"]) for r in rows)
    fam = collections.Counter(f for r in rows for f in r["families"])
    print(f"wrote {len(rows)} rows ({n_turns} questions, {n_turns / max(len(rows),1):.1f}/row) "
          f"to {args.out}")
    print(f"  dropped: {dict(dropped)}")
    print(f"  families: {len(fam)}")
    probe = sorted(f for f in fam if f.startswith("charxiv."))
    if probe:
        # A probe corpus deliberately trains CharXiv's own templates, including the
        # panel-layout ones the pnl.* assertion guards. Say so loudly: the only thing worse
        # than a benchmark-fitted corpus is one nobody realises is benchmark-fitted.
        print("\n  " + "!" * 68)
        print("  !! CEILING-PROBE CORPUS -- benchmark-fitted, NOT SHIPPABLE")
        print(f"  !! {len(probe)} CharXiv template families: {', '.join(probe)}")
        print("  !! Trains templates 18/19, so the panel-layout transfer holdout is VOID.")
        print("  !! Any score from this corpus is an upper bound on score, not on skill,")
        print("  !! and is not comparable to published CharXiv numbers.")
        print("  " + "!" * 68 + "\n")
    else:
        assert not any(f.startswith("pnl.") for f in fam), \
            "HELD-OUT PRIMITIVE LEAKED INTO TRAINING -- panel-layout families must never train"
        print("  held-out primitive: absent (asserted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
