#!/usr/bin/env python3
"""Generate a ChartGym shard: PNGs + per-figure specs + questions.

One OS process per shard. matplotlib accumulates state across thousands of figures even
with `plt.close`, and a crash then costs one shard instead of the run.

Figures that fail `RenderAudit.validate` are discarded, not repaired: a spec/render
disagreement means we do not know which one the image shows, and a confidently wrong answer
is worse than a missing one. Rejection reasons are counted and written out, because a family
rejecting most of its candidates is a generator bug rather than a guard working.

Usage::

    python scripts/generate.py --out raw/train-v1/shard-00000 --n 1000 --seed-base 1000000
    python scripts/generate.py --out raw/eval-v1/shard-00000 --n 200 --eval-split
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from chartgym.families import emit_all  # noqa: E402
from chartgym.families_charxiv_probe import emit_probe  # noqa: E402
from chartgym.render import render  # noqa: E402
from chartgym.sample import sample_figure  # noqa: E402

DIFFICULTY_MIX = (("easy", 0.30), ("medium", 0.45), ("hard", 0.25))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--max-questions", type=int, default=16)
    ap.add_argument("--eval-split", action="store_true",
                    help="eval-only style sheets, and emit the held-out families")
    ap.add_argument("--charxiv-probe", action="store_true",
                    help="CEILING PROBE: emit CharXiv's own templates verbatim instead of "
                         "the capability families. Produces a corpus that is deliberately "
                         "benchmark-fitted and must never be shipped -- see "
                         "chartgym/families_charxiv_probe.py.")
    args = ap.parse_args()

    figures_dir = args.out / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    levels = [d for d, _ in DIFFICULTY_MIX]
    probs = [p for _, p in DIFFICULTY_MIX]

    rejects = collections.Counter()
    n_ok = n_q = n_na = 0
    with open(args.out / "specs.jsonl", "w") as fspec, open(args.out / "qa.jsonl", "w") as fqa:
        i = attempts = 0
        while i < args.n and attempts < args.n * 4:
            attempts += 1
            seed = args.seed_base + attempts
            rng = np.random.default_rng(seed)
            difficulty = str(rng.choice(levels, p=probs))
            fid = f"cg-{'eval' if args.eval_split else 'train'}-{seed:09d}"
            try:
                spec = sample_figure(fid, seed, difficulty, eval_split=args.eval_split)
                png, audit = render(spec)
            except Exception as exc:  # noqa: BLE001
                rejects[f"render:{type(exc).__name__}"] += 1
                continue
            if not audit.validate(spec):
                rejects[audit.problems[0].split(":", 1)[1].strip()[:44]] += 1
                continue
            if args.charxiv_probe:
                qs = emit_probe(spec, audit, np.random.default_rng(seed ^ 0x5EED))
            else:
                qs = emit_all(spec, audit, np.random.default_rng(seed ^ 0x5EED),
                              include_held_out=args.eval_split)
            if not qs:
                rejects["no questions emitted"] += 1
                continue
            # Cap per figure, at most 2 per family, keeping capability balance.
            sel, per_family = [], collections.Counter()
            for idx in rng.permutation(len(qs)):
                q = qs[int(idx)]
                if per_family[q["family"]] >= 2:
                    continue
                sel.append(q)
                per_family[q["family"]] += 1
                if len(sel) >= args.max_questions:
                    break
            (figures_dir / f"{fid}.png").write_bytes(png)
            fspec.write(json.dumps({
                "figure_id": fid, "seed": seed, "difficulty": difficulty,
                "grid_shape": list(spec.grid_shape), "n_panels": spec.n_panels,
                "total_labeled_ticks": spec.total_labeled_ticks(),
                "style_sheet": spec.style.style_sheet, "mpl_version": spec.mpl_version,
            }) + "\n")
            for k, q in enumerate(sel):
                q = dict(q, figure_id=fid, difficulty=difficulty,
                         question_id=f"{fid}-{k:02d}")
                fqa.write(json.dumps(q) + "\n")
                n_q += 1
                n_na += bool(q["is_na"])
            n_ok += 1
            i += 1

    (args.out / "reject_stats.json").write_text(json.dumps(dict(rejects), indent=1))
    print(f"{args.out.name}: {n_ok} figures, {n_q} questions "
          f"({n_q / max(n_ok, 1):.1f}/figure), NA share {100 * n_na / max(n_q, 1):.1f}%, "
          f"{sum(rejects.values())} rejected")
    for r, c in rejects.most_common(5):
        print(f"    reject {r}: {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
