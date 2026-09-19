"""
Per-task ``ctc-data build`` outputs -> ONE tagged, budgeted, shuffled SFT mix.

``ctc-data build --task X --rungs ... --out ROOT`` writes ``ROOT/X/train.jsonl`` per task, with no
task tag on the rows (a single-task file does not need one). Training on several tasks needs the
opposite: one shuffled file whose every row says which task it is, so the converter can dispatch
prompt construction per row. That is what this writes.

**The roster is the point, not a detail.** Two source restrictions are baked in below because they
are choices about what the mix measures, and a mix that quietly drifts off them is not the same
dataset:

* **retrieval -> MS MARCO only.** nq, hotpotqa, fiqa and scifact are all graded by the same
  ``retrieval`` spec. Training on several of them makes every retrieval row in-distribution and
  there is nothing left to generalise to, so the mix trains one and leaves the rest as held-out
  probes.
* **qdmatch -> NQ only**, for the same reason against ``qdmatch_hpqa``.

Held-out ladders (``fiqa``, ``scifact``, ``outlier_review``, ``contra_fever``) are refused outright,
not warned about: by the time a warning is read the checkpoint is trained and the OOD column is
meaningless.

Emits the mix plus a manifest recording the rung band, the per-task counts and the exact source
files, so the two length versions of the mix are distinguishable after the fact.

Example::

    # short version (2k-4k), debug slice
    python src/scripts/data/hybridish/build_ctc_sft_mix.py \\
        --root /data/ctc_hybridish/short --tasks qdmatch_nq msmarco \\
        --per-task 8000 --band 2k-4k \\
        --out /data/ctc_hybridish/mix_short.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from collections import defaultdict
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("ctc_sft_mix")

#: Ladders that exist to measure generalisation. Training on one destroys the only thing it is for.
HELD_OUT = {"fiqa", "scifact", "outlier_review", "contra_fever", "redundancy"}

#: One ladder per grading spec, so no spec is trained on two of its own sources. The values are the
#: ladders this mix is ALLOWED to draw that spec from; anything else sharing the spec is held out by
#: construction and stays a clean probe.
SPEC_SOURCE = {
    "retrieval": "msmarco",
    "qdmatch": "qdmatch_nq",
}

#: Ladder -> grading spec, for the SPEC_SOURCE check.
TASK_SPEC = {
    "nq": "retrieval",
    "msmarco": "retrieval",
    "fiqa": "retrieval",
    "scifact": "retrieval",
    "qdmatch_nq": "qdmatch",
    "qdmatch_hpqa": "qdmatch",
}

#: The full in-train roster, when ``--tasks`` is not given.
DEFAULT_ROSTER = [
    "msmarco",
    "qdmatch_nq",
    "contradiction",
    "outlier",
    "oolong",
    "xabsence",
    "absence",
    "strmatch",
    "reorder",
    "textgroups",
    "cycle",
    "groups4",
    "mathmatch",
    "rerank",
    "grouping_labeled",
    "hotpotqa",
]


def check_roster(tasks: List[str], allow_spec_collision: bool) -> None:
    """
    Refuse a roster that trains a held-out ladder or two sources of one grading spec.

    :param tasks: The requested ladders.
    :param allow_spec_collision: Permit two ladders sharing a spec (say so in the manifest).

    :raises SystemExit: On a held-out ladder, or a spec collision without the override.
    """
    bad = sorted(set(tasks) & HELD_OUT)
    if bad:
        raise SystemExit(
            f"{', '.join(bad)} is held out: it exists to measure generalisation to an unseen "
            "corpus, and training on it makes every number from it in-distribution."
        )
    by_spec: Dict[str, List[str]] = defaultdict(list)
    for t in tasks:
        spec = TASK_SPEC.get(t)
        if spec:
            by_spec[spec].append(t)
    for spec, members in by_spec.items():
        allowed = SPEC_SOURCE.get(spec)
        if allowed and (len(members) > 1 or members[0] != allowed):
            if allow_spec_collision:
                log.warning(
                    "spec %r drawn from %s (policy says %s only) -- recorded in the manifest",
                    spec, ", ".join(sorted(members)), allowed,
                )
            else:
                raise SystemExit(
                    f"spec {spec!r} would be trained from {', '.join(sorted(members))}, but the "
                    f"mix policy draws it from {allowed!r} only, so the others stay clean probes. "
                    "Pass --allow-spec-collision to override deliberately."
                )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--root", required=True, help="the ctc-data --out dir holding <task>/train.jsonl")
    p.add_argument("--tasks", nargs="*", default=None, help=f"default: {' '.join(DEFAULT_ROSTER)}")
    p.add_argument("--out", required=True, help="output combined JSONL")
    p.add_argument("--per-task", type=int, default=0, help="cap rows per task (0 = all)")
    p.add_argument(
        "--band",
        default="",
        help="rung band these builds were made at, e.g. '2k-4k' or '2k-32k'. Recorded in the "
        "manifest and in every row's _band, so the two versions of the mix stay distinguishable",
    )
    p.add_argument("--cot-mode", default="none")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-spec-collision", action="store_true")
    p.add_argument(
        "--missing-ok",
        action="store_true",
        help="skip a task whose train.jsonl is absent instead of failing (default: fail, so a "
        "half-built roster cannot silently become a smaller mix)",
    )
    args = p.parse_args()

    tasks = args.tasks if args.tasks else list(DEFAULT_ROSTER)
    check_roster(tasks, args.allow_spec_collision)

    rng = random.Random(args.seed)
    rows: List[dict] = []
    by_task: Dict[str, int] = {}
    sources: Dict[str, str] = {}
    for task in tasks:
        path = os.path.join(args.root, task, "train.jsonl")
        if not os.path.exists(path):
            if args.missing_ok:
                log.warning("missing (skipped): %s", path)
                continue
            raise SystemExit(
                f"no train.jsonl for {task!r} at {path}. Build it first "
                f"(ctc-data build --task {task} --split train --rungs ... --out {args.root}), "
                "or pass --missing-ok to drop it deliberately."
            )
        task_rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                ex["_task"] = task
                ex["_cot_mode"] = args.cot_mode
                if args.band:
                    ex["_band"] = args.band
                task_rows.append(ex)
        if args.per_task and len(task_rows) > args.per_task:
            task_rows = rng.sample(task_rows, args.per_task)
        rows.extend(task_rows)
        by_task[task] = len(task_rows)
        sources[task] = path
        log.info("%-18s %6d rows", task, len(task_rows))

    if not rows:
        raise SystemExit("no rows selected")

    rng.shuffle(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for ex in rows:
            f.write(json.dumps(ex) + "\n")

    manifest = {
        "out": args.out,
        "root": args.root,
        "band": args.band,
        "seed": args.seed,
        "num_examples": len(rows),
        "num_tasks": len(by_task),
        "by_task": by_task,
        "sources": sources,
        "spec_source_policy": SPEC_SOURCE,
        "held_out": sorted(HELD_OUT),
        "allow_spec_collision": args.allow_spec_collision,
    }
    with open(args.out + ".manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("wrote %s (%d rows, %d tasks)", args.out, len(rows), len(by_task))


if __name__ == "__main__":
    main()
