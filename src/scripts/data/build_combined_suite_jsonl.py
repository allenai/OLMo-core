"""
Build ONE combined, tagged unified-JSONL from the per-task corpus-reasoning suite train files,
ready for src/scripts/data/convert_unified_to_sft.py (-> AI2/olmo-core SFT shards).

Reads ``suite_manifest.tsv`` (file<TAB>task<TAB>cot_mode<TAB>split<TAB>bytes) + the per-task JSONL
in ``--data-dir`` (land them on weka with land_suite_eval_to_weka_gantry.sh, or download locally
with download_suite_data.sh). For each selected TRAIN file it streams rows, attaches ``_task`` /
``_cot_mode`` (so the converter dispatches per row), samples up to a budget, asserts the held-out
tasks never leak, shuffles, and writes the combined JSONL + a sidecar manifest.

Held out (eval-only; asserted absent): redundancy, beir_scifact, beir_fiqa (see HELD_OUT_GLOBS).

Examples::

    # ~1k debug build across all in-train tasks (balanced), for end-to-end pipeline tests
    python src/scripts/data/build_combined_suite_jsonl.py \\
        --data-dir /weka/.../prasanns/cr_suite_data \\
        --total-budget 1000 --out /weka/.../cr_suite_data/suite_combined_debug.jsonl

    # full build: up to N examples per (task x rung) file
    python src/scripts/data/build_combined_suite_jsonl.py \\
        --data-dir /weka/.../prasanns/cr_suite_data \\
        --per-rung-budget 2000 --out /weka/.../cr_suite_data/suite_combined.jsonl
"""

import argparse
import fnmatch
import json
import logging
import math
import os
import random
from collections import defaultdict
from typing import List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("build_combined")

# Held-out tasks/sources: never in train (eval-only generalization probes). Matched on filename.
HELD_OUT_GLOBS = ["redundancy_*", "beir_scifact_*", "beir_fiqa_*"]


def is_held_out(filename: str) -> bool:
    return any(fnmatch.fnmatch(filename, g) for g in HELD_OUT_GLOBS)


def is_cot_variant(filename: str) -> bool:
    return "cotmix" in filename or filename.endswith("_cot.jsonl")


def normalize_example(ex: dict, task: str) -> dict:
    """Reconcile schema variants so every row matches what build_prompt(task=...) expects.

    rerank: the HELMET train files are ``{qid, query, ctxs:[{id,label,score,text}]}`` (graded
    relevance), but the rerank eval + build_prompt use the standard ``{documents, queries,
    gold_doc_indices}`` schema. Convert ctxs -> documents and derive binary qrels from the graded
    label (label > 0 == relevant), so train matches the standard-rerank eval distribution.
    """
    if task == "rerank" and "ctxs" in ex and "documents" not in ex:
        ctxs = ex["ctxs"]
        ex = {
            "documents": [{"text": c.get("text", "")} for c in ctxs],
            "queries": [ex.get("query", "")],
            "gold_doc_indices": [i for i, c in enumerate(ctxs) if int(c.get("label", 0)) > 0],
            "answers": [""],
            "source": ex.get("source", "msmarco_helmet_rerank"),
            "_task": ex.get("_task"),
            "_cot_mode": ex.get("_cot_mode"),
        }
    return ex


def read_manifest(path: str) -> List[dict]:
    rows = []
    with open(path) as f:
        next(f)  # header
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            file, task, cot_mode, split, nbytes = line.split("\t")
            rows.append({"file": file, "task": task, "cot_mode": cot_mode, "split": split})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="dir with suite_manifest.tsv + the per-task JSONL")
    ap.add_argument("--out", required=True, help="output combined JSONL path")
    ap.add_argument("--manifest", default=None, help="manifest path (default: <data-dir>/suite_manifest.tsv)")
    ap.add_argument("--total-budget", type=int, default=0,
                    help="approx TOTAL examples across all tasks (debug builds), balanced per task "
                         "(per_task = total / n_tasks).")
    ap.add_argument("--per-task-budget", type=int, default=0,
                    help="EQUAL rows per task (distributed across that task's rungs). The default "
                         "balancing mode for the full build.")
    ap.add_argument("--per-rung-budget", type=int, default=0,
                    help="max examples taken from EACH (task x rung) file (0 = all). Not task-balanced.")
    ap.add_argument("--tasks", nargs="*", default=None, help="only these task types (default: all in-train)")
    ap.add_argument("--cot", choices=("both", "plain", "cot"), default="both",
                    help="include both plain+CoT files, plain-only, or cot-only.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    manifest_path = args.manifest or os.path.join(args.data_dir, "suite_manifest.tsv")
    rows = read_manifest(manifest_path)

    # Select train files: drop held-out, apply task / cot filters.
    selected = []
    held_out_seen = []
    for r in rows:
        if r["split"] != "train":
            continue
        if is_held_out(r["file"]):
            held_out_seen.append(r["file"])
            continue
        if args.tasks and r["task"] not in args.tasks:
            continue
        cot = is_cot_variant(r["file"])
        if args.cot == "plain" and cot:
            continue
        if args.cot == "cot" and not cot:
            continue
        selected.append(r)

    if not selected:
        raise SystemExit("No files selected (check --data-dir / --tasks / --cot).")
    log.info(f"selected {len(selected)} train files across "
             f"{len(set(r['task'] for r in selected))} tasks; excluded {len(held_out_seen)} held-out files")

    if sum(bool(x) for x in (args.total_budget, args.per_task_budget, args.per_rung_budget)) > 1:
        raise SystemExit("Pass at most one of --total-budget / --per-task-budget / --per-rung-budget.")

    # Group selected files by task so we can balance rows ACROSS tasks (equal-per-task).
    files_by_task: dict = defaultdict(list)
    for r in selected:
        files_by_task[r["task"]].append(r)
    n_tasks = len(files_by_task)

    if args.per_task_budget:
        per_task_target = args.per_task_budget
    elif args.total_budget:
        per_task_target = max(1, math.ceil(args.total_budget / n_tasks))
    else:
        per_task_target = 0  # all (or capped per-file by --per-rung-budget)

    def read_file(r) -> list:
        path = os.path.join(args.data_dir, r["file"])
        if not os.path.exists(path):
            log.warning(f"missing file (skipped): {r['file']}")
            return []
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                if "ex" in ex and "documents" not in ex:
                    ex = ex["ex"]
                ex["_task"] = r["task"]
                ex["_cot_mode"] = r["cot_mode"]
                ex = normalize_example(ex, r["task"])  # schema reconciliation (e.g. rerank)
                rows.append(ex)
        return rows

    rng = random.Random(args.seed)
    out_rows: list = []
    by_task: dict = defaultdict(int)
    for task, files in files_by_task.items():
        # per-file cap: from --per-rung-budget, else spread the per-task target across this task's files.
        if args.per_rung_budget:
            per_file_cap = args.per_rung_budget
        elif per_task_target:
            per_file_cap = max(1, math.ceil(per_task_target / len(files)))
        else:
            per_file_cap = 0
        task_rows: list = []
        for r in files:
            rows_in_file = read_file(r)
            if per_file_cap and len(rows_in_file) > per_file_cap:
                rows_in_file = rng.sample(rows_in_file, per_file_cap)
            task_rows.extend(rows_in_file)
        # trim to an equal per-task total
        if per_task_target and len(task_rows) > per_task_target:
            task_rows = rng.sample(task_rows, per_task_target)
        out_rows.extend(task_rows)
        by_task[task] = len(task_rows)

    # Defense in depth: assert nothing held-out leaked.
    leaked = [ex for ex in out_rows if is_held_out(str(ex.get("source", "")) + ".jsonl")]
    assert not leaked, f"{len(leaked)} held-out rows leaked into the combined set!"

    rng.shuffle(out_rows)
    if args.total_budget and len(out_rows) > args.total_budget:
        out_rows = out_rows[: args.total_budget]
        # recompute by_task after the trim
        by_task = defaultdict(int)
        for ex in out_rows:
            by_task[ex["_task"]] += 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for ex in out_rows:
            f.write(json.dumps(ex) + "\n")

    sidecar = {
        "out": args.out,
        "num_examples": len(out_rows),
        "num_tasks": len(by_task),
        "by_task": dict(sorted(by_task.items())),
        "held_out_globs": HELD_OUT_GLOBS,
        "held_out_files_excluded": sorted(held_out_seen),
        "cot": args.cot,
        "budget_mode": (
            f"per_task={args.per_task_budget}" if args.per_task_budget
            else f"total={args.total_budget}" if args.total_budget
            else f"per_rung={args.per_rung_budget}" if args.per_rung_budget
            else "all"
        ),
        "per_task_target": per_task_target,
        "seed": args.seed,
        "n_files": len(selected),
    }
    with open(args.out + ".manifest.json", "w") as f:
        json.dump(sidecar, f, indent=2)
    log.info("DONE: %d examples across %d tasks -> %s", len(out_rows), len(by_task), args.out)
    log.info("by_task: %s", json.dumps(dict(sorted(by_task.items()))))


if __name__ == "__main__":
    main()
