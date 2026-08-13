"""
Build the OLMo-3-vocabulary SFT shards for the HiLS / Olmo-3 arms, one out-dir per task.

Reads ``suite_manifest.tsv`` and selects each task's TRAIN files itself rather than taking
hand-copied globs, because the traps here are all in file selection:

* **NQ must be the p10 build.** ``hn<N>`` is the hard-negative COUNT, so p10 means ``hn ~ 10% of
  k``. The banned 98%-hard build (``hn98`` / ``hn198`` / ``hn498``) sits in the same directory
  under near-identical names. Everything is *evaluated* on the p10 ladder, so training on the other
  one is a train/eval mismatch that costs real NQ points.
* **``retrieval`` is overloaded** in the manifest: hotpotqa, msmarco, niah and the beir sets all
  carry that task name. Selecting by task alone would silently widen NQ into a different mixture
  than our Qwen3.5 arms trained on, so NQ is selected by FILENAME.
* **scifact / fiqa are held out** and scored as OOD ladders; they must never enter training.

Per-task out-dirs, not one combined corpus: the mixture is applied later by
``sft_shard_dataset.mix_documents`` at dataset level, and a pre-combined corpus has no mixing stage
(the sampling weights would silently do nothing).

Run on a weka-mounted CPU gantry node.
"""

import argparse
import os
import subprocess
import sys
from typing import Dict, List

# Selection per task. `include` are substrings ALL of which must appear in the filename; `exclude`
# any of which disqualifies it. Deliberately explicit rather than a glob, so a new file landing in
# cr_suite_data cannot silently join the mixture.
TASK_SELECTORS: Dict[str, Dict[str, object]] = {
    "contra": {
        "manifest_task": "contradiction",
        "include": ["contradiction_train_pubmed_both_", "_k3"],
        "exclude": ["cotmix"],  # the enumerate-CoT variant is a different target format
    },
    "nq": {
        "manifest_task": None,  # selected by filename: `retrieval` is overloaded
        "include": ["nq_train_"],
        "exclude": ["hn98", "hn198", "hn398", "hn498", "hn48", "hn18"],  # keep only the p10 build
        "require_p10": True,
    },
    "rerank": {
        "manifest_task": "rerank",
        "include": ["msmarco_helmet_rerank_train_"],
        "exclude": [],
    },
    "outlier": {"manifest_task": "outlier", "include": ["_train"], "exclude": []},
    "oolong": {"manifest_task": "oolong", "include": ["_train"], "exclude": []},
}

HELD_OUT = ["beir_scifact", "beir_fiqa", "redundancy_"]


def read_manifest(path: str) -> List[dict]:
    rows = []
    with open(path) as fh:
        next(fh)  # header
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            f, task, cot_mode, split, nbytes = line.split("\t")
            rows.append({"file": f, "task": task, "cot_mode": cot_mode, "split": split})
    return rows


def select(rows: List[dict], task: str) -> List[dict]:
    """
    Pick the train files for one task.

    :raises SystemExit: If nothing matches, or if NQ's p10 requirement is not satisfied -- a silent
        empty selection would train an arm with a task missing and still converge.
    """
    sel = TASK_SELECTORS[task]
    out = []
    for r in rows:
        if r["split"] != "train":
            continue
        if any(h in r["file"] for h in HELD_OUT):
            continue
        if sel["manifest_task"] and r["task"] != sel["manifest_task"]:
            continue
        if not all(s in r["file"] for s in sel["include"]):  # type: ignore[operator]
            continue
        if any(s in r["file"] for s in sel["exclude"]):  # type: ignore[operator]
            continue
        out.append(r)
    if not out:
        raise SystemExit(f"[{task}] selected NO files -- refusing to build a mixture missing a task")
    if sel.get("require_p10"):
        bad = [r["file"] for r in out if "hn10" not in r["file"] and "hn2" not in r["file"]]
        if bad:
            raise SystemExit(
                f"[{task}] p10 required but selection includes non-p10 files: {bad}. "
                f"hn<N> is the hard-negative COUNT; p10 means hn ~ 10% of k."
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="cr_suite_data (manifest + JSONL)")
    ap.add_argument("--out-root", required=True, help="parent dir for the per-task out-dirs")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--chat-template", required=True)
    ap.add_argument("--eos-token-id", type=int, default=100257)
    ap.add_argument("--tasks", nargs="*", default=list(TASK_SELECTORS))
    ap.add_argument("--limit-per-file", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = read_manifest(os.path.join(args.data_dir, "suite_manifest.tsv"))
    plan = {t: select(rows, t) for t in args.tasks}

    print("=== selection ===")
    for task, files in plan.items():
        print(f"\n[{task}] {len(files)} file(s)")
        for r in files:
            print(f"    {r['file']}  (cot_mode={r['cot_mode']})")
    if args.dry_run:
        return 0

    converter = "src/scripts/data/convert_unified_to_sft.py"
    for task, files in plan.items():
        out_dir = os.path.join(args.out_root, task)
        # One converter invocation per cot_mode: the mode changes the target format, and mixing two
        # formats under one task would teach the model two different answers to the same prompt.
        by_mode: Dict[str, List[str]] = {}
        for r in files:
            by_mode.setdefault(r["cot_mode"], []).append(os.path.join(args.data_dir, r["file"]))
        for mode, paths in by_mode.items():
            dest = out_dir if len(by_mode) == 1 else f"{out_dir}_{mode}"
            cmd = [
                sys.executable, converter,
                "--tokenizer", args.tokenizer,
                "--eos-token-id", str(args.eos_token_id),
                "--landmark-token-id", "-1",
                "--chat-template", args.chat_template,
                "--out-dir", dest,
                "--task", task if task != "nq" else "nq",
                "--cot-mode", mode,
                "--input-jsonl", *paths,
            ]
            if args.limit_per_file:
                cmd += ["--limit-per-file", str(args.limit_per_file)]
            print(f"\n=== [{task}/{mode}] -> {dest}\n{' '.join(cmd)}", flush=True)
            rc = subprocess.call(cmd)
            if rc != 0:
                print(f"FAILED [{task}/{mode}] rc={rc}")
                return rc
    print("\nAll tasks converted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
