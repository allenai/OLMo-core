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
import json
import os
import subprocess
import sys
from typing import Dict, List

# Selection per task. `include` are substrings ALL of which must appear in the filename; `exclude`
# any of which disqualifies it. Deliberately explicit rather than a glob, so a new file landing in
# cr_suite_data cannot silently join the mixture.
TASK_SELECTORS: Dict[str, Dict[str, object]] = {
    "contra": {
        "converter_task": "contradiction",
        "manifest_task": "contradiction",
        "include": ["contradiction_train_pubmed_both_", "_k3"],
        "exclude": ["cotmix"],  # the enumerate-CoT variant is a different target format
    },
    "nq": {
        "converter_task": "nq",
        "manifest_task": None,  # selected by filename: `retrieval` is overloaded
        "include": ["nq_train_"],
        # `_cot` is a different target format; the mixture references one `nq` source, so taking
        # both would train two answer formats for one task under one weight.
        "exclude": ["_cot"],
        # p10 is enforced by RATIO below, not by a blocklist. A blocklist was tried and was wrong:
        # it listed hotpotqa's hn98/hn198 while NQ's banned files are hn99/hn199/hn19/hn49
        # (hn = k-1). Any new k would need a new entry, and a miss silently trains the 98%-hard
        # build. The ratio rule needs no maintenance and states the actual requirement.
        "require_p10": True,
    },
    "rerank": {
        "converter_task": "rerank",
        "manifest_task": "rerank",
        "include": ["msmarco_helmet_rerank_train_"],
        "exclude": [],
    },
    # `include` is empty for these two: the manifest's own `split` column already selects train,
    # and an extra "_train" substring filter is both redundant and wrong -- oolong's files are
    # named `..._splittrain.jsonl`, which does not contain "_train", so the filter silently
    # selected nothing and the guard had to catch it.
    "outlier": {"converter_task": "outlier", "manifest_task": "outlier", "include": [], "exclude": []},
    "oolong": {
        "converter_task": "oolong",
        "manifest_task": "oolong",
        "include": [],
        # ctx65536 documents cannot fit the 32k SFT window, so every one would be dropped by the
        # packer after being tokenized. Excluding them saves the conversion rather than changing
        # the mixture.
        "exclude": ["ctx65536"],
    },
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
        out = [r for r in out if _is_p10(r["file"])]
        if not out:
            raise SystemExit(
                f"[{task}] no p10 file found. hn<N> is the hard-negative COUNT and p10 means "
                f"hn <= {P10_MAX_RATIO:.0%} of k; the 98%-hard build (hn = k-1) is banned because "
                f"everything is EVALUATED on the p10 ladder."
            )
    return out


P10_MAX_RATIO = 0.15  # 10% hard negatives, with slack for rounding at small k


def _is_p10(filename: str) -> bool:
    """
    Decide from the filename whether a retrieval file is the p10 (10%-hard-negative) build.

    Positive rule rather than a blocklist: the banned families differ per task (hotpotqa uses
    ``hn98``/``hn198``, NQ uses ``hn99``/``hn199``/``hn19``/``hn49``), so a blocklist has to be
    maintained per task and a miss silently trains the wrong build.

    :param filename: e.g. ``nq_train_k100_hn10_2500.jsonl``.

    :returns: ``True`` if ``hn / k`` is within the p10 ratio.
    """
    import re

    k = re.search(r"_k(\d+)", filename)
    hn = re.search(r"_hn(\d+)", filename)
    if not (k and hn):
        return False
    k_n, hn_n = int(k.group(1)), int(hn.group(1))
    return k_n > 0 and hn_n / k_n <= P10_MAX_RATIO


def _prepare(row: dict, task: str, args) -> str:
    """
    Return the JSONL path to feed the converter, normalizing the schema first when needed.

    rerank's HELMET train files are ``{qid, query, ctxs:[...]}`` while ``build_prompt`` expects
    ``{documents, queries, gold_doc_indices}``. Without this, EVERY rerank row fails with
    ``KeyError: 'documents'`` and the converter writes zero instances **while exiting 0** -- the
    mixture then trains with rerank silently absent (observed: 8000/8000 rows skipped, job green).

    :returns: Either the original path or a normalized copy under ``--out-root/_normalized``.
    """
    src = os.path.join(args.data_dir, row["file"])
    if task != "rerank":
        return src

    # hils_sft -> memexpress -> train -> scripts, then /data. Three levels, not two: with two this
    # resolves to src/scripts/train/data, which does not exist, and the import fails at runtime.
    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data")
    )
    from build_combined_suite_jsonl import normalize_example  # type: ignore[import-not-found]

    dest_dir = os.path.join(args.out_root, "_normalized")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, row["file"])
    if os.path.exists(dest):
        return dest
    n = 0
    with open(src) as fin, open(dest, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            ex = normalize_example(json.loads(line), task)
            # normalize_example carries `_task`/`_cot_mode` through as None (they are populated by
            # the combined-corpus path, not present in the raw files). The converter reads them
            # with dict.get(key, default), which returns the stored None rather than falling back
            # to --task/--cot-mode -- so a present-but-null key defeats the CLI flags. Drop them
            # and let the flags apply.
            ex = {k: v for k, v in ex.items() if not (k in ("_task", "_cot_mode") and v is None)}
            fout.write(json.dumps(ex) + "\n")
            n += 1
    print(f"    normalized {row['file']}: {n} rows -> {dest}", flush=True)
    return dest


def _instances_written(out_dir: str) -> int:
    """Count instances actually on disk, to catch a converter that wrote nothing but exited 0."""
    import glob as _glob

    return len(_glob.glob(os.path.join(out_dir, "token_ids_part_*.npy")))


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
        files = [dict(r, path=_prepare(r, task, args)) for r in files]
        # One converter invocation per cot_mode: the mode changes the target format, and mixing two
        # formats under one task would teach the model two different answers to the same prompt.
        by_mode: Dict[str, List[str]] = {}
        for r in files:
            by_mode.setdefault(r["cot_mode"], []).append(r["path"])
        for mode, paths in by_mode.items():
            dest = out_dir if len(by_mode) == 1 else f"{out_dir}_{mode}"
            cmd = [
                sys.executable, converter,
                "--tokenizer", args.tokenizer,
                "--eos-token-id", str(args.eos_token_id),
                "--landmark-token-id", "-1",
                "--chat-template", args.chat_template,
                "--out-dir", dest,
                # The CANONICAL task name, not our short key: it selects the prompt builder, and
                # these must be the same names the eval ladder uses as LTASK (contra ->
                # contradiction) or training and eval render different prompts for one task.
                "--task", str(TASK_SELECTORS[task]["converter_task"]),
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
            # rc==0 is NOT success: the converter exits 0 after skipping every row (a schema
            # mismatch skips per-row and only warns). A task that wrote no shards would leave the
            # mixture missing that task while the whole build looked green.
            if _instances_written(dest) == 0:
                print(f"FAILED [{task}/{mode}] converter exited 0 but wrote NO shards to {dest} -- "
                      f"every row was skipped (check the build_prompt warnings above for a schema "
                      f"mismatch).")
                return 3
    print("\nAll tasks converted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
