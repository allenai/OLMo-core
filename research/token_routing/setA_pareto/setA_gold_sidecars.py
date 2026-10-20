"""Re-tokenize a built CTC SFT set with gold sidecars, and prove the tokens did not change.

The soft-token (token-drop) arm needs ``gold_fingerprints.json`` next to each shard: a row whose
fingerprint is MISSING trains uncompressed (every document kept real). setA was tokenized without
sidecars, so this re-runs the set's own converter call -- arguments read from its ``manifest.json``,
never retyped -- with ``--emit-gold-sidecar --gold-sidecar-empty-for-missing`` into a sibling shard
dir, then byte-compares every token/mask part against the shipped shards. Any difference is a
failure: the sidecar must describe the exact rows the dense arm trained on.

    python research/token_routing/setA_pareto/setA_gold_sidecars.py \
        --root /weka/.../ctc_sft_sets/setA_max20_evaliid --jobs 11 --num-proc 4

Writes ``<root>/<--out-dir>/<task>/`` plus ``<root>/<--out-dir>/sidecar_report.json``.
"""

import argparse
import concurrent.futures as cf
import filecmp
import glob
import json
import os
import subprocess
import sys
import time


def _cmd(root: str, conv: dict, task: dict, out_dir: str, num_proc: int, repo: str) -> list:
    cmd = [
        sys.executable,
        os.path.join(repo, "recipes", "data", "convert_unified_to_document_landmark.py"),
        "--emit", "dense", "--task", task["spec"], "--chunk-by", task.get("chunk_by", "document"),
        "--marker-set", conv["marker_set"], "--tokenizer", conv["tokenizer"],
        "--query-position", conv["query_position"], "--cot-mode", conv["cot_mode"],
        "--seq-len", str(conv["seq_len"]),
        "--input-jsonl", os.path.join(root, "per_task", task["name"], "train.jsonl"),
        "--out-dir", out_dir, "--emit-gold-sidecar", "--gold-sidecar-empty-for-missing",
        "--num-proc", str(num_proc),
    ]
    if not conv.get("doc_markers", True):
        cmd.append("--no-doc-markers")
    return cmd


def _one(job):
    root, conv, task, ref_dir, out_dir, num_proc, repo = job
    t0 = time.time()
    rec = {"task": task["name"], "out": out_dir}
    if not os.path.exists(os.path.join(out_dir, "gold_fingerprints.json")):
        os.makedirs(out_dir, exist_ok=True)
        p = subprocess.run(_cmd(root, conv, task, out_dir, num_proc, repo), capture_output=True, text=True)
        if p.returncode != 0:
            rec.update(status="CONVERT_FAILED", tail=(p.stdout + p.stderr).strip().splitlines()[-5:])
            return rec
    ref = sorted(os.path.basename(f) for f in glob.glob(os.path.join(ref_dir, "*_part_*.npy"))
                 + glob.glob(os.path.join(ref_dir, "labels_mask_*.npy")))
    new = sorted(os.path.basename(f) for f in glob.glob(os.path.join(out_dir, "*_part_*.npy"))
                 + glob.glob(os.path.join(out_dir, "labels_mask_*.npy")))
    ref, new = sorted(set(ref)), sorted(set(new))
    if ref != new:
        rec.update(status="PART_LIST_DIFFERS", ref=ref, new=new)
        return rec
    diff = [f for f in ref if not filecmp.cmp(os.path.join(ref_dir, f), os.path.join(out_dir, f), shallow=False)]
    m_ref = json.load(open(os.path.join(ref_dir, "metadata.json")))
    m_new = json.load(open(os.path.join(out_dir, "metadata.json")))
    side = json.load(open(os.path.join(out_dir, "gold_fingerprints.json")))
    n_rows = int(m_new["num_instances"])
    nonempty = sum(1 for v in side.values() if v)
    rec.update(
        status="OK" if not diff and m_ref["num_instances"] == n_rows else "BYTES_DIFFER",
        parts=len(ref), differing_parts=diff, rows=n_rows, rows_ref=m_ref["num_instances"],
        sidecar_entries=len(side), sidecar_nonempty=nonempty,
        # entries < rows only if two rows share identical content (one fingerprint)
        rows_without_entry=n_rows - len(side), seconds=round(time.time() - t0),
    )
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="set root holding manifest.json and per_task/")
    ap.add_argument("--ref-dir", default="", help="shipped shard dir name (default: manifest shards_dir)")
    ap.add_argument("--out-dir", default="", help="new shard dir name (default: <ref-dir>_gold)")
    ap.add_argument("--tasks", default="", help="comma list; default every task with rows")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--num-proc", type=int, default=4, help="converter workers per task")
    ap.add_argument("--repo", default=os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3)))
    a = ap.parse_args()
    man = json.load(open(os.path.join(a.root, "manifest.json")))
    conv = man["convert"]
    print("manifest convert args:", json.dumps(conv), flush=True)
    ref_name = a.ref_dir or conv.get("shards_dir") or f"shards_{conv['marker_set']}"
    out_name = a.out_dir or f"{ref_name}_gold"
    want = set(a.tasks.split(",")) if a.tasks else None
    roster = [t for t in man["roster"] if man["per_task"].get(t["name"], {}).get("rows")
              and (want is None or t["name"] in want)]
    jobs = [(a.root, conv, t, os.path.join(a.root, ref_name, t["name"]),
             os.path.join(a.root, out_name, t["name"]), a.num_proc, a.repo) for t in roster]
    print(f"{len(jobs)} tasks: {[t['name'] for t in roster]} -> {out_name}", flush=True)
    t0, recs = time.time(), []
    with cf.ProcessPoolExecutor(max_workers=a.jobs) as pool:
        for i, rec in enumerate(pool.map(_one, jobs), 1):
            recs.append(rec)
            el = time.time() - t0
            print(f"[{i}/{len(jobs)}] {rec['task']}: {rec['status']} rows={rec.get('rows')} "
                  f"sidecar={rec.get('sidecar_entries')} nonempty={rec.get('sidecar_nonempty')} "
                  f"| {el:.0f}s, ETA {el / i * (len(jobs) - i):.0f}s", flush=True)
            if rec["status"] != "OK":
                print("   ", json.dumps(rec)[:800], flush=True)
    json.dump(recs, open(os.path.join(a.root, out_name, "sidecar_report.json"), "w"), indent=1)
    bad = [r["task"] for r in recs if r["status"] != "OK"]
    print("ALL_OK" if not bad else f"FAILED: {bad}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
