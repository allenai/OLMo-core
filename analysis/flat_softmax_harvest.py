#!/usr/bin/env python3
"""Harvest scores for the flat-softmax ablation sweep from Beaker job logs.

The datalake dashboard upload was 404-ing cluster-wide when the sweep launched, so results are pulled
directly from each experiment's Beaker logs instead of the dashboard. Reads the manifest produced at
launch (``analysis/flat_softmax_sweep_manifest.csv``) and, for every experiment that has finalized,
extracts the printed final metrics.

RULER: the runner prints ``ruler_<subtask>__<len>::std: <score>`` and ``ruler_all__<len>::suite:
<score>`` near the end of the log -- we capture those. (exit=1 from the datalake 404 does NOT mean the
metrics are missing; they are computed and printed before the upload step.)

HELMET / SFT: metrics are written to the weka output dir, not just stdout; this script records the
job status and, for RULER, the parsed scores. Extend the HELMET/SFT parsing once their output layout
is confirmed (HELMET -> /weka/oe-training-default/tylerr/data/eval/helmet/output/<model>_<suffix>/;
SFT -> the per-run --results-dir given at launch).

Usage:
  ~/miniconda3/envs/cookin-lc-olmo3/bin/python analysis/flat_softmax_harvest.py \
      [--manifest analysis/flat_softmax_sweep_manifest.csv] [--out analysis/flat_softmax_results.csv]
"""
import argparse
import csv
import json
import re
import subprocess

RULER_LINE = re.compile(r"(ruler_[a-z0-9_]+__\d+::(?:std|suite)):\s*([0-9.]+)")


def beaker_status(exp_id):
    try:
        out = subprocess.run(
            ["beaker", "experiment", "get", exp_id, "--format", "json"],
            capture_output=True, text=True, timeout=60,
        ).stdout
        e = json.loads(out)[0]
        j = (e.get("jobs") or [{}])[-1]
        s = j.get("status", {}) or {}
        return ("finalized" if s.get("finalized") else "running", s.get("exitCode"))
    except Exception as ex:
        return (f"err:{ex}", None)


def beaker_logs(exp_id):
    try:
        return subprocess.run(
            ["beaker", "experiment", "logs", exp_id],
            capture_output=True, text=True, timeout=120,
        ).stdout
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="analysis/flat_softmax_sweep_manifest.csv")
    ap.add_argument("--out", default="analysis/flat_softmax_results.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.manifest)))
    results = []
    for r in rows:
        status, code = beaker_status(r["experiment_id"])
        scores = {}
        if r["harness"] == "RULER" and status == "finalized":
            for k, v in RULER_LINE.findall(beaker_logs(r["experiment_id"])):
                scores[k] = float(v)
        results.append({**r, "status": status, "exit_code": code,
                        "n_scores": len(scores), "scores_json": json.dumps(scores)})
        print(f"{r['harness']:6s} {r['checkpoint']:15s} {r['variant']:5s} {r['unit']:8s} "
              f"{status:10s} exit={code} scores={len(scores)}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nWrote {args.out} ({len(results)} rows)")


if __name__ == "__main__":
    main()
