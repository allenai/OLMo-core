#!/usr/bin/env python3
"""Finalize freshly-synthesized oolong rungs into the v2 eval layout.

Reads the per-rung generator output (generate_oolong_ladder_data.py, run per rung
into $GEN/<label>/), keeps >=500 examples/rung, and writes them to
eval500_v2/oolong/oolong_test_synth_ctx{ctx}_spliteval.jsonl (the names the v2
ladder spec expects). These are NEW synthetic item-combinations, so they are
DISJOINT from the oolong training split (the v1 eval overlapped its own train
split — a contamination bug this fixes) while keeping the same question-type +
corpus-type distribution across rungs. We assert disjointness against the v1
oolong splittrain files and print the type/corpus distribution per rung.
"""
import argparse
import glob
import json
import os
from collections import Counter

DATA = "/scratch/users/prasann/corpus-reasoning/data"
V2_OOLONG = "/scratch/users/prasann/cpt_data/eval500_v2/oolong"


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def train_keys():
    """Signature set of every v1 oolong *training* example, to guarantee the v2
    eval is disjoint from what the model was trained on."""
    keys = set()
    for p in glob.glob(f"{DATA}/oolong_test_synth_ctx*_splittrain.jsonl"):
        for ex in load(p):
            q = (ex.get("queries") or [""])[0]
            body = ex["documents"][0]["text"] if ex.get("documents") else ""
            keys.add(q + "||" + body[:200])
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-root", required=True,
                    help="dir with per-rung subdirs (8k/16k/32k) of generator output")
    ap.add_argument("--min-per-rung", type=int, default=500)
    ap.add_argument("--rungs", default="8k:8192,16k:16384,32k:32768")
    args = ap.parse_args()
    os.makedirs(V2_OOLONG, exist_ok=True)
    tkeys = train_keys()
    print(f"loaded {len(tkeys)} v1 oolong TRAIN signatures (for disjointness check)")

    problems = []
    for spec in args.rungs.split(","):
        label, ctx = spec.split(":")
        ctx = int(ctx)
        srcs = glob.glob(os.path.join(args.gen_root, label, "*.jsonl"))
        if not srcs:
            print(f"[{label}] NO generator output in {args.gen_root}/{label} — skipping");
            problems.append(f"{label}: no gen output"); continue
        rows = []
        for s in srcs:
            rows.extend(load(s))
        # drop any example that collides with a training signature (belt-and-suspenders).
        clean = [ex for ex in rows if (
            (ex.get("queries") or [""])[0] + "||" +
            (ex["documents"][0]["text"][:200] if ex.get("documents") else "")) not in tkeys]
        n_overlap = len(rows) - len(clean)
        out_path = os.path.join(V2_OOLONG, f"oolong_test_synth_ctx{ctx}_spliteval.jsonl")
        # cap to exactly --min-per-rung (=500); generator order is random so the first N keep
        # the same dataset/task-group distribution. flag if we somehow have fewer than N.
        keep = clean[:args.min_per_rung]
        with open(out_path, "w") as f:
            for ex in keep:
                f.write(json.dumps(ex) + "\n")
        ds = Counter(ex["_meta"].get("dataset") for ex in keep)
        tg = Counter(ex["_meta"].get("task_group") for ex in keep)
        flag = "" if len(keep) >= args.min_per_rung else f"  <<< ONLY {len(keep)} < {args.min_per_rung}"
        print(f"[{label} ctx{ctx}] gen={len(rows)} train-overlap-dropped={n_overlap} "
              f"-> kept {len(keep)}{flag}")
        print(f"    dataset dist:    {dict(ds)}")
        print(f"    task_group dist: {dict(tg)}")
        print(f"    -> {out_path}")
        if len(keep) < args.min_per_rung:
            problems.append(f"{label}: {len(keep)}<{args.min_per_rung}")
    print("\n" + ("PROBLEMS: " + "; ".join(problems) if problems else "oolong v2 finalized OK"))


if __name__ == "__main__":
    main()
