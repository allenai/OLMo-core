#!/usr/bin/env python
"""Parallel wrapper around convert_unified_to_document_landmark.py: split the input JSONL into N
chunks, tokenize them concurrently (the converter is single-threaded and CPU-bound on huge
long-context examples), then merge the per-chunk npy shards + metadata into one output dir.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess

REPO = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core"
PY = "/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python"
CONV = f"{REPO}/src/scripts/data/convert_unified_to_document_landmark.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--marker-set", required=True)
    ap.add_argument("--seq-len", type=int, default=262144)
    ap.add_argument("--nproc", type=int, default=7)
    args = ap.parse_args()

    tmp = args.out_dir + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)

    lines = open(args.input_jsonl).read().splitlines()
    n = args.nproc
    chunks = [lines[i::n] for i in range(n)]  # round-robin -> even length distribution
    procs = []
    for i, ch in enumerate(chunks):
        cpath = f"{tmp}/chunk_{i:02d}.jsonl"
        with open(cpath, "w") as f:
            f.write("\n".join(ch) + "\n")
        odir = f"{tmp}/out_{i:02d}"
        env = dict(os.environ, HF_HOME="/scratch/users/prasann/huggingface-cache",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                   PYTHONWARNINGS="ignore", PYTHONPATH=f"{REPO}/src")
        cmd = [PY, CONV, "--emit", "dense", "--task", "contradiction", "--chunk-by", "document",
               "--cot-mode", "none", "--query-position", "both", "--seq-len", str(args.seq_len),
               "--tokenizer", args.tokenizer, "--marker-set", args.marker_set,
               "--input-jsonl", cpath, "--out-dir", odir]
        procs.append((i, subprocess.Popen(cmd, env=env, stdout=open(f"{tmp}/log_{i:02d}.txt", "w"),
                                          stderr=subprocess.STDOUT)))
    rc = 0
    for i, p in procs:
        r = p.wait()
        print(f"chunk {i:02d} rc={r}", flush=True)
        rc = rc or r
    if rc:
        raise SystemExit(f"a chunk converter failed (rc={rc}); see {tmp}/log_*.txt")

    # merge shards (renumber) + aggregate metadata
    os.makedirs(args.out_dir, exist_ok=True)
    gi = 0
    agg = None
    for i in range(n):
        odir = f"{tmp}/out_{i:02d}"
        toks = sorted(glob.glob(f"{odir}/token_ids_part_*.npy"))
        masks = sorted(glob.glob(f"{odir}/labels_mask_*.npy"))
        assert len(toks) == len(masks), f"{odir}: {len(toks)} tok vs {len(masks)} mask shards"
        for t, m in zip(toks, masks):
            shutil.copy(t, f"{args.out_dir}/token_ids_part_{gi:06d}.npy")
            shutil.copy(m, f"{args.out_dir}/labels_mask_part_{gi:06d}.npy")
            gi += 1
        md = json.load(open(f"{odir}/metadata.json"))
        if agg is None:
            agg = dict(md)
        else:
            for k in ("num_instances", "num_dropped", "num_tokens", "num_loss_tokens"):
                agg[k] = agg.get(k, 0) + md.get(k, 0)
            agg["max_example_len"] = max(agg.get("max_example_len", 0), md.get("max_example_len", 0))
            agg["min_example_len"] = min(agg.get("min_example_len", 1 << 60), md.get("min_example_len", 1 << 60))
    json.dump(agg, open(f"{args.out_dir}/metadata.json", "w"), indent=2)
    print(f"MERGED {gi} shards -> {args.out_dir}  num_instances={agg['num_instances']} "
          f"num_dropped={agg['num_dropped']} max_example_len={agg['max_example_len']}", flush=True)
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
