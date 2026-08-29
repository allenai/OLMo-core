#!/bin/bash
# 128k transfer test: n=880 train pool + held-out CE shard + 600-example eval rung + 4 tokenized
# arms, staged to S3.
#
# Pilot (job 3483938) measured at n=880: median 128,335 tok/ex (145.8 tok/doc), max 130,820,
# gen 3.3 ex/s, K_med 98. So n=880 IS the 128k rung; no adjustment needed.
#
# Tokenisation is the bottleneck (~12k tok/s single-process on a 128k-token example), so every
# arm is split into line-chunks, tokenised 8-way in parallel, and the RAW binary outputs are
# `cat`-concatenated back into a single part. convert_unified_to_sft.py writes with
# ndarray.tofile(), i.e. headerless raw uint32 / raw bool -- concatenation is byte-exact.
#
# The interpreter is deliberately the same NFS corpus-reasoning-eval env that produced every
# existing arm: a different transformers version could render the chat template differently and
# silently make these shards non-comparable with the 2k-32k arms.
#SBATCH --job-name=outlier-128k-build
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --nodelist=mooney
#SBATCH --output=/data/prasann/joblogs/outlier_128k_build_%j.log
#SBATCH --chdir=/data/prasann
set -uo pipefail
export HOME=/data/prasann/home HF_HOME=/data/prasann/hf_cache TMPDIR=/data/prasann/tmp
mkdir -p "$HOME" "$TMPDIR" /data/prasann/joblogs
GENSRC=/data/prasann/repo/OLMo-core/src
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-eval/bin/python
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
WORK=/data/prasann/outlier_lengthmix
POOL=/data/prasann/single_task_ladders_20k/wiki100w_article_pool_smoke.pkl
N880_MEDIAN=128335
CHUNKDIR="$TMPDIR/outlier128k_chunks"
NPAR=8

echo "=== [128k build] HOST=$(hostname) START=$(date '+%F %T') ==="
df -h /data | tail -1

# ---- 1. n=880 pool: 1900 examples -> train 1600 / heldout 300 ----
if [ ! -s "$WORK/outlier_lm_n880_k3_1900.jsonl" ]; then
  echo "--- [1] generate 1900 examples at n=880 $(date '+%T') ---"
  T0=$(date +%s)
  PYTHONPATH="$GENSRC" $PY "$GENSRC/corpus_reasoning/data/generate_wiki_outlier_data.py" \
    --num-examples 1900 --min-docs 880 --max-docs 880 --num-outliers 3 \
    --majority-mode articles --min-run 4 --max-run 14 \
    --pool-cache "$POOL" --eval-frac 0 --seed 305 \
    --out-dir "$WORK" --out-name outlier_lm_n880_k3_1900 || { echo "!!! gen FAILED"; exit 1; }
  echo "[timing] n880 gen elapsed=$(( $(date +%s) - T0 ))s"
else
  echo "[skip] n880 pool exists"
fi

$PY - <<'PYEOF' || exit 1
import pathlib
work = pathlib.Path("/data/prasann/outlier_lengthmix")
src = work / "outlier_lm_n880_k3_1900.jsonl"
tr, ho = work / "outlier_lm_n880_train.jsonl", work / "outlier_lm_n880_heldout.jsonl"
if tr.exists() and ho.exists():
    print(f"[skip] n880 split (train={sum(1 for _ in open(tr))})")
else:
    lines = src.read_text().splitlines()
    assert len(lines) >= 1890, f"only {len(lines)} rows"
    tr.write_text("\n".join(lines[:-300]) + "\n")
    ho.write_text("\n".join(lines[-300:]) + "\n")
    print(f"n880 split: train={len(lines)-300} heldout=300 (DISJOINT)")
PYEOF

# ---- 2. eval rung: 600 held-out examples, same builder+pool as every other rung ----
EVALF="$WORK/eval_rungs/outlier128k/outlier/outlier_wiki100w_n880_k3_eval_600.jsonl"
if [ ! -s "$EVALF" ]; then
  echo "--- [2] build 600-example eval rung at n=880 $(date '+%T') ---"
  T0=$(date +%s)
  PYTHONPATH="$GENSRC" $PY "$GENSRC/corpus_reasoning/data/build_v2_outlier_ladder.py" \
    --num-examples 600 --pool-cache "$POOL" --rungs "880:128k" \
    --out-root "$WORK/eval_rungs/outlier128k" || { echo "!!! eval rung FAILED"; exit 1; }
  echo "[timing] eval rung elapsed=$(( $(date +%s) - T0 ))s"
fi
# canonical native-eval location + ctc-suite rung naming
mkdir -p "$WORK/eval500_local/outlier/outlier" "$WORK/eval_rungs/outlier"
cp -f "$EVALF" "$WORK/eval500_local/outlier/outlier/outlier_wiki100w_n880_k3_eval_600.jsonl"
cp -f "$EVALF" "$WORK/eval_rungs/outlier/rung_131072.jsonl"
wc -l "$EVALF" "$WORK/eval500_local/outlier/outlier/outlier_wiki100w_n880_k3_eval_600.jsonl"

# ---- 3. compose the four new arms (jsonl space, nested prefixes, seed 7113) ----
echo "--- [3] compose arms $(date '+%T') ---"
$PY "$REPO/debug/outlier_lengthmix_scaling/compose_arms_128k.py" \
  --work "$WORK" --n880-median "$N880_MEDIAN" || { echo "!!! compose FAILED"; exit 1; }

# ---- 4. chunked parallel tokenisation ----
# tokchunks <input.jsonl> <out-dir> <n-chunks>
tokchunks () {
  local SRC=$1 OUT=$2 NC=$3 NAME
  NAME=$(basename "$OUT")
  [ -s "$OUT/metadata.json" ] && { echo "[skip] tok $NAME"; return 0; }
  local CD="$CHUNKDIR/$NAME"
  rm -rf "$CD"; mkdir -p "$CD" "$OUT"
  split -n "l/$NC" -d -a 3 "$SRC" "$CD/chunk_"
  echo "--- [4] tokenising $NAME: $(ls "$CD"/chunk_* | wc -l) chunks $(date '+%T') ---"
  local i=0
  for C in "$CD"/chunk_*; do
    while [ "$(jobs -rp | wc -l)" -ge "$NPAR" ]; do sleep 5; done
    (
      PYTHONPATH="$GENSRC" $PY "$GENSRC/corpus_reasoning/data/convert_unified_to_sft.py" \
        --task outlier --input "$C" --out-dir "${C}_tok" \
        --tokenizer Qwen/Qwen3.5-0.8B-Base --max-seq-len 140000 --query-position after \
        --eos 248044 >"${C}.log" 2>&1 \
        || { echo "!!! chunk FAILED $NAME $(basename "$C")"; touch "$CD/FAILED"; }
    ) &
    i=$((i+1))
  done
  wait
  [ -f "$CD/FAILED" ] && { echo "!!! tokenise FAILED for $NAME"; touch "$OUT/FAILED"; return 1; }
  # raw-binary concat back to a single part (tofile() output is headerless)
  cat "$CD"/chunk_*_tok/token_ids_part_000000.npy > "$OUT/token_ids_part_000000.npy" || return 1
  cat "$CD"/chunk_*_tok/labels_mask_000000.npy    > "$OUT/labels_mask_000000.npy"    || return 1
  $PY - "$CD" "$OUT" "$SRC" <<'PYEOF' || return 1
import glob, json, os, sys
cd, out, src = sys.argv[1], sys.argv[2], sys.argv[3]
metas = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(cd, "chunk_*_tok/metadata.json")))]
m = dict(metas[0])
m.update(input=src, num_instances=sum(x["num_instances"] for x in metas),
         num_skipped=sum(x["num_skipped"] for x in metas),
         num_tokens=sum(x["num_tokens"] for x in metas),
         num_loss_tokens=sum(x["num_loss_tokens"] for x in metas),
         max_len=max(x["max_len"] for x in metas), n_chunks=len(metas))
m.pop("median_len", None)   # recomputed exactly in the verify pass
json.dump(m, open(os.path.join(out, "metadata.json"), "w"), indent=2)
print("MERGED:", json.dumps(m))
PYEOF
  rm -rf "$CD"
}

tokchunks "$WORK/arms/p128k_500.jsonl"   "$WORK/arms_tokenized/p128k_500"  8
tokchunks "$WORK/arms/p128k_1500.jsonl"  "$WORK/arms_tokenized/p128k_1500" 16
tokchunks "$WORK/arms/mix_sx64M.jsonl"   "$WORK/arms_tokenized/mix_sx64M"  8
tokchunks "$WORK/arms/mix_sx200M.jsonl"  "$WORK/arms_tokenized/mix_sx200M" 16
tokchunks "$WORK/outlier_lm_n880_heldout.jsonl" "$WORK/tokenized/n880_heldout" 4
ls "$WORK"/arms_tokenized/*/FAILED "$WORK"/tokenized/*/FAILED 2>/dev/null && { echo "!!! FAILURES"; exit 1; }

# ---- 5. verify: exact length distribution, truncation check, seq-len feasibility ----
echo "--- [5] verify $(date '+%T') ---"
$PY - <<'PYEOF' || exit 1
import json, pathlib
import numpy as np
EOS = 248044
work = pathlib.Path("/data/prasann/outlier_lengthmix")
targets = [("arms_tokenized/p128k_500", 500), ("arms_tokenized/p128k_1500", 1500),
           ("arms_tokenized/mix_sx64M", None), ("arms_tokenized/mix_sx200M", None),
           ("tokenized/n880_heldout", 300)]
report = {}
ok = True
for rel, expect in targets:
    d = work / rel
    md = json.load(open(d / "metadata.json"))
    ids = np.fromfile(d / "token_ids_part_000000.npy", dtype=np.uint32)
    mask = np.fromfile(d / "labels_mask_000000.npy", dtype=bool)
    ends = np.flatnonzero(ids == EOS)
    lens = np.diff(np.concatenate(([-1], ends)))
    row = {
        "n_instances": int(md["num_instances"]), "num_skipped": int(md["num_skipped"]),
        "num_tokens": int(ids.size), "num_loss_tokens": int(mask.sum()),
        "len_median": int(np.median(lens)), "len_max": int(lens.max()),
        "len_p99": int(np.percentile(lens, 99)),
        "over_131072": int((lens > 131072).sum()), "over_140000": int((lens > 140000).sum()),
        "eos_delimited_instances": int(lens.size),
    }
    report[d.name] = row
    if md["num_skipped"] != 0:
        print(f"!!! {d.name}: {md['num_skipped']} instances SKIPPED (over max-seq-len 140000)")
        ok = False
    if ids.size != mask.size:
        print(f"!!! {d.name}: ids/mask length mismatch {ids.size} vs {mask.size}"); ok = False
    if lens.size != md["num_instances"]:
        print(f"!!! {d.name}: EOS count {lens.size} != num_instances {md['num_instances']}"); ok = False
    if expect is not None and md["num_instances"] != expect:
        print(f"!!! {d.name}: expected {expect} instances, got {md['num_instances']}"); ok = False
    print(f"[{d.name}] " + json.dumps(row))
(work / "REPORT_128k.json").write_text(json.dumps(report, indent=2))
print("\nTRUNCATION CHECK:", "PASS (0 skipped everywhere)" if ok else "FAIL")
tot = sum(r["over_131072"] for r in report.values())
print(f"SEQ-LEN NOTE: {tot} instances exceed 131072 tokens across all new shards "
      f"(these cannot fit a seq_len=131072 training window).")
PYEOF

# ---- 6. stage to S3 ----
echo "--- [6] S3 sync $(date '+%T') ---"
export AWS_PROFILE=S3
DST=s3://ai2-llm/checkpoints/prasanns/outlier_lengthmix
for A in p128k_500 p128k_1500 mix_sx64M mix_sx200M; do
  aws s3 sync "$WORK/arms_tokenized/$A" "$DST/arms/$A" --only-show-errors \
    || { echo "!!! s3 sync FAILED $A"; exit 1; }
  echo "[s3] $DST/arms/$A"
done
aws s3 sync "$WORK/tokenized/n880_heldout" "$DST/heldout/n880" --only-show-errors || exit 1
aws s3 cp "$EVALF" "$DST/eval_rungs/outlier128k/outlier_wiki100w_n880_k3_eval_600.jsonl" \
  --only-show-errors || exit 1
aws s3 cp "$WORK/arms/MANIFEST_128k.json" "$DST/arms/MANIFEST_128k.json" --only-show-errors || exit 1
aws s3 cp "$WORK/REPORT_128k.json" "$DST/arms/REPORT_128k.json" --only-show-errors || exit 1
echo "[s3] $DST/eval_rungs/outlier128k/"

du -sh "$WORK"/arms_tokenized/p128k_500 "$WORK"/arms_tokenized/p128k_1500 \
       "$WORK"/arms_tokenized/mix_sx64M "$WORK"/arms_tokenized/mix_sx200M \
       "$WORK"/tokenized/n880_heldout
df -h /data | tail -1
echo "=== [128k build] DONE $(date '+%F %T') ==="
