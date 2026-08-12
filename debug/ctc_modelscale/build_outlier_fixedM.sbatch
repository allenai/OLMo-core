#!/bin/bash
# Build the OUTLIER FIXED-M row: 20k training examples + 5 eval rungs with the category count M
# pinned, as the control for the suite's existing scale-K outlier row.
#
# Pilot (job 3440285) verified the pin holds and the pool can supply it:
#     n band      ex   n med   K med   K range   docs/topic
#     (0,20)      10      16     3.0    (3,3)          6.5
#     (20,40)     42      28     3.0    (3,3)         12.8
#     (40,80)     69      61     3.0    (3,3)         29.0
#     (80,160)   156     120     3.0    (3,3)         58.8
#     (160,999)  123     193     3.0    (3,3)         95.0
# K constant, docs-per-topic growing with n -- M fixed while N scales, which is the whole point.
#
# K=3 is the scale-K ladder's value at its SMALLEST rung (measured EVAL_K in
# debug/outlier_iid_rebuild/verify_outlier_iid.py: n=22 -> K=3, n=55 -> 7, n=110 -> 13, n=220 -> 25).
# Pinning there means the fixed-M and scale-K curves START together and diverge only in whether M
# grows -- the comparison the paper draws as dashed vs solid lines in Fig 4, and plan §6's T3.
#
# Eval rungs reuse the scale-K ladder's doc counts (n = 14/28/57/111/220, the measured medians of
# eval_rungs/outlier/rung_*.jsonl) and its rung labels, so both rows sit on one x-axis with N
# matched rung-for-rung.
#
#   sbatch debug/ctc_modelscale/build_outlier_fixedM.sbatch
#SBATCH --job-name=outlier-fixedM-build
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=sneetches
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=/data/prasann/joblogs/outlier_fixedM_build_%j.log
#SBATCH --chdir=/data/prasann

set -uo pipefail
export HOME=/data/prasann/home
export HF_HOME=/data/prasann/hf_cache
export TMPDIR=/data/prasann/tmp
mkdir -p "$HOME" "$TMPDIR" /data/prasann/joblogs

REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
GENSRC=$REPO/src
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
GEN="$GENSRC/corpus_reasoning/data/generate_wiki_outlier_data.py"
WORK=/data/prasann/outlier_fixedM
POOL=/data/prasann/single_task_ladders_20k/wiki100w_article_pool_smoke.pkl
RUNGDIR=/scratch/users/prasann/ctc_suite_staged/eval_rungs_fixedM/outlier_fixedM
SHARD=/data/prasann/ctc_suite/shards_final/outlier_fixedM_train
K=3
mkdir -p "$WORK" "$RUNGDIR"

echo "=== HOST=$(hostname) K=$K START=$(date '+%F %T') ==="
ls -la "$POOL" || { echo "!!! pool missing"; exit 1; }
PINK=(--majority-mode partition --simple-ratio 0
      --mixed-min-k $K --mixed-max-k $K --mixed-max-k-cap $K
      --num-outliers 3 --pool-cache "$POOL" --eval-frac 0)

echo "=== 1. eval rungs (500 each, fixed n per rung, seed 9000+) ==="
# rung label : docs -- doc counts are the measured medians of the scale-K ladder's rungs, so the
# two rows are matched on N at every rung.
RUNGS=("2048:14" "4096:28" "8192:57" "16384:111" "32768:220")
for spec in "${RUNGS[@]}"; do
  LAB="${spec%%:*}"; N="${spec##*:}"
  if [ -f "$RUNGDIR/rung_${LAB}.jsonl" ]; then echo "  [skip] rung_${LAB} exists"; continue; fi
  echo "  --- rung $LAB (n=$N) ---"
  PYTHONPATH="$GENSRC" $PY "$GEN" --num-examples 500 --min-docs "$N" --max-docs "$N" \
    "${PINK[@]}" --seed $(( 9000 + LAB % 1000 )) \
    --out-dir "$WORK/eval" --out-name "outlier_fixedM_k${K}_n${N}_500" || exit 1
  cp "$WORK/eval/outlier_fixedM_k${K}_n${N}_500.jsonl" "$RUNGDIR/rung_${LAB}.jsonl"
done
wc -l "$RUNGDIR"/rung_*.jsonl

echo "=== 2. train pool (20k, n ~ U[14,220], seed 42) ==="
if [ ! -f "$WORK/outlier_fixedM_k${K}_20000.jsonl" ]; then
  PYTHONPATH="$GENSRC" $PY "$GEN" --num-examples 20000 --min-docs 14 --max-docs 220 \
    "${PINK[@]}" --seed 42 --out-dir "$WORK" --out-name "outlier_fixedM_k${K}_20000" || exit 1
fi

echo "=== 3. decontaminate train against the fixed-M eval rungs ==="
$PY - <<PY
import glob, json, os
def sig(ex):
    docs = ex.get("documents") or []
    body = "".join((d.get("text","") if isinstance(d,dict) else str(d))[:60] for d in docs[:5])
    return str(ex.get("answers")) + "||" + body

keys = set()
for p in sorted(glob.glob("$RUNGDIR/rung_*.jsonl")):
    n = 0
    for line in open(p):
        if line.strip():
            keys.add(sig(json.loads(line))); n += 1
    print(f"  {os.path.basename(p)}: {n}")
print(f"  eval signature set: {len(keys)}")
kept = dropped = 0
with open("$WORK/outlier_fixedM_k${K}_20000_clean.jsonl", "w") as w:
    for line in open("$WORK/outlier_fixedM_k${K}_20000.jsonl"):
        if not line.strip(): continue
        if sig(json.loads(line)) in keys: dropped += 1
        else: w.write(line); kept += 1
print(f"  kept={kept} dropped={dropped} ({100*dropped/max(1,kept+dropped):.2f}%)")
PY
[ $? -ne 0 ] && exit 1

echo "=== 4. tokenize to a CTC-suite shard ==="
rm -rf "$SHARD"; mkdir -p "$SHARD"
PYTHONPATH="$GENSRC" $PY "$REPO/src/scripts/data/convert_unified_to_document_landmark.py" \
  --emit dense --task outlier --chunk-by document --cot-mode none \
  --marker-set qwen3_5 --tokenizer Qwen/Qwen3.5-0.8B-Base \
  --seq-len 40960 --shard-tokens 60000000 \
  --input-jsonl "$WORK/outlier_fixedM_k${K}_20000_clean.jsonl" --out-dir "$SHARD"
rc=$?; echo "  convert rc=$rc"; [ $rc -ne 0 ] && exit $rc

echo "=== 5. part/byte consistency ==="
$PY - <<PY
import glob, json, os
d = "$SHARD"
tok = sorted(glob.glob(os.path.join(d,"token_ids_part_*.npy")))
lab = sorted(glob.glob(os.path.join(d,"labels_mask_part_*.npy")))
m = json.load(open(os.path.join(d,"metadata.json")))
tb = sum(os.path.getsize(p) for p in tok); lb = sum(os.path.getsize(p) for p in lab)
print(f"token parts={len(tok)} bytes={tb}; label parts={len(lab)} bytes={lb}")
print(f"num_instances={m['num_instances']} max_example_len={m['max_example_len']}")
print("VERDICT:", "PASS" if (len(tok)==len(lab) and tb==m["num_tokens"]*4
                             and lb==m["num_tokens"]) else "FAIL -- truncated")
PY
echo "=== DONE $(date '+%F %T') ==="
