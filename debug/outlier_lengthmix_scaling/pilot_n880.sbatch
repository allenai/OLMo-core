#!/bin/bash
# PILOT for the 128k transfer test: measure tokens/example and generation wall-clock at n=880 docs.
# Generates 10 train-style examples + 5 eval-ladder examples at n=880, tokenizes with the CTC-suite
# Qwen3.5 tokenizer at max-seq-len 200000 (deliberately over 140k so NOTHING is skipped and we can
# see the true length distribution), and prints ndocs / num_categories / token stats.
#SBATCH --job-name=outlier-n880-pilot
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --nodelist=mooney
#SBATCH --output=/data/prasann/joblogs/outlier_n880_pilot_%j.log
#SBATCH --chdir=/data/prasann
set -uo pipefail
export HOME=/data/prasann/home HF_HOME=/data/prasann/hf_cache TMPDIR=/data/prasann/tmp
mkdir -p "$HOME" "$TMPDIR" /data/prasann/joblogs
GENSRC=/data/prasann/repo/OLMo-core/src
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-eval/bin/python
WORK=/data/prasann/outlier_lengthmix
POOL=/data/prasann/single_task_ladders_20k/wiki100w_article_pool_smoke.pkl
PILOT="$WORK/pilot_n880"
mkdir -p "$PILOT"

echo "=== [pilot n880] HOST=$(hostname) START=$(date '+%F %T') ==="
df -h /data | tail -1

# ---- 1. train-side generator at n=880 (10 examples), timed ----
echo "--- [1] generate 10 train examples n=880 $(date '+%T') ---"
T0=$(date +%s)
PYTHONPATH="$GENSRC" $PY "$GENSRC/corpus_reasoning/data/generate_wiki_outlier_data.py" \
  --num-examples 10 --min-docs 880 --max-docs 880 --num-outliers 3 \
  --majority-mode articles --min-run 4 --max-run 14 \
  --pool-cache "$POOL" --eval-frac 0 --seed 301 \
  --out-dir "$PILOT" --out-name outlier_lm_n880_pilot10
RC=$?
T1=$(date +%s)
echo "[timing] train-gen 10 examples rc=$RC elapsed=$((T1-T0))s"
[ $RC -ne 0 ] && { echo "!!! train gen FAILED"; exit 1; }

# ---- 2. eval-ladder builder at n=880 (5 examples), timed ----
echo "--- [2] build 5 eval-ladder examples n=880 $(date '+%T') ---"
T0=$(date +%s)
PYTHONPATH="$GENSRC" $PY "$GENSRC/corpus_reasoning/data/build_v2_outlier_ladder.py" \
  --num-examples 5 --pool-cache "$POOL" --rungs "880:128k" \
  --out-root "$PILOT/ladder"
RC=$?
T1=$(date +%s)
echo "[timing] eval-ladder 5 examples rc=$RC elapsed=$((T1-T0))s"
[ $RC -ne 0 ] && { echo "!!! eval ladder FAILED"; exit 1; }

# ---- 3. tokenize both with a DELIBERATELY oversized max-seq-len (no skipping) ----
for TAG in train eval; do
  if [ "$TAG" = train ]; then SRC="$PILOT/outlier_lm_n880_pilot10.jsonl";
  else SRC="$PILOT/ladder/outlier/outlier_wiki100w_n880_k3_eval_600.jsonl"; fi
  echo "--- [3.$TAG] tokenize $SRC $(date '+%T') ---"
  OUT="$PILOT/tok_$TAG"; mkdir -p "$OUT"
  PYTHONPATH="$GENSRC" $PY "$GENSRC/corpus_reasoning/data/convert_unified_to_sft.py" \
    --task outlier --input "$SRC" --out-dir "$OUT" \
    --tokenizer Qwen/Qwen3.5-0.8B-Base --max-seq-len 200000 --query-position after \
    --eos 248044 || echo "!!! tokenize FAILED $TAG"
done

# ---- 4. stats ----
$PY - <<'PYEOF'
import json, pathlib, statistics
p = pathlib.Path("/data/prasann/outlier_lengthmix/pilot_n880")
for tag, src in [("train", p/"outlier_lm_n880_pilot10.jsonl"),
                 ("eval",  p/"ladder/outlier/outlier_wiki100w_n880_k3_eval_600.jsonl")]:
    if not src.exists():
        print(f"[{tag}] MISSING {src}"); continue
    nd, nk, nchars = [], [], []
    for line in open(src):
        d = json.loads(line)
        nd.append(len(d["documents"]))
        nk.append(d["meta"].get("num_categories"))
        nchars.append(sum(len(x["text"]) for x in d["documents"]))
    print(f"[{tag}] rows={len(nd)} ndocs min/med/max={min(nd)}/{statistics.median(nd)}/{max(nd)} "
          f"K min/med/max={min(nk)}/{statistics.median(nk)}/{max(nk)} "
          f"chars/ex med={statistics.median(nchars):,.0f}")
    m = p/f"tok_{tag}"/"metadata.json"
    if m.exists():
        md = json.load(open(m))
        print(f"[{tag}] TOKENS: n={md['num_instances']} skipped={md['num_skipped']} "
              f"median_len={md['median_len']:,} max_len={md['max_len']:,} "
              f"mean={md['num_tokens']/max(md['num_instances'],1):,.0f} "
              f"tok/doc={md['median_len']/statistics.median(nd):.1f}")
PYEOF
echo "=== [pilot n880] DONE $(date '+%F %T') ==="
