#!/bin/bash
# Outlier TRAINING pool for the xlong5 2k->256k roots, scale-K throughout.
#
# TARGET: replaces shards_full/outlier_train in
#   /weka/oe-training-default/ai2-llm/checkpoints/prasanns/xlong5_2k256k_qwen35_qafter/
# (outlier is query-position agnostic per that root's README -- "outlier is a deliberate no-op",
# its qafter and both arms are the same data -- so this one build serves both roots.)
#
# WHY NOT BAND THE OLD POOL. gen_banded_expand.py grows examples with build_xlong_rungs'
# expand_example, whose outlier entry uses pool="self_nongold": fillers come from the example's OWN
# non-gold docs, so added documents join topics that already exist and K never grows. That is the
# exact defect just found in the outlier xlong EVAL rungs (K pinned at 25 from n=451 to n=14419).
# Banding through it would have frozen K in TRAINING too -- and because the eval had the same
# freeze, train and eval would have agreed while both measured the wrong task. Undetectable.
#
# INSTEAD: generate directly at every band with --majority-mode articles, which fills with whole
# articles of U[min-run,max-run] chunks and lets K emerge -- the same rule build_v2_outlier_ladder.py
# uses for the eval. K then scales ~n/9.5 at every length, matching the rebuilt scale-K eval
# (64k K=48, 128k K=96, 256k K=191, 512k K=378, docs/topic ~9.5 throughout).
#
# BANDS from debug/xlong_5task/bands.py with outlier's calibration tokens = -5.4 + 144.328*n_docs:
#   2-4k n=14-28 x6000 | 4-8k 28-57 x5000 | 8-16k 57-114 x4000 | 16-32k 114-227 x2700
#   32-64k 227-454 x1400 | 64-128k 454-908 x600 | 128-256k 908-1816 x300   (= 20,000 examples)
# Short-skewed exactly as the original build, so only the K structure changes.
#
# --maj-outlier-gap 2: every majority topic exceeds the 3-doc outlier by >=2, so "outlier is the
# rarest topic" has real margin instead of the historical 1-document margin.
#SBATCH --job-name=outlier-xlong5-train
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=20:00:00
#SBATCH --nodelist=mooney
#SBATCH --output=/data/prasann/joblogs/outlier_xlong5_train_%j.log
#SBATCH --chdir=/data/prasann
set -uo pipefail

export HOME=/data/prasann/home
export HF_HOME=/data/prasann/hf_cache
export TMPDIR=/data/prasann/tmp
mkdir -p "$HOME" "$TMPDIR" /data/prasann/joblogs

GENSRC=/data/prasann/repo/OLMo-core/src
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-eval/bin/python
POOL=/data/prasann/single_task_ladders_20k/wiki100w_article_pool_smoke.pkl
OUT=/data/prasann/outlier_xlong5_train
mkdir -p "$OUT/parts"

echo "=== [outlier-xlong5-train] HOST=$(hostname) START=$(date '+%F %T') ==="
ls -la "$POOL" || { echo "!!! pool missing"; exit 1; }
df -h /data | tail -1

# band: lo_n hi_n n_examples
BANDS="14:28:6000 28:57:5000 57:114:4000 114:227:2700 227:454:1400 454:908:600 908:1816:300"
for b in $BANDS; do
  LO="${b%%:*}"; rest="${b#*:}"; HI="${rest%%:*}"; NEX="${rest##*:}"
  NAME="outlier_band_n${LO}-${HI}"
  if [ -f "$OUT/parts/${NAME}.jsonl" ]; then echo "[skip] $NAME"; continue; fi
  echo "[band] n=$LO-$HI  examples=$NEX  $(date '+%F %T')"
  PYTHONPATH="$GENSRC" $PY "$GENSRC/corpus_reasoning/data/generate_wiki_outlier_data.py" \
    --num-examples "$NEX" \
    --min-docs "$LO" --max-docs "$HI" \
    --num-outliers 3 \
    --majority-mode articles \
    --min-run 5 --max-run 14 \
    --maj-outlier-gap 2 \
    --pool-cache "$POOL" \
    --eval-frac 0 --seed 42 \
    --out-dir "$OUT/parts" --out-name "$NAME" || { echo "!!! band $LO-$HI FAILED"; exit 1; }
done

echo ""
echo "=== concatenate + shuffle (bands must be interleaved; a sharded consumer must not get one band per shard) ==="
PYTHONPATH="$GENSRC" $PY - <<'PYEOF'
import glob, json, random
parts = sorted(glob.glob("/data/prasann/outlier_xlong5_train/parts/*.jsonl"))
rows = []
for p in parts:
    n = 0
    for line in open(p):
        rows.append(line.rstrip("\n")); n += 1
    print(f"  {p.split('/')[-1]}: {n}")
random.Random(42).shuffle(rows)
out = "/data/prasann/outlier_xlong5_train/outlier_scalek_2k-256k_k3_20000.jsonl"
with open(out, "w") as f:
    for r in rows: f.write(r + "\n")
print(f"\nwrote {len(rows)} examples -> {out}")
PYEOF

echo ""
echo "=== VERIFY: K scales with n, gap>=2, across the whole band range ==="
PYTHONPATH="$GENSRC" $PY - <<'PYEOF'
import json, statistics
from collections import Counter
rows = []
for i, l in enumerate(open("/data/prasann/outlier_xlong5_train/outlier_scalek_2k-256k_k3_20000.jsonl")):
    if i >= 4000: break
    rows.append(json.loads(l))
print(f"{'n band':<14} {'ex':>5} {'K_med':>6} {'K_range':>10} {'docs/topic':>11} {'min_gap':>8}")
for lo, hi in [(14,28),(28,57),(57,114),(114,227),(227,454),(454,908),(908,1816)]:
    sel = [r for r in rows if lo <= len(r["documents"]) <= hi]
    if not sel: print(f"n={lo}-{hi:<9} (none)"); continue
    ks = [r["meta"]["num_categories"] for r in sel]
    nd = statistics.median([len(r["documents"]) for r in sel])
    gaps = []
    for r in sel[:60]:
        cd = r["meta"].get("category_distribution") or {}
        minor = r["meta"].get("minority_label")
        maj = [c for t, c in cd.items() if t != minor]
        if maj: gaps.append(min(maj) - r["meta"].get("num_outliers", 3))
    print(f"n={lo}-{hi:<9} {len(sel):>5} {statistics.median(ks):>6} "
          f"{f'{min(ks)}-{max(ks)}':>10} {(nd-3)/(statistics.median(ks)-1):>11.1f} "
          f"{min(gaps) if gaps else '-':>8}")
print("\neval spec to match: K ~ n/9.5, docs/topic ~9.5, min_gap >= 2")
PYEOF
ls -la "$OUT"
echo "=== [outlier-xlong5-train] DONE $(date '+%F %T') ==="
