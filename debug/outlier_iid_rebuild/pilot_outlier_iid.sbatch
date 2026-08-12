#!/bin/bash
# Outlier training data, rebuilt to be IID with the v2 eval ladder on the M (category) axis.
#
# THE MISMATCH (measured 2026-08-11, see records/contradiction-train-eval-non-iid.md §2).
# `build_v2_outlier_ladder.py` is scale-K: it fills each rung with whole articles, so the number of
# majority topics grows with n. The training file does not. Measured docs per majority topic:
#     eval  n=220: 217 docs over 25 majority topics  ~=  8.7 docs each   (K = 23-28)
#     train n~220:                                   ~= 40 docs each     (K = 2-10, median 6.5)
# A 4.5x difference, on precisely the axis the O(NM) claim rests on. The eval's top rung poses a
# 25-category problem where training's maximum ever observed was 10.
#
# TWO CAUSES, BOTH FIXED HERE.
#  (SUPERSEDED APPROACH, kept for the record: fitting --chunks-per-article 9 to the eval's observed
#   K. Rejected -- re-deriving the eval's structure through a different mechanism is exactly how this
#   bug arose. --majority-mode articles SHARES the eval's fill rule instead.)
#  1. `--chunks-per-article 25` sets max_k = 1 + ceil(n_maj/25). Fitting the eval's K against
#     K = 1 + ceil((n-3)/cpa) over all four rungs gives cpa = 9 (reproduces K=7 and K=13 exactly,
#     lands inside the observed spread at n=22 and n=220). Also raise the cap: the default
#     --mixed-max-k-cap 16 CANNOT express the eval's K=25 at all.
#  2. K is drawn UNIFORMLY: build_mixed_example does `K = rng.randint(k_min, k_max)`, so training
#     gets K ~ U[2, max_k] while the eval sits in a tight band near the top. Flags alone cannot fix
#     this, so `--mixed-min-k-frac` was added to generate_wiki_outlier_data.py (defaults to 0.0 =
#     historical behaviour, so no existing build changes).
#
# Also: --simple-ratio 0. Simple mode is a single majority article (K=2); the eval's minimum K is 3,
# so it contains no such examples, while ~half of training below n=60 were.
#
# POOL. Train used wiki100w_article_pool.pkl (6.2G); the eval builder defaults to
# wiki100w_article_pool_smoke.pkl (92k articles). Different pools = different article-length
# distributions = another axis of drift. This uses the EVAL's pool.
#
# This is the PILOT: 400 examples, verify K(n) and docs-per-majority against the eval before
# spending the full 20k build.
#SBATCH --job-name=outlier-iid-pilot
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --nodelist=mooney
#SBATCH --output=/data/prasann/joblogs/outlier_iid_pilot_%j.log
#SBATCH --chdir=/data/prasann
set -uo pipefail

export HOME=/data/prasann/home
export HF_HOME=/data/prasann/hf_cache
export TMPDIR=/data/prasann/tmp
mkdir -p "$HOME" "$TMPDIR" /data/prasann/joblogs

GENSRC=/data/prasann/repo/OLMo-core/src
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-eval/bin/python
WORK=/data/prasann/outlier_iid
POOL=/data/prasann/single_task_ladders_20k/wiki100w_article_pool_smoke.pkl
mkdir -p "$WORK"

echo "=== [outlier-iid-pilot] HOST=$(hostname) START=$(date '+%F %T') ==="
ls -la "$POOL" || { echo "!!! eval article pool not staged at $POOL"; exit 1; }

PYTHONPATH="$GENSRC" $PY "$GENSRC/corpus_reasoning/data/generate_wiki_outlier_data.py" \
  --num-examples 400 \
  --min-docs 14 --max-docs 220 \
  --num-outliers 3 \
  --majority-mode articles \
  --min-run 4 --max-run 14 \
  --pool-cache "$POOL" \
  --eval-frac 0 \
  --seed 42 \
  --out-dir "$WORK" \
  --out-name outlier_iid_pilot || { echo "!!! generation FAILED"; exit 1; }

echo ""
echo "=== VERIFY: K(n) and docs-per-majority vs the eval ladder ==="
PYTHONPATH="$GENSRC" $PY "$GENSRC/../debug/outlier_iid_rebuild/verify_outlier_iid.py" \
  --train "$WORK/outlier_iid_pilot.jsonl" \
  --out "$WORK/outlier_iid_pilot_audit.json"

echo "=== [outlier-iid-pilot] DONE $(date '+%F %T') ==="
