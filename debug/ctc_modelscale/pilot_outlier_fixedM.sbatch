#!/bin/bash
# PILOT: outlier with M (category count) held CONSTANT while N grows -- the fixed-M control the
# CTC suite never built (plan §6 criterion T3: "fixed-M outlier gap stays flat while scale-k
# widens"), and the paper's dashed lines in Fig 4 ("we ablate keeping M constant ... and find it
# behaves like an O(N) task in that setting").
#
# THE CONTRAST. The suite's existing outlier row is scale-K: `--majority-mode articles` fills each
# corpus with WHOLE articles of U[4,14] chunks, so the number of majority topics EMERGES and grows
# with n. Measured on the v2 eval ladder (debug/outlier_iid_rebuild/verify_outlier_iid.py):
#       n= 22 -> K=3 (3-5)    n= 55 -> K=7 (5-11)
#       n=110 -> K=13 (10-17) n=220 -> K=25 (23-28)
# The fixed-M arm pins K at the SMALLEST rung's value (K=3) for every n, so both curves start at
# the same point and differ only in whether M scales. That is what makes the pair readable as one
# figure.
#
# WHY A PILOT FIRST. Pinning K=3 at n=220 means ~108 chunks per majority topic, and the pool's
# articles are mostly far shorter (`--min-article-chunks 4`, `--chunks-per-article` default 25).
# The generator may simply be unable to build those examples -- the same class of silent-capacity
# problem as the oolong packer. `--max-build-retries` masks it as slowness, not an error. So:
# 400 examples across the full n range, then VERIFY realized K and n before spending the 20k build.
#
#   sbatch debug/ctc_modelscale/pilot_outlier_fixedM.sbatch
#SBATCH --job-name=outlier-fixedM-pilot
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=sneetches
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --output=/data/prasann/joblogs/outlier_fixedM_pilot_%j.log
#SBATCH --chdir=/data/prasann

set -uo pipefail
export HOME=/data/prasann/home
export HF_HOME=/data/prasann/hf_cache
export TMPDIR=/data/prasann/tmp
mkdir -p "$HOME" "$TMPDIR" /data/prasann/joblogs

REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
GENSRC=$REPO/src
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
WORK=/data/prasann/outlier_fixedM
# The EVAL ladder's pool -- using the train pool instead is one of the two drifts that made the
# original outlier train/eval non-iid (records/contradiction-train-eval-non-iid.md §2).
POOL=/data/prasann/single_task_ladders_20k/wiki100w_article_pool_smoke.pkl
FIXED_K="${FIXED_K:-3}"
mkdir -p "$WORK"

echo "=== [outlier-fixedM-pilot] HOST=$(hostname) FIXED_K=$FIXED_K START=$(date '+%F %T') ==="
ls -la "$POOL" || { echo "!!! eval article pool not staged at $POOL"; exit 1; }

# --mixed-min-k = --mixed-max-k = --mixed-max-k-cap = K pins the category count three ways: the
# sampled draw, and the auto-grow that would otherwise raise K with num_docs.
# --simple-ratio 0: simple mode is a single majority article (K=2) and would contaminate the pin.
PYTHONPATH="$GENSRC" $PY "$GENSRC/corpus_reasoning/data/generate_wiki_outlier_data.py" \
  --num-examples 400 \
  --min-docs 14 --max-docs 220 \
  --num-outliers 3 \
  --majority-mode partition \
  --simple-ratio 0 \
  --mixed-min-k "$FIXED_K" --mixed-max-k "$FIXED_K" --mixed-max-k-cap "$FIXED_K" \
  --pool-cache "$POOL" \
  --eval-frac 0 \
  --seed 42 \
  --out-dir "$WORK" \
  --out-name outlier_fixedM_k${FIXED_K}_pilot400
rc=$?
echo "  generation rc=$rc $(date '+%T')"
[ $rc -ne 0 ] && { echo "!!! generation FAILED"; exit $rc; }

echo "=== VERIFY: is K actually constant, and does n still span the ladder? ==="
PYTHONPATH="$GENSRC" $PY - <<PY
import json, statistics, collections
p = "$WORK/outlier_fixedM_k${FIXED_K}_pilot400.jsonl"
rows = [json.loads(l) for l in open(p)]
print(f"examples: {len(rows)}")
bands = [(0,20),(20,40),(40,80),(80,160),(160,999)]
by = collections.defaultdict(list)
for ex in rows:
    n = len(ex["documents"])
    k = (ex.get("meta") or {}).get("num_categories")
    for lo, hi in bands:
        if lo <= n < hi:
            by[(lo,hi)].append((n, k)); break
print(f"{'n band':>12s} {'ex':>5s} {'n med':>7s} {'K med':>7s} {'K range':>10s} {'docs/topic':>11s}")
bad = []
for b in bands:
    v = by.get(b) or []
    if not v:
        print(f"{str(b):>12s} {'0':>5s}   -- none"); continue
    ns = [x for x,_ in v]; ks = [y for _,y in v if y is not None]
    if not ks:
        print(f"{str(b):>12s} {len(v):5d} {statistics.median(ns):7.0f}   K NOT RECORDED"); continue
    kmed = statistics.median(ks)
    dpt = (statistics.median(ns) - 3) / max(1, kmed - 1)
    print(f"{str(b):>12s} {len(v):5d} {statistics.median(ns):7.0f} {kmed:7.1f} "
          f"{str((min(ks),max(ks))):>10s} {dpt:11.1f}")
    if kmed != $FIXED_K:
        bad.append((b, kmed))
print()
if bad:
    print("VERDICT: FAIL -- K drifted from the pin in bands: " + str(bad))
    print("  K rising with n means the auto-grow cap did not hold; K falling means the pool could")
    print("  not supply enough chunks per topic at that n (the capacity limit this pilot tests).")
else:
    print("VERDICT: PASS -- K constant at $FIXED_K across the whole n range; safe to build 20k")
PY
echo "=== DONE $(date '+%F %T') ==="
