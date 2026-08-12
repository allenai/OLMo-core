#!/bin/bash
# Rebuild the outlier xlong eval rungs (64k-512k) as TRUE SCALE-K.
#
# THE DEFECT. The shipped xlong rungs were built by build_xlong_rungs.py's GENERIC expander:
#     "outlier": {"canonical": ".../outlier_wiki100w_n220_k3_eval_600.jsonl",
#                 "keep_all": True, "pool": "self_nongold"}
# expand_example pads an example with random fillers drawn from ITS OWN non-gold documents, so every
# added doc belongs to a topic that is already present. K never grows. Measured on the shipped
# rungs (25 examples each):
#     n=451(64k) 901(128k) 1802(256k) 3605(512k) 7209(1M) 14419(2M)  ->  K = 25 at EVERY rung
#     docs/topic 18.7 -> 37.4 -> 75.0 -> 150.1 -> 300.2 -> 600.7
# So M is pinned while N grows by 32x. The O(NM) claim REQUIRES M to scale with N; with M fixed
# these rungs measure the FIXED-K variant, which the paper's own Fig-4 dashed lines show behaves
# like an O(N) task with essentially no dense/chunked gap. The 2k-32k rungs (true scale-K,
# K=3/7/13/25) and the 64k+ rungs are therefore different tasks plotted as one curve.
#
# This is the same class of bug build_v2_outlier_ladder.py was written to avoid one level down
# ("the generic build_v2 shrink path is WRONG for outlier") -- the xlong path then used the generic
# EXPANDER anyway.
#
# THE FIX. No new builder needed: build_v2_outlier_ladder.py is scale-K by construction (fill with
# whole articles of U[min-run,max-run] chunks until the rung is full, K emergent) and already takes
# --rungs/--out-tmpl. Pointing it at the xlong doc counts gives K ~ n/9:
#     64k n=451 -> K~50 | 128k n=901 -> K~100 | 256k n=1802 -> K~200 | 512k n=3605 -> K~400
# instead of 25 everywhere. n values mirror the shipped rungs exactly so old and new sit on one
# x-axis and the difference is attributable to K alone.
#
# Written to a NEW root; the shipped rungs are left in place for comparison rather than overwritten.
#SBATCH --job-name=outlier-xlong-scalek
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=16:00:00
#SBATCH --nodelist=mooney
#SBATCH --output=/data/prasann/joblogs/outlier_xlong_scalek_%j.log
#SBATCH --chdir=/data/prasann
set -uo pipefail

export HOME=/data/prasann/home
export HF_HOME=/data/prasann/hf_cache
export TMPDIR=/data/prasann/tmp
mkdir -p "$HOME" "$TMPDIR" /data/prasann/joblogs

GENSRC=/data/prasann/repo/OLMo-core/src
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-eval/bin/python
POOL=/data/prasann/single_task_ladders_20k/wiki100w_article_pool_smoke.pkl
OUT=/data/prasann/outlier_xlong_scalek
mkdir -p "$OUT"

echo "=== [outlier-xlong-scalek] HOST=$(hostname) START=$(date '+%F %T') ==="
ls -la "$POOL" || { echo "!!! article pool missing: $POOL"; exit 1; }
df -h /data | tail -1

PYTHONPATH="$GENSRC" $PY "$GENSRC/corpus_reasoning/data/build_v2_outlier_ladder.py" \
  --num-examples 600 \
  --num-outliers 3 \
  --min-run 5 --max-run 14 \
  --maj-outlier-gap 2 \
  --rungs "451:64k,901:128k,1802:256k,3605:512k" \
  --out-tmpl "outlier_wiki100w_n{n}_k3_eval_xlong_{lab}.jsonl" \
  --pool-cache "$POOL" \
  --out-root "$OUT" || { echo "!!! build FAILED"; exit 1; }

echo ""
echo "=== VERIFY: K scales with n, AND every majority topic beats the outlier by >=2 ==="
PYTHONPATH="$GENSRC" $PY - <<'PYEOF'
import json, glob, os, statistics
SHIPPED = {451:25, 901:25, 1802:25, 3605:25}   # measured on the shipped rungs
print(f"{'file':<58} {'rows':>5} {'n':>6} {'K_med':>6} {'K_range':>10} {'docs/topic':>11} {'shipped_K':>10}")
for f in sorted(glob.glob("/data/prasann/outlier_xlong_scalek/**/*xlong*.jsonl", recursive=True),
                key=lambda p: int(__import__("re").search(r"_n(\d+)_", p).group(1))):
    rows = [json.loads(l) for i, l in enumerate(open(f)) if i < 40]
    ks = [r["meta"]["num_categories"] for r in rows if (r.get("meta") or {}).get("num_categories")]
    nd = [len(r["documents"]) for r in rows]
    if not ks: continue
    n = statistics.median(nd); k = statistics.median(ks)
    # invariant check: smallest majority topic vs the 3-doc outlier
    worst = None
    for r in rows[:20]:
        cd = (r.get("meta") or {}).get("category_distribution") or {}
        minor = (r.get("meta") or {}).get("minority_label")
        maj = [c for t, c in cd.items() if t != minor]
        if maj: worst = min(maj) if worst is None else min(worst, min(maj))
    gapok = "-" if worst is None else f"min_maj={worst} gap={worst-3}"
    print(f"{os.path.basename(f):<58} {len(rows):>5} {n:>6.0f} {k:>6} "
          f"{f'{min(ks)}-{max(ks)}':>10} {(n-3)/(k-1):>11.1f} {SHIPPED.get(int(n),'-'):>10}  {gapok}")
print("\nbase ladder for reference: n=22/55/110/220 -> K=3/7/13/25, docs/topic ~9 throughout")
PYEOF
ls -la "$OUT"
echo "=== [outlier-xlong-scalek] DONE $(date '+%F %T') ==="
