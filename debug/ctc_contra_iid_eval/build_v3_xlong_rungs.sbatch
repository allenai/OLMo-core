#!/bin/bash
# v3 xlong contradiction rungs: 64k / 128k / 256k / 512k, realistic-mode.
#
# WHY THIS IS A PREREQUISITE. v3 currently stops at 32k (n = 100/190/385/765). The 64k-2M rungs
# exist only for v2 (`contradiction_eval_pubmed_both_n{1525,3051,6102,12204,24408,48816}_k3_xlong_*`),
# so "eval up to 512k on v3" cannot run until the matching realistic-mode rungs exist. n values are
# mirrored EXACTLY from the v2 clean bundle so the two ladders sit on one x-axis.
#
# Expansion is LLM-free and re-seeds per example (seed*1_000_003 + i), so all rungs carry the SAME
# 500 examples as the 2k-32k v3 rungs -- only the filler count grows. The base
# (contra_iid_base500.jsonl) is the disjointness-filtered, --max-overlap 0.5 realistic build:
# Jaccard median 0.324 / p90 0.462 / hard cap 0.500 / 0.0% above, matching training's 0.306.
#
# ⚠ Like v2's own 64k+ rungs, these are NOT comparable to the <=32k rungs as a single curve without
# saying so: bigger n is a different task, not just a longer context.
#
# Sizes to expect (v2 equivalents): 64k ~125MB, 128k ~250MB, 256k ~500MB, 512k ~1.0GB.
#SBATCH --job-name=contra-v3-xlong
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --nodelist=mooney
#SBATCH --output=/data/prasann/joblogs/contra_v3_xlong_%j.log
#SBATCH --chdir=/data/prasann
set -uo pipefail

export HOME=/data/prasann/home
export HF_HOME=/data/prasann/hf_cache
export TMPDIR=/data/prasann/tmp
mkdir -p "$HOME" "$TMPDIR" /data/prasann/joblogs

GENSRC=/data/prasann/repo/OLMo-core/src
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-eval/bin/python
WORK=/data/prasann/contra_iid_eval
BASE="$WORK/raw/contra_iid_base500.jsonl"
OUT="$WORK/v3_xlong"
mkdir -p "$OUT"

echo "=== [contra-v3-xlong] HOST=$(hostname) START=$(date '+%F %T') ==="
[ -f "$BASE" ] || { echo "!!! base missing: $BASE"; exit 1; }
df -h /data | tail -1

# n values mirrored from the v2 clean bundle's xlong contra rungs
for spec in "1525:64k" "3051:128k" "6102:256k" "12204:512k"; do
  N="${spec%%:*}"; LAB="${spec##*:}"
  DST="$OUT/contradiction_eval_pubmed_realistic_n${N}_k3_xlong_${LAB}.jsonl"
  if [ -f "$DST" ]; then echo "[skip] $LAB already built"; continue; fi
  echo "[expand] n=$N ($LAB) $(date '+%F %T')"
  PYTHONPATH="$GENSRC" $PY "$GENSRC/corpus_reasoning/data/generate_pubmed_contradiction_data.py" \
    --expand-from-eval "$BASE" \
    --num-docs "$N" \
    --num-contradictions 3 \
    --mode realistic \
    --pool-abstracts 200000 \
    --seed 42 \
    --filler-pool-seed 43 \
    --output-dir "$OUT" || { echo "!!! expand n=$N FAILED"; continue; }
  SRC="$OUT/contradiction_eval_pubmed_realistic_n${N}_k3.jsonl"
  [ -f "$SRC" ] && mv "$SRC" "$DST"
  ls -la "$DST" 2>/dev/null
done

echo ""
echo "=== audit ==="
PYTHONPATH="$GENSRC" $PY - <<'PYEOF'
import json, re, statistics, glob, os
def toks(s): return set(re.findall(r"[a-z0-9]+", s.lower()))
def jac(a,b):
    A,B=toks(a),toks(b); return len(A&B)/max(1,len(A|B))
base_sig=None
print(f"{'file':<62} {'rows':>5} {'n':>6} {'jac':>6} {'>0.5':>6} {'same500':>8}")
for f in sorted(glob.glob("/data/prasann/contra_iid_eval/v3_xlong/*.jsonl"),
                key=lambda p: int(re.search(r"_n(\d+)_", p).group(1))):
    recs=[json.loads(l) for l in open(f)]
    ov=[]; sig=[]
    for r in recs:
        d=[x["text"] if isinstance(x,dict) else x for x in r["documents"]]
        sig.append(frozenset(frozenset((d[a-1].strip(), d[b-1].strip())) for a,b in r["gold_doc_indices"]))
        for a,b in r["gold_doc_indices"]: ov.append(jac(d[a-1], d[b-1]))
    same = True if base_sig is None else (sig == base_sig)
    if base_sig is None: base_sig = sig
    print(f"{os.path.basename(f):<62} {len(recs):>5} {len(recs[0]['documents']):>6} "
          f"{statistics.median(ov):>6.3f} {sum(1 for x in ov if x>0.5)/len(ov):>6.3f} {str(same):>8}")
print("\ntraining reference: jac 0.306, >0.5 0.000  |  both-mode: 0.388, 0.381")
PYEOF
echo "=== [contra-v3-xlong] DONE $(date '+%F %T') ==="
