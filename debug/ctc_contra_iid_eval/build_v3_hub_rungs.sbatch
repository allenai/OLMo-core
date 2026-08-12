#!/bin/bash
# Contradiction v3: iid (realistic-mode) eval rungs at the RESULTS-HUB n values.
#
# The hub's 5-task runs read contradiction_eval_pubmed_both_n{100,190,385,765}_k3.jsonl -- `both`
# mode -- while their training shard (single_task_ladders_v2/contradiction/metadata.json) names
# contradiction_train_pubmed_realistic_n50-950_k3.jsonl. Same mismatch as the CTC ladder, same
# shared source file. See records/contradiction-train-eval-non-iid.md.
#
# This reuses the base built and validated earlier today rather than generating new gold:
#   contra_iid_base500.jsonl -- 500 examples, realistic mode, Qwen2.5-7B-Instruct generator,
#   --max-overlap 0.5, and disjointness-filtered against the training file's 120,673 gold
#   sentences (21.3% of candidates dropped). Gold-pair Jaccard median 0.324 / mean 0.317 /
#   p90 0.462 / hard cap 0.500 / 0.0% above the cap -- matching training (0.306/0.307/0.459/0.500/0.0)
#   and NOT the `both` ladder (0.388/0.501/0.943/1.000/38.1%).
# Expansion is LLM-free and seeds per-example (seed*1_000_003 + i), so all four rungs carry the
# SAME 500 examples and only the filler count changes.
#SBATCH --job-name=contra-v3-rungs
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --nodelist=mooney
#SBATCH --output=/data/prasann/joblogs/contra_v3_rungs_%j.log
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
OUT="$WORK/v3_hub"
mkdir -p "$OUT"

echo "=== [contra-v3] HOST=$(hostname) START=$(date '+%F %T') ==="
[ -f "$BASE" ] || { echo "!!! validated base missing: $BASE"; exit 1; }
wc -l "$BASE"

for N in 100 190 385 765; do
  echo "[expand] n=$N $(date '+%F %T')"
  PYTHONPATH="$GENSRC" $PY "$GENSRC/corpus_reasoning/data/generate_pubmed_contradiction_data.py" \
    --expand-from-eval "$BASE" \
    --num-docs "$N" \
    --num-contradictions 3 \
    --mode realistic \
    --pool-abstracts 200000 \
    --seed 42 \
    --filler-pool-seed 43 \
    --output-dir "$OUT" || echo "!!! expand n=$N FAILED"
done

echo ""
echo "=== assembled v3 bundle files ==="
ls -la "$OUT"

echo ""
echo "=== audit: same 500 examples, gold-pair Jaccard vs training ==="
PYTHONPATH="$GENSRC" $PY - <<'PYEOF'
import json, re, statistics, glob, os
def toks(s): return set(re.findall(r"[a-z0-9]+", s.lower()))
def jac(a, b):
    A, B = toks(a), toks(b)
    return len(A & B) / max(1, len(A | B))
OUT = "/data/prasann/contra_iid_eval/v3_hub"
base_sig = None
print(f"{'file':<52} {'rows':>5} {'n_docs':>7} {'jac_med':>8} {'>0.5':>6} {'same500':>8}")
for n in (100, 190, 385, 765):
    p = os.path.join(OUT, f"contradiction_eval_pubmed_realistic_n{n}_k3.jsonl")
    if not os.path.exists(p):
        print(f"{os.path.basename(p):<52} MISSING"); continue
    recs = [json.loads(l) for l in open(p)]
    ov, sig = [], []
    for r in recs:
        d = [x["text"] if isinstance(x, dict) else x for x in r["documents"]]
        sig.append(frozenset(frozenset((d[a-1].strip(), d[b-1].strip()))
                             for a, b in r["gold_doc_indices"]))
        for a, b in r["gold_doc_indices"]:
            ov.append(jac(d[a-1], d[b-1]))
    if base_sig is None:
        base_sig, same = sig, True
    else:
        same = (sig == base_sig)
    print(f"{os.path.basename(p):<52} {len(recs):>5} {len(recs[0]['documents']):>7} "
          f"{statistics.median(ov):>8.3f} {sum(1 for x in ov if x>0.5)/len(ov):>6.3f} {str(same):>8}")
print("\ntraining reference: jac_med 0.306, >0.5 = 0.000")
print("both-mode ladder  : jac_med 0.388, >0.5 = 0.381")
PYEOF

echo "=== [contra-v3] DONE $(date '+%F %T') ==="
