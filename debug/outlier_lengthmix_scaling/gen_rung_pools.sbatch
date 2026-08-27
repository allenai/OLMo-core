#!/bin/bash
# Per-rung outlier pools for the length-mix data-scaling-law experiment (sparse-landmark vs full,
# Qwen3.5). Fixed doc-count pools matched to the token-calibrated eval rungs (n:label =
# 14:2048, 28:4096, 57:8192, 111:16384 — build_v2_outlier_ladder.py's canonical mapping), built
# with the M-axis-FIXED generator (--majority-mode articles, 2026-08-11) and the EVAL's article
# pool, mirroring debug/outlier_iid_rebuild/full_outlier_iid.sbatch exactly except: fixed n per
# pool (degenerate --min-docs=--max-docs) and per-pool seeds.
# Also rebuilds the matching 600-row eval ladder with the same builder+pool, splits a 300-example
# held-out CE shard off each pool, and tokenizes everything with the CTC-suite Qwen3.5 tokenizer
# (Qwen/Qwen3.5-0.8B-Base) at query_position=after (standing decision; outlier pins 'after').
#SBATCH --job-name=outlier-lm-pools
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --nodelist=mooney
#SBATCH --output=/data/prasann/joblogs/outlier_lengthmix_pools_%j.log
#SBATCH --chdir=/data/prasann
set -uo pipefail

export HOME=/data/prasann/home
export HF_HOME=/data/prasann/hf_cache
export TMPDIR=/data/prasann/tmp
mkdir -p "$HOME" "$TMPDIR" /data/prasann/joblogs

GENSRC=/data/prasann/repo/OLMo-core/src
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-eval/bin/python
WORK=/data/prasann/outlier_lengthmix
POOL=/data/prasann/single_task_ladders_20k/wiki100w_article_pool_smoke.pkl
mkdir -p "$WORK"
ls -la "$POOL" || { echo "!!! article pool missing at $POOL"; exit 1; }

echo "=== [outlier-lm] HOST=$(hostname) START=$(date '+%F %T') ==="

# ---- 1. train pools (fixed n, item 1,2,5,10 progress logged by the generator) ----
gen_pool () {  # n count seed
  local N=$1 CNT=$2 SEED=$3
  local NAME="outlier_lm_n${N}_k3_${CNT}"
  if [ -s "$WORK/${NAME}.jsonl" ]; then echo "[skip] $NAME exists"; return 0; fi
  echo "--- generating $NAME (seed $SEED) $(date '+%T') ---"
  PYTHONPATH="$GENSRC" $PY "$GENSRC/corpus_reasoning/data/generate_wiki_outlier_data.py" \
    --num-examples "$CNT" --min-docs "$N" --max-docs "$N" --num-outliers 3 \
    --majority-mode articles --min-run 4 --max-run 14 \
    --pool-cache "$POOL" --eval-frac 0 --seed "$SEED" \
    --out-dir "$WORK" --out-name "$NAME" || { echo "!!! gen FAILED: $NAME"; exit 1; }
}
gen_pool 14  9000 101
gen_pool 28  9000 102
gen_pool 57  7000 103
gen_pool 111 7000 104

# ---- 2. eval ladder (same builder + pool as the committed rung_2048/4096) ----
if [ ! -s "$WORK/eval_rungs/outlier/rung_16384.jsonl" ]; then
  echo "--- building eval ladder $(date '+%T') ---"
  mkdir -p "$WORK/eval_rungs/outlier"
  PYTHONPATH="$GENSRC" $PY "$GENSRC/corpus_reasoning/data/build_v2_outlier_ladder.py" \
    --num-examples 600 --pool-cache "$POOL" \
    --rungs "14:2048,28:4096,57:8192,111:16384" \
    --out-root "$WORK/eval_rungs/outlier" --out-tmpl "rung_{lab}.jsonl" \
    || { echo "!!! eval ladder FAILED"; exit 1; }
fi

# ---- 3. holdout split (last 300 of each pool -> held-out CE shard source) ----
$PY - <<'PYEOF'
import json, pathlib
work = pathlib.Path("/data/prasann/outlier_lengthmix")
for n, cnt in [(14,9000),(28,9000),(57,7000),(111,7000)]:
    src = work / f"outlier_lm_n{n}_k3_{cnt}.jsonl"
    tr, ho = work / f"outlier_lm_n{n}_train.jsonl", work / f"outlier_lm_n{n}_heldout.jsonl"
    if tr.exists() and ho.exists():
        print(f"[skip] split n{n}"); continue
    lines = src.read_text().splitlines()
    assert len(lines) >= cnt - 50, f"{src}: only {len(lines)} lines"
    tr.write_text("\n".join(lines[:-300]) + "\n")
    ho.write_text("\n".join(lines[-300:]) + "\n")
    print(f"split n{n}: train={len(lines)-300} heldout=300")
PYEOF

# ---- 4. tokenize (Qwen3.5 tokenizer, query_position=after) ----
tok () {  # split n
  local SPLIT=$1 N=$2
  local OUT="$WORK/tokenized/n${N}_${SPLIT}"
  if [ -s "$OUT/metadata.json" ]; then echo "[skip] tok n${N}_${SPLIT}"; return 0; fi
  mkdir -p "$OUT"
  PYTHONPATH="$GENSRC" $PY "$GENSRC/corpus_reasoning/data/convert_unified_to_sft.py" \
    --task outlier --input "$WORK/outlier_lm_n${N}_${SPLIT}.jsonl" --out-dir "$OUT" \
    --tokenizer Qwen/Qwen3.5-0.8B-Base --max-seq-len 40960 --query-position after \
    || { echo "!!! tokenize FAILED: n${N}_${SPLIT}"; exit 1; }
  cat "$OUT/metadata.json"
}
for N in 14 28 57 111; do tok train "$N"; tok heldout "$N"; done

# ---- 5. audit: K distribution per pool vs eval ladder ----
$PY - <<'PYEOF'
import json, pathlib, statistics
work = pathlib.Path("/data/prasann/outlier_lengthmix")
audit = {}
def kof(d):
    m = d.get("metadata") or {}
    for key in ("num_categories","K","k_majority","num_majority_topics"):
        if key in m: return m[key]
    return None
for n, cnt in [(14,9000),(28,9000),(57,7000),(111,7000)]:
    ks, docs = [], []
    with open(work / f"outlier_lm_n{n}_k3_{cnt}.jsonl") as f:
        for i, line in enumerate(f):
            if i >= 500: break
            d = json.loads(line); docs.append(len(d["documents"])); k = kof(d)
            if k is not None: ks.append(k)
    audit[f"train_n{n}"] = {"docs": [min(docs), max(docs)],
                            "K_med": statistics.median(ks) if ks else None,
                            "K_range": [min(ks), max(ks)] if ks else None}
label = {14:2048, 28:4096, 57:8192, 111:16384}
for n, lab in label.items():
    p = work / "eval_rungs/outlier" / f"rung_{lab}.jsonl"
    ks, docs = [], []
    with open(p) as f:
        for i, line in enumerate(f):
            if i >= 200: break
            d = json.loads(line); docs.append(len(d["documents"])); k = kof(d)
            if k is not None: ks.append(k)
    audit[f"eval_{lab}"] = {"rows_sampled": len(docs), "docs": [min(docs), max(docs)],
                            "K_med": statistics.median(ks) if ks else None}
(work / "audit.json").write_text(json.dumps(audit, indent=2))
print(json.dumps(audit, indent=2))
PYEOF

echo "=== [outlier-lm] DONE $(date '+%F %T') ==="
