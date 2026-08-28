#!/bin/bash
# Check-thread v2 data: EXTEND pools by appending (prefix-stable), add the 32k rung (n=220),
# build eval rung_32768, compose+tokenize the new arms (4-way parallel), sync S3.
#SBATCH --job-name=outlier-lm-v2
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --nodelist=mooney
#SBATCH --output=/data/prasann/joblogs/outlier_lm_v2_%j.log
#SBATCH --chdir=/data/prasann
set -uo pipefail
export HOME=/data/prasann/home HF_HOME=/data/prasann/hf_cache TMPDIR=/data/prasann/tmp
GENSRC=/data/prasann/repo/OLMo-core/src
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-eval/bin/python
WORK=/data/prasann/outlier_lengthmix
POOL=/data/prasann/single_task_ladders_20k/wiki100w_article_pool_smoke.pkl

echo "=== [v2] START $(date '+%F %T') ==="
gen () { # n count seed name
  local N=$1 CNT=$2 SEED=$3 NAME=$4
  [ -s "$WORK/${NAME}.jsonl" ] && { echo "[skip] $NAME"; return 0; }
  PYTHONPATH="$GENSRC" $PY "$GENSRC/corpus_reasoning/data/generate_wiki_outlier_data.py" \
    --num-examples "$CNT" --min-docs "$N" --max-docs "$N" --num-outliers 3 \
    --majority-mode articles --min-run 4 --max-run 14 \
    --pool-cache "$POOL" --eval-frac 0 --seed "$SEED" \
    --out-dir "$WORK" --out-name "$NAME" || { echo "!!! gen FAILED $NAME"; exit 1; }
}
# extensions (new seeds; APPENDED after the existing prefixes)
gen 14  15300 201 outlier_lm_n14_ext15300
gen 57  26000 203 outlier_lm_n57_ext26000
gen 220 8500  204 outlier_lm_n220_k3_8500

$PY - <<'PYEOF'
import pathlib
work = pathlib.Path("/data/prasann/outlier_lengthmix")
def extend(train, ext, expect):
    t = work / train
    lines = t.read_text().splitlines()
    if len(lines) >= expect:
        print(f"[skip] {train} already {len(lines)}"); return
    lines += (work / ext).read_text().splitlines()
    assert len(lines) >= expect - 60, (train, len(lines))
    t.write_text("\n".join(lines) + "\n")
    print(f"{train}: extended to {len(lines)}")
extend("outlier_lm_n14_train.jsonl", "outlier_lm_n14_ext15300.jsonl", 24000)
extend("outlier_lm_n57_train.jsonl", "outlier_lm_n57_ext26000.jsonl", 32700)
# n220: fresh split
n220 = (work / "outlier_lm_n220_k3_8500.jsonl").read_text().splitlines()
tr, ho = work / "outlier_lm_n220_train.jsonl", work / "outlier_lm_n220_heldout.jsonl"
if not tr.exists():
    tr.write_text("\n".join(n220[:-300]) + "\n"); ho.write_text("\n".join(n220[-300:]) + "\n")
    print(f"n220 split: train={len(n220)-300} heldout=300")
PYEOF

# eval rung 32768 (same builder/pool)
if [ ! -s "$WORK/eval_rungs/outlier/outlier/rung_32768.jsonl" ]; then
  PYTHONPATH="$GENSRC" $PY "$GENSRC/corpus_reasoning/data/build_v2_outlier_ladder.py" \
    --num-examples 600 --pool-cache "$POOL" --rungs "220:32768" \
    --out-root "$WORK/eval_rungs/outlier" --out-tmpl "rung_{lab}.jsonl" || exit 1
fi

# compose v2 arms (script now includes them; existing arms [skip] naturally via same content)
sed -i 's/^    ap.add_argument/    ap.add_argument/' /data/prasann/repo/OLMo-core/compose_arms_lengthmix.py 2>/dev/null
$PY /data/prasann/repo/OLMo-core/compose_arms_lengthmix.py --work "$WORK" || exit 1

tokarm () {
  local ARM=$1
  local OUT="$WORK/arms_tokenized/$ARM"
  [ -s "$OUT/metadata.json" ] && { echo "[skip] $ARM"; return 0; }
  mkdir -p "$OUT"
  PYTHONPATH="$GENSRC" $PY "$GENSRC/corpus_reasoning/data/convert_unified_to_sft.py" \
    --task outlier --input "$WORK/arms/$ARM.jsonl" --out-dir "$OUT" \
    --tokenizer Qwen/Qwen3.5-0.8B-Base --max-seq-len 40960 --query-position after \
    --eos 248044 || { echo "!!! tokenize FAILED $ARM"; touch "$WORK/arms_tokenized/FAILED_$ARM"; }
}
# 4-way parallel tokenization of the new arms + n220 heldout
tokarm m8k_mix &  tokarm p8k_5000 &  tokarm p8k_8000 &  tokarm p2k_1250 &
wait
tokarm p2k_2500 & tokarm p2k_10000 & tokarm p2k_20000 & tokarm p32k_2000 &
wait
mkdir -p "$WORK/tokenized/n220_heldout"
if [ ! -s "$WORK/tokenized/n220_heldout/metadata.json" ]; then
  PYTHONPATH="$GENSRC" $PY "$GENSRC/corpus_reasoning/data/convert_unified_to_sft.py" \
    --task outlier --input "$WORK/outlier_lm_n220_heldout.jsonl" \
    --out-dir "$WORK/tokenized/n220_heldout" --tokenizer Qwen/Qwen3.5-0.8B-Base \
    --max-seq-len 40960 --query-position after --eos 248044 || echo "!!! n220 heldout tok FAILED"
fi
ls "$WORK"/arms_tokenized/FAILED_* 2>/dev/null && exit 1

cp "$WORK/arms/MANIFEST.json" "$WORK/arms_tokenized/MANIFEST.json"
export AWS_PROFILE=S3
DST=s3://ai2-llm/checkpoints/prasanns/outlier_lengthmix
aws s3 sync "$WORK/arms_tokenized" "$DST/arms" --only-show-errors || exit 1
aws s3 sync "$WORK/tokenized/n220_heldout" "$DST/heldout/n220" --only-show-errors || exit 1
aws s3 sync "$WORK/eval_rungs/outlier/outlier" "$DST/eval_rungs/outlier" --only-show-errors || exit 1
echo "=== [v2] DONE $(date '+%F %T') ==="
