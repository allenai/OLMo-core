#!/bin/bash
# Re-tokenize the outlier length-mix rung pools with the CORRECT Qwen3.5 eos (248044).
# The first pass (job 3481048, canceled) used the converter's hardcoded Qwen3 eos 151643 --
# wrong answer-terminator for the qwen3_5 family. Pools/ladder/splits from that job are fine.
#SBATCH --job-name=outlier-lm-tok
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --nodelist=mooney
#SBATCH --output=/data/prasann/joblogs/outlier_lengthmix_tok_%j.log
#SBATCH --chdir=/data/prasann
set -uo pipefail
export HOME=/data/prasann/home
export HF_HOME=/data/prasann/hf_cache
export TMPDIR=/data/prasann/tmp

GENSRC=/data/prasann/repo/OLMo-core/src
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-eval/bin/python
WORK=/data/prasann/outlier_lengthmix
QWEN35_EOS=248044

echo "=== [outlier-lm-tok] HOST=$(hostname) START=$(date '+%F %T') ==="
grep -q "add_argument(\"--eos\"" "$GENSRC/corpus_reasoning/data/convert_unified_to_sft.py" \
  || { echo "!!! converter on mooney clone lacks --eos flag"; exit 1; }
rm -rf "$WORK/tokenized"   # first-pass shards have the wrong eos; rebuild all

tok () {
  local SPLIT=$1 N=$2
  local OUT="$WORK/tokenized/n${N}_${SPLIT}"
  mkdir -p "$OUT"
  echo "--- tokenizing n${N}_${SPLIT} $(date '+%T') ---"
  PYTHONPATH="$GENSRC" $PY "$GENSRC/corpus_reasoning/data/convert_unified_to_sft.py" \
    --task outlier --input "$WORK/outlier_lm_n${N}_${SPLIT}.jsonl" --out-dir "$OUT" \
    --tokenizer Qwen/Qwen3.5-0.8B-Base --max-seq-len 40960 --query-position after \
    --eos $QWEN35_EOS || { echo "!!! tokenize FAILED: n${N}_${SPLIT}"; exit 1; }
  $PY -c "
import json; m=json.load(open('$OUT/metadata.json'))
assert m['eos']==$QWEN35_EOS, m['eos']
print(json.dumps(m))"
done_all=1
}
for N in 14 28 57 111; do tok train "$N"; tok heldout "$N"; done
echo "=== [outlier-lm-tok] DONE $(date '+%F %T') ==="
