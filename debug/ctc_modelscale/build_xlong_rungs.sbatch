#!/bin/bash
# Build the length-generalization rungs: grow each task's own 2k-32k ladder past the 32k training
# ceiling, using that task's own document pool (see expand_ctc_rung.py for why the pre-existing
# contradiction/rung_131072.jsonl is NOT usable -- 47.1 tok/doc of FEVER/wiki filler against 15.6
# for the rest of its ladder).
#
# Targets are MEASURED token budgets, not rung labels: measured against the real tokenizer the
# suite's labels are off by ~1.5x (contradiction) and ~2.9x (niah), so the outputs are named by
# what they actually contain. Both targets sit under Qwen3.5's 262,144 native position ceiling, so
# they are native-RoPE measurements and need no YaRN serving copy.
#
#   sbatch debug/ctc_modelscale/build_xlong_rungs.sbatch
#SBATCH --job-name=ctcms-xlong-rungs
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --output=/data/prasann/joblogs/ctcms_xlong_rungs_%j.log

set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
RUNGS=/scratch/users/prasann/ctc_suite_staged/eval_rungs
OUT=/scratch/users/prasann/ctc_suite_staged/eval_rungs_xlong
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python

export HF_HOME=/data/prasann/hf_cache
export TMPDIR=/data/prasann/tmp
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$OUT" /data/prasann/joblogs

echo "=== HOST=$(hostname) START=$(date '+%F %T') ==="

# task : source rung (the largest well-formed one) : measured targets
SPECS=(
  "contradiction:$RUNGS/contradiction/rung_32768.jsonl:65536,131072"
  "niah:$RUNGS/niah/rung_32768.jsonl:65536,131072"
  "hotpotqa:$RUNGS/hotpotqa/rung_16384.jsonl:65536,131072"
  "nq:$RUNGS/nq/rung_8192.jsonl:65536,131072"
)

for spec in "${SPECS[@]}"; do
  TASK="${spec%%:*}"; rest="${spec#*:}"; SRC="${rest%%:*}"; TGT="${rest##*:}"
  echo "--- $TASK  src=$SRC  targets=$TGT ---"
  [ -f "$SRC" ] || { echo "!!! missing source $SRC -- skipping $TASK"; continue; }
  timeout 10000 "$PY" "$REPO/debug/ctc_modelscale/expand_ctc_rung.py" \
      --task "$TASK" --src "$SRC" --targets "$TGT" --out-dir "$OUT/$TASK"
  echo "--- $TASK rc=$? ---"
done

echo "=== built files ==="
ls -la "$OUT"/*/ 2>/dev/null
df -h /scratch | tail -1
echo "=== DONE $(date '+%F %T') ==="
