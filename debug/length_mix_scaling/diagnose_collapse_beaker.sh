#!/bin/bash
# Run diagnose_collapse.py on Beaker, where the length-mix responses live on weka.
#
# CPU-only: this only re-grades already-generated text, so it needs no GPU, no vLLM venv, no CUDA
# toolkit and no transformers shadow -- all of which the eval job needed and none of which are
# involved in parsing pairs out of a string. Keeping them out is what makes this job schedule in
# seconds instead of queueing for an H100.
set -uo pipefail
echo "=== diagnose_collapse HOST=$(hostname) START=$(date '+%F %T') ==="

LM=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_length_mix
OUT=$LM/eval_results
REPO=$(find / -maxdepth 3 -iname pyproject.toml 2>/dev/null | grep -v /opt/conda | grep -v cache | head -1 | xargs -r dirname)
[ -z "$REPO" ] && { echo "FATAL: repo not found"; exit 1; }
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
echo "REPO=$REPO"

# gantry --no-python skips `pip install -e .`, so olmo-core's transitive deps (dataclass_extensions)
# are absent -- same trap that killed the first eval job 4 s in.
python -c "import dataclass_extensions" 2>/dev/null || {
  echo "=== installing olmo-core deps $(date '+%F %T') ==="
  pip install -q -e "$REPO" 2>&1 | tail -5
}

ls -la "$OUT"/*responses.json 2>/dev/null | head -20
echo "--- rungs ---"
ls -la "$LM"/eval_rungs/ 2>/dev/null | head

PAIRS="${PAIRS:-A4:A4s2,C3:C3s2}"
python -u "$REPO/debug/length_mix_scaling/diagnose_collapse.py" \
  --eval-results "$OUT" --rungs-dir "$LM/eval_rungs" \
  --pairs "$PAIRS" --rungs 2048,8192,32768 \
  --out "$OUT/collapse_diagnosis.json" --dump-examples 6
RC=$?
echo "=== DONE rc=$RC $(date '+%F %T') ==="
exit $RC
