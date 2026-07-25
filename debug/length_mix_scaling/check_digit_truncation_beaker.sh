#!/bin/bash
set -uo pipefail
LM=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_length_mix
REPO=$(find / -maxdepth 3 -iname pyproject.toml 2>/dev/null | grep -v /opt/conda | grep -v cache | head -1 | xargs -r dirname)
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
python -c "import dataclass_extensions" 2>/dev/null || pip install -q -e "$REPO" 2>&1 | tail -3
python -u "$REPO/debug/length_mix_scaling/check_digit_truncation.py" \
  --eval-results "$LM/eval_results" --rungs-dir "$LM/eval_rungs" \
  --arms "${ARMS:-A4rr,A4s2,C3rr,C3s2,A4e}" --rung "${RUNG:-32768}" \
  --out "$LM/eval_results/digit_truncation_${RUNG:-32768}.json"
echo "=== DONE rc=$? ==="
