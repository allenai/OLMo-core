#!/bin/bash
# Wait for the 6 LR-sweep eval JSONs (fresh, post-resubmit), then print the f1 table.
set -uo pipefail
RESULTS=/scratch/users/prasann/stl_eval_results
CUTOFF=$(date -d "2026-08-27 19:55" +%s)
RUNS="lmx-full-lr2e5-4b-loc lmx-full-lr5e5-4b-loc lmx-full-lr1p2e4-4b-loc lmx-slm-lr2e5-4b-loc lmx-slm-lr5e5-4b-loc lmx-slm-lr1p2e4-4b-loc"
for i in $(seq 1 90); do
  missing=0
  for RUN in $RUNS; do
    f="$RESULTS/${RUN}_outlier_multirung.json"
    if [ ! -f "$f" ] || [ "$(stat -c %Y "$f")" -lt "$CUTOFF" ]; then missing=$((missing+1)); fi
  done
  echo "[harvest $i] waiting on $missing/6 $(date '+%T')"
  [ "$missing" = 0 ] && break
  sleep 60
done
echo "=== LR SWEEP RESULTS ==="
python3 - <<'PYEOF'
import json, pathlib
res = pathlib.Path("/scratch/users/prasann/stl_eval_results")
runs = ["lmx-full-lr2e5-4b-loc","lmx-full-lr5e5-4b-loc","lmx-full-lr1p2e4-4b-loc",
        "lmx-slm-lr2e5-4b-loc","lmx-slm-lr5e5-4b-loc","lmx-slm-lr1p2e4-4b-loc"]
print(f"{'run':34s} {'f1@3k':>8s} {'f1@8k':>8s}")
for r in runs:
    p = res / f"{r}_outlier_multirung.json"
    if not p.exists():
        print(f"{r:34s}  MISSING"); continue
    d = json.loads(p.read_text())
    def get(rung):
        v = d.get(rung) or d.get(f"outlier_{rung}") or {}
        if isinstance(v, dict):
            return v.get("set_f1", v.get("f1", v.get("score")))
        return v
    print(f"{r:34s} {str(get('3k')):>8s} {str(get('8k')):>8s}")
PYEOF
