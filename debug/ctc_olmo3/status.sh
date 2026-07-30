#!/bin/bash
# One-screen status of the OLMo-3 CTC arm: Berkeley slurm jobs + Beaker experiments.
# Usage: bash debug/ctc_olmo3/status.sh
set -uo pipefail
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$HOME/.local/bin:$PATH

echo "===== BERKELEY ($(date '+%F %T')) ====="
squeue -u prasann -o '%.10i %.30j %.9T %.14R' | grep -E "olmo3|JOBID" || true
for f in /scratch/users/prasann/ctc_suite_logs/train_ctc-olmo3-*.log; do
  [ -f "$f" ] || continue
  echo "-- $(basename "$f")"
  grep -oE "\[step=[0-9]+/[0-9]+[^]]*\]" "$f" | tail -1
  grep -E "saved model-only checkpoint|=== DONE" "$f" | tail -1
done

echo "===== BEAKER ====="
for id in $(cat debug/ctc_olmo3/experiment_ids.txt 2>/dev/null); do
  timeout 90 beaker experiment get "$id" --format json 2>/dev/null | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)[0]
except Exception:
    print('  (could not read)'); raise SystemExit
j=(d.get('jobs') or [{}])[-1]
st=j.get('status',{})
# jobs[-1].status is the ONLY truth (the table view lies); report the latest state key present.
order=['finalized','started','scheduled','created','canceled']
state=next((k for k in order if k in st), '?')
extra=''
if 'finalized' in st:
    extra=f\"  exitCode={st.get('exitCode')}\"
print(f\"  {d['name'][:52]:52s} {state}{extra}  https://beaker.org/ex/{d['id']}\")
"
done
