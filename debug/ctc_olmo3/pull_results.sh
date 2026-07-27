#!/bin/bash
# Pull the Beaker-side OLMo-3 rung results (written to weka, relayed to S3 by beaker_eval.sh) into
# the repo's results/ctc_suite tree, then print the per-rung table for both tasks and both arms.
# The Berkeley-side evals write into results/ctc_suite directly, so this only fetches what ran on
# Beaker; running it twice is harmless (aws s3 sync is idempotent).
set -uo pipefail
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd "$REPO"

STAGE=/scratch/users/prasann/ctc_olmo3_results
mkdir -p "$STAGE"
AWS_PROFILE=S3 aws s3 sync s3://ai2-llm/checkpoints/prasanns/ctc_olmo3/results "$STAGE" --only-show-errors
echo "staged -> $STAGE"

# Copy only the per-rung summary JSONs into the repo (the .raw/.generations dumps stay on /scratch:
# they are large and the repo keeps few-KB artifacts only).
find "$STAGE" -name 'rung_*.json' ! -name '*.raw.json' ! -name '*.generations.json' | while read -r f; do
  rel="${f#$STAGE/}"
  mkdir -p "$REPO/results/ctc_suite/$(dirname "$rel")"
  cp -f "$f" "$REPO/results/ctc_suite/$rel"
done

python3 - <<'EOF'
import glob, json, os

REPO = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core"
rows = {}
for p in glob.glob(f"{REPO}/results/ctc_suite/*/olmo3-7b_*/rung_*.json"):
    if p.endswith((".raw.json", ".generations.json")):
        continue
    d = json.load(open(p))
    rows[(d["task"], d["arm"], d["rung_tokens"])] = d

if not rows:
    print("no OLMo-3 results yet")
    raise SystemExit

print(f"{'task':<14}{'arm':<13}{'rung':>6}  {'metric':>7}  {'eval_size':>9}  {'parse':>6}  metric_name")
for (task, arm, rung) in sorted(rows):
    d = rows[(task, arm, rung)]
    aux = d.get("aux_metrics", {})
    flag = "  <-- WARN eval_size<500" if d.get("eval_size", 0) < 500 else ""
    print(f"{task:<14}{arm:<13}{rung:>6}  {d['metric_value']:>7.3f}  "
          f"{d.get('eval_size', 0):>9}  {aux.get('parse_rate', float('nan')):>6.3f}  "
          f"{d['metric_name']}{flag}")
EOF
