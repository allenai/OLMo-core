#!/bin/bash
# Wait for the LR harvest, pick per-arch best LR by mean f1 over 3k+8k, sync v2 arms to weka,
# then launch wave 2 at the winning LRs. Prints the LR table for the record.
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd "$REPO"
RESULTS=/scratch/users/prasann/stl_eval_results
CUTOFF=$(date -d "2026-08-27 20:05" +%s)
RUNS="lmx-full-lr2e5-4b-loc lmx-full-lr5e5-4b-loc lmx-full-lr1p2e4-4b-loc lmx-slm-lr2e5-4b-loc lmx-slm-lr5e5-4b-loc lmx-slm-lr1p2e4-4b-loc"

for i in $(seq 1 100); do
  missing=0
  for RUN in $RUNS; do
    f="$RESULTS/${RUN}_outlier_multirung.json"
    { [ -f "$f" ] && [ "$(stat -c %Y "$f")" -ge "$CUTOFF" ]; } || missing=$((missing+1))
  done
  echo "[w2gate $i] waiting on $missing/6 evals $(date '+%T')"
  [ "$missing" = 0 ] && break
  sleep 60
done

python3 - > /tmp/lr_winners.txt <<'PYEOF'
import json, pathlib
res = pathlib.Path("/scratch/users/prasann/stl_eval_results")
lrs = {"lr2e5": "2e-5", "lr5e5": "5e-5", "lr1p2e4": "1.2e-4"}
print("run\tf1@3k\tf1@8k\tmean")
best = {}
for arch in ("full", "slm"):
    scores = {}
    for tag, lr in lrs.items():
        p = res / f"lmx-{arch}-{tag}-4b-loc_outlier_multirung.json"
        if not p.exists(): continue
        d = json.loads(p.read_text())
        def g(rung):
            v = d.get(rung) or d.get(f"outlier_{rung}") or {}
            return v.get("set_f1", v.get("f1", v.get("score"))) if isinstance(v, dict) else v
        a, b = g("3k"), g("8k")
        vals = [x for x in (a, b) if isinstance(x, (int, float))]
        m = sum(vals)/len(vals) if vals else -1
        scores[lr] = m
        print(f"lmx-{arch}-{tag}\t{a}\t{b}\t{m:.4f}")
    if scores:
        best[arch] = max(scores, key=scores.get)
print("WINNER_FULL", best.get("full", "5e-5"))
print("WINNER_SLM", best.get("slm", "5e-5"))
PYEOF
cat /tmp/lr_winners.txt
LR_FULL=$(grep WINNER_FULL /tmp/lr_winners.txt | awk '{print $2}')
LR_SLM=$(grep WINNER_SLM /tmp/lr_winners.txt | awk '{print $2}')
echo "[w2gate] winners: full=$LR_FULL slm=$LR_SLM"

echo "[w2gate] syncing v2 arms to weka"
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
NAME=outlier-lm-weka-sync4 PRIORITY=urgent \
  S3_PREFIX=s3://ai2-llm/checkpoints/prasanns/outlier_lengthmix \
  DEST_REL=ai2-llm/checkpoints/prasanns/outlier_lengthmix \
  bash src/scripts/train/memexpress/singletask_ladder/stage_eval500_v2_to_weka_gantry.sh 2>&1 | tail -1
sleep 420   # sync is a few GB of small files; give it 7 min then verify via launch failures if any

PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export PYTHONPATH="$REPO/src"
WEKA_ROOT=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
BASE="$WEKA_ROOT/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim"
LOGD=debug/outlier_lengthmix_scaling/launches; mkdir -p "$LOGD"
launch () { # arm seqlen variant lr
  local ARM=$1 SEQ=$2 variant=$3 lr=$4
  local vtag=full; [ "$variant" = "sparselandmark" ] && vtag=slm
  local PACK_FLAG=""; [ "$variant" = "full" ] && PACK_FLAG="--pack"
  local RUN="lmx-${vtag}-${ARM//_/}-4b"
  echo "[w2gate] launch $RUN (lr=$lr)"
  timeout 240 $PY -u src/scripts/train/memexpress/ctc_suite/beaker_ctc_suite.py \
    --task outlier --variant "$variant" --model-scale 4b --model-family qwen3_5 \
    --run-name "$RUN" --num-nodes 1 --epochs 1 --seq-len "$SEQ" --lr "$lr" $PACK_FLAG \
    --global-batch 8 --micro-batch-instances 1 \
    --data-root "$WEKA_ROOT/outlier_lengthmix/arms/$ARM" \
    --base-checkpoint "$BASE" --wandb-group outlier-lengthmix-checks \
    launch > "$LOGD/${RUN}.log" 2>&1 &
  sleep 3
}
for arm_seq in "m8k_mix:16384" "p8k_5000:16384" "p8k_8000:16384" "p2k_1250:4096" "p2k_2500:4096" "p2k_10000:4096" "p2k_20000:4096"; do
  IFS=: read -r ARM SEQ <<< "$arm_seq"
  launch "$ARM" "$SEQ" full "$LR_FULL"
  launch "$ARM" "$SEQ" sparselandmark "$LR_SLM"
done
wait
echo "[w2gate] wave 2 submitted at winners full=$LR_FULL slm=$LR_SLM"
grep -l "Traceback\|ERROR" "$LOGD"/lmx-*.log 2>/dev/null || echo "no errors in launch logs"
