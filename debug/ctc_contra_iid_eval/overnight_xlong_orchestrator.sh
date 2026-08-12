#!/usr/bin/env bash
# Overnight orchestrator for the v3 xlong (64k-512k) contradiction passes.
#
# Runs unattended on the LOGIN node (needs gantry + beaker creds, which compute nodes lack).
# Waits for the rung build, stages S3 -> weka, then launches Pass B split by YaRN group.
#
# CHAIN
#   1. wait for slurm job $RUNG_JOB (contra-v3-xlong) to leave the queue
#   2. verify all four rung files exist and have >=500 rows      -> abort loudly if not
#   3. aws s3 sync  mooney:/data/.../v3_xlong -> s3 _eval_bundle_eval500_v3/contra
#   4. gantry S3 -> weka sync, wait for it to succeed
#   5. build the yarn2 serving copies for 256k/512k
#   6. launch Pass B: native (64k,128k) on the plain step, yarn2 (256k,512k) on the _yarn2 copy
#
# WHY PASS B IS CONTRA-ONLY: the v3 change only touches contradiction; nq/outlier/rerank/oolong
# xlong rungs are identical to v2 and these checkpoints already have those rows in results-hub.
# Running them would duplicate, not measure. Deviation from the skill's
# "--task contra,nq,outlier,rerank,oolong" is deliberate and recorded in the ledger.
#
# NOTE ON MISSING FLAGS: the skill calls for --eval-tag and --xlong-only. NEITHER EXISTS in the
# launcher or the on-node runner (checked: 0 occurrences of --eval-tag/--xlong-only/EVAL_TAG/
# XLONG_ONLY). --results-dir -> EVAL_OUT_DIR is the supported equivalent for collision-avoidance
# and is used here. Without --xlong-only the xlong passes ALSO re-run 2k-32k at bs=1; that is
# wasted GPU, not a wrong number, and there is no supported way to suppress it today.
set -uo pipefail

RUNG_JOB="${RUNG_JOB:-3438651}"
SRC=/net/mooney/data/prasann/contra_iid_eval/v3_xlong
S3=s3://ai2-llm/checkpoints/prasanns/_eval_bundle_eval500_v3/contra
A=/weka/oe-training-default/ai2-llm/checkpoints/amandab
R=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/_eval_results
LOG=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/ctc_contra_iid_eval/overnight.log
export PATH="/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH"
export AWS_PROFILE=S3
cd /accounts/projects/berkeleynlp/prasann/projects/OLMo-core

say () { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "=== orchestrator start (waiting on slurm $RUNG_JOB) ==="
while squeue -j "$RUNG_JOB" -h 2>/dev/null | grep -q .; do sleep 120; done
say "rung build left the queue"

MISSING=0
for spec in "1525:64k" "3051:128k" "6102:256k" "12204:512k"; do
  N="${spec%%:*}"; LAB="${spec##*:}"
  F="$SRC/contradiction_eval_pubmed_realistic_n${N}_k3_xlong_${LAB}.jsonl"
  if [ ! -f "$F" ]; then say "!!! MISSING rung $LAB ($F)"; MISSING=1; continue; fi
  ROWS=$(wc -l < "$F")
  say "  $LAB n=$N rows=$ROWS"
  [ "$ROWS" -ge 500 ] || { say "!!! $LAB has only $ROWS rows (<500 floor)"; MISSING=1; }
done
[ "$MISSING" = "0" ] || { say "ABORTING: rung build incomplete -- nothing launched"; exit 1; }

say "=== s3 sync ==="
aws s3 sync "$SRC" "$S3" --only-show-errors || { say "!!! s3 sync failed"; exit 1; }
aws s3 ls "$S3/" | tee -a "$LOG"

say "=== gantry s3 -> weka ==="
S3_PREFIX=s3://ai2-llm/checkpoints/prasanns/_eval_bundle_eval500_v3 \
DEST_REL=ai2-llm/checkpoints/prasanns/_eval_bundle_eval500_v3 \
NAME=stage-v3-xlong-weka PRIORITY=urgent \
  bash src/scripts/train/memexpress/singletask_ladder/stage_eval500_v2_to_weka_gantry.sh >>"$LOG" 2>&1
say "weka sync submitted; waiting 10 min for it to land"
sleep 600

say "=== yarn2 copies for 256k/512k ==="
read -r -d '' YJOB <<'EOS' || true
set -uo pipefail
A=/weka/oe-training-default/ai2-llm/checkpoints/amandab
for spec in "q35-4b-dense-xlong5-dolci25-256k/step560" \
            "q35-4b-fastcomplm-xlong5-dolci25-256k/step560" \
            "q35-4b-fastcomplm-xlong5-dolci25-256k-ep1/step634"; do
  SRC="$A/$spec"; DST="${SRC}_yarn2"
  [ -d "$DST" ] && { echo "[skip] $DST exists"; continue; }
  mkdir -p "$DST"
  for f in "$SRC"/*; do ln -sfn "$f" "$DST/$(basename "$f")"; done
  rm -f "$DST/config.json"
  python - <<PY
import json
c=json.load(open("$SRC/config.json"))
m=c.setdefault("model",{})
# olmo-core config: patch the attention rope scaling in place, factor 2 (256k/512k per the
# YaRN-by-rung rule; Qwen3.5's native ceiling is 262144 and eval raises the cap ~10% over label)
def patch(d):
    if isinstance(d,dict):
        if "rope" in d and isinstance(d["rope"],dict):
            d["rope"].setdefault("scaling",{})
            d["rope"]["scaling"].update({"name":"yarn","factor":2.0,
                                         "original_max_position_embeddings":262144})
        for v in d.values(): patch(v)
    elif isinstance(d,list):
        for v in d: patch(v)
patch(m)
json.dump(c,open("$DST/config.json","w"),indent=2)
print("wrote $DST/config.json")
PY
done
ls -la $A/*_yarn2 2>/dev/null | head
EOS
gantry run --name build-yarn2-copies --workspace ai2/flex2 --budget ai2/oe-other \
  --cluster ai2/neptune --cluster ai2/saturn --weka oe-training-default:/weka/oe-training-default \
  --cpus 4 --gpus 0 --priority urgent --allow-dirty --timeout 0 --yes -- bash -c "$YJOB" >>"$LOG" 2>&1
say "yarn2 copy job submitted; waiting 8 min"
sleep 480

say "=== Pass B launches ==="
launch_b () {  # run step variant tag ckpt rungs
  say "  Pass B $4 rungs=$6"
  PYTHONPATH=src python src/scripts/train/memexpress/singletask_ladder/run_q4b_beaker_multirung_eval.py \
    "$1" ai2/saturn --task contra --variant "$3" --ckpt "$5" \
    --ladder-version v3 --xlong --xlong-rungs "$6" \
    --results-dir "$R/$4" --priority urgent >>"$LOG" 2>&1
}
for spec in "q35-4b-dense-xlong5-dolci25-256k:step560:dense:v3_q35_dense" \
            "q35-4b-fastcomplm-xlong5-dolci25-256k:step560:compressive:v3_q35_fcl" \
            "q35-4b-fastcomplm-xlong5-dolci25-256k-ep1:step634:compressive:v3_q35_fclep1"; do
  RUN="${spec%%:*}"; rest="${spec#*:}"; STEP="${rest%%:*}"; rest="${rest#*:}"; VAR="${rest%%:*}"; TAG="${rest##*:}"
  launch_b "$RUN" "$STEP" "$VAR" "${TAG}_xlong_native" "$A/$RUN/$STEP"        "64k,128k"
  launch_b "$RUN" "$STEP" "$VAR" "${TAG}_xlong_yarn2"  "$A/$RUN/${STEP}_yarn2" "256k,512k"
done

say "=== orchestrator done: Pass B submitted for all 3 checkpoints ==="
