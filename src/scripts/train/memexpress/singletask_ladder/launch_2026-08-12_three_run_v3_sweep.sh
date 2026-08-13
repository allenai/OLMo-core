#!/usr/bin/env bash
# Queue the v2/v3 eval sweep for the three runs that finished 2026-08-11/12:
#
#   q35-4b-dense-xlong5-qboth-dolci25-256k          step2240   dense     (xlong5 @ 256k)
#   q35-4b-fastlm-5task-dolci25-33344-datamatch     step10858  landmark  (data-vs-compute ablation)
#   q35-4b-fastlm-5task-dolci25-33344-tokenmatch    step10515  landmark  (data-vs-compute ablation)
#
# LADDER ASSIGNMENT (per the requested matrix, not the run-evals default of "v2 for everything").
# Every "does v3 exist for this task" call below was checked against weka, not inferred from docs --
# the v3 root holds exactly `contra` and `outlier` as real dirs plus nq/oolong/rerank symlinks:
#
#   contra                  -> v3 ONLY. v2 scores a realistic-mode-trained model on both-mode gold;
#                              all three runs train realistic-mode contradiction, so the v2 number
#                              measures a task none of them was trained for.
#   outlier BASE (2k-32k)   -> v2 only. v3's base rungs are BYTE-IDENTICAL to v2's (md5-verified,
#                              all four rungs) -- only the xlong rungs were rebuilt. Running both
#                              here would be one measurement filed under two names.
#   outlier XLONG (64k+)    -> v2 AND v3. Here they genuinely differ: v3 scales K with n, v2 pins
#                              K at 25 while n grows 32x. Both are real, non-comparable numbers.
#   nq / rerank / oolong    -> ONE copy, run as v2. In the v3 bundle these are directory SYMLINKS
#                              back at v2_clean, so a v3 run reads byte-identical files. Labeling
#                              that result v3 would assert a measurement that was never made.
#   outlier_review          -> ONE copy, v2. It reads {root}/outlier/outlier_review_matched_n*,
#                              which does resolve under v3 -- but those four files are md5-identical
#                              to v2's, so v3 would again be a relabel, not a measurement.
#   fiqa / scifact          -> ONE copy, v2. They read {root}/beir/, and v3 HAS NO beir DIRECTORY.
#   contra_fever            -> ONE copy, v2. It reads {root}/contra/...fever..., and v3's rebuilt
#                              contra dir contains only realistic-mode PubMed files.
#   (the four OOD tasks have no xlong rung files at all -> base ladder only)
#
# All three checkpoints live under checkpoints/amandab/, not the runner's default prasanns/, so
# --ckpt and --results-dir are both explicit; without them the runner globs an empty prasanns/<run>.
#
# Usage:  [DRY=1] src/scripts/train/memexpress/singletask_ladder/launch_2026-08-12_three_run_v3_sweep.sh
set -uo pipefail

L=src/scripts/train/memexpress/singletask_ladder/run_q4b_beaker_multirung_eval.py
CKPT_ROOT=/weka/oe-training-default/ai2-llm/checkpoints/amandab
CLUSTER="${CLUSTER:-ai2/jupiter-cirrascale-2}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3.5-0.8B}"   # Qwen3.5 family; Qwen3-4B would score ~0.000 silently
QPOS="${QPOS:-both}"                          # all three trained on query-both shards
DRY="${DRY:-0}"
DRY_FLAG=""; [ "$DRY" = "1" ] && DRY_FLAG="--dry-run"

# v2 base = everything except contra (v3-only). outlier IS here: its base rungs are v2==v3.
V2_BASE_TASKS="outlier,nq,rerank,oolong,fiqa,scifact,outlier_review,contra_fever"
# v3 base = contra alone. outlier is deliberately absent: identical files to v2 at these rungs.
V3_BASE_TASKS="contra"
# xlong rung files exist only for the doc-pool five; contra is v3-only, so v2 xlong is the other four.
V2_XLONG_TASKS="outlier,nq,rerank,oolong"
# v3 xlong = contra (realistic mode) + outlier (scale-K); both are real rebuilds at these rungs.
V3_XLONG_TASKS="contra,outlier"

# run_name : step : variant
RUNS=(
  "q35-4b-dense-xlong5-qboth-dolci25-256k:step2240:dense"
  "q35-4b-fastlm-5task-dolci25-33344-datamatch:step10858:landmark"
  "q35-4b-fastlm-5task-dolci25-33344-tokenmatch:step10515:landmark"
)

submit() {  # submit <run> <ckpt> <variant> <tasks> <eval-tag> [extra flags...]
  local run="$1" ckpt="$2" variant="$3" tasks="$4" tag="$5"; shift 5
  echo
  echo "############################################################"
  echo "# $run | $tasks | tag=$tag | $*"
  echo "############################################################"
  PYTHONPATH=src python "$L" "$run" "$CLUSTER" \
    --task "$tasks" \
    --variant "$variant" \
    --ckpt "$ckpt" \
    --results-dir "$CKPT_ROOT/$run/eval" \
    --tokenizer "$TOKENIZER" \
    --query-position "$QPOS" \
    --eval-tag "$tag" \
    --priority urgent \
    $DRY_FLAG "$@"
}

for entry in "${RUNS[@]}"; do
  IFS=':' read -r RUN STEP VARIANT <<< "$entry"
  CKPT="$CKPT_ROOT/$RUN/$STEP"
  YARN2="${CKPT}_yarn2"

  # ---- base ladder (2k-32k) --------------------------------------------------------------
  submit "$RUN" "$CKPT" "$VARIANT" "$V2_BASE_TASKS" base
  submit "$RUN" "$CKPT" "$VARIANT" "$V3_BASE_TASKS" base --ladder-version v3

  # ---- xlong, native RoPE (64k/128k are inside Qwen3.5's 262,144 ceiling) ------------------
  submit "$RUN" "$CKPT" "$VARIANT" "$V2_XLONG_TASKS" xlong-native \
    --xlong --xlong-only --xlong-rungs 64k,128k
  submit "$RUN" "$CKPT" "$VARIANT" "$V3_XLONG_TASKS" xlong-native \
    --ladder-version v3 --xlong --xlong-only --xlong-rungs 64k,128k

  # ---- xlong, YaRN factor 2 (256k/512k) ---------------------------------------------------
  # 256k is in this group, not the native one: prompts run 0.4-4% OVER the rung label, so the
  # realized cap crosses the 262,144 ceiling even though the label sits exactly on it.
  # These _yarn2 copies were built with --old-context-len 262144, NOT the make_yarn_copy.py default
  # of the SFT window -- at 33344 the default would have capped the landmark arms' reach at 66,688.
  submit "$RUN" "$YARN2" "$VARIANT" "$V2_XLONG_TASKS" xlong-yarn2 \
    --xlong --xlong-only --xlong-rungs 256k,512k
  submit "$RUN" "$YARN2" "$VARIANT" "$V3_XLONG_TASKS" xlong-yarn2 \
    --ladder-version v3 --xlong --xlong-only --xlong-rungs 256k,512k
done

echo
echo "=== all submissions issued (DRY=$DRY) ==="
