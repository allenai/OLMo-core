#!/usr/bin/env bash
# RELAUNCH of the 2026-08-12 sweep, under a v3-first ladder policy.
#
# Why a relaunch: 48 of the original 63 jobs died at startup with "EVAL500: unbound variable" --
# the v2 arm of the bundle-selection chain had been dropped when fast/v3 were added, so under
# `set -u` every v2 job failed before any GPU work (fixed in 654d76678). The 15 v3 jobs were
# unaffected. Separately, the two landmark _yarn2 copies were clobbered back to a wrong
# old_context_len, invalidating the 4 landmark yarn2 jobs; those copies have been rebuilt.
#
# LADDER POLICY (changed from the 2026-08-12 launch): run v3 wherever v3 exists.
#
#   contra / outlier            -> v3. Real rebuilt dirs.
#   nq / rerank / oolong        -> v3. Directory symlinks to v2_clean, so identical bytes; the v3
#                                  label is a deliberate simplification, NOT a new measurement.
#                                  ⚠ these numbers are directly comparable to existing v2 hub rows
#                                  despite the differing eval_version.
#   outlier_review              -> v3. Its files live inside v3's real outlier/ dir.
#   fiqa / scifact              -> v2 ONLY. They read {root}/beir/, and the v3 root HAS NO beir/.
#   contra_fever                -> v2 ONLY. It reads {root}/contra/...fever..., and v3's rebuilt
#                                  contra/ contains only realistic-mode PubMed files.
#
# The three v2-only tasks are OOD probes with no xlong rungs, so they are base-ladder only.
#
# Usage:  [DRY=1] [ONLY=<run-substring>] [PASSES=base,xlong-native,xlong-yarn2] \
#           src/scripts/.../relaunch_2026-08-13_three_run_v3first.sh
#
# PASSES exists so the base/native passes can go out immediately while the _yarn2 copies are still
# being rebuilt -- only the yarn2 pass reads them.
set -uo pipefail

L=src/scripts/train/memexpress/singletask_ladder/run_q4b_beaker_multirung_eval.py
CKPT_ROOT=/weka/oe-training-default/ai2-llm/checkpoints/amandab
CLUSTER="${CLUSTER:-ai2/jupiter-cirrascale-2}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3.5-0.8B}"
QPOS="${QPOS:-both}"
DRY="${DRY:-0}"; DRY_FLAG=""; [ "$DRY" = "1" ] && DRY_FLAG="--dry-run"
ONLY="${ONLY:-}"
PASSES="${PASSES:-base,xlong-native,xlong-yarn2}"

want_pass() {  # want_pass <name> -> 0 if this pass is selected
  case ",${PASSES}," in *",$1,"*) return 0 ;; *) return 1 ;; esac
}

# Everything that resolves under the v3 root.
V3_BASE_TASKS="contra,outlier,nq,rerank,oolong,outlier_review"
V3_XLONG_TASKS="contra,outlier,nq,rerank,oolong"   # outlier_review has no xlong rungs
# The three the v3 bundle cannot serve at all.
V2_ONLY_TASKS="fiqa,scifact,contra_fever"

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
    --task "$tasks" --variant "$variant" --ckpt "$ckpt" \
    --results-dir "$CKPT_ROOT/$run/eval" \
    --tokenizer "$TOKENIZER" --query-position "$QPOS" \
    --eval-tag "$tag" --priority urgent \
    $DRY_FLAG "$@"
}

for entry in "${RUNS[@]}"; do
  IFS=':' read -r RUN STEP VARIANT <<< "$entry"
  [ -n "$ONLY" ] && case "$RUN" in *"$ONLY"*) ;; *) continue ;; esac
  CKPT="$CKPT_ROOT/$RUN/$STEP"
  YARN2="${CKPT}_yarn2"

  # ---- base ladder (2k-32k) ---------------------------------------------------------------
  if want_pass base; then
    submit "$RUN" "$CKPT" "$VARIANT" "$V3_BASE_TASKS"  base --ladder-version v3
    submit "$RUN" "$CKPT" "$VARIANT" "$V2_ONLY_TASKS"  base
  fi

  # ---- xlong, native RoPE (64k/128k, inside Qwen3.5's 262,144 ceiling) ---------------------
  if want_pass xlong-native; then
    submit "$RUN" "$CKPT" "$VARIANT" "$V3_XLONG_TASKS" xlong-native \
      --ladder-version v3 --xlong --xlong-only --xlong-rungs 64k,128k
  fi

  # ---- xlong, YaRN factor 2 (256k/512k) ----------------------------------------------------
  # _yarn2 copies were rebuilt with --old-context-len 262144; make_yarn_copy.py's default would
  # key off the 33344 SFT window and cap the landmark arms at 66,688 nominal reach.
  if want_pass xlong-yarn2; then
    submit "$RUN" "$YARN2" "$VARIANT" "$V3_XLONG_TASKS" xlong-yarn2 \
      --ladder-version v3 --xlong --xlong-only --xlong-rungs 256k,512k
  fi
done

echo
echo "=== all submissions issued (DRY=$DRY) ==="
