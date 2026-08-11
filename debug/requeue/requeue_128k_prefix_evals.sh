#!/usr/bin/env bash
# Re-run the 128k xlong rungs that were poisoned by the pre-`18d129f67` maxlen cap.
#
# WHY: any --xlong eval launched before 18d129f67 (2026-07-29 16:53 PDT) capped prompts at
# `rung_label + 1024` -> MAX_LENGTH=132096 at 128k, which is BELOW what the 128k rung builds.
# The prompt tail (question + output-format instruction) was cut off and the rung scored ~0.000
# at parse_rate 1.0. Full diagnosis: records/xlong-128k-maxlen-truncation.md
#
# The 69 affected rows were deleted from results-hub (contradiction 14, contra 9, outlier 23,
# nq 23) across 22 checkpoints. This script regenerates exactly those cells.
#
# The fix needs no flag: the current runner computes MAX_LENGTH = label*1.10 + 2048 = 146227 at
# 128k, and eval_lc_native.py re-raises an undersized --max-length by the same rule. Just launch
# from a checkout at or after 18d129f67.
#
# --xlong-only is deliberate: the 2k-32k rungs for these checkpoints are already valid (they were
# far under even the broken cap), so re-running them would burn GPU to reproduce existing numbers.
#
# top-k: NOT passed. GenerationConfig.landmark_top_k_fraction defaults to 0.1, which is what every
# deleted row used (landmark_top_k_percentage=0.1). Passing --landmark-top-k-blocks would CHANGE
# the config rather than reproduce it.
#
# Usage:
#   bash debug/requeue/requeue_128k_prefix_evals.sh            # dry run, prints every command
#   SUBMIT=1 bash debug/requeue/requeue_128k_prefix_evals.sh   # actually submit
#   CLUSTER=ai2/jupiter-cirrascale-2 SUBMIT=1 bash ...         # override the cluster
set -euo pipefail

CLUSTER="${CLUSTER:-ai2/neptune}"
SUBMIT="${SUBMIT:-0}"
LAUNCHER="src/scripts/train/memexpress/singletask_ladder/run_q4b_beaker_multirung_eval.py"

[ -f "$LAUNCHER" ] || { echo "run me from the OLMo-core repo root"; exit 2; }

# each entry: run_name|task|checkpoint|extra_flags
JOBS=(
  "q35-4b-fastcomplm-xlong5-dolci25-256k-ep1|contra|/weka/oe-training-default/ai2-llm/checkpoints/amandab/q35-4b-fastcomplm-xlong5-dolci25-256k-ep1/step634|--variant compressive --tokenizer Qwen/Qwen3.5-0.8B"
  "q35-4b-fastcomplm-xlong5-dolci25-256k-ep1|nq|/weka/oe-training-default/ai2-llm/checkpoints/amandab/q35-4b-fastcomplm-xlong5-dolci25-256k-ep1/step634|--variant compressive --tokenizer Qwen/Qwen3.5-0.8B"
  "q35-4b-fastcomplm-xlong5-dolci25-256k-ep1|outlier|/weka/oe-training-default/ai2-llm/checkpoints/amandab/q35-4b-fastcomplm-xlong5-dolci25-256k-ep1/step634|--variant compressive --tokenizer Qwen/Qwen3.5-0.8B"
  "q35-4b-fastcomplm-xlong5-dolci25-256k|contra|/weka/oe-training-default/ai2-llm/checkpoints/amandab/q35-4b-fastcomplm-xlong5-dolci25-256k/step560|--variant compressive --tokenizer Qwen/Qwen3.5-0.8B"
  "q35-4b-fastcomplm-xlong5-dolci25-256k|nq|/weka/oe-training-default/ai2-llm/checkpoints/amandab/q35-4b-fastcomplm-xlong5-dolci25-256k/step560|--variant compressive --tokenizer Qwen/Qwen3.5-0.8B"
  "q35-4b-fastcomplm-xlong5-dolci25-256k|outlier|/weka/oe-training-default/ai2-llm/checkpoints/amandab/q35-4b-fastcomplm-xlong5-dolci25-256k/step560|--variant compressive --tokenizer Qwen/Qwen3.5-0.8B"
  "q4b-comp-gate-temp-5task-dolci25-32k|contra|/weka/oe-training-default/ai2-llm/checkpoints/amandab/q4b-comp-gate-temp-5task-dolci25-32k/step8550|--variant compressive"
  "q4b-comp-gate-temp-5task-dolci25-32k|nq|/weka/oe-training-default/ai2-llm/checkpoints/amandab/q4b-comp-gate-temp-5task-dolci25-32k/step8550|--variant compressive"
  "q4b-comp-gate-temp-5task-dolci25-32k|outlier|/weka/oe-training-default/ai2-llm/checkpoints/amandab/q4b-comp-gate-temp-5task-dolci25-32k/step8550|--variant compressive"
  "q4b-comp-blocklocal-5task-32k-nocpt|contra|/weka/oe-training-default/ai2-llm/checkpoints/prasanns/q4b-comp-blocklocal-5task-32k-nocpt/step8550|--variant compressive"
  "q4b-comp-blocklocal-5task-32k-nocpt|nq|/weka/oe-training-default/ai2-llm/checkpoints/prasanns/q4b-comp-blocklocal-5task-32k-nocpt/step8550|--variant compressive"
  "q4b-comp-blocklocal-5task-32k-nocpt|outlier|/weka/oe-training-default/ai2-llm/checkpoints/prasanns/q4b-comp-blocklocal-5task-32k-nocpt/step8550|--variant compressive"
  "q4b-comp-partialrope-5task-32k-nocpt|contra|/weka/oe-training-default/ai2-llm/checkpoints/prasanns/q4b-comp-partialrope-5task-32k-nocpt/step8550|--variant compressive"
  "q4b-comp-partialrope-5task-32k-nocpt|nq|/weka/oe-training-default/ai2-llm/checkpoints/prasanns/q4b-comp-partialrope-5task-32k-nocpt/step8550|--variant compressive"
  "q4b-comp-partialrope-5task-32k-nocpt|outlier|/weka/oe-training-default/ai2-llm/checkpoints/prasanns/q4b-comp-partialrope-5task-32k-nocpt/step8550|--variant compressive"
  "q4b-compressive-5task-32k-nocpt-fixdata|contra|/weka/oe-training-default/ai2-llm/checkpoints/prasanns/q4b-compressive-5task-32k-nocpt-fixdata/step8550|--variant compressive"
  "q4b-compressive-5task-32k-nocpt-fixdata|nq|/weka/oe-training-default/ai2-llm/checkpoints/prasanns/q4b-compressive-5task-32k-nocpt-fixdata/step8550|--variant compressive"
  "q4b-compressive-5task-32k-nocpt-fixdata|outlier|/weka/oe-training-default/ai2-llm/checkpoints/prasanns/q4b-compressive-5task-32k-nocpt-fixdata/step8550|--variant compressive"
  "q4b-dense-5task-32k-nocpt-longer|contra|/weka/oe-training-default/ai2-llm/checkpoints/prasanns/q4b-dense-5task-32k-nocpt-longer/step21400|--variant dense"
  "q4b-dense-5task-32k-nocpt-longer|nq|/weka/oe-training-default/ai2-llm/checkpoints/prasanns/q4b-dense-5task-32k-nocpt-longer/step21400|--variant dense"
  "q4b-dense-5task-32k-nocpt-longer|outlier|/weka/oe-training-default/ai2-llm/checkpoints/prasanns/q4b-dense-5task-32k-nocpt-longer/step21400|--variant dense"
  "q4b-dense-5task-dolci25-32k-nocpt|contra|/weka/oe-training-default/ai2-llm/checkpoints/prasanns/q4b-dense-5task-dolci25-32k-nocpt/step10700|--variant dense"
  "q4b-dense-5task-dolci25-32k-nocpt|nq|/weka/oe-training-default/ai2-llm/checkpoints/prasanns/q4b-dense-5task-dolci25-32k-nocpt/step10700|--variant dense"
  "q4b-dense-5task-dolci25-32k-nocpt|outlier|/weka/oe-training-default/ai2-llm/checkpoints/prasanns/q4b-dense-5task-dolci25-32k-nocpt/step10700|--variant dense"
  "q4b-dense-5task-dolci50-32k-nocpt|contra|/weka/oe-training-default/ai2-llm/checkpoints/prasanns/q4b-dense-5task-dolci50-32k-nocpt/step10700|--variant dense"
  "q4b-dense-5task-dolci50-32k-nocpt|nq|/weka/oe-training-default/ai2-llm/checkpoints/prasanns/q4b-dense-5task-dolci50-32k-nocpt/step10700|--variant dense"
  "q4b-dense-5task-dolci50-32k-nocpt|outlier|/weka/oe-training-default/ai2-llm/checkpoints/prasanns/q4b-dense-5task-dolci50-32k-nocpt/step10700|--variant dense"
  "q4b-dense-5task-v2-32k-nocpt|contra|/weka/oe-training-default/ai2-llm/checkpoints/prasanns/q4b-dense-5task-v2-32k-nocpt/step1465|--variant dense"
  "q4b-dense-5task-v2-32k-nocpt|nq|/weka/oe-training-default/ai2-llm/checkpoints/prasanns/q4b-dense-5task-v2-32k-nocpt/step1465|--variant dense"
  "q4b-dense-5task-v2-32k-nocpt|outlier|/weka/oe-training-default/ai2-llm/checkpoints/prasanns/q4b-dense-5task-v2-32k-nocpt/step1465|--variant dense"
  "q4b-dense-dolci-32k-nocpt|contra|/weka/oe-training-default/ai2-llm/checkpoints/prasanns/q4b-dense-dolci-32k-nocpt/step10700|--variant dense"
  "q4b-dense-dolci-32k-nocpt|nq|/weka/oe-training-default/ai2-llm/checkpoints/prasanns/q4b-dense-dolci-32k-nocpt/step10700|--variant dense"
  "q4b-dense-dolci-32k-nocpt|outlier|/weka/oe-training-default/ai2-llm/checkpoints/prasanns/q4b-dense-dolci-32k-nocpt/step10700|--variant dense"
  "q4b-comp-block128-5task-dolci25-nocpt|contra|/weka/oe-training-default/ai2-llm/checkpoints/q4b-comp-block128-5task-dolci25-nocpt/step8550|--variant compressive"
  "q4b-comp-block128-5task-dolci25-nocpt|nq|/weka/oe-training-default/ai2-llm/checkpoints/q4b-comp-block128-5task-dolci25-nocpt/step8550|--variant compressive"
  "q4b-comp-block128-5task-dolci25-nocpt|outlier|/weka/oe-training-default/ai2-llm/checkpoints/q4b-comp-block128-5task-dolci25-nocpt/step8550|--variant compressive"
  "q4b-comp-block128-cptmix-5task-32k|contra|/weka/oe-training-default/ai2-llm/checkpoints/q4b-comp-block128-cptmix-5task-32k/step1465|--variant compressive"
  "q4b-comp-block128-cptmix-5task-32k|nq|/weka/oe-training-default/ai2-llm/checkpoints/q4b-comp-block128-cptmix-5task-32k/step1465|--variant compressive"
  "q4b-comp-block128-cptmix-5task-32k|outlier|/weka/oe-training-default/ai2-llm/checkpoints/q4b-comp-block128-cptmix-5task-32k/step1465|--variant compressive"
  "q4b-comp-block16-5task-dolci25-nocpt|contra|/weka/oe-training-default/ai2-llm/checkpoints/q4b-comp-block16-5task-dolci25-nocpt/step8550|--variant compressive"
  "q4b-comp-block16-5task-dolci25-nocpt|nq|/weka/oe-training-default/ai2-llm/checkpoints/q4b-comp-block16-5task-dolci25-nocpt/step8550|--variant compressive"
  "q4b-comp-block16-5task-dolci25-nocpt|outlier|/weka/oe-training-default/ai2-llm/checkpoints/q4b-comp-block16-5task-dolci25-nocpt/step8550|--variant compressive"
  "q4b-comp-block32-5task-dolci25-nocpt|contra|/weka/oe-training-default/ai2-llm/checkpoints/q4b-comp-block32-5task-dolci25-nocpt/step8550|--variant compressive"
  "q4b-comp-block32-5task-dolci25-nocpt|nq|/weka/oe-training-default/ai2-llm/checkpoints/q4b-comp-block32-5task-dolci25-nocpt/step8550|--variant compressive"
  "q4b-comp-block32-5task-dolci25-nocpt|outlier|/weka/oe-training-default/ai2-llm/checkpoints/q4b-comp-block32-5task-dolci25-nocpt/step8550|--variant compressive"
  "q4b-comp-block32-cptmix-5task-32k|contra|/weka/oe-training-default/ai2-llm/checkpoints/q4b-comp-block32-cptmix-5task-32k/step1465|--variant compressive"
  "q4b-comp-block32-cptmix-5task-32k|nq|/weka/oe-training-default/ai2-llm/checkpoints/q4b-comp-block32-cptmix-5task-32k/step1465|--variant compressive"
  "q4b-comp-block32-cptmix-5task-32k|outlier|/weka/oe-training-default/ai2-llm/checkpoints/q4b-comp-block32-cptmix-5task-32k/step1465|--variant compressive"
  "q4b-comp-block64-5task-dolci25-nocpt|contra|/weka/oe-training-default/ai2-llm/checkpoints/q4b-comp-block64-5task-dolci25-nocpt/step8550|--variant compressive"
  "q4b-comp-block64-5task-dolci25-nocpt|nq|/weka/oe-training-default/ai2-llm/checkpoints/q4b-comp-block64-5task-dolci25-nocpt/step8550|--variant compressive"
  "q4b-comp-block64-5task-dolci25-nocpt|outlier|/weka/oe-training-default/ai2-llm/checkpoints/q4b-comp-block64-5task-dolci25-nocpt/step8550|--variant compressive"
  "q4b-mcl-block64-2lm-max-5task-dolci25-32k-nocpt|contra|/weka/oe-training-default/ai2-llm/checkpoints/q4b-mcl-block64-2lm-max-5task-dolci25-32k-nocpt/step8550|--variant compressive"
  "q4b-mcl-block64-2lm-max-5task-dolci25-32k-nocpt|nq|/weka/oe-training-default/ai2-llm/checkpoints/q4b-mcl-block64-2lm-max-5task-dolci25-32k-nocpt/step8550|--variant compressive"
  "q4b-mcl-block64-2lm-max-5task-dolci25-32k-nocpt|outlier|/weka/oe-training-default/ai2-llm/checkpoints/q4b-mcl-block64-2lm-max-5task-dolci25-32k-nocpt/step8550|--variant compressive"
  "q4b-mcl-block64-2lm-mean-5task-dolci25-32k-nocpt|contra|/weka/oe-training-default/ai2-llm/checkpoints/q4b-mcl-block64-2lm-mean-5task-dolci25-32k-nocpt/step8550|--variant compressive"
  "q4b-mcl-block64-2lm-mean-5task-dolci25-32k-nocpt|nq|/weka/oe-training-default/ai2-llm/checkpoints/q4b-mcl-block64-2lm-mean-5task-dolci25-32k-nocpt/step8550|--variant compressive"
  "q4b-mcl-block64-2lm-mean-5task-dolci25-32k-nocpt|outlier|/weka/oe-training-default/ai2-llm/checkpoints/q4b-mcl-block64-2lm-mean-5task-dolci25-32k-nocpt/step8550|--variant compressive"
  "q4b-mcl-block64-4lm-max-5task-dolci25-32k-nocpt|contra|/weka/oe-training-default/ai2-llm/checkpoints/q4b-mcl-block64-4lm-max-5task-dolci25-32k-nocpt/step8550|--variant compressive"
  "q4b-mcl-block64-4lm-max-5task-dolci25-32k-nocpt|nq|/weka/oe-training-default/ai2-llm/checkpoints/q4b-mcl-block64-4lm-max-5task-dolci25-32k-nocpt/step8550|--variant compressive"
  "q4b-mcl-block64-4lm-max-5task-dolci25-32k-nocpt|outlier|/weka/oe-training-default/ai2-llm/checkpoints/q4b-mcl-block64-4lm-max-5task-dolci25-32k-nocpt/step8550|--variant compressive"
  "q4b-mcl-block64-4lm-mean-5task-dolci25-32k-nocpt|contra|/weka/oe-training-default/ai2-llm/checkpoints/q4b-mcl-block64-4lm-mean-5task-dolci25-32k-nocpt/step8550|--variant compressive"
  "q4b-mcl-block64-4lm-mean-5task-dolci25-32k-nocpt|nq|/weka/oe-training-default/ai2-llm/checkpoints/q4b-mcl-block64-4lm-mean-5task-dolci25-32k-nocpt/step8550|--variant compressive"
  "q4b-mcl-block64-4lm-mean-5task-dolci25-32k-nocpt|outlier|/weka/oe-training-default/ai2-llm/checkpoints/q4b-mcl-block64-4lm-mean-5task-dolci25-32k-nocpt/step8550|--variant compressive"
)

echo "=== requeue 128k pre-fix evals | 63 jobs | cluster=$CLUSTER | submit=$SUBMIT ==="
n=0
for entry in "${JOBS[@]}"; do
  IFS='|' read -r run task ckpt extra <<< "$entry"
  n=$((n+1))
  cmd=(python "$LAUNCHER" "$run" "$CLUSTER"
       --task "$task" --ckpt "$ckpt"
       --xlong --xlong-only --xlong-rungs 128k
       --eval-tag refix128k --priority urgent)
  [ -n "$extra" ] && cmd+=($extra)
  [ "$SUBMIT" = "1" ] || cmd+=(--dry-run)
  echo "--- [$n/63] $run / $task"
  PYTHONPATH=src "${cmd[@]}"
done
echo "=== done: $n jobs ==="

# MANUAL: these checkpoints had an unusable weka path in results-hub and were not
#   checkpoints/prasanns/q4b-dense-5task-32k-nocpt-fixdata
# Resolve the absolute /weka/... step dir, then add them above.
