#!/usr/bin/env bash
#
# Launch Molmo2 stage-1 on B300 (holmes) with the fastest settings measured to leave the
# objective unchanged.
#
# Usage:
#   src/scripts/train/molmo2_stage1_launch.sh <run-name> [nodes] [extra args...]
#
#   nodes defaults to 2. Anything after it is appended verbatim, so per-run overrides work:
#     molmo2_stage1_launch.sh my-run 2 --model_size=8b
#     molmo2_stage1_launch.sh baseline-run 2 --perf=off      # released recipe, no tuning
#
# ---------------------------------------------------------------------------------------------
# Why these settings
#
# Measured on holmes (B300, ~268 GiB/GPU), 150 steps, median of the last 50, against the
# released Molmo2-4B-Pretrain recipe. All three knobs are *mathematically neutral*: train CE
# agrees with the baseline to within 0.004 nats over 150 steps, and PR #828 added a test
# asserting checkpointed and uncheckpointed gradients are equal under dropout. This is not a
# quality-for-speed trade.
#
#   2 nodes                          ex/s/dev   step s   MFU     active mem   CE@150
#   released recipe                    5.957     2.027   14.1%     44.6 GiB    2.195
#   + ac_config=null                   6.767     1.801   16.3%    ~95   GiB    2.195
#   + rank_microbatch_size=20480       7.750     1.551   18.8%    174.8 GiB    2.194
#   + response_logits_only  (C2)       8.232     1.484   19.4%    124.5 GiB    2.191   <- default
#
#   ac_config=null          +13.6%  the released recipe enables activation checkpointing for
#                                   H100-class memory; B300 has ~268 GiB and the baseline used 62.
#   rank_microbatch_size    +17%    at 2 nodes each rank owns 8 sequences, so mb8 makes gradient
#                                   accumulation 1 and halves FSDP all-gather traffic.
#   response_logits_only    +3.7%   and -46 GiB, by not materialising dense mb x 2560 x 151936
#                                   logits. That headroom is what lets the other two coexist.
#
# Node-count constraint: the microbatch is a subdivision of each rank's share, so it can never
# exceed global_batch / world_size. Global batch is 128 sequences, so mb8 requires 8 seq/rank,
# which exists only at 2 nodes (16 GPUs). At 4 nodes each rank owns 4 sequences and gradient
# accumulation is already 1, so only the other two knobs apply (~+13%, measured).
#
# Deliberately NOT set - each was measured and is worse:
#   --data_prefetch_workers=8                     -4.7%, and the loader's share of step time rose
#                                                 to 17.8%; it oversubscribes the CPUs the ranks use
#   --train_module.compile_vision=false           -10.4%; the compile pays for itself despite the
#                                                 ViT recompiling on packer crop-count changes
#   --train_module.dp_config.param_dtype=bfloat16 +1.9% only, and the sole change altering numerics
#                                                 (the released recipe is fsdp.precision: float)
#
# For reference, mm_olmo's molmo3 stack on the same hardware runs 1.619 s/step at 2 nodes: slower
# than this config, faster than the un-tuned released recipe.
# ---------------------------------------------------------------------------------------------

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $(basename "$0") <run-name> [nodes=2] [extra args...]" >&2
  exit 2
fi

NAME="$1"; shift
NODES="${1:-2}"
if [[ "${NODES}" =~ ^[0-9]+$ ]]; then shift; else NODES=2; fi

# `--perf=off` runs the released recipe unchanged, for A/B against the tuning above.
PERF=on
ARGS=()
for a in "$@"; do
  case "$a" in
    --perf=off) PERF=off ;;
    --perf=on)  PERF=on ;;
    *) ARGS+=("$a") ;;
  esac
done

PERF_ARGS=()
if [[ "${PERF}" == "on" ]]; then
  # Neutral on every node count: skip activation checkpointing, skip dense logits.
  PERF_ARGS+=(--train_module.ac_config=null --train_module.response_logits_only=true)
  # mb8 = 8 sequences x 2560 tokens. Only divides evenly when a rank owns 8 sequences,
  # i.e. exactly 2 nodes at the recipe's 128-sequence global batch.
  if [[ "${NODES}" -eq 2 ]]; then
    PERF_ARGS+=(--train_module.rank_microbatch_size=20480)
  else
    echo "note: ${NODES} nodes -> keeping the default microbatch; mb8 needs 8 seq/rank (2 nodes)." >&2
  fi
fi

# Set PYTHON to pick a specific interpreter; otherwise the active env must have olmo_core.
PYTHON="${PYTHON:-python}"
if ! "${PYTHON}" -c 'import olmo_core' >/dev/null 2>&1; then
  echo "error: '${PYTHON}' cannot import olmo_core." >&2
  echo "       Activate the olmo-core environment, or set PYTHON=/path/to/python." >&2
  exit 1
fi

# cu130 image: B300 is sm_103, which the default cu128 image predates.
exec "${PYTHON}" src/scripts/train/Molmo2-Stage1.py launch "${NAME}" \
  --launch.workspace=ai2/molmofication \
  --launch.clusters=[ai2/holmes] \
  --launch.num_nodes="${NODES}" \
  --launch.priority=urgent \
  --launch.retries=10 \
  --launch.beaker_image=akshitab/olmo-core-tch2100cu130-2026-07-03 \
  "--launch.post_setup=pip install -U 'datasets>=4,<6' && pip uninstall -y torchaudio || true" \
  "--launch.env_vars=[{name: NCCL_DEBUG, value: WARN}, {name: OLMO2_FLEX_ATTN, value: '1'}, {name: HF_HOME, value: /weka/oe-training-default/jasonr/hf-home}]" \
  "${PERF_ARGS[@]}" \
  "${ARGS[@]}"
