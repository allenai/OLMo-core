#!/usr/bin/env bash
#
# Sweep launcher for long-context OLMo-2 7B experiments.
#
# This is a lightweight alternative to the Python sweep script: it just calls
# ``OLMo2-7B-long-context.py`` with the right OmegaConf overrides.
#
# USAGE
#   ./OLMo2-7B-long-context-sweep.sh CLUSTER_NAME
#
# Example:
#   ./OLMo2-7B-long-context-sweep.sh ai2/pluto-cirrascale
#
# The script submits each experiment to Beaker with the ``launch`` sub-command.
# It disables the profiler and enables WandB logging so you can inspect
# "throughput/TPS (actual avg)" for global TPS numbers.
# -----------------------------------------------------------------------------
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 CLUSTER_NAME" >&2
  exit 1
fi

CLUSTER=$1
PY_SCRIPT="src/scripts/train/profile/OLMo2-7B-long-context.py"
DATE=$(date +%y%m%d)

# CONSTANTS -------------------------------------------------------------------
CONTEXT_LEN=$((4*16384))   # 65,536
TOKENS_PER_BATCH_FACTOR=32 # default factor for 16-GPU runs

# Helper to compute global batch size (tokens) given factor.
calc_gbs () {
  local factor=$1
  echo $((factor * CONTEXT_LEN))
}

# -----------------------------------------------------------------------------
# EXPERIMENT DEFINITIONS
# name  bs_factor  num_gpus  tp  cp  ac_enabled  gqa_ratio
# -----------------------------------------------------------------------------
CONFIG_MATRIX=(
  "tp8_cp2_dp1_gqa 32 16 8 2 false 0.25"
  "tp8_dp2_gqa 32 16 8 none false 0.25"
  "tp4_cp2_dp2_gqa 32 16 4 2 false 0.25"
  "tp4_cp4_dp1_gqa 32 16 4 4 false 0.25"
)

# -----------------------------------------------------------------------------
# MAIN LOOP --------------------------------------------------------------------
# -----------------------------------------------------------------------------
for cfg in "${CONFIG_MATRIX[@]}"; do
  read -r NAME BS_FACTOR NUM_GPUS TP CP AC GQA <<< "$cfg"

  RUN_NAME="${NAME}-${DATE}-$(date +%H%M%S)"
  GLOBAL_BS=$(calc_gbs "$BS_FACTOR")
  NUM_NODES=$((NUM_GPUS/8))

  echo "[INFO] Launching $RUN_NAME (GPUs=$NUM_GPUS, nodes=$NUM_NODES, tp=$TP, cp=$CP, ac=$AC, gqa=$GQA)"

  # ---------------------------------------------------------------
  # Build override list
  # ---------------------------------------------------------------
  OVERRIDES=(
    "--trainer.callbacks.profiler.enabled=True"
    "--trainer.callbacks.wandb.enabled=True"
    "--data_loader.global_batch_size=${GLOBAL_BS}"
    "--launch.num_nodes=${NUM_NODES}"
    "--launch.num_gpus=8"
  )

  # TP override
  if [[ "$TP" != "none" ]]; then
    OVERRIDES+=("--train_module.tp_config.degree=${TP}")
  else
    OVERRIDES+=("--train_module.tp_config=null")
  fi

  # CP override
  if [[ "$CP" != "none" ]]; then
    OVERRIDES+=("--train_module.cp_config.degree=${CP}")
  else
    OVERRIDES+=("--train_module.cp_config=null")
  fi

  # AC override (disable if needed)
  if [[ "$AC" == "false" ]]; then
    OVERRIDES+=("--train_module.ac_config=null")
  fi

  # GQA / n_kv_heads override
  if [[ "$GQA" == "none" ]]; then
    OVERRIDES+=("--model.block.attention.n_kv_heads=null")
  fi

  # Combine overrides array into single command call.
  echo "[CMD] python $PY_SCRIPT launch $RUN_NAME $CLUSTER ${OVERRIDES[@]}"
  python "$PY_SCRIPT" launch "$RUN_NAME" "$CLUSTER" "${OVERRIDES[@]}"

done