#!/usr/bin/env bash

set -euo pipefail

# Submit a Beaker (gantry) job to convert an OLMo checkpoint to Hugging Face format.
#
# Example:
#   bash src/scripts/beaker/gantry_convert_to_hf.sh \
#     -i /weka/oe-training-default/ai2-llm/checkpoints/sanjaya/olmo2-7B-sft-router/router-sft/step1212 \
#     -o /weka/oe-training-default/sanjaya/flexolmo/checkpoints/router-sft/step1212-hf \
#     -s 4096

INPUT=""
OUTPUT=""
SEQ_LEN=4096
DEVICE="cuda"
SKIP_VALIDATION=1

# Gantry / Beaker defaults (can be overridden by flags or env vars)
BUDGET=${BUDGET:-"ai2/oe-base"}
WORKSPACE=${WORKSPACE:-"ai2/flex2"}
CLUSTER=${CLUSTER:-"ai2/jupiter-cirrascale-2"}
PRIORITY=${PRIORITY:-"urgent"}
GPUS=${GPUS:-1}

# Optional env secrets (set to secret names when needed, leave empty to skip)
ENV_SECRET_HF_TOKEN=${ENV_SECRET_HF_TOKEN:-""}
ENV_SECRET_AWS_ACCESS_KEY_ID=${ENV_SECRET_AWS_ACCESS_KEY_ID:-""}
ENV_SECRET_AWS_SECRET_ACCESS_KEY=${ENV_SECRET_AWS_SECRET_ACCESS_KEY:-""}

usage() {
  echo "Usage: $0 -i INPUT_DIR -o OUTPUT_DIR [-s SEQ_LEN] [--device cpu|cuda] [--no-skip-validation] [--cluster CLUSTER] [--budget BUDGET] [--workspace WORKSPACE] [--priority PRIORITY] [--gpus N]" >&2
  exit 1
}

ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input) INPUT="$2"; shift 2;;
    -o|--output) OUTPUT="$2"; shift 2;;
    -s|--seq-len) SEQ_LEN="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --no-skip-validation) SKIP_VALIDATION=0; shift 1;;
    --cluster) CLUSTER="$2"; shift 2;;
    --budget) BUDGET="$2"; shift 2;;
    --workspace) WORKSPACE="$2"; shift 2;;
    --priority) PRIORITY="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --env-secret-hf-token) ENV_SECRET_HF_TOKEN="$2"; shift 2;;
    --env-secret-aws-access-key-id) ENV_SECRET_AWS_ACCESS_KEY_ID="$2"; shift 2;;
    --env-secret-aws-secret-access-key) ENV_SECRET_AWS_SECRET_ACCESS_KEY="$2"; shift 2;;
    --) shift; ARGS+=("$@"); break;;
    *) echo "Unknown arg: $1" >&2; usage;;
  esac
done

if [[ -z "$INPUT" || -z "$OUTPUT" ]]; then
  usage
fi

# Determine Weka bucket from either input or output path
WEKA_BUCKET=""
if [[ "$INPUT" == /weka/* ]]; then
  WEKA_BUCKET=$(echo "$INPUT" | awk -F/ '{print $3}')
fi
if [[ -z "$WEKA_BUCKET" && "$OUTPUT" == /weka/* ]]; then
  WEKA_BUCKET=$(echo "$OUTPUT" | awk -F/ '{print $3}')
fi

# Create a safe short job name
INPUT_BASE=$(basename "$INPUT")
SAFE_INPUT=$(echo "$INPUT_BASE" | sed 's/[^a-zA-Z0-9_-]//g' | cut -c1-20)
JOB_NAME="convert-hf-${SAFE_INPUT}-$(date +%y%m%d%H%M)"

echo "Submitting gantry job: $JOB_NAME" >&2
echo "  Input : $INPUT" >&2
echo "  Output: $OUTPUT" >&2
echo "  SeqLen: $SEQ_LEN" >&2
echo "  Device: $DEVICE" >&2
echo "  GPUs  : $GPUS" >&2
if [[ -n "$WEKA_BUCKET" ]]; then
  echo "  Weka  : $WEKA_BUCKET" >&2
fi

GANTRY_ARGS=(
  run
  --name "$JOB_NAME"
  --budget "$BUDGET"
  --workspace "$WORKSPACE"
  --cluster "$CLUSTER"
  --priority "$PRIORITY"
  --gpus "$GPUS"
  --install "conda shell.bash activate base; pip install -e '.[all]'; pip freeze;"
)

if [[ -n "$WEKA_BUCKET" ]]; then
  GANTRY_ARGS+=(--weka "${WEKA_BUCKET}:/weka/${WEKA_BUCKET}")
fi

if [[ -n "$ENV_SECRET_HF_TOKEN" ]]; then
  GANTRY_ARGS+=(--env-secret "HF_TOKEN=${ENV_SECRET_HF_TOKEN}")
fi
if [[ -n "$ENV_SECRET_AWS_ACCESS_KEY_ID" ]]; then
  GANTRY_ARGS+=(--env-secret "AWS_ACCESS_KEY_ID=${ENV_SECRET_AWS_ACCESS_KEY_ID}")
fi
if [[ -n "$ENV_SECRET_AWS_SECRET_ACCESS_KEY" ]]; then
  GANTRY_ARGS+=(--env-secret "AWS_SECRET_ACCESS_KEY=${ENV_SECRET_AWS_SECRET_ACCESS_KEY}")
fi

CONVERT_CMD=(
  "PYTHONPATH=."
  python src/examples/huggingface/convert_checkpoint_to_hf.py
  -i "$INPUT"
  -o "$OUTPUT"
  -s "$SEQ_LEN"
  --device "$DEVICE"
)
if [[ $SKIP_VALIDATION -eq 1 ]]; then
  CONVERT_CMD+=(--skip-validation)
fi
if [[ ${#ARGS[@]} -gt 0 ]]; then
  CONVERT_CMD+=("${ARGS[@]}")
fi

set -x
gantry "${GANTRY_ARGS[@]}" -- bash -c "${CONVERT_CMD[*]}"
set +x


