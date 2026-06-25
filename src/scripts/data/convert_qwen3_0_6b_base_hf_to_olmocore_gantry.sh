#!/usr/bin/env bash
# Download Qwen/Qwen3-0.6B-Base from HuggingFace to Weka and convert to OLMo-core format via Gantry.
#
# Step 1: snapshot_download to a staging path on Weka.
# Step 2: src/examples/huggingface/convert_checkpoint_from_hf.py to produce an OLMo-core checkpoint.
#
# Output: /weka/oe-training-default/ai2-llm/checkpoints/amandab/Qwen3-0.6B-Base-olmocore
#
# Overridable via env vars:
#   CLUSTER (ai2/jupiter-cirrascale-2)  WORKSPACE (ai2/flex2)  BUDGET (ai2/oe-other)
#   WEKA (oe-training-default)  PRIORITY (urgent)  HF_SECRET (amandab_HF_TOKEN)
set -euo pipefail

CLUSTER="${CLUSTER:-ai2/jupiter-cirrascale-2}"
WORKSPACE="${WORKSPACE:-ai2/flex2}"
BUDGET="${BUDGET:-ai2/oe-other}"
WEKA="${WEKA:-oe-training-default}"
PRIORITY="${PRIORITY:-urgent}"
HF_SECRET="${HF_SECRET:-amandab_HF_TOKEN}"

HF_MODEL_ID="Qwen/Qwen3-0.6B-Base"
WEKA_ROOT="/weka/${WEKA}"
HF_STAGING="${WEKA_ROOT}/ai2-llm/checkpoints/amandab/Qwen3-0.6B-Base-hf"
OUTPUT_DIR="${WEKA_ROOT}/ai2-llm/checkpoints/amandab/Qwen3-0.6B-Base-olmocore"

gantry run \
  --name "qwen3-0.6b-base-hf2olmocore" \
  --description "Download Qwen3-0.6B-Base from HF and convert to OLMo-core checkpoint" \
  --workspace "${WORKSPACE}" \
  --budget "${BUDGET}" \
  --cluster "${CLUSTER}" \
  --gpus 0 \
  --priority "${PRIORITY}" \
  --weka "${WEKA}:${WEKA_ROOT}" \
  --allow-dirty \
  --shared-memory 32GiB \
  --timeout 0 \
  --env-secret "HF_TOKEN=${HF_SECRET}" \
  --env HF_HUB_ENABLE_HF_TRANSFER=1 \
  --yes \
  -- bash -c "
set -euo pipefail
pip install 'huggingface_hub>=0.24' hf_transfer && \
echo '=== Step 1: downloading ${HF_MODEL_ID} to ${HF_STAGING} ===' && \
python -c \"
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${HF_MODEL_ID}',
    local_dir='${HF_STAGING}',
    token=os.environ['HF_TOKEN'],
)
print('Download complete.')
\" && \
echo '=== Step 2: converting to OLMo-core format ===' && \
python src/examples/huggingface/convert_checkpoint_from_hf.py \
  --checkpoint-input-path '${HF_STAGING}' \
  --model-arch qwen3_0_6b \
  --tokenizer qwen3 \
  --output-dir '${OUTPUT_DIR}'
"
