#!/usr/bin/env bash
# Download a Qwen3-4B(-Base) checkpoint from HuggingFace to Weka and convert it to OLMo-core
# format via Gantry. Optionally bake in YaRN RoPE scaling for long-context eval.
#
# Step 1: snapshot_download the HF model to a staging path on Weka.
# Step 2: src/examples/huggingface/convert_checkpoint_from_hf.py -> OLMo-core checkpoint.
#
# Output naming (under /weka/<WEKA>/ai2-llm/checkpoints/amandab/):
#   Qwen/Qwen3-4B          , no yarn  -> Qwen3-4B-olmocore
#   Qwen/Qwen3-4B          , yarn 2.0 -> Qwen3-4B-yarn2-olmocore
#   Qwen/Qwen3-4B-Base     , no yarn  -> Qwen3-4B-Base-olmocore
#   Qwen/Qwen3-4B-Base     , yarn 2.0 -> Qwen3-4B-Base-yarn2-olmocore
#
# Usage:
#   HF_MODEL_ID=Qwen/Qwen3-4B                 bash convert_qwen3_hf_to_olmocore_gantry.sh
#   HF_MODEL_ID=Qwen/Qwen3-4B YARN_FACTOR=2.0 bash convert_qwen3_hf_to_olmocore_gantry.sh
#   HF_MODEL_ID=Qwen/Qwen3-4B-Base            bash convert_qwen3_hf_to_olmocore_gantry.sh
#   HF_MODEL_ID=Qwen/Qwen3-4B-Base YARN_FACTOR=2.0 bash convert_qwen3_hf_to_olmocore_gantry.sh
#
# Overridable via env vars:
#   HF_MODEL_ID (Qwen/Qwen3-4B)  YARN_FACTOR (unset -> no scaling)
#   CLUSTER (ai2/jupiter-cirrascale-2)  WORKSPACE (ai2/flex2)  BUDGET (ai2/oe-other)
#   WEKA (oe-training-default)  PRIORITY (urgent)  HF_SECRET (amandab_HF_TOKEN)
set -euo pipefail

HF_MODEL_ID="${HF_MODEL_ID:-Qwen/Qwen3-4B}"
YARN_FACTOR="${YARN_FACTOR:-}"

CLUSTER="${CLUSTER:-ai2/jupiter-cirrascale-2}"
WORKSPACE="${WORKSPACE:-ai2/flex2}"
BUDGET="${BUDGET:-ai2/oe-other}"
WEKA="${WEKA:-oe-training-default}"
PRIORITY="${PRIORITY:-urgent}"
HF_SECRET="${HF_SECRET:-amandab_HF_TOKEN}"

# Derive checkpoint names from the model id and (optional) yarn factor.
MODEL_BASENAME="${HF_MODEL_ID##*/}"                 # e.g. Qwen3-4B or Qwen3-4B-Base
YARN_SUFFIX=""
YARN_FLAG=""
if [ -n "${YARN_FACTOR}" ]; then
  YARN_SUFFIX="-yarn${YARN_FACTOR%.*}"              # 2.0 -> -yarn2
  # Skip validation for YaRN: the converter compares the olmo-core forward pass against the
  # vanilla HF model (no scaling), so the intentional RoPE remapping makes logits diverge. The
  # weight conversion is identical to the (already-validated) no-yarn path; only runtime RoPE
  # inv_freqs differ, so there is nothing extra to validate here.
  YARN_FLAG="--yarn-factor ${YARN_FACTOR} --skip-validation"
fi

WEKA_ROOT="/weka/${WEKA}"
HF_STAGING="${WEKA_ROOT}/ai2-llm/checkpoints/amandab/${MODEL_BASENAME}-hf"
OUTPUT_DIR="${WEKA_ROOT}/ai2-llm/checkpoints/amandab/${MODEL_BASENAME}${YARN_SUFFIX}-olmocore"

JOB_NAME="$(echo "${MODEL_BASENAME}${YARN_SUFFIX}-hf2olmocore" | tr '[:upper:]' '[:lower:]')"

echo "==> ${HF_MODEL_ID} -> ${OUTPUT_DIR} (yarn_factor=${YARN_FACTOR:-none})"

gantry run \
  --name "${JOB_NAME}" \
  --description "Download ${HF_MODEL_ID} from HF and convert to OLMo-core (yarn=${YARN_FACTOR:-none})" \
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
echo '=== Step 2: converting to OLMo-core format (yarn=${YARN_FACTOR:-none}) ===' && \
python src/examples/huggingface/convert_checkpoint_from_hf.py \
  --checkpoint-input-path '${HF_STAGING}' \
  --model-arch qwen3_4b \
  --tokenizer qwen3 \
  ${YARN_FLAG} \
  --output-dir '${OUTPUT_DIR}'
"
