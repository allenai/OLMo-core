#!/usr/bin/env bash
# Launch the Dolci-Instruct-SFT -> Qwen3 SFT conversion on a Beaker CPU node via gantry.
#
# This wraps src/scripts/data/convert_dolci_instruct_sft.py. It runs on a single CPU node with the
# weka bucket mounted; the dataset is downloaded from HuggingFace into the container and the
# resulting token_ids_part_*.npy / labels_mask_*.npy shards are written to --out-dir on weka.
#
# Usage:
#   src/scripts/data/convert_dolci_instruct_sft_gantry.sh \
#       --out-dir /weka/oe-training-default/amandab/dolci-instruct-sft/qwen3
#
#   # Qwen3.5 (vocab 248320): the eos/landmark ids MUST be overridden alongside --tokenizer.
#   NAME=dolci-instruct-sft-convert-qwen35 \
#   src/scripts/data/convert_dolci_instruct_sft_gantry.sh \
#       --out-dir /weka/oe-training-default/amandab/dolci-instruct-sft/qwen35 \
#       --tokenizer Qwen/Qwen3.5-0.8B --eos-token-id 248044 --landmark-token-id 248200
#
# Any extra args are forwarded verbatim to convert_dolci_instruct_sft.py (e.g. --limit, --max-seq-len).
#
# Overridable via env vars:
#   CLUSTER (ai2/jupiter-cirrascale-2)  WORKSPACE (ai2/flex2)  BUDGET (ai2/oe-other)
#   WEKA (oe-training-default)  PRIORITY (urgent)  CPUS (32)  HF_SECRET (amandab_HF_TOKEN)
#   NAME (dolci-instruct-sft-convert) -- give each tokenizer variant its own Beaker job name
#
# NOTE: gantry runs the code at your current *committed* git HEAD, so commit (and push) the
# converter before launching -- uncommitted working-tree changes are not shipped to the node.
set -euo pipefail

OUT_DIR="/weka/oe-training-default/amandab/dolci-instruct-sft/qwen3"
if [[ "$*" == *"--out-dir"* ]]; then
  OUT_DIR=""  # caller supplied their own --out-dir; don't inject a default
fi

CLUSTER="${CLUSTER:-ai2/jupiter-cirrascale-2}"
WORKSPACE="${WORKSPACE:-ai2/flex2}"
BUDGET="${BUDGET:-ai2/oe-other}"
WEKA="${WEKA:-oe-training-default}"
PRIORITY="${PRIORITY:-urgent}"   # all jobs launch at urgent priority
CPUS="${CPUS:-32}"
HF_SECRET="${HF_SECRET:-amandab_HF_TOKEN}"
NAME="${NAME:-dolci-instruct-sft-convert}"

gantry run \
  --name "${NAME}" \
  --description "Convert allenai/Dolci-Instruct-SFT -> Qwen SFT npy (token_ids + labels_mask)" \
  --workspace "${WORKSPACE}" \
  --budget "${BUDGET}" \
  --cluster "${CLUSTER}" \
  --python-manager conda \
  --system-python \
  --weka "${WEKA}:/weka/${WEKA}" \
  --cpus "${CPUS}" \
  --gpus 0 \
  --priority "${PRIORITY}" \
  --allow-dirty \
  --shared-memory 32GiB \
  --timeout 0 \
  --env-secret "HF_TOKEN=${HF_SECRET}" \
  --env HF_HUB_ENABLE_HF_TRANSFER=1 \
  --env TOKENIZERS_PARALLELISM=true \
  --install "pip install datasets transformers numpy tqdm jinja2 'huggingface_hub>=0.24' hf_transfer" \
  --yes \
  -- python src/scripts/data/convert_dolci_instruct_sft.py \
  ${OUT_DIR:+--out-dir "$OUT_DIR"} "$@"
