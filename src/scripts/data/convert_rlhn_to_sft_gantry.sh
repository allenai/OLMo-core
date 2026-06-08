#!/usr/bin/env bash
# Launch the rlhn-100K -> Qwen3 SFT conversion on a Beaker CPU node via gantry.
#
# This wraps src/scripts/data/convert_rlhn_to_sft.py. It runs on a single CPU node with the weka
# bucket mounted; the dataset is downloaded from HuggingFace into the container and the resulting
# token_ids_part_*.npy / labels_mask_*.npy shards are written to --out-dir on weka.
#
# Usage:
#   src/scripts/data/convert_rlhn_to_sft_gantry.sh \
#       --out-dir /weka/oe-training-default/ai2-llm/checkpoints/$USER/rlhn_sft_qwen \
#       --max-seq-len 8192
#
# Any extra args are forwarded verbatim to convert_rlhn_to_sft.py (e.g. --max-doc-chars, --limit).
#
# Overridable via env vars:
#   CLUSTER (ai2/jupiter-cirrascale-2)  WORKSPACE (ai2/flex2)  BUDGET (ai2/oe-other)
#   WEKA (oe-training-default)  PRIORITY (urgent)  CPUS (32)  HF_SECRET (amandab_HF_TOKEN)
#
# NOTE: gantry runs the code at your current *committed* git HEAD, so commit (and push) the
# converter before launching -- uncommitted working-tree changes are not shipped to the node.
set -euo pipefail

if [[ "$#" -eq 0 || "$*" != *"--out-dir"* ]]; then
  echo "error: you must pass --out-dir (on the mounted weka bucket). Example:" >&2
  echo "  $0 --out-dir /weka/oe-training-default/ai2-llm/checkpoints/\$USER/rlhn_sft_qwen --max-seq-len 8192" >&2
  exit 2
fi

CLUSTER="${CLUSTER:-ai2/jupiter-cirrascale-2}"
WORKSPACE="${WORKSPACE:-ai2/flex2}"
BUDGET="${BUDGET:-ai2/oe-other}"
WEKA="${WEKA:-oe-training-default}"
PRIORITY="${PRIORITY:-urgent}"   # all jobs launch at urgent priority
CPUS="${CPUS:-32}"
HF_SECRET="${HF_SECRET:-amandab_HF_TOKEN}"

gantry run \
  --name "rlhn-sft-convert" \
  --description "Convert rlhn-100K -> Qwen3 SFT npy (token_ids + labels_mask)" \
  --workspace "${WORKSPACE}" \
  --budget "${BUDGET}" \
  --cluster "${CLUSTER}" \
  --weka "${WEKA}:/weka/${WEKA}" \
  --cpus "${CPUS}" \
  --gpus 0 \
  --priority "${PRIORITY}" \
  --shared-memory 32GiB \
  --env-secret "HF_TOKEN=${HF_SECRET}" \
  --env HF_HUB_ENABLE_HF_TRANSFER=1 \
  --env TOKENIZERS_PARALLELISM=true \
  --install "pip install datasets transformers numpy tqdm 'huggingface_hub>=0.24' hf_transfer" \
  --yes \
  -- python src/scripts/data/convert_rlhn_to_sft.py "$@"
