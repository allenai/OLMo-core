#!/usr/bin/env bash
# Launch the RAG-QA (NQ + HotpotQA) -> Qwen3 SFT conversion on a Beaker CPU node via gantry.
# This is the "NEAR" rung of the HELMET-RAG in-distribution experiment.
#
# This wraps src/scripts/data/convert_rag_tasks_to_sft.py. It runs on a single CPU node with the
# weka bucket mounted. Unlike the rlhn converter, the input data is NOT downloaded from HF -- you
# pass --input-jsonl pointing at corpus-reasoning unified-JSONL files that already live on weka
# (generate them first with corpus-reasoning's generate_nq_training_data.py /
# generate_hotpotqa_data.py and copy the resulting data/*.jsonl onto weka). Only the Qwen3
# tokenizer is fetched from HF. The token_ids_part_*.npy / labels_mask_*.npy shards are written to
# --out-dir on weka.
#
# Usage (quote the globs so the container expands them, not your shell):
#   src/scripts/data/convert_rag_to_sft_gantry.sh \
#       --input-jsonl '/weka/oe-training-default/ai2-llm/checkpoints/$USER/rag_jsonl/nq_train_*.jsonl' \
#                     '/weka/oe-training-default/ai2-llm/checkpoints/$USER/rag_jsonl/hotpotqa_train_*.jsonl' \
#       --target-tokens 123456789 \
#       --out-dir /weka/oe-training-default/ai2-llm/checkpoints/$USER/rag_sft_qwen/nq_hotpotqa_near
#
# Set --target-tokens to amandab/rlhn_sft_qwen_63k's metadata.json "num_tokens" so the NEAR set
# matches the FAR (RLHN) training-token count. Any extra args forward verbatim to the converter
# (e.g. --query-position, --max-seq-len, --limit).
#
# Overridable via env vars:
#   CLUSTER (ai2/jupiter-cirrascale-2)  WORKSPACE (ai2/flex2)  BUDGET (ai2/oe-other)
#   WEKA (oe-training-default)  PRIORITY (urgent)  CPUS (32)  HF_SECRET (amandab_HF_TOKEN)
#
# NOTE: gantry runs the code at your current *committed* git HEAD, so commit (and push) the
# converter before launching -- uncommitted working-tree changes are not shipped to the node.
set -euo pipefail

if [[ "$#" -eq 0 || "$*" != *"--out-dir"* || "$*" != *"--input-jsonl"* ]]; then
  echo "error: you must pass --input-jsonl and --out-dir (both on the mounted weka bucket)." >&2
  echo "  $0 --input-jsonl '/weka/.../nq_train_*.jsonl' '/weka/.../hotpotqa_train_*.jsonl' \\" >&2
  echo "     --target-tokens <rlhn num_tokens> --out-dir /weka/.../rag_sft_qwen/nq_hotpotqa_near" >&2
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
  --name "rag-sft-convert" \
  --description "Convert NQ+HotpotQA RAG-QA -> Qwen3 SFT npy (token_ids + labels_mask); NEAR rung" \
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
  -- python src/scripts/data/convert_rag_tasks_to_sft.py "$@"
