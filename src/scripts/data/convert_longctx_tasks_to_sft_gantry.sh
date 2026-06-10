#!/usr/bin/env bash
# Launch the longctx-task (oolong / contradiction) -> Qwen3 SFT conversion on a Beaker CPU node
# via gantry.
#
# This wraps src/scripts/data/convert_longctx_tasks_to_sft.py. It runs on a single CPU node with
# the weka bucket mounted and the unified JSONL inputs mounted at /input from a Beaker dataset
# (create it locally first with `beaker dataset create <dir-with-jsonl>`). The resulting
# token_ids_part_*.npy / labels_mask_*.npy shards are written to --out-dir on weka.
#
# Usage:
#   # one-time: upload the jsonl staging dir as a Beaker dataset
#   beaker dataset create --name longctx-task-jsonl ~/Desktop/Projects/longctx_data
#
#   INPUT_DATASET=<user>/longctx-task-jsonl \
#   src/scripts/data/convert_longctx_tasks_to_sft_gantry.sh \
#       --task oolong \
#       --input-jsonl '/input/oolong_test_synth_ctx*_splittrain.jsonl' \
#       --out-dir /weka/oe-training-default/ai2-llm/checkpoints/$USER/longctx_sft_qwen/oolong
#
#   INPUT_DATASET=<user>/longctx-task-jsonl \
#   src/scripts/data/convert_longctx_tasks_to_sft_gantry.sh \
#       --task contradiction --cot-mode enumerate \
#       --input-jsonl '/input/contradiction_train_pubmed_both_n*_k3.jsonl' \
#       --out-dir /weka/oe-training-default/ai2-llm/checkpoints/$USER/longctx_sft_qwen/contradiction_cot
#
# Any extra args are forwarded verbatim to convert_longctx_tasks_to_sft.py (e.g. --limit).
#
# Overridable via env vars:
#   CLUSTER (ai2/jupiter-cirrascale-2)  WORKSPACE (ai2/flex2)  BUDGET (ai2/oe-other)
#   WEKA (oe-training-default)  PRIORITY (urgent)  CPUS (16)  NAME (longctx-sft-convert)
#   INPUT_DATASET (required) -- Beaker dataset with the unified JSONL, mounted at /input
#
# NOTE: gantry runs the code at your current *committed* git HEAD, so commit (and push) the
# converter before launching -- uncommitted working-tree changes are not shipped to the node.
set -euo pipefail

if [[ "$#" -eq 0 || "$*" != *"--out-dir"* || "$*" != *"--task"* ]]; then
  echo "error: you must pass --task and --out-dir (on the mounted weka bucket). See header." >&2
  exit 2
fi

if [[ -z "${INPUT_DATASET:-}" ]]; then
  echo "error: INPUT_DATASET env var is required (Beaker dataset with the unified JSONL)." >&2
  exit 2
fi

CLUSTER="${CLUSTER:-ai2/jupiter-cirrascale-2}"
WORKSPACE="${WORKSPACE:-ai2/flex2}"
BUDGET="${BUDGET:-ai2/oe-other}"
WEKA="${WEKA:-oe-training-default}"
PRIORITY="${PRIORITY:-urgent}"
CPUS="${CPUS:-16}"
NAME="${NAME:-longctx-sft-convert}"

gantry run \
  --name "${NAME}" \
  --description "Convert longctx task data (oolong/contradiction) -> Qwen3 SFT npy (token_ids + labels_mask)" \
  --workspace "${WORKSPACE}" \
  --budget "${BUDGET}" \
  --cluster "${CLUSTER}" \
  --python-manager conda \
  --system-python \
  --weka "${WEKA}:/weka/${WEKA}" \
  --dataset "${INPUT_DATASET}:/input" \
  --cpus "${CPUS}" \
  --gpus 0 \
  --priority "${PRIORITY}" \
  --allow-dirty \
  --shared-memory 32GiB \
  --timeout 0 \
  --env TOKENIZERS_PARALLELISM=true \
  --install "pip install transformers numpy tqdm jinja2 'huggingface_hub>=0.24'" \
  --yes \
  -- python src/scripts/data/convert_longctx_tasks_to_sft.py "$@"
