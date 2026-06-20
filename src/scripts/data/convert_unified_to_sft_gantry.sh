#!/usr/bin/env bash
# Tokenize corpus-reasoning unified JSONL (any task) -> Qwen3 SFT npy shards on a Beaker CPU node.
# Wraps src/scripts/data/convert_unified_to_sft.py. Reads the unified JSONL straight from weka
# (land it first with land_suite_eval_to_weka_gantry.sh -> prasanns/cr_suite_data) and writes
# token_ids_part_*.npy / labels_mask_*.npy to --out-dir on weka.
#
# Usage (one task family at a time; pass the manifest's cot_mode for that task):
#   src/scripts/data/convert_unified_to_sft_gantry.sh \
#       --task contradiction --cot-mode template \
#       --input-jsonl '/weka/oe-training-default/ai2-llm/checkpoints/prasanns/cr_suite_data/contradiction_train_pubmed_both_n*_k3.jsonl' \
#       --out-dir /weka/oe-training-default/ai2-llm/checkpoints/prasanns/suite_it_sft_qwen/contradiction
#
# Any extra args forward verbatim to convert_unified_to_sft.py (e.g. --limit, --query-position).
#
# Overridable: CLUSTER (ai2/jupiter-cirrascale-2) WORKSPACE (ai2/flex2) BUDGET (ai2/oe-other)
#   WEKA (oe-training-default) PRIORITY (normal) CPUS (16) NAME (suite-sft-convert)
#
# NOTE: gantry ships your committed git HEAD -- commit + push the converter before launching.
set -euo pipefail

if [[ "$#" -eq 0 || "$*" != *"--out-dir"* || "$*" != *"--input-jsonl"* ]]; then
  echo "error: pass --input-jsonl and --out-dir (both on the mounted weka bucket). See header." >&2
  exit 2
fi

CLUSTER="${CLUSTER:-ai2/jupiter-cirrascale-2}"
WORKSPACE="${WORKSPACE:-ai2/flex2}"
BUDGET="${BUDGET:-ai2/oe-other}"
WEKA="${WEKA:-oe-training-default}"
PRIORITY="${PRIORITY:-normal}"
CPUS="${CPUS:-16}"
NAME="${NAME:-suite-sft-convert}"

CLUSTER_ARGS=()
IFS=',' read -ra _CLUSTERS <<< "${CLUSTER}"
for c in "${_CLUSTERS[@]}"; do CLUSTER_ARGS+=(--cluster "$c"); done

gantry run \
  --name "${NAME}" \
  --description "Tokenize unified suite JSONL -> Qwen3 SFT npy (token_ids + labels_mask)" \
  --workspace "${WORKSPACE}" \
  --budget "${BUDGET}" \
  "${CLUSTER_ARGS[@]}" \
  --python-manager conda \
  --system-python \
  --weka "${WEKA}:/weka/${WEKA}" \
  --cpus "${CPUS}" \
  --gpus 0 \
  --priority "${PRIORITY}" \
  --allow-dirty \
  --shared-memory 32GiB \
  --timeout 0 \
  --env TOKENIZERS_PARALLELISM=true \
  --install "pip install -e . && pip install transformers numpy jinja2" \
  --yes \
  -- python src/scripts/data/convert_unified_to_sft.py "$@"
