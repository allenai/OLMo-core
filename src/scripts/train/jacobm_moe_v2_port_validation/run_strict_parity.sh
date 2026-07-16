#!/usr/bin/env bash

set -euo pipefail

REFERENCE_REPO=${REFERENCE_REPO:-/weka/oe-adapt-default/jacobm/olmoe3/olmo-ddp-migration/OLMo-core}
CANDIDATE_REPO=${CANDIDATE_REPO:-/weka/oe-adapt-default/jacobm/olmoe3/olmo-ddp-migration/OLMo-core-moe-v2-core-validation}
CHECKPOINT=${CHECKPOINT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmo-ddp/pretraining/pt-275m-intwide-hybrid-gdn-ev1-cx1-lr1p6e-3-r1/step16108}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmo-ddp/port-validation/0cdcc8b81/parity/275m-cx1-step16108}
SEQUENCE_LENGTH=${SEQUENCE_LENGTH:-128}

mkdir -p "${ARTIFACT_ROOT}" /results
export CUDA_SCALE_LAUNCH_QUEUES=4x
export OLMO_SHARED_FS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
unset S3_PROFILE

REFERENCE_ARTIFACT="${ARTIFACT_ROOT}/reference.pt"
CANDIDATE_ARTIFACT="${ARTIFACT_ROOT}/candidate.pt"
REPORT="${ARTIFACT_ROOT}/strict_parity.json"

PYTHONPATH="${REFERENCE_REPO}/src" torchrun --standalone --nproc-per-node=1 \
  "${REFERENCE_REPO}/src/scripts/export_moe_checkpoint_logits.py" \
  "${CHECKPOINT}" "${REFERENCE_ARTIFACT}" \
  --model-kind olmo-ddp \
  --sequence-length "${SEQUENCE_LENGTH}" \
  --capture-intermediates \
  2>&1 | tee /results/reference.log

PYTHONPATH="${CANDIDATE_REPO}/src" torchrun --standalone --nproc-per-node=1 \
  "${CANDIDATE_REPO}/src/scripts/train/jacobm_moe_v2_port_validation/export_logits.py" \
  "${CHECKPOINT}" "${CANDIDATE_ARTIFACT}" \
  --input-artifact "${REFERENCE_ARTIFACT}" \
  --sequence-length "${SEQUENCE_LENGTH}" \
  2>&1 | tee /results/candidate.log

PYTHONPATH="${CANDIDATE_REPO}/src" python \
  "${CANDIDATE_REPO}/src/scripts/train/jacobm_moe_v2_port_validation/compare_artifacts.py" \
  "${REFERENCE_ARTIFACT}" "${CANDIDATE_ARTIFACT}" "${REPORT}" \
  2>&1 | tee /results/comparison.log

cp "${REPORT}" /results/strict_parity.json

if [[ ${OLMOE3_PORT_SUBMIT_POSTGATE:-0} == 1 ]]; then
  beaker experiment create \
    "${CANDIDATE_REPO}/src/scripts/train/jacobm_moe_v2_port_validation/beaker_postgate.yaml" \
    --workspace ai2/OLMo-3-moe-experiments \
    --name jacobm-moe-v2-core-port-postgate-0cdcc8b81 \
    --format json \
    | tee /results/postgate_submission.json
fi
