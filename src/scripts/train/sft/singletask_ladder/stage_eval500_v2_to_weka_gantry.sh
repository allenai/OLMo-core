#!/usr/bin/env bash
# Sync the v2 eval ladders from s3 onto weka via a Beaker CPU gantry job, so
# `--ladder-version v2` evals can read them (weka is not mounted at Berkeley).
#
# Prereq: the v2 data is already on s3 (run upload_lc_eval_bundle.sh first, which
# pushes /scratch/.../eval500_v2 -> s3://ai2-llm/.../_eval_bundle_eval500_v2).
# This job just mirrors that s3 prefix onto the mounted weka bucket. AWS creds
# come from the PRASANNS_AWS_* beaker secrets (S3 profile = real AWS, port 443).
#
# The sync command is INLINE, so this job needs no in-repo file (gantry still
# clones OLMo-core, but --install true skips the package build -- no olmo_core needed).
#
# Usage:
#   src/scripts/train/sft/singletask_ladder/stage_eval500_v2_to_weka_gantry.sh
# Overridable: CLUSTER WORKSPACE BUDGET WEKA PRIORITY CPUS NAME S3_PREFIX DEST_REL
set -euo pipefail

CLUSTER="${CLUSTER:-ai2/jupiter-cirrascale-2}"
WORKSPACE="${WORKSPACE:-ai2/flex2}"
BUDGET="${BUDGET:-ai2/oe-other}"
WEKA="${WEKA:-oe-training-default}"
PRIORITY="${PRIORITY:-normal}"
CPUS="${CPUS:-8}"
NAME="${NAME:-stage-eval500-v2-weka}"
S3_PREFIX="${S3_PREFIX:-s3://ai2-llm/checkpoints/prasanns/_eval_bundle_eval500_v2}"
DEST_REL="${DEST_REL:-ai2-llm/checkpoints/prasanns/_eval_bundle_eval500_v2}"

CLUSTER_ARGS=()
IFS=',' read -ra _CLUSTERS <<< "${CLUSTER}"
for c in "${_CLUSTERS[@]}"; do CLUSTER_ARGS+=(--cluster "$c"); done

DEST="/weka/${WEKA}/${DEST_REL}"

# Inline job: materialize AWS creds from the injected secrets, sync s3 -> weka, verify.
read -r -d '' JOB <<'EOS' || true
set -euo pipefail
mkdir -p ~/.aws
printf '%s\n' "$AWS_CREDS" > ~/.aws/credentials
printf '%s\n' "$AWS_CFG"   > ~/.aws/config
command -v aws >/dev/null 2>&1 || pip install -q awscli
echo "=== s3 -> weka sync: $S3_PREFIX -> $OUT ==="
AWS_PROFILE=S3 aws s3 sync "$S3_PREFIX" "$OUT"
echo "=== landed on weka ($OUT): ==="
find "$OUT" -name '*.jsonl' | sort | while read -r f; do printf '%6s  %s\n' "$(wc -l < "$f")" "$f"; done
EOS

gantry run \
  --name "${NAME}" \
  --description "Sync v2 eval ladders s3 -> weka (${DEST_REL})" \
  --workspace "${WORKSPACE}" \
  --budget "${BUDGET}" \
  "${CLUSTER_ARGS[@]}" \
  --weka "${WEKA}:/weka/${WEKA}" \
  --cpus "${CPUS}" \
  --gpus 0 \
  --priority "${PRIORITY}" \
  --allow-dirty \
  --install true \
  --timeout 0 \
  --env-secret "AWS_CREDS=PRASANNS_AWS_CREDENTIALS" \
  --env-secret "AWS_CFG=PRASANNS_AWS_CONFIG" \
  --env "S3_PREFIX=${S3_PREFIX}" \
  --env "OUT=${DEST}" \
  --yes \
  -- bash -c "${JOB}"

echo "Launched ${NAME}: syncing ${S3_PREFIX} -> ${DEST}"
