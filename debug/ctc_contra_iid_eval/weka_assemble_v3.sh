#!/bin/bash
# Assemble _eval_bundle_eval500_v3 ON WEKA: sync the two rebuilt tasks from S3, symlink the three
# unchanged ones from v2_clean, then inventory what an eval can actually read.
#
# WHY THIS EXISTS SEPARATELY FROM THE S3 PUSH. An S3-only publish leaves every rung MISSING at eval
# time while the job still exits 0 -- weka is what the runner reads. This is the second half of the
# eval-bundle staging two-step, and it must run before any v3 launch.
#
# v3 = v2_clean with TWO tasks replaced:
#   contra  -> realistic-mode, IID with the training generator, rungs 2k..1M
#   outlier -> TRUE scale-K (K ~ n/9.5, min majority-vs-outlier gap 2), xlong rungs 64k..1M
# and THREE tasks byte-identical to v2_clean: nq, rerank, oolong.
#
# The three unchanged tasks are SYMLINKED rather than copied. Copying moves several GB (nq's 2M rung
# alone is 4.5G) and leaves "identical to v2" as a claim someone has to re-verify; a symlink makes it
# true by construction, costs nothing, and propagates any later v2 fix instead of silently diverging.
#
# NOTE the outlier asymmetry: v3 replaces outlier's XLONG rungs only (64k+). The 2k-32k base rungs
# were already scale-K and are the same files as v2 -- the freeze was introduced by the generic
# xlong expander (pool="self_nongold"), which never touched the base ladder.
set -uo pipefail
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
cd /accounts/projects/berkeleynlp/prasann/projects/OLMo-core

WORK='
set -uo pipefail
export PATH=/opt/conda/bin:$PATH
command -v aws >/dev/null 2>&1 || python -m pip install -q awscli
mkdir -p ~/.aws
printf "%s" "$AWS_CREDS" > ~/.aws/credentials
printf "%s" "$AWS_CFG" > ~/.aws/config
export AWS_PROFILE=S3

P=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
V2="$P/_eval_bundle_eval500_v2_clean"
V3="$P/_eval_bundle_eval500_v3"
S3=s3://ai2-llm/checkpoints/prasanns/_eval_bundle_eval500_v3

[ -d "$V2" ] || { echo "!!! v2_clean missing at $V2 -- cannot symlink the unchanged tasks"; exit 1; }
mkdir -p "$V3"

echo "=== 1. sync rebuilt tasks S3 -> weka ==="
for t in contra outlier; do
  # a stale symlink here would make aws-sync write THROUGH it into v2_clean
  [ -L "$V3/$t" ] && { echo "  [$t] removing stale symlink"; rm -f "$V3/$t"; }
  mkdir -p "$V3/$t"
  echo "  [$t] syncing..."
  aws s3 sync "$S3/$t" "$V3/$t" --only-show-errors || { echo "!!! sync $t FAILED"; exit 1; }
  echo "  [$t] $(du -sh "$V3/$t" | cut -f1)"
done

echo ""
echo "=== 2. symlink unchanged tasks -> v2_clean ==="
for t in nq rerank oolong; do
  if [ -e "$V3/$t" ] && [ ! -L "$V3/$t" ]; then
    echo "  [$t] real directory present, leaving alone"; continue
  fi
  ln -sfn "$V2/$t" "$V3/$t"
  echo "  [$t] -> $(readlink "$V3/$t")  ($(ls "$V3/$t"/*.jsonl 2>/dev/null | wc -l) jsonl)"
done

echo ""
echo "=== 3. inventory: what an eval can actually read ==="
for t in contra outlier nq rerank oolong; do
  kind=$([ -L "$V3/$t" ] && echo "symlink->v2" || echo "REPLACED")
  printf "\n  [%s] %s\n" "$t" "$kind"
  ls "$V3/$t"/*.jsonl 2>/dev/null | while read -r f; do
    printf "    %8s rows  %10s  %s\n" "$(wc -l < "$f")" "$(du -h "$f" | cut -f1)" "$(basename "$f")"
  done
done

echo ""
echo "=== 4. provenance ==="
for j in disjointness_report.json iid_rung_audit.json; do
  [ -f "$V3/contra/$j" ] && { echo "--- $j ---"; cat "$V3/contra/$j"; echo; }
done

echo ""
echo "NOTE: v3 contra and outlier are NOT comparable to their v2 counterparts (contra changes"
echo "  perturbation mode realistic-vs-both; outlier xlong changes K scaling). nq/rerank/oolong ARE"
echo "  the same files as v2 and compare directly."
'

gantry run --name contra-v3-weka-assemble -w ai2/flex2 -b ai2/oe-other \
  --cluster 'ai2/*-cirrascale*' --gpus 0 --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --weka oe-training-default:/weka/oe-training-default \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --no-python --allow-dirty --timeout 0 --yes -- bash -c "$WORK"
