#!/bin/bash
# Part 2 of 3: assemble _eval_bundle_eval500_v2_clean on S3 from what is ALREADY clean, using
# server-side copies (no bytes travel through this host).
#
# Only the rungs the weka audit marked BAD get rebuilt (part 1 uploads those). Everything the audit
# marked OK is copied as-is -- there is no reason to regenerate a rung that already passes:
#
#   copied from _eval_bundle_eval500_v2 (audited OK):
#     contra   2k/8k/16k/32k          nq      3k/8k/16k/32k + 512k/1M/2M
#     outlier  3k/8k/16k/32k + 512k/1M/2M     rerank  3k/8k/16k + 512k/1M/2M
#     the OOD probes (fiqa/scifact/outlier_review/contra_fever) verbatim
#   copied from xlong5_2k256k_qwen35/eval:
#     ALL oolong rungs 2k..2M -- the live root was missing 2k/4k/64k/128k/256k, and the build
#     bundle already has them at >=500, so this is a copy and not a regeneration.
#
# NOT extended to 2M: the OOD probes (fiqa/scifact/outlier_review) top out at 16k/32k by design --
# they are held-out generalization probes with their own subsampled CE pools, not length ladders.
# contra_fever is intentionally FEVER-sourced and is a separate setting, not part of this ladder.
set -uo pipefail
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
export AWS_PROFILE=S3

LIVE=s3://ai2-llm/checkpoints/prasanns/_eval_bundle_eval500_v2
BUILD=s3://ai2-llm/checkpoints/prasanns/xlong5_2k256k_qwen35/eval
CLEAN=s3://ai2-llm/checkpoints/prasanns/_eval_bundle_eval500_v2_clean

cp1() {  # cp1 <src-key> ; server-side single-object copy, keeps the same relative path
  aws s3 cp "$LIVE/$1" "$CLEAN/$1" --only-show-errors && echo "  ok   $1" || echo "  FAIL $1"
}

echo "=== base rungs (audited eval_size>=500, 0.00% FEVER) ==="
for k in \
  contra/contradiction_eval_pubmed_both_n100_k3.jsonl \
  contra/contradiction_eval_pubmed_both_n190_k3.jsonl \
  contra/contradiction_eval_pubmed_both_n385_k3.jsonl \
  contra/contradiction_eval_pubmed_both_n765_k3.jsonl \
  nq/nq_validation_k20_600.jsonl nq/nq_validation_k50_600.jsonl \
  nq/nq_validation_k100_600.jsonl nq/nq_validation_k200_600.jsonl \
  outlier/outlier_wiki100w_n22_k3_eval_600.jsonl outlier/outlier_wiki100w_n55_k3_eval_600.jsonl \
  outlier/outlier_wiki100w_n110_k3_eval_600.jsonl outlier/outlier_wiki100w_n220_k3_eval_600.jsonl \
  rerank/msmarco_trainhn_eval_k20_500.jsonl rerank/msmarco_trainhn_eval_k50_500.jsonl \
  rerank/msmarco_trainhn_eval_k100_500.jsonl ; do
  cp1 "$k"
done

echo "=== nq/outlier/rerank 512k/1M/2M (already 500 examples and FEVER-free) ==="
for k in \
  nq/nq_validation_k3342_xlong_512k.jsonl nq/nq_validation_k6683_xlong_1M.jsonl \
  nq/nq_validation_k13366_xlong_2M.jsonl \
  outlier/outlier_wiki100w_n3605_k3_eval_xlong_512k.jsonl \
  outlier/outlier_wiki100w_n7209_k3_eval_xlong_1M.jsonl \
  outlier/outlier_wiki100w_n14419_k3_eval_xlong_2M.jsonl \
  rerank/msmarco_trainhn_eval_k6121_xlong_512k.jsonl \
  rerank/msmarco_trainhn_eval_k12242_xlong_1M.jsonl \
  rerank/msmarco_trainhn_eval_k24485_xlong_2M.jsonl ; do
  cp1 "$k"
done

echo "=== oolong: ALL rungs 2k..2M from the build bundle (live root lacked 2k/4k/64k/128k/256k) ==="
aws s3 sync "$BUILD/oolong" "$CLEAN/oolong" --only-show-errors
echo "  oolong objects: $(aws s3 ls "$CLEAN/oolong/" | wc -l)"

echo "=== OOD probes, verbatim (16k/32k ceilings are by design; contra_fever is FEVER on purpose) ==="
for sub in beir outlier contra; do
  aws s3 sync "$LIVE/$sub" "$CLEAN/$sub" --only-show-errors \
    --exclude '*_xlong_*' --exclude '*.manifest.json'
  echo "  synced $sub"
done

echo
echo "=== clean bundle inventory ==="
for sub in contra nq outlier rerank oolong beir; do
  echo "  $sub: $(aws s3 ls "$CLEAN/$sub/" 2>/dev/null | wc -l) objects"
done
