#!/usr/bin/env bash
# Relay the two OLMo-3-7B contradiction checkpoints from weka to S3 so Berkeley can evaluate them.
#
#   bash debug/ctc_crossfamily/relay_olmo3_contra_to_s3.sh
#
# ── WHY ───────────────────────────────────────────────────────────────────────────────────────
# The cross-family contradiction figure has to be re-measured on `contradiction_iid`, because the
# three families were each scored on a DIFFERENT ladder:
#
#   Qwen3.5-4B     eval_rungs/contradiction_clean   dense .8427   (harvested, vLLM harness)
#   OLMo-3-7B      eval_rungs/contradiction         dense .8291   (native evaluator)
#   Llama-3.2-3B   eval_rungs/contradiction         dense .7434   (native, ']]'-truncated re-score)
#
# Those are different corpora, so the numbers were never comparable. `contradiction_iid` is the
# ladder that matches the `contradiction_train` shard every arm was trained on, and it is where
# Qwen -- the gold standard -- reads .9895/.9843/.9760/.9652/.9463 dense and
# .8613/.8337/.8039/.7586/.6991 chunked-mix. So OLMo and Llama get re-run there.
#
# Llama needs no transfer: both contradiction arms already sit on cubbins' node-local disk
# (/data/prasann/ctc_suite/ckpts/llama32-3b-contra-{full,chunked-mix}), and both hotpotqa arms on
# sneetches. Only these two OLMo checkpoints are weka-only, and Berkeley cannot mount weka -- hence
# the S3 hop.
#
# ⚠ THE -swa CHECKPOINTS, NOT THE FIRST PAIR. `ctc-olmo3-7b-contra-full` (no suffix, a copy of which
# is already on sneetches) is the SUPERSEDED no-sliding-window run: it disabled SWA on every layer,
# which costs the base model ~41x in CE before any training (olmo3_swa_ablation.py), and both arms
# then trained from a wrecked starting point and emitted one constant answer for all 500 examples.
# The valid arms keep Olmo's native 3:1 sliding:full backbone and chunk only the 8 full-attention
# layers -- the faithful counterpart of what the Qwen3.5 arms do. Those are the names below.
set -uo pipefail
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$HOME/.local/bin:$PATH

CLUSTERS=(--cluster ai2/ceres --cluster ai2/saturn --cluster ai2/neptune --cluster ai2/jupiter)
WEKA_ROOT=/weka/oe-training-default
CKPT_ROOT="$WEKA_ROOT/ai2-llm/checkpoints/prasanns/ctc_suite/ckpts"
S3=s3://ai2-llm/checkpoints/prasanns/_transfer

FULL=ctc-olmo3-7b-contra-full-swa-20260727T020006-0700
CMIX=ctc-olmo3-7b-contra-cmix-swa2-20260727T020625-0700

gantry run \
  --name ctc-olmo3-contra-relay-s3 \
  --description "Relay OLMo-3-7B contradiction SWA checkpoints weka -> S3 for Berkeley iid re-eval" \
  --workspace ai2/flex2 --budget ai2/oe-other \
  "${CLUSTERS[@]}" --gpus 0 --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --weka oe-training-default:"$WEKA_ROOT" \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --no-python --allow-dirty --timeout 0 --yes \
  -- bash -c "
set -uo pipefail
mkdir -p ~/.aws
printf '%s\n' \"\${AWS_CREDS}\" > ~/.aws/credentials
printf '%s\n' \"\${AWS_CFG}\" > ~/.aws/config
command -v aws >/dev/null || pip install -q awscli
for CK in $FULL $CMIX; do
  SRC='$CKPT_ROOT'/\$CK
  echo \"=== \$CK\"
  if [ ! -f \"\$SRC/model_and_optim/.metadata\" ]; then
    echo \"FATAL: \$SRC has no model_and_optim/.metadata -- refusing to relay an unfinalized checkpoint\"; exit 3
  fi
  ls -l \"\$SRC/model_and_optim/.metadata\"
  AWS_PROFILE=S3 aws s3 sync \"\$SRC\" '$S3'/\$CK --only-show-errors
  echo \"pushed -> $S3/\$CK\"
  AWS_PROFILE=S3 aws s3 ls '$S3'/\$CK/model_and_optim/ --summarize | tail -2
done
echo '=== relay done ==='
" 2>&1 | grep -oE "beaker.org/ex/[A-Z0-9]+" | head -1
