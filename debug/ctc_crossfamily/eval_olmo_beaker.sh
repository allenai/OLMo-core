#!/usr/bin/env bash
# Evaluate an OLMo-family CTC checkpoint on Beaker with the NATIVE olmo-core evaluator, reading the
# checkpoint from weka and the rung data from S3.
#
#   CKPT=<weka-ckpt-dirname> ARM=full LADDER=contradiction_iid TASK=contradiction \
#     bash debug/ctc_crossfamily/eval_olmo_beaker.sh
#
# ── WHY NOT THE vLLM PIPELINE ─────────────────────────────────────────────────────────────────
# eval_pipeline_cu129_apt.sh exports the checkpoint to HF via export_olmo_to_hf.py, whose
# resolve_olmo_model() supports ONLY model_type qwen3 / qwen3_5 and raises for anything else. An
# OLMo checkpoint cannot be exported at all, so that route is closed rather than merely slower.
# The native evaluator loads the distcp directly, needs no venv build, and starts in seconds.
#
# ── WHY NOT RELAY THE CHECKPOINTS TO BERKELEY ─────────────────────────────────────────────────
# weka is already mounted on Beaker, so an on-cluster job reads these 28GB checkpoints in place.
# Relaying them weka -> S3 -> Berkeley was ~60GB of transfer to avoid a queue that the eager
# clusters (ceres/saturn/neptune) do not have. Run here, on those clusters, instead.
#
# ── WHY contradiction_iid ─────────────────────────────────────────────────────────────────────
# The three families were each scored on a DIFFERENT contradiction ladder -- Qwen on
# contradiction_clean, OLMo and Llama on contradiction -- which are distinct corpora, so the
# cross-family numbers were never comparable. contradiction_iid is the ladder that matches the
# contradiction_train shard every arm trained on, and it is where the Qwen3.5-4B reference reads
# .9895/.9843/.9760/.9652/.9463 dense and .8613/.8337/.8039/.7586/.6991 chunked-mix.
#
# ⚠ MAXLEN IS FIXED AT 32768 AND THAT IS DELIBERATE.
# Contradiction rungs hold a FIXED corpus size but wildly varying claim lengths: at "rung 4096" the
# median prompt is ~6,969 tokens and the longest is 23,796. The driver's default of rung+2048 skips
# every over-long prompt and scores it 0 while parse_rate still reads 1.0 -- that silently zeroed
# 354/500 examples per rung on the first OLMo pass and made the two arms look tied. 32768 is what
# the published OLMo ladder used (500/500 covered at 2k/4k, 489/500 at 8k/16k), so matching it
# exactly is also what keeps this comparable to those numbers.
#
# ⚠ TOKENIZER AND MARKER IDS ARE PASSED EXPLICITLY. run_rung_eval defaults to the Qwen3.5 tokenizer
# and Qwen marker ids; pointed at an OLMo checkpoint those defaults do not crash, they produce
# plausible numbers from a mis-tokenized prompt. dolma2 ids come from RESERVED_IDS['olmo3'].
set -uo pipefail
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$HOME/.local/bin:$PATH
cd /accounts/projects/berkeleynlp/prasann/projects/OLMo-core

CKPT="${CKPT:?weka checkpoint dirname under ctc_suite/ckpts/ -- read it from the launch log}"
ARM="${ARM:?full or chunked-mix}"
TASK="${TASK:-contradiction}"          # run_rung_eval catalog key
LADDER="${LADDER:-contradiction_iid}"  # S3 rung tree under _transfer/ctc_eval_rungs/
RUNGS="${RUNGS:-2560 4096 8192 16384}"
MAXLEN="${MAXLEN:-32768}"
NGPU="${NGPU:-2}"
MODEL_SCALE="${MODEL_SCALE:-olmo3-7b}"
RESULT_PREFIX="${RESULT_PREFIX:-ctc_crossfamily_results}"
PRIORITY="${PRIORITY:-urgent}"
REF="${REF:-$(git rev-parse HEAD)}"
LOGD=debug/ctc_crossfamily/launches
mkdir -p "$LOGD"

case "$ARM" in
  full)        VARIANT=dense ;;
  chunked-mix) VARIANT=chunked ;;
  *) echo "FATAL: ARM must be full|chunked-mix, got '$ARM'"; exit 2 ;;
esac

if ! git merge-base --is-ancestor "$REF" "origin/$(git rev-parse --abbrev-ref HEAD)" 2>/dev/null; then
  echo "FATAL: $REF is not on the remote -- gantry clones from the remote; push first"; exit 2
fi

WEKA=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
TOK="$WEKA/ctc_olmo3/tokenizer"
LEDGER="$LOGD/LAUNCH_LEDGER.tsv"
[ -f "$LEDGER" ] || printf 'launched_at\tckpt\ttask\tladder\tarm\trung\texperiment_id\n' > "$LEDGER"

echo "ckpt=$CKPT arm=$ARM variant=$VARIANT task=$TASK ladder=$LADDER maxlen=$MAXLEN"
for R in $RUNGS; do
  NAME="xf-$(echo "$MODEL_SCALE-$TASK-$ARM-r$R" | tr '_.' '--')"
  LOG="$LOGD/${CKPT}_${ARM}_r${R}.log"
  gantry run --name "$NAME" \
    -w ai2/flex2 -b ai2/oe-other \
    --cluster ai2/ceres --cluster ai2/saturn --cluster ai2/neptune --cluster ai2/jupiter \
    --ref "$REF" --gpus "$NGPU" --priority "$PRIORITY" \
    --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
    --weka oe-training-default:/weka/oe-training-default \
    --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
    --no-python --allow-dirty --timeout 0 --yes \
    -- bash -c "
set -uo pipefail
mkdir -p ~/.aws
printf '%s\n' \"\${AWS_CREDS}\" > ~/.aws/credentials
printf '%s\n' \"\${AWS_CFG}\" > ~/.aws/config
command -v aws >/dev/null || pip install -q awscli
REPO=\$(find / -maxdepth 3 -iname pyproject.toml 2>/dev/null | grep -v /opt/conda | grep -v /root/.cache | head -1 | xargs -r dirname)
export PYTHONPATH=\"\$REPO/src\"
python -m pip install --quiet --no-deps 'dataclass-extensions>=0.3.0' 2>&1 | tail -3
# ⚠ GDN-hybrid checkpoints (Olmo-Hybrid-7B) NEED flash-linear-attention TO BUILD AT ALL.
# GatedDeltaNet.__init__ asserts has_fla() and then imports FusedRMSNormGated from fla.modules,
# and the baked olmo-core image ships neither -- all 16 Olmo-Hybrid evals died on that assert after
# loading, while the OLMo-3 evals passed because OLMo-3 has no GDN layers. Installed WITH deps under
# a PIP_CONSTRAINT pinning the image's torch: --no-deps yields a partial package whose fla.modules
# is missing, and has_fla() -- literally 'fla is not None' -- does not catch that.
python -c 'import torch; print(torch.__version__)' > /tmp/torchver.txt
printf 'torch==%s\n' \$(cat /tmp/torchver.txt) > /tmp/pipconstraint.txt
PIP_CONSTRAINT=/tmp/pipconstraint.txt python -m pip install --quiet 'flash-linear-attention==0.4.1' einops 2>&1 | tail -5
python -c 'from fla.modules import FusedRMSNormGated; print(FusedRMSNormGated)'
export TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

CKPT_DIR=$WEKA/ctc_suite/ckpts/$CKPT
[ -f \"\$CKPT_DIR/model_and_optim/.metadata\" ] || { echo 'FATAL: no model_and_optim/.metadata -- would evaluate an untrained model'; exit 3; }
[ -d '$TOK' ] || { echo 'FATAL: dolma2 marker tokenizer missing on weka'; exit 3; }

EV=/root/rung_${R}.jsonl
AWS_PROFILE=S3 aws s3 cp s3://ai2-llm/checkpoints/prasanns/_transfer/ctc_eval_rungs/$LADDER/rung_${R}.jsonl \"\$EV\" --only-show-errors
[ -s \"\$EV\" ] || { echo 'FATAL: rung jsonl empty/missing on S3'; exit 3; }

cd \"\$REPO\"
python -u -m scripts.eval.ctc_suite.run_rung_eval \
  --task $TASK --ckpt \"\$CKPT_DIR\" --variant $VARIANT --arm $ARM --model-scale $MODEL_SCALE \
  --rung-tokens $R --eval-jsonl \"\$EV\" --max-length $MAXLEN \
  --tokenizer '$TOK' --doc-start-id 100266 --doc-end-id 100267 --eos-token-id 100257 \
  --trained-ctx '2k-32k-joint-uniform' --batch-size 1 --small-eval-ok --nproc $NGPU \
  --out-root /root/out
rc=\$?
echo \"run_rung_eval rc=\$rc\"
# Preserve each artifact's own basename: run_rung_eval emits rung_N.json, rung_N.raw.json and
# rung_N.generations.json, and collapsing them onto one key destroys the per-example predictions
# needed for any later re-score.
for f in \$(find /root/out -name '*.json'); do
  AWS_PROFILE=S3 aws s3 cp \"\$f\" s3://ai2-llm/checkpoints/prasanns/_transfer/$RESULT_PREFIX/${LADDER}__${MODEL_SCALE}_${ARM}/\$(basename \"\$f\") --only-show-errors && echo \"pushed \$(basename \"\$f\")\"
done
exit \$rc
" > "$LOG" 2>&1
  E=$(grep -oE 'beaker\.org/ex/[A-Z0-9]+' "$LOG" | head -1 | sed 's#.*/##')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "$CKPT" "$TASK" "$LADDER" "$ARM" "$R" "${E:-SUBMIT-FAILED}" >> "$LEDGER"
  echo "  rung $R -> ${E:-FAILED (see $LOG)}"
done
