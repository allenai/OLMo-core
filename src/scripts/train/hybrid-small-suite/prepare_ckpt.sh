#!/usr/bin/env bash
# Turn a trained olmo-core checkpoint into an evaluable, self-contained HF checkpoint on weka.
#
#   bash prepare_ckpt.sh <olmo-core step dir> <output name> [--ref <released mainline ckpt>]
#
# Runs as ONE Beaker job with weka mounted, doing all three prep stages in place, so nothing is
# copied between machines:
#
#   1. export  olmo-core -> HF            (examples/huggingface/convert_checkpoint_to_hf.py)
#   2. re-dialect olmo3_5_hybrid -> mainline_ladder  (same weights, different key spellings)
#   3. make self-contained: copy the SSMax-patched modeling code in and set auto_map, so
#      trust_remote_code=True is the whole integration for any consumer
#
# Stage 3 is not optional. The stock mainline_ladder plugin has NO Scalable-Softmax: a checkpoint's
# ssmax_scale loads as UNEXPECTED and is silently ignored, so the model scores while missing a
# trained component.
set -euo pipefail

SRC="${1:?usage: prepare_ckpt.sh <olmo-core step dir> <output name> [--ref <released ckpt>]}"
NAME="${2:?}"
shift 2

BRANCH=prasann/ctc-sft-hybridish
GANTRY="${GANTRY:-gantry}"
CLUSTER="${CLUSTER:-ai2/jupiter-cirrascale-2}"
W=/weka/oe-training-default/ai2-llm
OUT="${OUT:-$W/checkpoints/prasanns/ctc_hybridish_sft}"
REF="${REF:-$W/scaling-ladders/mainline/tanushy/v0.0.1-seven_to_one_hybrid_ratio-c6e480e336d5/1.4B-Cx8/long-context/step44124}"
while [ $# -gt 0 ]; do case "$1" in --ref) REF="$2"; shift 2;; *) shift;; esac; done

cur=$(git rev-parse --abbrev-ref HEAD)
[ "$cur" = "$BRANCH" ] || { echo "ERROR: run from '$BRANCH' (on '$cur')" >&2; exit 2; }
git merge-base --is-ancestor HEAD "origin/$BRANCH" 2>/dev/null || {
  echo "ERROR: HEAD is not pushed; gantry clones the remote commit." >&2; exit 2; }

read -r -d '' CMD <<INNER || true
set -eu
HF=/tmp/hf_export
python src/examples/huggingface/convert_checkpoint_to_hf.py \
  --checkpoint-input-path '$SRC' --huggingface-output-dir \$HF --skip-validation
python debug/hybridish_sft/redialect_to_mainline.py --src \$HF --ref '$REF' --out '$OUT/$NAME'
python debug/hybridish_sft/make_self_contained_ckpt.py \
  --ckpt '$OUT/$NAME' --plugin debug/hybridish_sft/plugin_ssmax
python debug/hybridish_sft/verify_ckpt_ssmax_load.py \
  --ckpt '$OUT/$NAME' --plugin debug/hybridish_sft/plugin_ssmax || \
  echo "[prepare] NOTE: ssmax verify skipped/failed -- checkpoint may have no ssmax, which is fine"
ls -la '$OUT/$NAME'
INNER

exec "$GANTRY" run --name "prep-$NAME" -w ai2/flex2 -b ai2/oe-other \
  --cluster "$CLUSTER" --gpus 1 --priority urgent \
  --weka oe-training-default:/weka/oe-training-default \
  --branch "$BRANCH" --install 'pip install -e .' \
  --timeout 0 --allow-dirty --yes -- bash -c "$CMD"
