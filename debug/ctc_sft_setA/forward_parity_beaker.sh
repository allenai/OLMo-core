#!/usr/bin/env bash
# One single-GPU Beaker job: score the same setA training sequences with a checkpoint loaded under
# the TRAINING OLMo-core (prasann/ctc-setA-sft) and then under the EVAL one (prasann/landmark, what
# olmo-eval's launch_ctc_suite.py installs), and compare. See forward_parity.py.
#
#   bash debug/ctc_sft_setA/forward_parity_beaker.sh <weka step dir> [name]
#
# Runs the PUSHED commit of the current branch (gantry clones it), so commit + push first.
set -euo pipefail
CKPT=${1:?usage: forward_parity_beaker.sh <weka step dir> [name]}
NAME=${2:-setA-forward-parity-$(date -u +%Y%m%dT%H%M)}
TRAIN_REF=${TRAIN_REF:-prasann/ctc-setA-sft}
EVAL_REF=${EVAL_REF:-prasann/landmark}
GIT=https://github.com/allenai/OLMo-core.git
P=debug/ctc_sft_setA/forward_parity.py
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$HOME/.local/bin:$PATH

CMD="set -euo pipefail
python $P dump --ckpt $CKPT --out /tmp/train.pt
pip install -q --no-deps --force-reinstall 'ai2-olmo-core @ git+$GIT@$EVAL_REF'
python $P dump --ckpt $CKPT --out /tmp/eval.pt
python $P compare /tmp/train.pt /tmp/eval.pt"

gantry run --name "$NAME" -w ai2/flex2 -b ai2/oe-other --cluster ai2/jupiter-cirrascale-2 \
  --gpus 1 --priority urgent --weka oe-training-default:/weka/oe-training-default \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --python-manager conda --system-python \
  --install "pip install 'ai2-olmo-core[fla] @ git+$GIT@$TRAIN_REF' dataclass-extensions" \
  --timeout 0 --shared-memory 32GiB --allow-dirty --yes -- bash -c "$CMD"
