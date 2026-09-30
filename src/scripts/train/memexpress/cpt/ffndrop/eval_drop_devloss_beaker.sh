#!/bin/bash
# One Beaker GPU job per checkpoint: drop-robustness dev loss (eval_drop_devloss.py) on cpt_dev.
#   CKPT=<weka .../model_and_optim> NAME=<label> bash src/scripts/train/memexpress/cpt/ffndrop/eval_drop_devloss_beaker.sh
#   RUN=fdcpt-q35-4b-drop75l12-u1B bash ...   (waits for the run's final model-only export)
set -uo pipefail
RUN="${RUN:-}"; CKPT="${CKPT:-}"; NAME="${NAME:-$RUN}"; ROWS="${ROWS:-32}"; RATES="${RATES:-0,0.25,0.5,0.75}"; BRANCH="${BRANCH:-prasann/landmark}"
WEKA=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
[ -n "$RUN$CKPT" ] || { echo "set RUN or CKPT"; exit 1; }
read -r -d '' WORK <<EOF
set -uo pipefail
export PYTHONWARNINGS=ignore PATH=/opt/conda/bin:\$PATH
PYB=/opt/conda/bin/python; \$PYB -m pip install -q -e '.[all]' 2>&1 | tail -1
\$PYB -c "import torch, fla, olmo_core; print(torch.__version__)" || { echo "!!! deps missing"; exit 1; }
CK="$CKPT"
if [ -z "\$CK" ]; then
  CK=$WEKA/ctc_suite/ckpts/$RUN/model_and_optim; W=0
  while [ ! -f "\$CK/.metadata" ]; do
    [ \$W -ge 43200 ] && { echo "!!! no final export for $RUN after 12h"; exit 2; }
    sleep 120; W=\$((W+120))
  done; echo "checkpoint: \$CK (waited \${W}s)"
fi
PYTHONPATH=src \$PYB src/scripts/train/memexpress/cpt/ffndrop/eval_drop_devloss.py --ckpt \$CK --dev $WEKA/softdetach_cpt/shards/cpt_dev --rows $ROWS --rates $RATES --out $WEKA/ffndrop_cpt/devloss/${NAME}.json
RC=\$?; echo "rc=\$RC"; exit \$RC
EOF
gantry run --name "fdcpt-eval-$NAME-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster 'ai2/jupiter*' --gpus 1 --cpus 8 --memory 120GiB --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --install false --branch "$BRANCH" --allow-dirty \
  --weka oe-training-default:/weka/oe-training-default \
  --timeout 0 --yes -- bash -c "$WORK"
