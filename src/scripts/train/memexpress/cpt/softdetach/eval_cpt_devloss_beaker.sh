#!/bin/bash
# One Beaker GPU job per finished run: dev loss on the held-out cpt_dev shard (eval_cpt_devloss.py).
#   RUN=sdcpt-q35-4b-sd20-u64M ARM=sd20 bash src/scripts/train/memexpress/cpt/softdetach/eval_cpt_devloss_beaker.sh
set -uo pipefail
RUN="${RUN:?}"; ARM="${ARM:-dense}"; ROWS="${ROWS:-32}"; BRANCH="${BRANCH:-prasann/landmark}"
WEKA=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
read -r -d '' WORK <<EOF
set -uo pipefail
export PYTHONWARNINGS=ignore PATH=/opt/conda/bin:\$PATH
PYB=/opt/conda/bin/python; \$PYB -m pip install -q -e '.[all]' 2>&1 | tail -1
\$PYB -c "import torch, fla, olmo_core; print(torch.__version__)" || { echo "!!! deps missing"; exit 1; }
CK=\$(ls -d $WEKA/ctc_suite/ckpts/$RUN/model_and_optim $WEKA/ctc_suite/ckpts/$RUN/step*/model_and_optim 2>/dev/null | sort -V | tail -1)
[ -n "\$CK" ] || { echo "!!! no checkpoint for $RUN"; exit 2; }
PYTHONPATH=src \$PYB src/scripts/train/memexpress/cpt/softdetach/eval_cpt_devloss.py --ckpt \$CK --dev $WEKA/softdetach_cpt/shards/cpt_dev --arm $ARM --rows $ROWS --out $WEKA/softdetach_cpt/devloss/${RUN}.json
RC=\$?; echo "rc=\$RC"; exit \$RC
EOF
gantry run --name "sdcpt-eval-$RUN-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster 'ai2/jupiter*' --gpus 1 --cpus 8 --memory 120GiB --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --install false --branch "$BRANCH" --allow-dirty \
  --weka oe-training-default:/weka/oe-training-default \
  --timeout 0 --yes -- bash -c "$WORK"
