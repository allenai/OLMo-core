#!/bin/bash
# List the weka paths the 16k-64k data-scaling campaign (debug/ds64) needs: 64k eval rungs, tokenizers, bases.
W=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
read -r -d '' WORK <<EOW
for d in _eval_bundle_eval500_v2/contra _eval_bundle_eval500_v2/nq _eval_bundle_eval500_v2/outlier _eval_bundle_eval500_v2_clean/oolong _eval_bundle_eval500_v3/contra outlier_lengthmix/eval_rungs/nq outlier_lengthmix/eval_rungs/outlier hf_tokenizers ctc_suite/bases flop_scaling35/shards flop_scaling/shards; do echo "== \$d"; ls -la $W/\$d 2>&1 | head -40; done
EOW
gantry run --name "ds64-weka-ls-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other --cluster ai2/jupiter-cirrascale-2 --cluster ai2/ceres-cirrascale --cluster ai2/saturn-cirrascale --cluster ai2/neptune-cirrascale --gpus 0 --cpus 2 --memory 4GiB --priority urgent --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --install false --weka oe-training-default:/weka/oe-training-default --allow-dirty --timeout 0 --yes -- bash -c "$WORK" 2>&1 | grep -oE "ex/[A-Z0-9]{26}" | head -1 | cut -d/ -f2
