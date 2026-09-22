#!/bin/bash
# Beaker-native (weka-mounted, CPU) build of the soft-detach CPT shards, records/softdetach-cpt-plan.md.
# Train shard = 128M row-tokens from source parts 0-7; dev shard = 64 rows from the LAST source part
# (held out: never read by any training arm). Both are marker-wrapped 127x512 rows (build_cpt_shard.py).
#   bash src/scripts/train/memexpress/cpt/softdetach/build_cpt_shard_beaker.sh
set -uo pipefail
SRC=/weka/oe-training-default/ai2-llm/checkpoints/amandab/dolma3_longmino_mix_sample15B_qwen3_5
WEKA=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/softdetach_cpt
BRANCH="${BRANCH:-prasann/landmark}"
read -r -d '' WORK <<EOF
set -uo pipefail
export PYTHONWARNINGS=ignore
PYB=/opt/conda/bin/python; \$PYB -m pip install -q -e . 2>&1 | tail -1
\$PYB -c "import numpy, olmo_core" || { echo "!!! deps missing"; exit 1; }
N=\$(ls $SRC/part-*.npy | wc -l); echo "source parts: \$N"; LAST=\$((N-1))
mkdir -p $WEKA/shards
[ -s $WEKA/shards/cpt_u128M/metadata.json ] || PYTHONPATH=src \$PYB src/scripts/train/memexpress/cpt/softdetach/build_cpt_shard.py --src $SRC --out $WEKA/shards/cpt_u128M --tokens 128000000 --parts 0-7 || exit 1
[ -s $WEKA/shards/cpt_dev/metadata.json ] || PYTHONPATH=src \$PYB src/scripts/train/memexpress/cpt/softdetach/build_cpt_shard.py --src $SRC --out $WEKA/shards/cpt_dev --tokens 4161600 --parts \$LAST-\$LAST || exit 1
# 1B-token shard for the high-end sd20 points (256M/512M/1B tokens as --max-tokens prefixes): source
# parts 0-27 (parts 0-7 = the 128M shard's source, re-read here -- the SAME first 128M rows in the
# same order, so cpt_u128M is a prefix; part 28 stays the held-out dev part)
[ -s $WEKA/shards/cpt_u1B/metadata.json ] || PYTHONPATH=src \$PYB src/scripts/train/memexpress/cpt/softdetach/build_cpt_shard.py --src $SRC --out $WEKA/shards/cpt_u1B --tokens 1000000000 --parts 0-27 || exit 1
for s in cpt_u128M cpt_dev cpt_u1B; do \$PYB -c "import json;m=json.load(open('$WEKA/shards/\$s/metadata.json'));print('\$s',{k:m[k] for k in ('num_instances','num_tokens','num_loss_tokens','max_example_len','source_parts')})"; done
ls -la $WEKA/shards/*; echo "=== DONE ==="
EOF
gantry run --name "softdetach-cpt-data-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster 'ai2/jupiter*' --cluster 'ai2/neptune*' --cluster 'ai2/ceres*' --cluster 'ai2/saturn*' --gpus 0 --cpus 8 --memory 160GiB --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --install false --branch "$BRANCH" --allow-dirty \
  --weka oe-training-default:/weka/oe-training-default \
  --timeout 0 --yes --no-logs -- bash -c "$WORK"
