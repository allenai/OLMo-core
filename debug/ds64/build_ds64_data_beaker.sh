#!/bin/bash
# Beaker-NATIVE data build for the uniform 16k-64k data-scaling campaign (records/ds64-scaling-plan.md):
# per-rung train pools via ctc-data (seed pools from the HF Hub), uniform-share nested arms at
# 32M/64M/128M, Qwen3.5 MARKER tokenization (both the dense and the soft-token arms train on the
# same marker-wrapped shards) at seq 65536 with gold sidecars. Writes straight to weka.
#
#   TASK=outlier bash debug/ds64/build_ds64_data_beaker.sh
set -uo pipefail
TASK="${TASK:?set TASK=outlier|contradiction|nq|oolong}"
BUDGETS="${BUDGETS:-32M,64M,128M}"
# pool sizes = 128M / 4 rungs / rung tokens, +15%
POOL_16K="${POOL_16K:-2300}"; POOL_32K="${POOL_32K:-1150}"; POOL_48K="${POOL_48K:-770}"; POOL_56K="${POOL_56K:-660}"
WEKA=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ds64
TOKENIZER=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/hf_tokenizers/Qwen3.5-0.8B-Base
CTC_REPO="${CTC_REPO:-https://github.com/PrasannS/ctc.git}"
case "$TASK" in nq) CONV_TASK=retrieval; CHUNK_BY=document ;; oolong) CONV_TASK=oolong; CHUNK_BY=line ;; *) CONV_TASK=$TASK; CHUNK_BY=document ;; esac

read -r -d '' WORK <<EOF
set -uo pipefail
export PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
pip install -q "git+$CTC_REPO" 2>&1 | tail -1 || true
W=$WEKA/build/$TASK; mkdir -p \$W/pools \$W/arms $WEKA/shards
i=0
for R in 16k 32k 48k 56k; do
  case \$R in 16k) N=$POOL_16K;; 32k) N=$POOL_32K;; 48k) N=$POOL_48K;; 56k) N=$POOL_56K;; esac
  OUT=\$W/pools/${TASK}_\$R
  if [ -s \$OUT/$TASK/train.jsonl ] && [ \$(wc -l < \$OUT/$TASK/train.jsonl) -ge \$N ]; then echo "[skip] pool \$R"; continue; fi
  i=\$((i+1)); echo "--- pool $TASK \$R: \$N \$(date +%T) ---"
  ctc-data build --task $TASK --out \$OUT --split train --rungs \$R --train \$N --seed \$((4200+i)) --pool auto --force || { echo "!!! pool FAILED \$R"; exit 1; }
  echo "    -> \$(wc -l < \$OUT/$TASK/train.jsonl) rows"
done
python debug/ds64/compose_uniform_arms.py --task $TASK --pools-dir \$W/pools --out-dir \$W/arms --budgets $BUDGETS || exit 1
for f in \$W/arms/${TASK}_u*.jsonl; do
  ARM=\$(basename \$f .jsonl); OUT=$WEKA/shards/\$ARM
  if [ -s \$OUT/metadata.json ]; then echo "[skip] shard \$ARM"; continue; fi
  echo "--- tokenizing \$ARM \$(date +%T) ---"; mkdir -p \$OUT
  PYTHONPATH=src python src/scripts/data/convert_unified_to_document_landmark.py --input-jsonl \$f --task $CONV_TASK --out-dir \$OUT --emit dense --marker-set qwen3_5 --tokenizer $TOKENIZER --seq-len 65536 --query-position after --cot-mode none --chunk-by $CHUNK_BY --emit-gold-sidecar --num-proc 16 || { echo "!!! tokenize FAILED \$ARM"; exit 1; }
  python -c "import json;m=json.load(open('\$OUT/metadata.json'));print('    metadata:',{k:m.get(k) for k in ('num_instances','num_dropped','num_tokens','max_example_len','min_example_len','marker_set','query_position')})"
done
ls -la $WEKA/shards | grep $TASK; echo "=== DONE $TASK ==="
EOF

gantry run --name "ds64-data-$TASK-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster 'ai2/jupiter*' --cluster 'ai2/neptune*' --cluster 'ai2/ceres*' --cluster 'ai2/saturn*' --gpus 0 --cpus 16 --memory 120GiB --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --install true \
  --weka oe-training-default:/weka/oe-training-default \
  --allow-dirty \
  --timeout 0 --yes -- bash -c "$WORK" 2>&1 | grep -oE "ex/[A-Z0-9]{26}" | head -1 | cut -d/ -f2
