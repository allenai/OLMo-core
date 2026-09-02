#!/bin/bash
# Beaker-NATIVE short-heavy data build (Prasann 2026-09-02: build data on Beaker, not locally).
# Same four steps as build_shortheavy_data.sbatch, but as a CPU gantry job on a weka node that
# writes the shards straight to weka (no S3 relay). The ctc-data seed pools come from the HF Hub
# (Beaker nodes have internet), the converter + ctc package from the pushed commit of this repo.
#
#   TASK=outlier bash debug/flop_scaling/build_shortheavy_data_beaker.sh
#   TASK=contradiction BUDGETS=8M,16M,32M,48M POOL_2K=12000 POOL_4K=3600 POOL_8K=1100 POOL_16K=280 POOL_32K=80 \
#     bash debug/flop_scaling/build_shortheavy_data_beaker.sh
set -uo pipefail
TASK="${TASK:?set TASK=outlier|contradiction|nq|oolong}"
BUDGETS="${BUDGETS:-8M,16M,32M,64M,128M}"
POOL_2K="${POOL_2K:-31000}"; POOL_4K="${POOL_4K:-9300}"; POOL_8K="${POOL_8K:-2800}"
POOL_16K="${POOL_16K:-700}"; POOL_32K="${POOL_32K:-180}"
WEKA=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/flop_scaling
CTC_REPO="${CTC_REPO:-https://github.com/PrasannS/ctc.git}"   # the public ctc-data package (ctc-data CLI)
case "$TASK" in nq) CONV_TASK=retrieval; CHUNK_BY=document ;; oolong) CONV_TASK=oolong; CHUNK_BY=line ;; *) CONV_TASK=$TASK; CHUNK_BY=document ;; esac

read -r -d '' WORK <<EOF
set -uo pipefail
export PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false
pip install -q "git+$CTC_REPO" 2>&1 | tail -1 || true
W=$WEKA/build/$TASK; mkdir -p \$W/pools \$W/arms $WEKA/shards
i=0
for R in 2k 4k 8k 16k 32k; do
  case \$R in 2k) N=$POOL_2K;; 4k) N=$POOL_4K;; 8k) N=$POOL_8K;; 16k) N=$POOL_16K;; 32k) N=$POOL_32K;; esac
  OUT=\$W/pools/${TASK}_\$R
  if [ -s \$OUT/$TASK/train.jsonl ] && [ \$(wc -l < \$OUT/$TASK/train.jsonl) -ge \$N ]; then echo "[skip] pool \$R"; continue; fi
  i=\$((i+1)); echo "--- pool $TASK \$R: \$N \$(date +%T) ---"
  ctc-data build --task $TASK --out \$OUT --split train --rungs \$R --train \$N --seed \$((42+i)) --pool auto --force || { echo "!!! pool FAILED \$R"; exit 1; }
  echo "    -> \$(wc -l < \$OUT/$TASK/train.jsonl) rows"
done
python debug/flop_scaling/compose_shortheavy_arms.py --task $TASK --pools-dir \$W/pools --out-dir \$W/arms --budgets $BUDGETS || exit 1
for f in \$W/arms/${TASK}_sh*.jsonl; do
  ARM=\$(basename \$f .jsonl); OUT=$WEKA/shards/\$ARM
  if [ -s \$OUT/metadata.json ]; then echo "[skip] shard \$ARM"; continue; fi
  echo "--- tokenizing \$ARM \$(date +%T) ---"; mkdir -p \$OUT
  PYTHONPATH=src python src/scripts/data/convert_unified_to_document_landmark.py --input-jsonl \$f --task $CONV_TASK --out-dir \$OUT --emit dense --marker-set qwen3 --tokenizer Qwen/Qwen3-4B --seq-len 40960 --query-position after --cot-mode none --chunk-by $CHUNK_BY --emit-gold-sidecar --num-proc 16 || { echo "!!! tokenize FAILED \$ARM"; exit 1; }
  python -c "import json;m=json.load(open('\$OUT/metadata.json'));print('    metadata:',{k:m.get(k) for k in ('num_instances','task','marker_set','query_position')})"
done
ls -la $WEKA/shards | grep $TASK; echo "=== DONE $TASK ==="
EOF

gantry run --name "fs-data-$TASK-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster 'ai2/jupiter*' --cluster 'ai2/neptune*' --cluster 'ai2/ceres*' --cluster 'ai2/saturn*' --gpus 0 --cpus 16 --memory 120GiB --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --install true \
  --weka oe-training-default:/weka/oe-training-default \
  --allow-dirty \
  --timeout 0 --yes -- bash -c "$WORK"
