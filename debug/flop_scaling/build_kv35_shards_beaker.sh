#!/bin/bash
# Beaker-native marker re-tokenization of the taskscale arms (contradiction / oolong), whose JSONL
# lives only on weka (taskscale_lengthmix/arms/<task>_mix_s<B>M.jsonl). Writes
# flop_scaling35/shards/<task>_s<B>M_mk on weka directly. CPU gantry job on the olmo-core image.
#   TASK=contradiction BUDGETS="14M 28M 56M" bash debug/flop_scaling/build_kv35_shards_beaker.sh
set -uo pipefail
TASK="${TASK:?}"; BUDGETS="${BUDGETS:?}"
WEKA=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
case "$TASK" in oolong) CONV=oolong; CHUNK=line;; nq) CONV=retrieval; CHUNK=document;; *) CONV=$TASK; CHUNK=document;; esac
read -r -d '' WORK <<EOW
set -uo pipefail; export PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false
ls $WEKA/taskscale_lengthmix/ ; ls $WEKA/taskscale_lengthmix/arms | head -20
for B in $BUDGETS; do
  SRC=\$(ls $WEKA/taskscale_lengthmix/arms/${TASK}_mix_s\$B*.jsonl 2>/dev/null | head -1)
  [ -s "\$SRC" ] || { echo "!!! no JSONL for $TASK \$B under taskscale_lengthmix/arms"; continue; }
  OUT=$WEKA/flop_scaling35/shards/${TASK}_s\${B}_mk; [ -s \$OUT/metadata.json ] && { echo "[skip] \$OUT"; continue; }
  echo "--- $TASK \$B: \$SRC -> \$OUT \$(date +%T) ---"; mkdir -p \$OUT
  PYTHONPATH=src python src/scripts/data/convert_unified_to_document_landmark.py --input-jsonl \$SRC --task $CONV --out-dir \$OUT --emit dense --marker-set qwen3_5 --tokenizer Qwen/Qwen3.5-0.8B-Base --seq-len 65536 --query-position after --cot-mode none --chunk-by $CHUNK --emit-gold-sidecar --num-proc 16 || { echo "!!! FAILED \$B"; continue; }
  python -c "import json;m=json.load(open('\$OUT/metadata.json'));print('   ',{k:m.get(k) for k in ('num_instances','task','marker_set','query_position','max_example_len')})"
done
ls -la $WEKA/flop_scaling35/shards/ ; echo KVDATA_DONE
EOW
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
gantry run --name "fs35-kvdata-$TASK-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/jupiter-cirrascale-2 --gpus 0 --cpus 16 --memory 120GiB --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --install true \
  --weka oe-training-default:/weka/oe-training-default \
  --allow-dirty \
  --timeout 0 --yes -- bash -c "$WORK" 2>&1 | grep -E "beaker.org/ex|rror" | head -2
