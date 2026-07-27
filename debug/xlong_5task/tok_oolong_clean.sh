#!/bin/bash
# Re-tokenize the DECONTAMINATED oolong pool (train examples colliding with the 8k/16k/32k eval
# rungs removed -- see decontaminate_oolong_train.py). Emits one variant per invocation:
#
#   bash tok_oolong_clean.sh shards_chunked                    # WITH box markers (chunked arm)
#   bash tok_oolong_clean.sh shards_full --no-doc-markers      # NO markers (standard arm)
#
# Must live on shared NFS (this repo), not the session scratchpad: /tmp is node-local, so a script
# staged there is invisible to the compute node and srun dies with exit 127.
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
VARDIR=$1; EXTRA=${2:-}
OUT=/data/prasann/xlong5/$VARDIR/oolong_train
source /usr/local/linux/miniforge-3.13/etc/profile.d/conda.sh 2>/dev/null \
  || source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate /data/prasann/conda/envs/corpus-reasoning-olmo
export PYTHONPATH=$REPO/src HF_HOME=/data/prasann/hf_cache TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
rm -rf "$OUT"; mkdir -p "$OUT"
python $REPO/src/scripts/data/convert_unified_to_document_landmark.py \
  --emit dense --task oolong --chunk-by line --cot-mode none \
  --marker-set qwen3_5 --tokenizer /scratch/users/prasann/hf_models/Qwen3.5-4B-Base \
  --seq-len 262144 --shard-tokens 100000000 --num-proc 16 $EXTRA \
  --input-jsonl /data/prasann/xlong5/pools_oolong_clean/*.jsonl --out-dir "$OUT"
echo "TOKENIZE_DONE $VARDIR rc=$?"
