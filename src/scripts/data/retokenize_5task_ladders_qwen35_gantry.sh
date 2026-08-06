#!/usr/bin/env bash
# Build the **Qwen3.5** twin of the canonical 32k 5-task ladders from the existing Qwen3 shards.
#
# Why from shards and not from JSONL: the source pools for single_task_ladders_v2/* and
# single_task_ladders_p10/nq live only on the Berkeley cluster (see each shard's metadata.json --
# /data/prasann/single_task_ladders_20k/... and /scratch/users/prasann/nq_p10_20k/...) and are not
# on weka or S3. retokenize_sft_shards_qwen3_to_qwen35.py therefore decodes each document, parses
# (user, answer) back out via the fixed chat-template delimiters, and re-renders through Qwen3.5's
# OWN apply_chat_template with an offset-derived loss mask -- equivalent to a rebuild from source,
# and order-preserving (see that script's docstring).
#
# Outputs are the roots that _qwen35_5task_dolci25_32k_nocpt_common.py reads.
#
#   bash src/scripts/data/retokenize_5task_ladders_qwen35_gantry.sh
#
set -euo pipefail

P="${P:-/weka/oe-training-default/ai2-llm/checkpoints/prasanns}"
OUT_V2="${OUT_V2:-$P/single_task_ladders_v2_qwen35}"
OUT_P10="${OUT_P10:-$P/single_task_ladders_p10_qwen35}"
NAME="${NAME:-retok-5task-ladders-qwen35}"
# max_seq_len of the Qwen3 builds; keep identical so the length-based drop set stays comparable.
MAX_SEQ_LEN="${MAX_SEQ_LEN:-40960}"
# Qwen3.5 re-tokenization inflates a few near-cap instances past the cap, so allow a small skip rate.
MAX_SKIP_FRAC="${MAX_SKIP_FRAC:-0.01}"

gantry run --name "$NAME" -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/neptune --cluster ai2/ceres --cluster ai2/saturn --cluster ai2/jupiter \
  --gpus 0 --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --weka oe-training-default:/weka/oe-training-default \
  --env HF_HUB_DISABLE_XET=1 \
  --python-manager conda --system-python \
  --install true --allow-dirty --timeout 0 --yes -- bash -c "
set -euo pipefail
retok() {
  local src=\"\$1\" dst=\"\$2\"
  echo \"############ \$src -> \$dst\"
  python src/scripts/data/retokenize_sft_shards_qwen3_to_qwen35.py \
    --in-dir \"\$src\" --out-dir \"\$dst\" \
    --max-seq-len $MAX_SEQ_LEN --max-skip-frac $MAX_SKIP_FRAC
}
retok '$P/single_task_ladders_v2/contradiction' '$OUT_V2/contradiction'
retok '$P/single_task_ladders_v2/oolong'        '$OUT_V2/oolong'
retok '$P/single_task_ladders_v2/rerank'        '$OUT_V2/rerank'
retok '$P/single_task_ladders_v2/outlier'       '$OUT_V2/outlier'
retok '$P/single_task_ladders_p10/nq'           '$OUT_P10/nq'
echo '=== ALL DONE. Qwen3 instance counts for comparison:'
echo '    contradiction 20091 / oolong 21000 / rerank 20000 / outlier 19981 / nq 19967'
for d in '$OUT_V2'/*/ '$OUT_P10'/nq/; do
  echo \"--- \$d\"
  python -c \"import json;m=json.load(open('\$d/metadata.json'));print('instances',m['num_instances'],'read',m['num_documents_read'],'tokens',m['num_tokens'],'skip_frac',m['skip_fraction'])\"
done
"
