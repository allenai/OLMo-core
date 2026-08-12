#!/bin/bash
# Repair the TRUNCATED qdmatch_nq 20k shard.
#
# The 2026-07-19 conversion died partway through writing token parts: the shard has all FIVE
# labels_mask parts (245,984,267 bytes of bool = exactly the metadata's num_tokens) but only THREE
# of five token_ids parts (720MB of uint32 against the 984MB that num_tokens implies). The loader
# refuses it with `'label_mask_paths' should have the same length as 'source_paths'`, so every
# qdmatch_nq training run since has died at startup -- which is almost certainly why qdmatch_nq
# never landed in results/ctc_suite/dense_vs_chunked_table.md.
#
# The break is in the tokenized artifact, not the data: the source
# `qdmatch_nq_train_20k_mixedn.jsonl` is intact. So this re-runs the SAME converter invocation as
# `convert_ctc_final_dense_cubbins.sbatch` (index 10) on the SAME input -- no examples are
# regenerated, only re-tokenized.
#
#   sbatch debug/ctc_modelscale/repair_qdmatch_nq_shard.sbatch
#SBATCH --job-name=qdmatch-nq-repair
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/data/prasann/joblogs/qdmatch_nq_repair_%j.log

set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
CONV=$REPO/src/scripts/data/convert_unified_to_document_landmark.py
SRC=/data/prasann/ctc_suite_data/qdmatch_nq/final20k/qdmatch_nq_train_20k_mixedn.jsonl
OUTDIR=/data/prasann/ctc_suite/shards_final/qdmatch_nq_train_repaired

echo "=== HOST=$(hostname) START=$(date '+%F %T') ==="
[ -f "$SRC" ] || { echo "MISSING INPUT: $SRC"; exit 2; }
echo "  input lines=$(wc -l < "$SRC")"

source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate /data/prasann/conda/envs/corpus-reasoning-olmo
export PYTHONPATH="$REPO/src" HF_HOME=/data/prasann/hf_cache
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"
python "$CONV" --emit dense --task qdmatch --chunk-by document --cot-mode none \
  --marker-set qwen3_5 --tokenizer Qwen/Qwen3.5-0.8B-Base \
  --seq-len 40960 --shard-tokens 60000000 \
  --input-jsonl "$SRC" --out-dir "$OUTDIR"
rc=$?
echo "  convert rc=$rc $(date '+%T')"

echo "=== VERIFY ==="
ls -la "$OUTDIR"
python - <<PY
import glob, json, os
d = "$OUTDIR"
tok = sorted(glob.glob(os.path.join(d, "token_ids_part_*.npy")))
lab = sorted(glob.glob(os.path.join(d, "labels_mask_part_*.npy")))
m = json.load(open(os.path.join(d, "metadata.json")))
tb = sum(os.path.getsize(p) for p in tok)
lb = sum(os.path.getsize(p) for p in lab)
print(f"token parts={len(tok)} bytes={tb}")
print(f"label parts={len(lab)} bytes={lb}")
print(f"metadata num_instances={m['num_instances']} num_tokens={m['num_tokens']}")
ok = (len(tok) == len(lab)
      and tb == m["num_tokens"] * 4          # uint32
      and lb == m["num_tokens"])             # bool
print("VERDICT:", "PASS" if ok else "FAIL -- still inconsistent")
PY
echo "=== DONE $(date '+%F %T') ==="
