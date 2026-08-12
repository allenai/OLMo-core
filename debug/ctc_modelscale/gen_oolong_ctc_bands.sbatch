#!/bin/bash
# Generate an OOLONG training pool with the 5-TASK-MIX source/setup but the CTC-SUITE length
# distribution (user, 2026-08-12: "source / setup should be the same as the 5task mix ... just
# length distributions and other things should be based on the ctc suite (e.g. data only for
# in-domain eval rungs, evenly split)").
#
# Same generator, same knobs, same offline item substrate as debug/xlong_5task/gen_oolong.sbatch --
# only the band table and the per-band counts change:
#
#   5task mix : 7 bands 2k->256k, counts DECREASING per band (1500/1250/1000/675/350/150/75 per
#               shard) so the long tail stays affordable.
#   this build: 5 bands 2k->32k ONLY -- exactly the lengths the CTC oolong eval rungs cover
#               (measured label/actual 0.95-1.00, so oolong's rung labels really are token counts)
#               -- with counts EVENLY split: 1000/shard x 4 shards x 5 bands = 20,000 examples,
#               4,000 per band.
#
# Band edges [LO,HI] are copied verbatim from the 5task build so each band is the same slice of the
# ladder, just evenly weighted. NO --item-regex anywhere (the leak fix); --pool-max-ctx and the
# tokenizer are kept identical to the 5task build so the item substrate is unchanged.
#
# Seeds 3000+ keep these item-combinations disjoint from the 5task train pool (42-69) and from its
# eval builds (9000-9019, 9500+). A separate decontamination pass against the CTC oolong eval rungs
# still runs afterwards -- seeds alone do not guarantee disjointness.
#
# ⚠ CAPACITY TRAP (gen_oolong_ultra.sbatch): the packer draws items WITHOUT REPLACEMENT from one
# dataset's pool and silently emits a SHORT example if that pool runs dry. Far less likely at <=32k
# than at 512k+, but the realized band lengths MUST be checked before this pool is used --
# check_realized_bands.py does that, and it is not optional.
#
#   sbatch --array=0-19 debug/ctc_modelscale/gen_oolong_ctc_bands.sbatch
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=mooney
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=ctcms-oolong-gen
#SBATCH --output=/data/prasann/ctc_oolong_ctc/logs/gen_%a_%A.log

set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
LO=(2048 4096 8192 16384 32768)
HI=(3973 7946 15892 31784 63569)
PER=1000                                  # per shard, per band -> 4000/band, 20000 total
B=$(( SLURM_ARRAY_TASK_ID / 4 )); S=$(( SLURM_ARRAY_TASK_ID % 4 ))
OUT=/data/prasann/ctc_oolong_ctc/pools/band${B}_shard${S}
mkdir -p "$OUT" /data/prasann/ctc_oolong_ctc/logs

source /usr/local/linux/miniforge-3.13/etc/profile.d/conda.sh 2>/dev/null \
  || source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate /data/prasann/conda/envs/corpus-reasoning-olmo
export PYTHONPATH=$REPO/src HF_HOME=/data/prasann/hf_cache HF_DATASETS_CACHE=/data/prasann/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
# The dataset is cached; forcing offline avoids the 429 IP rate-limit that killed tasks in the
# 5task build when many array tasks hit the HF dataset API at once.
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "=== band=$B shard=$S len=[${LO[$B]},${HI[$B]}] n=$PER HOST=$(hostname) $(date '+%F %T') ==="
python $REPO/src/corpus_reasoning/data/generate_oolong_ladder_data.py \
  --num-examples $PER --len-min ${LO[$B]} --len-max ${HI[$B]} \
  --pool-max-ctx 262144 --tokenizer /data/prasann/hf_models/Qwen3.5-4B-Base \
  --seed $(( 3000 + SLURM_ARRAY_TASK_ID )) --num-shards 4 --shard-index $S --output-dir "$OUT"
echo "rc=$? done $(date '+%F %T')"
