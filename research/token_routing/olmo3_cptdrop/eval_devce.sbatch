#!/bin/bash
# Full-context dev CE (eval_devce.py) for OLMo 3 checkpoints on one H200 of the data node (sbatch -w <node>).
# Each SPEC is label:ckpt_dir:yarn_factor (ckpt_dir relative to $ROOT or absolute; yarn none|4|8), scored on
# every shard in SHARDS; one JSON per (spec, shard) in $OUT_DIR. CPU tests run first unless TESTS=0.
#   sbatch -w mooney --export=ALL,EVAL_GIT_COMMIT=$(git rev-parse --short HEAD),CODE=/data/prasann/olmo3_cptdrop/code_eval \
#     research/token_routing/olmo3_cptdrop/eval_devce.sbatch
# Default SPECS = the sanity check: stage2 at YaRN x4 (the CPT starting state) vs stage3 at YaRN x8.
# Needs the repo tree staged at $CODE first (login node):
#   git archive HEAD | srun -p jsteinhardt --qos=preemptive_high -w <node> -c1 --mem=2G -t 5 \
#     bash -c 'rm -rf $CODE && mkdir -p $CODE && tar -x -C $CODE'
#SBATCH --job-name=o3cd-devce
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/data/prasann/olmo3_cptdrop/logs/%x-%j.out
set -euo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
ROOT=/data/prasann/olmo3_cptdrop
CODE=${CODE:-$ROOT/code_eval}
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
SPECS=${SPECS:-"stage2-yarn4:base/stage2-step47684:4 stage3-yarn8:base/stage3-step11921:8"}
SHARDS=${SHARDS:-"dev32k short4k"}
OUT_DIR=${OUT_DIR:-$ROOT/devce}
export HF_HOME="$ROOT/hf_cache" TMPDIR="$ROOT/tmp" PYTHONPATH="$CODE/src:$CODE/research/token_routing/olmo3_cptdrop"
export TRITON_CACHE_DIR="$ROOT/tmp/triton" EVAL_GIT_COMMIT="${EVAL_GIT_COMMIT:-unknown}"
mkdir -p "$TMPDIR" "$OUT_DIR"
cd "$CODE"
if [[ "${TESTS:-1}" == 1 ]]; then
  CUDA_VISIBLE_DEVICES= "$PY" -m pytest -q -p no:cacheprovider research/token_routing/olmo3_cptdrop/test_eval_devce_cpu.py
fi
for SPEC in $SPECS; do
  IFS=: read -r LABEL CK YF <<<"$SPEC"
  [[ "$CK" == /* ]] || CK="$ROOT/$CK"
  for SH in $SHARDS; do
    OPS_RUN_ID="o3cd-devce-${LABEL}-${SH}-${SLURM_JOB_ID}" OPS_EXPERIMENT=olmo3_cptdrop OPS_KIND=eval \
    OPS_PARENT_RUN="$(basename "$CK")" OPS_CKPT_PATH="$CK" OPS_TASK="devce_${SH}" OPS_EVAL_VARIANT="yarn${YF}" \
    OPS_EXPECTED_MINUTES=15 OPS_FIRST_OUTPUT=600 OPS_STALL=900 \
    bash "$REPO/recipes/env/ops_job.sh" \
      "$PY" -u research/token_routing/olmo3_cptdrop/eval_devce.py \
        --ckpt "$CK" --yarn-factor "$YF" --dev "$ROOT/data/$SH" --label "$LABEL" \
        --out "$OUT_DIR/${LABEL}_${SH}.json"
  done
done
