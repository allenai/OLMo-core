#!/bin/bash
# Convert OLMo 3 7B HF revisions to OLMo-core model-only distcp on the chosen node's /data (sbatch -w <node>), with the
# check_conversion.py fp32/bf16 parity check (the step-2 load check). The converter's own validation is
# skipped: it compares bf16 logits at atol 1e-4, which bf16 kernel noise alone fails (max 0.25 on stage 2).
#   stage2-step47684 = the CPT start (8k, no YaRN); stage3-step11921 = the long-context reference (YaRN x8).
# Needs the repo tree staged at $CODE first (login node):
#   git -C $REPO archive HEAD | srun -p jsteinhardt --qos=preemptive_high -w <node> -c1 --mem=2G -t 5 \
#     bash -c 'mkdir -p /data/prasann/olmo3_cptdrop/code && tar -x -C /data/prasann/olmo3_cptdrop/code'
#SBATCH --job-name=o3cd-convert
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --output=/data/prasann/olmo3_cptdrop/logs/%x-%j.out
set -euo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
ROOT=/data/prasann/olmo3_cptdrop
CODE=$ROOT/code
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export HF_HOME="$ROOT/hf_cache" TMPDIR="$ROOT/tmp" PYTHONPATH="$CODE/src"
mkdir -p "$TMPDIR" "$ROOT/base"
cd "$CODE"
for REV in ${REVS:-stage2-step47684 stage3-step11921}; do
  # the converter reads <path>/config.json through cached_path, so it needs a local copy, not a repo id
  HFDIR="$ROOT/hf/$REV"
  "$PY" -c "from huggingface_hub import snapshot_download as s; s('allenai/Olmo-3-1025-7B', revision='$REV', local_dir='$HFDIR')"
  OPS_RUN_ID="o3cd-convert-${REV}-${SLURM_JOB_ID}" OPS_EXPERIMENT=olmo3_cptdrop OPS_KIND=data \
  OPS_EXPECTED_MINUTES=40 OPS_FIRST_OUTPUT=900 OPS_STALL=1800 \
  bash "$REPO/recipes/env/ops_job.sh" \
    "$PY" -u src/examples/huggingface/convert_checkpoint_from_hf.py \
      --checkpoint-input-path "$HFDIR" \
      --model-arch olmo3_7b --tokenizer dolma2 \
      --output-dir "$ROOT/base/$REV" --device cuda --skip-validation
  YF=0; [[ "$REV" == stage3-* ]] && YF=8
  cp "$REPO/research/token_routing/olmo3_cptdrop/check_conversion.py" "$ROOT/check_conversion.py"
  "$PY" -u "$ROOT/check_conversion.py" --hf-dir "$HFDIR" --olmo-dir "$ROOT/base/$REV" \
    --dev "${PARITY_DEV:-$ROOT/data/dev32k}" --real-len 6144 --yarn-factor "$YF" | tee "$ROOT/base/$REV/parity.txt"
  grep -q "PARITY PASS" "$ROOT/base/$REV/parity.txt"
done
