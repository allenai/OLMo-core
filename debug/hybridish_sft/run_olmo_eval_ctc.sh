#!/usr/bin/env bash
# Run the CTC suite (olmo-eval, held-out, 500/rung) on a hybridish SFT checkpoint, via Beaker.
#
#   bash run_olmo_eval_ctc.sh <ckpt-name> <tag> [smoke]
#
# Launches from the olmo-eval clone so gantry ships THAT repo at the pushed ctc-suite branch.
# The checkpoints are self-contained (auto_map + the SSMax-patched modeling code lives in the
# checkpoint dir), so the container needs no plugin install -- trust_remote_code is the whole
# integration, and it is what keeps ssmax_scale from being silently dropped.
# The base image must carry torch: olmo-eval sizes its GPU plan from torch.cuda.device_count(),
# and the .[hf] extra pulls only transformers, so a torch-less image fails with the misleading
# "Not enough GPUs. Need 1 ... but only 0 available" even though Beaker allocated one.
# Eval data is the PUBLIC HF dataset PrasannSinghal/ctc-suite-eval: no weka mount needed for it,
# only for the checkpoints.
set -euo pipefail

CKPT="${1:?usage: run_olmo_eval_ctc.sh <ckpt-name> <tag> [smoke]}"
TAG="${2:?}"
MODE="${3:-full}"      # 'smoke', 'full', or a single task name to shard by
OE=/accounts/projects/berkeleynlp/prasann/projects/olmo-eval
WEKA=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_hybridish_sft
GANTRY="${GANTRY:-/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/gantry}"

SPECS=/tmp/claude-3018/-accounts-projects-berkeleynlp-prasann-projects-OLMo-core/7919ea25-3230-4b77-8a18-91d843d1c793/scratchpad/ctc_specs.txt
if [ "$MODE" = "smoke" ]; then
  # One cheap rung, a handful of instances: proves load + format + parse + score end to end.
  TASKS="-t ctc_nq:r2k -o limit=8 -t ctc_contradiction:r2k -o limit=8"
elif [ "$MODE" = "full" ]; then
  TASKS=""
  for s in $(cat "$SPECS"); do TASKS="$TASKS -t $s"; done
else
  # Shard by task: one job per (model, task) so the 8-task sweep runs concurrently rather than
  # serially. A 500-per-rung ladder to 32k is hours on one GPU; eight of them in sequence is not
  # an overnight job.
  TASKS=""
  for s in $(cat "$SPECS"); do
    case "$s" in "$MODE":*) TASKS="$TASKS -t $s";; esac
  done
  [ -n "$TASKS" ] || { echo "no rungs matched task '$MODE'"; exit 2; }
fi

cd "$OE"
exec "$GANTRY" run --name "ctceval-$TAG" -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/jupiter-cirrascale-2 --gpus 1 --priority urgent \
  --weka oe-training-default:/weka/oe-training-default \
  --branch prasann/ctc-suite-grader-fixes \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --install 'pip install ".[hf]" && pip install -U "transformers>=5.13" && python -c "import olmo_eval.cli,torch,transformers;print(\"olmo_eval\",olmo_eval.__file__);print(\"torch\",torch.__version__,torch.cuda.device_count(),\"gpu\");print(\"transformers\",transformers.__version__);assert torch.cuda.device_count()>0"' \
  --timeout 0 --allow-dirty --yes \
  -- olmo-eval run \
       --harness default \
       -o provider.kind=hf \
       -o provider.trust_remote_code=true \
       -o provider.dtype=bfloat16 \
       -m "$WEKA/$CKPT" \
       $TASKS \
       --output-dir "/results/$TAG" \
       --save-predictions
