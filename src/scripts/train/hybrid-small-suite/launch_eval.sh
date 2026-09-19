#!/usr/bin/env bash
# Score a checkpoint on the CTC suite (olmo-eval, held-out, 500 per rung), via Beaker.
#
#   bash launch_eval.sh <ckpt-name> <tag> smoke        # ~3 min, proves the path end to end
#   bash launch_eval.sh <ckpt-name> <tag> ctc_nq       # one task, all rungs to 32k
#   bash launch_eval.sh <ckpt-name> <tag> full         # all 39 task x rung runs
#
# <ckpt-name> is a directory name under $WEKA (see below), not a full path.
# The olmo-eval branch carrying the CTC suite is pinned here, and that clone is made by gantry --
# you do not need it checked out.
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
# gantry needs a checkout of allenai/olmo-eval to read the branch/commit from. Point OE at one.
OE="${OE:-$HOME/projects/olmo-eval}"
WEKA=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_hybridish_sft
GANTRY="${GANTRY:-/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/gantry}"
# Default 64 OOMs on the long-context tasks: a single 20 GiB activation on top of ~29 GiB lost to
# allocator fragmentation. CHUNK shrinks the batch; expandable_segments reclaims the fragmentation.
CHUNK="${CHUNK:-64}"

SPECS="${SPECS:-$(dirname "$0")/ctc_task_rungs.txt}"
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
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --timeout 0 --allow-dirty --yes \
  -- olmo-eval run \
       --harness default \
       -o provider.kind=hf \
       -o provider.trust_remote_code=true \
       -o provider.dtype=bfloat16 \
       -o batching.chunk_size=$CHUNK \
       -m "$WEKA/$CKPT" \
       $TASKS \
       --output-dir "/results/$TAG" \
       --save-predictions
