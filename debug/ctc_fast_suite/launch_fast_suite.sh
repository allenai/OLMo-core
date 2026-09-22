#!/bin/bash
# Launch the full CTC suite as 8 single-GPU Beaker jobs, sized to finish in about an hour.
#
#   bash debug/ctc_fast_suite/launch_fast_suite.sh <plan.json> <model> <tag>
#
# <plan.json> is size_the_suite.py's output: it carries `shard_cells`, one CELLS string per job,
# already LPT-balanced against MEASURED per-cell cost. Balance is the whole point of planning this
# rather than sharding by task -- 8 jobs finish when the SLOWEST finishes, and a per-task split puts
# every 256k rung of the long tasks in one job while another runs nothing but 2k rungs.
#
# <model> is a vLLM-loadable path. For an olmo-exported checkpoint that means a VL serving copy
# (see the qwen35-4b-vllm-load-recipe record), NOT the raw export; a stock HF Qwen3.5 needs no such
# thing. ⚠ The 256k rung sits ON Qwen3.5's 262,144 position ceiling and the realized p90 crosses it,
# so any shard carrying r256k must point at a YaRN factor-2 copy -- build it with
# debug/ctx_ceiling_4b/make_yarn_copy.py and launch those shards separately.
#
# Why this and not `olmo-eval run`: olmo-eval pins vllm==0.19.1, which predates Qwen3.5/GDN
# entirely. Grading is unaffected -- the runner scores through the SAME vendored ctc specs
# olmo-eval's CTCScorer calls, in the same three steps (stop, parse, score).
set -uo pipefail
PLAN="${1:?usage: launch_fast_suite.sh <plan.json> <model> <tag>}"
MODEL="${2:?}"
TAG="${3:?}"
CLUSTER="${CLUSTER:-ai2/jupiter-cirrascale-2}"
MAXLEN="${MAXLEN:-146227}"     # 128k + the 10% margin the rung builder calibrates against
GANTRY="${GANTRY:-/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/gantry}"
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python

N=$($PY -c "import json;print(len(json.load(open('$PLAN'))['shard_cells']))")
echo "=== launching $N shards | model=$MODEL | tag=$TAG | cluster=$CLUSTER ==="
$PY -c "
import json; p=json.load(open('$PLAN'))
print('  planned wall-clock: %.0f min (slowest shard); %.2f GPU-h total'
      % (p['est_wall_clock_min'], p['est_total_gpu_hours']))
print('  shard minutes:', p['shard_loads_min'])
"

for i in $(seq 0 $((N-1))); do
  CELLS=$($PY -c "import json;print(json.load(open('$PLAN'))['shard_cells'][$i])")
  echo "--- shard $i: $(echo "$CELLS" | tr ',' '\n' | wc -l) cells ---"
  $GANTRY run --name "ctcsuite-${TAG}-s${i}" -w ai2/flex2 -b ai2/oe-other \
    --cluster "$CLUSTER" --gpus 1 --priority urgent \
    --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
    --weka oe-training-default:/weka/oe-training-default \
    --branch prasann/landmark --allow-dirty --install 'true' --timeout 0 --yes \
    --env "TAG=${TAG}-s${i}" --env "MODEL=$MODEL" --env "MAXLEN=$MAXLEN" \
    --env "CELLS=$CELLS" --env SAVE_GENERATIONS=1 \
    -- bash debug/ctc_fast_suite/run_bench_beaker.sh 2>&1 | grep -E "Experiment:|beaker.org/ex" | head -2
done
echo "=== all $N shards submitted ==="
