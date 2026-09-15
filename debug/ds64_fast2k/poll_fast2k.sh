#!/bin/bash
# Blocking poller for the fast2k loop: refresh state.json, launch the 2k-rung eval for every run
# that finishes, launch the WARM arms once their source run's export exists, and stop when every
# run has a finished eval.
#   bash debug/ds64_fast2k/poll_fast2k.sh [max_cycles] [sleep_s] [warm_arms]
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
cd /accounts/projects/berkeleynlp/prasann/projects/OLMo-core
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
N=${1:-300}; S=${2:-120}; WARM=${3:-}
for i in $(seq 1 "$N"); do
  echo "=== cycle $i $(date +%T) ==="
  $PY debug/ds64_fast2k/launch_fast2k.py status
  # Warm arms: launch once their --base-checkpoint source run is DONE (its model_and_optim export
  # is written after fit()), and only once. The logic lives in warm_ready.py on purpose -- inlining
  # it as a heredoc inside $( ) inside this loop silently never fired.
  if [ -n "$WARM" ]; then
    GO=$($PY debug/ds64_fast2k/warm_ready.py "$WARM")
    if [ -n "$GO" ]; then
      echo "--- launching warm arms: $GO"
      $PY debug/ds64_fast2k/launch_fast2k.py --arms "$GO" --budgets 4M launch
    fi
  fi
  DONE=$($PY debug/ds64_fast2k/pending.py needs_eval)
  if [ -n "$DONE" ]; then
    echo "--- launching evals: $DONE"
    $PY debug/ds64_fast2k/launch_fast2k.py eval --runs "$DONE"
  fi
  ALL=$($PY debug/ds64_fast2k/pending.py summary)
  echo "$ALL"
  [ "$ALL" = "ALLDONE" ] && break
  sleep "$S"
done
echo "=== poller exit $(date +%T) ==="
