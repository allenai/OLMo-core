#!/bin/bash
# Login-node watcher: when a CPT run's Beaker job exits, submit its drop-robustness dev-loss eval
# (eval_drop_devloss_beaker.sh) so no GPU sits idle waiting for a checkpoint. Run detached:
#   setsid nohup bash research/ffn_compute/ffndrop/chain_evals.sh RUN=EXPID ... \
#     >> $FFNDROP_LOG_DIR/chain_evals.log 2>&1 &
REPO_ROOT=${REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}
cd "$REPO_ROOT"
FFNDROP_LOG_DIR=${FFNDROP_LOG_DIR:-/data/prasann/logs}
FFNDROP_CONDA_BIN=${FFNDROP_CONDA_BIN:-/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin}
export PATH=$FFNDROP_CONDA_BIN:$HOME/.local/bin:$PATH
declare -A PENDING
for kv in "$@"; do PENDING[${kv%%=*}]=${kv#*=}; done
for i in $(seq 1 144); do  # 24h at 10 min
  for run in "${!PENDING[@]}"; do
    id=${PENDING[$run]}
    st=$(timeout 120 beaker experiment get "$id" --format json 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);e=d[0] if isinstance(d,list) else d;s=e['jobs'][-1]['status'];print(('exited:%s' % s.get('exitCode')) if s.get('exited') else ('running' if s.get('started') else 'pending'))" 2>/dev/null)
    echo "$(date '+%m-%d %H:%M') $run $id ${st:-?}"
    if [[ "$st" == exited:* && "$st" != "exited:0" ]]; then
      # a failed run has no final export; the eval would idle a GPU for 12h waiting for it
      echo "$(date '+%m-%d %H:%M') $run FAILED ($st) -- no eval submitted; check the job log"
      unset "PENDING[$run]"; continue
    fi
    if [[ "$st" == exited:* ]]; then
      out=$(RUN=$run bash research/ffn_compute/ffndrop/eval_drop_devloss_beaker.sh 2>&1 | grep -oE "ex/[A-Z0-9]{26}" | head -1)
      echo "$(date '+%m-%d %H:%M') $run finished ($st) -> eval ${out:-SUBMIT FAILED}"
      printf "EVAL\t%s\t%s\t%s\t%s\n" "$run" "$st" "${out#ex/}" "$(date '+%Y-%m-%d %H:%M')" >> research/ffn_compute/ffndrop/EVAL_LEDGER.tsv
      unset "PENDING[$run]"
    fi
  done
  [ ${#PENDING[@]} -eq 0 ] && { echo "all evals submitted"; exit 0; }
  sleep 600
done
