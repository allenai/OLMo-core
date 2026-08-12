#!/usr/bin/env bash
# End-to-end `reorder` build on horton, the node holding the Gutenberg cache.
#
# `sedthh/gutenberg_english` is ~11 GB of arrow shards at /data/prasann/hf/datasets on horton's
# NODE-LOCAL disk. Reading it over /net/horton is the ~5 MB/s NFS layer and an 11 GB memory-map
# will not finish, so this must run on horton itself.
#
# Run:  srun -p jsteinhardt -q preemptive_high -w horton -c 8 --mem 64G \
#           bash debug/ctc_three_task_port/build_reorder_horton.sh
set -euo pipefail

REPO=/accounts/projects/berkeleynlp/prasann/projects/newolmocore/OLMo-core
PY=/scratch/users/prasann/conda/envs/corpus-reasoning/bin/python

export HF_HOME=/data/prasann/hf
export HF_DATASETS_CACHE=/data/prasann/hf/datasets
export HF_HUB_OFFLINE=1
export NLTK_DATA=/data/prasann/nltk_data
export TMPDIR=/data/prasann/tmp
export PYTHONPATH="$REPO/ctc/src"
mkdir -p "$TMPDIR"

cd "$REPO"
exec "$PY" -m ctc.data.cli build \
  --task reorder --split "${SPLIT:-train}" --train "${TRAIN:-64}" --rungs "${RUNGS:-2k}" \
  --out "${OUT:-$REPO/debug/ctc_three_task_port/build}" \
  -C max_books="${MAX_BOOKS:-2000}"
