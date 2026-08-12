#!/bin/bash
# Build the absence ladder on horton, where the Gutenberg arrow cache is node-local.
#
# The 11 GB `sedthh/gutenberg_english` cache lives on horton's own /data; reading it over
# /net/horton/... instead is the ~5 MB/s NFS path and the memory-map will not finish. Run with:
#   srun -p berkeleynlp -w horton --cpus-per-task=8 --mem=96G -t 90 \
#        bash debug/absence_port/build_absence_horton.sh <extra ctc-data build args>
set -euo pipefail
cd /accounts/projects/berkeleynlp/prasann/projects/newolmocore/OLMo-core
export HF_HOME=/data/prasann/hf
export HF_DATASETS_CACHE=/data/prasann/hf/datasets
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NLTK_DATA=/accounts/projects/berkeleynlp/prasann/nltk_data
export PYTHONPATH=ctc/src
exec /scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python -m ctc.data.cli "$@"
