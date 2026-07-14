#!/bin/bash
# CPU pytest sweep for the dilation_cycle rotation + positional-deprecation change, on horton's
# node-local /data conda env (fast imports vs NFS). No GPU needed.
# NFS-safe: job log on node-local /data; the pytest summary is mirrored to the repo debug/ on exit.
#   sbatch debug/run_dilation_tests.sbatch
#SBATCH --job-name=dilation-tests
#SBATCH --partition=berkeleynlp
#SBATCH --qos=preemptive_high_sewonm
#SBATCH --nodelist=horton
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/data/prasann/joblogs/dilation_tests_%j.log
set -uo pipefail

REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
[ -x "$PY" ] || PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python  # NFS fallback
OUT="$REPO/debug/dilation_tests_result.txt"
mkdir -p /data/prasann/joblogs

cd "$REPO"
echo "host=$(hostname)  python=$PY" | tee "$OUT"
PYTHONPATH="$REPO/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PY" -m pytest -q \
  src/test/nn/attention/ \
  src/test/nn/transformer/config_test.py \
  src/test/nn/transformer/block_test.py \
  2>&1 | tee -a "$OUT"
echo "EXIT_CODE=${PIPESTATUS[0]}" | tee -a "$OUT"
