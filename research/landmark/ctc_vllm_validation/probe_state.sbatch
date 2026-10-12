#!/bin/bash
#SBATCH --job-name=probe-state
#SBATCH --partition=berkeleynlp
#SBATCH --qos=preemptive_high_sewonm
#SBATCH --nodelist=horton
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:06:00
#SBATCH --output=/data/prasann/joblogs/probe_state_%j.log
set -uo pipefail
echo "=== $(date '+%F %T') HOST=$(hostname) ==="
echo "--- /data/prasann top-level ---"
ls -la /data/prasann/ 2>&1
echo "--- conda envs on /data ---"
ls -la /data/prasann/conda/envs/ 2>&1
echo "--- ctc_vllm_venv? ---"
ls -la /data/prasann/ctc_vllm_venv/bin/python 2>&1
if [ -x /data/prasann/ctc_vllm_venv/bin/python ]; then
  echo "--- ctc_vllm_venv vllm/torch version (should be near-instant if venv healthy) ---"
  time /data/prasann/ctc_vllm_venv/bin/python -c "import vllm, torch; print('vllm', vllm.__version__); print('torch', torch.__version__)" 2>&1
fi
echo "--- disk free /data ---"
df -h /data 2>&1
echo "--- 4b hf export present? ---"
ls -la /data/prasann/ctc_suite/hf_exports_4b/ctc-s5-contra-full-4b/ 2>&1
echo "--- 0.8b hf exports present? ---"
find /data/prasann/ctc_suite -maxdepth 2 -iname "*0.8b*" -o -iname "*hf_export*" 2>&1 | head -20
echo "=== DONE $(date '+%F %T') ==="
