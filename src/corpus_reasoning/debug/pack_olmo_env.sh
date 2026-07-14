#!/bin/bash
#SBATCH --job-name=pack-olmo-env
#SBATCH --partition=berkeleynlp
#SBATCH --qos=preemptive_high_sewonm
#SBATCH --nodelist=horton
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/users/prasann/corpus-reasoning/outputs/logs/batch_logs/pack_olmo_env_%j.log
#SBATCH --error=/scratch/users/prasann/corpus-reasoning/outputs/logs/batch_logs/pack_olmo_env_%j.log
#SBATCH --open-mode=append
set -uo pipefail
set -x

echo "[$(date -Iseconds)] node=$SLURMD_NODENAME packing olmo env"
free -g | head -2

eval "$(conda shell.bash hook)"
conda activate base
python -m pip install --user -q conda-pack 2>&1 | tail -2 || true

ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
OUT=/scratch/users/prasann/olmo-env.tar.gz
rm -f "$OUT"

python -c "from conda_pack.cli import main; main()" \
    -p "$ENV" -o "$OUT" --ignore-editable-packages --n-threads 8 --force
rc=$?
echo "[$(date -Iseconds)] conda-pack rc=$rc"
ls -lh "$OUT" || true
