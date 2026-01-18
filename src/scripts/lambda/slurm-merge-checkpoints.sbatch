#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=merge-checkpoints
#SBATCH --output=/data/ai2/logs/merge-checkpoints/%j.log

# Create logs directory if it doesn't exist
mkdir -p /data/ai2/logs/merge-checkpoints

cd /data/ai2/$USERNAME/OLMo-core

export GOOGLE_APPLICATION_CREDENTIALS=/data/ai2/google/credentials.json

# Activate Python virtual env
source /data/ai2/uv/OLMo-core-$USERNAME/bin/activate

python src/scripts/merge_core_checkpoints.py \
    --model /data/ai2/checkpoints/willm/linear-rnns/OLMo3.1-7B-6T-30h-midtrain-deux/step23842/ \
    --model /data/ai2/checkpoints/willm/linear-rnns/OLMo3.1-7B-6T-30h-midtrain-deux-ing2/step23842/ \
    --output /data/ai2/checkpoints/willm/linear-rnns/OLMo3.1-7B-6T-30h-midtrain-deux-soup/step23842/
