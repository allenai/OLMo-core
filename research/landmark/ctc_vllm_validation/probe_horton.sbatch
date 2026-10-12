#!/bin/bash
#SBATCH --job-name=probe-horton
#SBATCH --partition=berkeleynlp
#SBATCH --qos=preemptive_high_sewonm
#SBATCH --nodelist=horton
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/data/prasann/joblogs/probe_horton_%j.log
set -uo pipefail
echo "=== $(date '+%F %T') ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
echo "--- ps for smoke job ---"
ps -eo pid,ppid,etimes,pcpu,pmem,stat,cmd | grep -E "python|smoke_4b" | grep -v grep
echo "--- py-spy dump if available ---"
PY_SMOKE_PID=$(ps -eo pid,cmd | grep "smoke_4b.sbatch" | grep -v grep | awk '{print $1}' | head -1)
echo "smoke sbatch shell pid guess: $PY_SMOKE_PID"
