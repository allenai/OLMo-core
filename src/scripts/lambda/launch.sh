#!/bin/bash

git pull

# Define the script you want to submit
JOB_SCRIPT="${1:-src/scripts/lambda/slurm-test-job.sbatch}"

# Check for requirement env vars.
if [ -z "$WANDB_API_KEY" ]; then
    echo "Error: WANDB_API_KEY environment variable is not set."
    exit 1
fi
# for env_var in "BEAKER_TOKEN" "WANDB_API_KEY"; do
#     if [[ -z "${!env_var+x}" ]]; then
#         log_error "Required environment variable '$env_var' is empty"
#         exit 1
#     fi
# done

# Find an open port to use for distributed training.
echo "Submitting job script: $JOB_SCRIPT"

# Submit the job and capture the output (the Job ID).
# The --parsable option ensures only the Job ID is returned.
JOB_ID=$(sbatch --export=WANDB_API_KEY --output='/data/ai2/logs/%j/node_%n.log' --parsable "$JOB_SCRIPT")

# Check if the submission was successful (sbatch returns a non-zero exit code on failure).
if [ $? -eq 0 ]; then
    echo "Submitted job with ID: $JOB_ID"
else
    echo "Job submission failed."
    exit 1
fi

# Loop until the job status is no longer PENDING (PD).
echo "Waiting for job to start..."
while squeue -j "$JOB_ID" | grep " PD " > /dev/null; do
    sleep 2
done

# Loop until the log file is created.
LOG_FILE="/data/ai2/logs/$JOB_ID/node_0.log"
echo "Waiting on log file at $LOG_FILE..."
while [ ! -f "$LOG_FILE" ]; do
    sleep 1
done

# Stream the log file from the first task.
tail -n +1 -f "$LOG_FILE"
