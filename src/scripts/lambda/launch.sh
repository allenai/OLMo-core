#!/bin/bash

git pull

# Define the script you want to submit
JOB_SCRIPT="${1:-src/scripts/lambda/slurm-test-job.sbatch}"

# Check that BEAKER_TOKEN env var is set.
if [ -z "$BEAKER_TOKEN" ]; then
    echo "Error: BEAKER_TOKEN environment variable is not set."
    exit 1
fi

# Find an open port to use for distributed training.
echo "Submitting job script: $JOB_SCRIPT"

# Submit the job and capture the output (the Job ID).
# The --parsable option ensures only the Job ID is returned.
JOB_ID=$(sbatch --export=BEAKER_TOKEN --parsable "$JOB_SCRIPT")

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
LOG_FILE="/data/ai2/logs/$JOB_ID/task_0.log"
echo "Waiting on log file at $LOG_FILE..."
while [ ! -f "$LOG_FILE" ]; do
    sleep 1
done

# Stream the log file from the first task.
tail -n +1 -f "$LOG_FILE"
