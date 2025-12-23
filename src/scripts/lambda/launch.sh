#!/bin/bash

git pull

# Define the script you want to submit
JOB_SCRIPT="${1:-src/scripts/lambda/slurm-test-job.sbatch}"

echo "Submitting job script: $JOB_SCRIPT"

# Submit the job and capture the output (the Job ID).
# The --parsable option ensures only the Job ID is returned.
JOB_ID=$(sbatch --parsable "$JOB_SCRIPT")

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
echo "Waiting on log file at /data/ai2/logs/$JOB_ID..."
while [ ! -f "/data/ai2/logs/$JOB_ID.log" ]; do
    sleep 1
done

# Stream the log file from the first task.
tail -n +1 -f "/data/ai2/logs/$JOB_ID/task_0.log"
