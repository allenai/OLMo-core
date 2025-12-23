#!/bin/bash

# Define the script you want to submit
JOB_SCRIPT="${1:-src/scripts/lambda/slurm-test-job.sbatch}"

echo "Submitting job script: $JOB_SCRIPT"

# Submit the job and capture the output (the Job ID)
# The --parsable option ensures only the Job ID is returned.
JOB_ID=$(sbatch --parsable "$JOB_SCRIPT")

# Check if the submission was successful (sbatch returns a non-zero exit code on failure)
if [ $? -eq 0 ]; then
    echo "Submitted job with ID: $JOB_ID"
else
    echo "Job submission failed."
fi

