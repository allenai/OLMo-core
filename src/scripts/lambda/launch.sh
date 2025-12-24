#!/bin/bash

# Define the script you want to submit
JOB_SCRIPT=$1
shift

RUN_NAME=$1
shift

NODES=$1
shift

for var in "JOB_SCRIPT" "RUN_NAME" "NODES"; do
    if [ -z "${!var}" ]; then
        echo "Usage: $0 <job_script.sbatch> <run_name> <nodes>"
        exit 1
    fi
done

# After we start tailing, always print the job ID + log file path again on exit
TAIL_STARTED=0
JOB_ID=""
LOG_FILE=""

on_exit() {
    local exit_code=$?
    if [ "$TAIL_STARTED" -eq 1 ]; then
        echo
        echo "Job ID: $JOB_ID"
        echo "Log file: $LOG_FILE"
    fi
    # Preserve original exit code.
    :
}

on_signal() {
    local sig="$1"
    case "$sig" in
        INT) exit 130 ;;  # 128 + SIGINT(2)
        TERM) exit 143 ;; # 128 + SIGTERM(15)
        *) exit 128 ;;
    esac
}

trap on_exit EXIT
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM

# Check for requirement env vars.
if [ -z "$WANDB_API_KEY" ]; then
    echo "Error: WANDB_API_KEY environment variable is not set."
    exit 1
fi
if [ -z "$USERNAME" ]; then
    echo "Error: USERNAME environment variable is not set (e.g. 'petew', 'tylerr')."
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
JOB_ID=$(sbatch --export=WANDB_API_KEY,USERNAME --output="/data/ai2/logs/${RUN_NAME}/%j.log" --nodes="$NODES" --gpus-per-node=8 --ntasks-per-node=1 --parsable "$JOB_SCRIPT")

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
LOG_FILE="/data/ai2/logs/$RUN_NAME/$JOB_ID.log"
echo "Waiting on log file at $LOG_FILE..."
while [ ! -f "$LOG_FILE" ]; do
    sleep 1
done

# Stream the log file from the first task.
TAIL_STARTED=1
tail -n +1 -f "$LOG_FILE"
