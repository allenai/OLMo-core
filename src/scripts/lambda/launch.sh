#!/bin/bash
#
# Launch a Slurm job and stream its logs in real-time.

# Source helper functions.
. ./src/scripts/lambda/utils.sh

JOB_SCRIPT=$1
shift

RUN_NAME=$1
shift

NODES=$1
shift

for var in "JOB_SCRIPT" "RUN_NAME" "NODES"; do
    if [ -z "${!var}" ]; then
        log_error "Usage: $0 <job_script.sbatch> <run_name> <nodes> [args...]"
        exit 1
    fi
done

if [ -z "$SLACK_WEBHOOK_URL" ]; then
    log_warning "SLACK_WEBHOOK_URL environment variable is not set."
fi

JOB_ID=$(launch_job "$JOB_SCRIPT" "$RUN_NAME" "$NODES" "$@")
if ! [ $? -eq 0 ]; then
    exit 1
fi

LOG_FILE="${LOGS_DIR}/${RUN_NAME}/${JOB_ID}.log"

# On keyboard interrupt, print some useful information before exiting.
on_interrupt() {
    log_warning "Caught interrupt signal. Checking job status..."
    local exit_code=0
    if ! job_completed "$JOB_ID"; then
        log_info "Job $JOB_ID may still be running."
        echo "You can check the job status with:"
        echo "  squeue -j $JOB_ID"
        echo ""
        echo "Or cancel the job with:"
        echo "  scancel $JOB_ID"
        exit_code=1
    elif job_succeeded "$JOB_ID"; then
        log_info "Job $JOB_ID completed successfully."
    else
        log_warning "Job $JOB_ID may have failed."
        exit_code=1
    fi
    echo ""
    echo "The main log file is located at '$LOG_FILE'. Use this command to grep through it:"
    echo "  cat $LOG_FILE | less -R"
    echo ""
    exit $exit_code
}

trap on_interrupt SIGINT

# Loop until the job status is no longer PENDING (PD).
log_info "Waiting for job to start..."
while job_pending "$JOB_ID"; do
    sleep 2
done

# Loop until the log file is created.
log_info "Waiting on log file at $LOG_FILE..."
while [ ! -f "$LOG_FILE" ]; do
    if job_completed "$JOB_ID" && ! job_succeeded "$JOB_ID"; then
        log_error "Job $JOB_ID ended before log file was created."
        exit 1
    fi
    sleep 2
done

# Stream the log file from the first task.
log_info "Streaming logs (stay tuned)..."
tail -n +1 -f "$LOG_FILE"
