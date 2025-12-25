#!/bin/bash
#
# Launch a Slurm job with retries until it succeeds, and send Slack notifications on failure.

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
        log_error "Usage: $0 <job_script.sbatch> <run_name> <nodes>"
        exit 1
    fi
done

if [ -z "$SLACK_WEBHOOK_URL" ]; then
    log_warning "SLACK_WEBHOOK_URL environment variable is not set."
fi

while true; do
    JOB_ID=$(launch_job "$JOB_SCRIPT" "$RUN_NAME" "$NODES")
    if ! [ $? -eq 0 ]; then
        log_warning "Retrying in 5 seconds..."
        sleep 5
        continue
    fi

    LOG_FILE="${LOGS_DIR}/${RUN_NAME}/${JOB_ID}.log"

    # Loop until the job status is no longer PENDING (PD).
    log_info "Waiting for job $JOB_ID '$RUN_NAME' to start..."
    while job_pending "$JOB_ID"; do
        sleep 2
    done
    
    # Loop until the log file is created.
    log_info "Waiting on log file at '$LOG_FILE'..."
    while [ ! -f "$LOG_FILE" ]; do
        if job_completed "$JOB_ID" && ! job_succeeded "$JOB_ID"; then
            log_error "Job $JOB_ID '$RUN_NAME' ended before log file was created."
            return 1
        fi
        sleep 2
    done

    log_info "Log file detected at '$LOG_FILE'."
    echo ""
    echo "Use this command to grep through the log file:"
    echo "  cat $LOG_FILE | less -R"
    echo ""
    echo "Or use this command to stream it live:"
    echo "  tail -n +1 -f $LOG_FILE"
    echo ""

    # Wait for job to complete.
    log_info "Waiting on job $JOB_ID '$RUN_NAME'..."
    while ! job_completed "$JOB_ID"; do
        sleep 5
    done

    if job_succeeded "$JOB_ID"; then
        log_info "Job $JOB_ID '$RUN_NAME' completed successfully."
        exit 0
    else
        log_error "Job $JOB_ID '$RUN_NAME' failed."
        if [ -n "$SLACK_WEBHOOK_URL" ]; then
            post_to_slack "Slurm job $JOB_ID '$RUN_NAME' failed!"
        fi
        log_warning "Retrying in 5 seconds..."
        sleep 5
    fi
done
