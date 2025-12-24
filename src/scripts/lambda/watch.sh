#!/bin/bash

# Source helper functions.
. ./src/scripts/lambda/utils.sh

JOB_ID=$1
shift

if [ -z "$JOB_ID" ]; then
    log_error "Usage: $0 <job_id>"
    exit 1
fi
if [ -z "$SLACK_WEBHOOK_URL" ]; then
    log_error "SLACK_WEBHOOK_URL environment variable is not set."
    exit 1
fi

job_name=$(get_job_name "$JOB_ID")

if job_completed "$JOB_ID"; then
    log_info "Job $JOB_ID '$job_name' is no longer runnig."
    exit 0
fi

while ! job_completed "$JOB_ID"; do
    sleep 5
done

if job_succeeded "$JOB_ID"; then
    log_info "Job $JOB_ID '$job_name' completed successfully."
    exit 0
else
    log_error "Job $JOB_ID '$job_name' failed."
    exit 1
fi
