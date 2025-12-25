#!/bin/bash

export CORDONED_NODES_FILE=/data/ai2/cordoned-nodes.txt
export LOGS_DIR=/data/ai2/logs

function path_prepend {
  for ((i=$#; i>0; i--)); do
      ARG=${!i}
      if [[ -d "$ARG" ]] && [[ ":$PATH:" != *":$ARG:"* ]]; then
          export PATH="$ARG${PATH:+":$PATH"}"
      fi
  done
}

path_prepend /data/ai2/bin/

function log_debug {
    local script_name
    script_name=$(basename "$0")
    echo -e >&2 "\e[36m[$(date '+%Y-%m-%d %H:%M:%S')]\e[0m \e[90mDEBUG  \e[0m [$script_name, $(hostname), node=${SLURM_NODEID:-?}] $1"
}

function log_info {
    local script_name
    script_name=$(basename "$0")
    echo -e >&2 "\e[36m[$(date '+%Y-%m-%d %H:%M:%S')]\e[0m \e[34mINFO   \e[0m [$script_name, $(hostname), node=${SLURM_NODEID:-?}] $1"
}

function log_warning {
    local script_name
    script_name=$(basename "$0")
    echo -e >&2 "\e[36m[$(date '+%Y-%m-%d %H:%M:%S')]\e[0m \e[33mWARNING\e[0m [$script_name, $(hostname), node=${SLURM_NODEID:-?}] $1"
}

function log_error {
    local script_name
    script_name=$(basename "$0")
    echo -e >&2 "\e[36m[$(date '+%Y-%m-%d %H:%M:%S')]\e[0m \e[31mERROR  \e[0m [$script_name, $(hostname), node=${SLURM_NODEID:-?}] $1"
}

function die {
  log_error "$1"
  exit 1
}

function have_cmd {
  command -v "$1" >/dev/null 2>&1
}

function node_0_only {
    if [ -z "$SLURM_NODEID" ] || [ "$SLURM_NODEID" -eq 0 ]; then
        "$@"
    fi
}

# usage: with_retries MAX_RETRIES(INT) COMMAND(TEXT) [ARGS(ANY)...]
function with_retries {
    local max_retries="$1"
    shift 1
    local attempts=0

    while true; do
        "$@" && return 0

        if ((++attempts >= max_retries)); then
            log_error "Retries exceeded."
            return 1
        else
            local pause_seconds=$((2**(attempts-1)))
            if ((pause_seconds > 30)); then
                pause_seconds=30
            fi
            log_warning "Attempt ${attempts}/${max_retries} failed. Retrying in ${pause_seconds} second(s)..."
            sleep "$pause_seconds"
        fi
    done
}

function get_job_name {
    sacct -j "$1" --format=JobName%60 --noheader | head -n 1 | xargs
}

function job_pending {
    if squeue -j "$1" | grep " PD " > /dev/null; then
        return 0
    else
        return 1
    fi
}

function job_completed {
    if squeue -j "$1" --noheader | grep "$1" > /dev/null; then
        return 1
    else
        return 0
    fi
}

function job_succeeded {
    if ! job_completed "$1"; then
        return 1
    fi

    local exit_status
    exit_status=$(sacct -j "$1" --format=exitcode --parsable2 --noheader | tail -n 1)
    # exit_status has the form '<exitcode>:<signal>'. '0:0' indicates success.
    if [[ "$exit_status" == "0:0" ]]; then
        return 0
    else
        return 1
    fi
}

function post_to_slack {
    if [ -z "$SLACK_WEBHOOK_URL" ]; then
        log_error "SLACK_WEBHOOK_URL is not set."
        return 1
    fi

    local encoded_text
    encoded_text=$(printf '%s' "$1" | jq -sR .)
    curl -X POST -H 'Content-type: application/json' --data "{\"text\":$encoded_text}" "$SLACK_WEBHOOK_URL" || return 1
    return 0
}

function launch_job {
    local JOB_SCRIPT="$1"
    local RUN_NAME="$2"
    local NODES="$3"
    for var in "JOB_SCRIPT" "RUN_NAME" "NODES"; do
        if [ -z "${!var}" ]; then
            log_error "Usage: launch_job <job_script.sbatch> <run_name> <nodes>"
            exit 1
        fi
    done

    if [ ! -f "$JOB_SCRIPT" ]; then
        log_error "Job script '$JOB_SCRIPT' does not exist."
        exit 1
    fi

    # Check for requirement env vars.
    if [ -z "$WANDB_API_KEY" ]; then
        log_error "WANDB_API_KEY environment variable is not set."
        exit 1
    fi
    if [ -z "$USERNAME" ]; then
        log_error "USERNAME environment variable is not set (e.g. 'petew', 'tylerr')."
        exit 1
    fi
    
    SBATCH_ARGS=(
        --export="WANDB_API_KEY,USERNAME,HOME"
        --job-name="$RUN_NAME"
        --output="${LOGS_DIR}/${RUN_NAME}/%j.log"
        --nodes="$NODES"
        --gpus-per-node=8
        --ntasks-per-node=1
        --parsable
    )
    
    # Check for cordoned nodes and exclude them.
    if [ -f "$CORDONED_NODES_FILE" ]; then
        cordoned_nodes=$(grep -v '^#' "$CORDONED_NODES_FILE" | tr '\n' ',' | sed 's/,$//')
        formatted_cordoned_nodes=$(echo "$cordoned_nodes" | tr ',' '\n' | sed 's/^/ • /')
        cordoned_nodes_count=$(echo "$formatted_cordoned_nodes" | wc -l)
        log_warning "$cordoned_nodes_count cordoned nodes detected:"
        echo >&2 "$formatted_cordoned_nodes"
        SBATCH_ARGS+=(--exclude="$cordoned_nodes")
    else
        log_warning "No cordoned nodes file found at '$CORDONED_NODES_FILE'."
    fi
    
    # Find an open port to use for distributed training.
    log_info "Submitting job script: $JOB_SCRIPT"
    
    # Submit the job and capture the output (the Job ID).
    # The --parsable option ensures only the Job ID is returned.
    JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" "$JOB_SCRIPT")
    
    # Check if the submission was successful (sbatch returns a non-zero exit code on failure).
    if [ $? -eq 0 ]; then
        log_info "Submitted slurm job $JOB_ID '$RUN_NAME'."
    else
        log_error "Job submission failed."
        return 1
    fi

    echo "$JOB_ID"
    return 0
}
