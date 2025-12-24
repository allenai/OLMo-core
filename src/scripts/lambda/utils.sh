#!/bin/bash

function log_debug {
    local script_name
    script_name=$(basename "$0")
    echo -e "\e[36m[$(date '+%Y-%m-%d %H:%M:%S')]\e[0m \e[90mDEBUG  \e[0m [$script_name, $(hostname), node=${SLURM_NODEID:-?}] $1"
}

function log_info {
    local script_name
    script_name=$(basename "$0")
    echo -e "\e[36m[$(date '+%Y-%m-%d %H:%M:%S')]\e[0m \e[34mINFO   \e[0m [$script_name, $(hostname), node=${SLURM_NODEID:-?}] $1"
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

function path_prepend {
  for ((i=$#; i>0; i--)); do
      ARG=${!i}
      if [[ -d "$ARG" ]] && [[ ":$PATH:" != *":$ARG:"* ]]; then
          export PATH="$ARG${PATH:+":$PATH"}"
      fi
  done
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

path_prepend /data/ai2/bin/
