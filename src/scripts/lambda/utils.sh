#!/bin/bash

function log_debug {
    echo -e "\e[36m[$(date '+%Y-%m-%d %H:%M:%S')]\e[0m \e[90m\e[1mDEBUG\e[0m       [$(hostname), node=${SLURM_NODEID:-?}] $1"
}

function log_info {
    echo -e "\e[36m[$(date '+%Y-%m-%d %H:%M:%S')]\e[0m \e[34m\e[1mINFO\e[0m        [$(hostname), node=${SLURM_NODEID:-?}] $1"
}

function log_warning {
    echo -e >&2 "\e[36m[$(date '+%Y-%m-%d %H:%M:%S')]\e[0m \e[33m\e[1mWARNING\e[0m [$(hostname), node=${SLURM_NODEID:-?}] $1"
}

function log_error {
    echo -e >&2 "\e[36m[$(date '+%Y-%m-%d %H:%M:%S')]\e[0m \e[31m\e[1mERROR\e[0m   [$(hostname), node=${SLURM_NODEID:-?}] $1"
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

path_prepend /data/ai2/bin/
