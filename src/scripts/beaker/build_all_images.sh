#!/usr/bin/env bash
#
# Build and register the olmo-core Beaker image matrix, with minimal intervention.
#
# Checks the prerequisites (Docker daemon, Beaker CLI + auth, jq) up front, then builds each image in
# turn via its Makefile target. Builds continue past failures — a summary is printed at the end and
# the script exits non-zero if any image failed, so it can be kicked off and left unattended.
#
# The CUDA-13 images are built for sm_90/100/103, so each runs on H100 + B200 + B300 (no separate
# GPU-generation image). FA4 (flash_attn.cute) is CUDA-13 only — the `flash-attn-4` package ships a
# `cu13` extra and no `cu12` — so there is no CUDA-12.8 FA4 image.
#
# Usage:
#   src/scripts/beaker/build_all_images.sh                 # build the whole matrix
#   src/scripts/beaker/build_all_images.sh cu130-fa4-rma   # build only these (space-separated)
#
# Requires a Beaker token: either `beaker account login` beforehand, or export BEAKER_TOKEN.

set -uo pipefail

# The image matrix, in build order. The CUDA-13 base is built before its FA4/RMA variants so they
# reuse its (cached) heavy build stage.
ALL_SUFFIXES=(
    cu128            # tch2100cu128-<date>          H100/B200
    cu128-rma        # tch2100cu128-rma-<date>      H100/B200, symm-mem/RMA
    cu130            # tch2110cu130-<date>          H100/B200/B300
    cu130-fa4        # tch2110cu130-fa4-<date>      + flash_4
    cu130-rma        # tch2110cu130-rma-<date>      + symm-mem/RMA
    cu130-fa4-rma    # tch2110cu130-fa4-rma-<date>  H100/B200/B300, flash_4 + symm-mem/RMA
)

# Resolve repo root (this script lives at src/scripts/beaker/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

log()  { printf '\033[1;34m[build-all]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[build-all]\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[1;31m[build-all]\033[0m %s\n' "$*" >&2; }

# --- Prerequisite checks (fail fast, before any long build) ---------------------------------------
check_prereqs() {
    local ok=1

    for tool in docker beaker jq make; do
        if ! command -v "${tool}" >/dev/null 2>&1; then
            err "'${tool}' not found on PATH."
            ok=0
        fi
    done
    [ "${ok}" -eq 1 ] || return 1

    if ! docker info >/dev/null 2>&1; then
        err "Docker daemon is not running (or not reachable). Start Docker and retry."
        ok=0
    fi

    # Beaker auth: `whoami` fails if there is no valid token / login.
    if ! beaker account whoami >/dev/null 2>&1; then
        err "Not authenticated with Beaker."
        err "  Run 'beaker account login', or export BEAKER_TOKEN=<your token> and retry."
        ok=0
    fi

    [ "${ok}" -eq 1 ] || return 1

    log "Beaker user: $(beaker account whoami --format=json | jq -r '.[0].name')"
    log "Prerequisites OK."
}

# Resolve the image tag a target produces (for the summary), without building.
resolve_tag() {
    make -n "beaker-image-$1" 2>/dev/null | grep -oE 'olmo-core:tch[0-9a-z.-]+' | head -1
}

# --- Select targets -------------------------------------------------------------------------------
declare -a SUFFIXES
if [ "$#" -gt 0 ]; then
    SUFFIXES=("$@")
    for s in "${SUFFIXES[@]}"; do
        if ! printf '%s\n' "${ALL_SUFFIXES[@]}" | grep -qx "${s}"; then
            err "Unknown image '${s}'. Valid: ${ALL_SUFFIXES[*]}"
            exit 2
        fi
    done
else
    SUFFIXES=("${ALL_SUFFIXES[@]}")
fi

# --- Run ------------------------------------------------------------------------------------------
check_prereqs || exit 1

log "Building ${#SUFFIXES[@]} image(s): ${SUFFIXES[*]}"
declare -a RESULTS

for suffix in "${SUFFIXES[@]}"; do
    target="beaker-image-${suffix}"
    tag="$(resolve_tag "${suffix}")"
    log "==== Building ${target}  (${tag:-unknown tag}) ===="
    start=$(date +%s)
    if make "${target}"; then
        dur=$(( $(date +%s) - start ))
        RESULTS+=("OK|${target}|${tag}|${dur}")
        log "==== DONE ${target} in $((dur / 60))m$((dur % 60))s ===="
    else
        dur=$(( $(date +%s) - start ))
        RESULTS+=("FAIL|${target}|${tag}|${dur}")
        err "==== FAILED ${target} after $((dur / 60))m$((dur % 60))s — continuing ===="
    fi
done

# --- Summary --------------------------------------------------------------------------------------
echo
log "================ Summary ================"
failures=0
for r in "${RESULTS[@]}"; do
    IFS='|' read -r status target tag dur <<< "${r}"
    human="$((dur / 60))m$((dur % 60))s"
    if [ "${status}" = "OK" ]; then
        printf '  \033[1;32m✓\033[0m %-28s %-32s %s\n' "${target}" "${tag}" "${human}"
    else
        printf '  \033[1;31m✗\033[0m %-28s %-32s %s (FAILED)\n' "${target}" "${tag}" "${human}"
        failures=$((failures + 1))
    fi
done

if [ "${failures}" -gt 0 ]; then
    err "${failures} of ${#RESULTS[@]} build(s) failed."
    exit 1
fi
log "All ${#RESULTS[@]} image(s) built and registered."
