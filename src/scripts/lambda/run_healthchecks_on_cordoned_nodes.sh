#!/usr/bin/env bash

set -euo pipefail

# Source helper functions.
. ./src/scripts/lambda/utils.sh

# Run healthchecks.sh on all nodes listed in /data/ai2/cordoned-nodes.txt
#
# This script reads the cordoned nodes file and runs healthchecks.sh on each node using srun.
#
# Usage:
#   ./src/scripts/lambda/run_healthchecks_on_cordoned_nodes.sh

NODES_FILE="/data/ai2/cordoned-nodes.txt"
HEALTHCHECKS_SCRIPT="./src/scripts/lambda/healthchecks.sh"

if [[ ! -f "$NODES_FILE" ]]; then
  die "Nodes file not found: $NODES_FILE"
fi

if [[ ! -f "$HEALTHCHECKS_SCRIPT" ]]; then
  die "Healthchecks script not found: $HEALTHCHECKS_SCRIPT"
fi

log_info "Reading nodes from: $NODES_FILE"

# Read nodes, filtering out comments and empty lines
nodes=()
while IFS= read -r line; do
  # Skip comments and empty lines
  if [[ -z "${line// /}" ]] || [[ "$line" =~ ^[[:space:]]*# ]]; then
    continue
  fi
  # Trim whitespace
  node=$(echo "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
  if [[ -n "$node" ]]; then
    nodes+=("$node")
  fi
done < "$NODES_FILE"

if [[ ${#nodes[@]} -eq 0 ]]; then
  log_warning "No nodes found in $NODES_FILE (after filtering comments)"
  exit 0
fi

log_info "Found ${#nodes[@]} node(s) to check: ${nodes[*]}"

# Track results
failed_nodes=()
passed_nodes=()

# Run healthchecks on each node
for node in "${nodes[@]}"; do
  log_info "Running healthchecks on node: $node"

  # Use srun to run healthchecks.sh on the specific node
  # --nodelist specifies which node to run on
  # --ntasks=1 --nodes=1 ensures we run on exactly one node
  # --kill-on-bad-exit=1 ensures srun exits non-zero if the script fails
  if srun \
    --nodelist="$node" \
    --ntasks=1 \
    --nodes=1 \
    --kill-on-bad-exit=1 \
    bash "$HEALTHCHECKS_SCRIPT"; then
    log_info "✓ Healthcheck PASSED on node: $node"
    passed_nodes+=("$node")
  else
    log_error "✗ Healthcheck FAILED on node: $node"
    failed_nodes+=("$node")
  fi
done

# Summary
log_info ""
log_info "=========================================="
log_info "Healthcheck Summary"
log_info "=========================================="
log_info "Total nodes checked: ${#nodes[@]}"
log_info "Passed: ${#passed_nodes[@]}"
log_info "Failed: ${#failed_nodes[@]}"

if [[ ${#passed_nodes[@]} -gt 0 ]]; then
  log_info ""
  log_info "Passed nodes:"
  for node in "${passed_nodes[@]}"; do
    log_info "  ✓ $node"
  done
fi

if [[ ${#failed_nodes[@]} -gt 0 ]]; then
  log_error ""
  log_error "Failed nodes:"
  for node in "${failed_nodes[@]}"; do
    log_error "  ✗ $node"
  done
  exit 1
fi

log_info ""
log_info "All healthchecks passed!"
exit 0
