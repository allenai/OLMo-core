#!/usr/bin/env bash
# Launch a GPU Beaker job that runs Claude Code in "Remote Control" mode, so you can monitor/steer
# it from your phone (claude.ai/code or the Claude app) -- NO inbound SSH / VPN needed (the agent
# only makes outbound HTTPS to api.anthropic.com). The job clones OLMo-core (this branch) + installs
# it, so the agent has the repo, the GPU, and weka.
#
# ── ONE-TIME PREREQS (do these once, from a machine where you can log in to Claude) ──
#   1. Generate a long-lived OAuth token (Remote Control does NOT work with an API key; needs a
#      Pro/Max/Team/Enterprise login, Claude Code >= 2.1.51):
#          claude setup-token            # prints a 1-year CLAUDE_CODE_OAUTH_TOKEN
#   2. Store it as a Beaker secret in the workspace:
#          beaker secret write -w ai2/flex2 CLAUDE_CODE_OAUTH_TOKEN '<the-token>'
#
# ── RUN ──
#   scripts/launch_remote_control_agent.sh
#   GPUS=1 TASK_BRIEF=landmark-sparse-decode-task.md scripts/launch_remote_control_agent.sh
#
# ── CONNECT FROM PHONE ──
#   Open claude.ai/code (or the Claude app -> Code) and find the session by NAME (${NAME}); or grab
#   the session URL printed in the job logs: `beaker experiment logs <id> | grep claude.ai/code`.
set -euo pipefail

GPUS="${GPUS:-1}"
CLUSTER="${CLUSTER:-ai2/jupiter-cirrascale-2}"
PRIORITY="${PRIORITY:-high}"
BUDGET="${BUDGET:-ai2/oe-other}"
WORKSPACE="${WORKSPACE:-ai2/flex2}"
WEKA="${WEKA:-oe-training-default}"
IMAGE="${IMAGE:-beaker://tylerr/olmo-core-tch291cu128-2025-11-25}"
NAME="${NAME:-landmark-gpu-agent}"
SECRET="${SECRET:-CLAUDE_CODE_OAUTH_TOKEN}"   # name of the Beaker secret holding the OAuth token
BRANCH="${BRANCH:-prasann/landmark}"
TASK_BRIEF="${TASK_BRIEF:-landmark-sparse-decode-task.md}"  # optional: seed the agent with this brief

CLUSTER_ARGS=()
IFS=',' read -ra _C <<< "${CLUSTER}"; for c in "${_C[@]}"; do CLUSTER_ARGS+=(--cluster "$c"); done

# The job: install Claude Code (native, no node), put it on PATH, and start Remote Control inside the
# gantry-cloned repo. gantry's --install already did `pip install -e .`, so the agent can run GPU
# tests. `claude remote-control` blocks (server mode) -> the job stays up until you stop it.
read -r -d '' CMD <<EOF || true
set -e
curl -fsSL https://claude.ai/install.sh | bash
export PATH="\$HOME/.local/bin:\$PATH"
claude --version
echo "Starting Claude Code Remote Control as '${NAME}'. Find it at https://claude.ai/code (by name) or via the URL below."
claude remote-control --name "${NAME}"
EOF

gantry run \
  --name "${NAME}" \
  --description "Claude Code Remote Control agent (GPU) on ${BRANCH}" \
  --workspace "${WORKSPACE}" \
  --budget "${BUDGET}" \
  "${CLUSTER_ARGS[@]}" \
  --weka "${WEKA}:/weka/${WEKA}" \
  --gpus "${GPUS}" \
  --priority "${PRIORITY}" \
  --allow-dirty \
  --shared-memory 32GiB \
  --timeout 0 \
  --env "ANTHROPIC_DEFAULT_TASK_BRIEF=${TASK_BRIEF}" \
  --secret-env "CLAUDE_CODE_OAUTH_TOKEN=${SECRET}" \
  --install "pip install -e ." \
  --yes \
  -- bash -c "${CMD}"

echo
echo "Launched. Connect from your phone: claude.ai/code (or the Claude app) -> find session '${NAME}'."
echo "Or get the session URL: beaker experiment logs <exp-id> 2>/dev/null | grep -m1 'claude.ai/code'"
echo "First message to send it (from the phone): \"Read ${TASK_BRIEF} in this repo and implement + GPU-validate it.\""
