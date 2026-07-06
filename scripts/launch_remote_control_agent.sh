#!/usr/bin/env bash
# Launch a GPU Beaker job that runs Claude Code in "Remote Control" mode, so you can monitor/steer
# it from your phone (claude.ai/code or the Claude app) -- NO inbound SSH / VPN needed (the agent
# only makes outbound HTTPS to api.anthropic.com). The job clones OLMo-core (this branch) + installs
# it, so the agent has the repo, the GPU, and weka.
#
# ── ABOUT THE TOKEN ──
#   The OAuth token has to reach the GPU node (the agent runs there). With --secret-env Beaker
#   injects it as an env var and MASKS it -- the value never appears in the command, the (publicly
#   viewable) experiment spec, or the captured job I/O. The only way to read the value is
#   `beaker secret read` by a member of the workspace -- exactly like your existing flex2 secrets
#   (PRASANNS_WANDB_API_KEY, PRASANNS_BEAKER_TOKEN, ...). So we keep it in flex2 under a
#   PRASANNS_-prefixed name, consistent with those, and revocable (re-run `claude setup-token`).
#
#   Source-of-truth stays on your LAPTOP: export CLAUDE_CODE_OAUTH_TOKEN locally and this script
#   pushes/refreshes the flex2 secret at launch (so you never paste it into a command/log). If you
#   want it gone from flex2 between runs, set EPHEMERAL=1 to delete the secret right after the job is
#   submitted (the running job keeps the env var; you'd re-launch if it gets preempted).
#
# ── ONE-TIME: generate the token (machine where you can log in to Claude; needs Pro/Max/Team/
#   Enterprise -- API keys do NOT work; Claude Code >= 2.1.51) ──
#       claude setup-token            # prints a 1-year token; keep it on your laptop, e.g.
#       echo 'export CLAUDE_CODE_OAUTH_TOKEN=...' >> ~/.zshrc.local   # (chmod 600), or use your keychain
#
# ── RUN (from your laptop, with CLAUDE_CODE_OAUTH_TOKEN exported) ──
#   scripts/launch_remote_control_agent.sh
#   EPHEMERAL=1 TASK_BRIEF=landmark-sparse-decode-task.md scripts/launch_remote_control_agent.sh
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
SECRET="${SECRET:-PRASANNS_CLAUDE_CODE_OAUTH_TOKEN}"   # flex2 secret name (user-prefixed, like the others)
BRANCH="${BRANCH:-prasann/landmark}"
TASK_BRIEF="${TASK_BRIEF:-landmark-sparse-decode-task.md}"  # optional: seed the agent with this brief

CLUSTER_ARGS=()
IFS=',' read -ra _C <<< "${CLUSTER}"; for c in "${_C[@]}"; do CLUSTER_ARGS+=(--cluster "$c"); done

# Just-in-time: if your laptop has CLAUDE_CODE_OAUTH_TOKEN exported, push it to the ${SECRET} secret
# now (source-of-truth stays on the laptop; the value never goes into a command/log). Otherwise we
# assume the secret already exists in ${WORKSPACE}.
if [ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then
  printf '%s' "${CLAUDE_CODE_OAUTH_TOKEN}" | beaker secret write -w "${WORKSPACE}" "${SECRET}"
  echo "[secret] wrote ${SECRET} in ${WORKSPACE} from local CLAUDE_CODE_OAUTH_TOKEN (value via stdin, not args/logs)"
fi

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

# --timeout 0 -> gantry submits and returns immediately (doesn't block on the long-running agent).
GANTRY_OUT=$(gantry run \
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
  --no-logs \
  --env "ANTHROPIC_DEFAULT_TASK_BRIEF=${TASK_BRIEF}" \
  --secret-env "CLAUDE_CODE_OAUTH_TOKEN=${SECRET}" \
  --install "pip install -e ." \
  --yes \
  -- bash -c "${CMD}" 2>&1 | tee /dev/stderr)

EXP_ID=$(echo "${GANTRY_OUT}" | grep -oE 'beaker\.org/(ex|workloads)/[A-Za-z0-9]+' | head -1 | sed 's#.*/##')

# EPHEMERAL=1: once the job is running (secret already injected into the container's env), delete the
# flex2 secret so it doesn't sit there between runs. The running agent keeps its env var; re-launch
# re-creates it if the job is ever preempted.
if [ "${EPHEMERAL:-0}" = "1" ] && [ -n "${EXP_ID:-}" ]; then
  echo "[ephemeral] waiting for the job to start, then deleting ${SECRET}..."
  for _ in $(seq 1 90); do
    st=$(beaker experiment get "${EXP_ID}" 2>/dev/null | tail -1 | grep -oE 'running|succeeded|failed|stopped' | head -1)
    [ -n "${st}" ] && break
    sleep 10
  done
  beaker secret delete -w "${WORKSPACE}" "${SECRET}" 2>&1 | tail -1 && \
    echo "[ephemeral] deleted ${SECRET} (running agent retains the env var; re-run to recreate on preemption)."
fi

echo
echo "Launched (${EXP_ID:-see above}). Connect from your phone: claude.ai/code (or the Claude app) -> session '${NAME}'."
echo "Or get the URL: beaker experiment logs ${EXP_ID:-<exp-id>} 2>/dev/null | grep -m1 'claude.ai/code'"
echo "First message to send it: \"Read ${TASK_BRIEF} in this repo and implement + GPU-validate it.\""
