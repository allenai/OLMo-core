#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-ai2/OLMo-3-moe-experiments}"
BUDGET="${BUDGET:-ai2/oe-other}"
HAMMOND_HOST="${HAMMOND_HOST:-hammond-cs-aus-452.reviz.ai2.in}"
SSH_USER="${SSH_USER:-jacobm}"
SESSION_NAME="${SESSION_NAME:-jacobm-olmoe3-control}"
IMAGE="${IMAGE:-beaker://ai2/cuda12.8-dev-ubuntu22.04-torch2.6.0}"
PROJECT_DIR="${PROJECT_DIR:-/weka/oe-adapt-default/jacobm/olmoe3}"
REPO_DIR="${REPO_DIR:-${PROJECT_DIR}/OLMo-core}"
BRANCH="${BRANCH:-jacobm/olmoe-dev-v2}"
CODEX_VERSION="${CODEX_VERSION:-0.136.0}"
CODEX_TARGET="${CODEX_TARGET:-x86_64-unknown-linux-musl}"

BEAKER_CONFIG_SECRET="${BEAKER_CONFIG_SECRET:-jacobm_beaker_config}"
GIT_CONFIG_SECRET="${GIT_CONFIG_SECRET:-jacobm_git_config}"
GITHUB_SSH_KEY_SECRET="${GITHUB_SSH_KEY_SECRET:-jacobm_github_ssh_key}"
CODEX_AUTH_SECRET="${CODEX_AUTH_SECRET:-jacobm_codex_auth_json}"
CODEX_CONFIG_SECRET="${CODEX_CONFIG_SECRET:-jacobm_codex_config_toml}"

if [[ "${REFRESH_SECRETS:-0}" == "1" ]]; then
  beaker secret write -w "${WORKSPACE}" "${BEAKER_CONFIG_SECRET}" < "${HOME}/.beaker/config.yml"
  beaker secret write -w "${WORKSPACE}" "${GIT_CONFIG_SECRET}" < "${HOME}/.gitconfig"
  beaker secret write -w "${WORKSPACE}" "${GITHUB_SSH_KEY_SECRET}" < "${HOME}/.ssh/id_ed25519"
  beaker secret write -w "${WORKSPACE}" "${CODEX_AUTH_SECRET}" < "${HOME}/.codex/auth.json"
  beaker secret write -w "${WORKSPACE}" "${CODEX_CONFIG_SECRET}" < "${HOME}/.codex/config.toml"
fi

create_log="$(mktemp)"
beaker session create \
  --detach \
  --bare \
  --hostname "${HAMMOND_HOST}" \
  --name "${SESSION_NAME}" \
  --cpus 4 \
  --memory 16GiB \
  --gpus 0 \
  --workspace "${WORKSPACE}" \
  --budget "${BUDGET}" \
  --image "${IMAGE}" \
  --mount src=weka,ref=oe-training-default,dst=/weka/oe-training-default \
  --mount src=weka,ref=oe-adapt-default,dst=/weka/oe-adapt-default \
  --mount "src=secret,ref=${BEAKER_CONFIG_SECRET},dst=/root/.beaker/config.yml" \
  --mount "src=secret,ref=${GIT_CONFIG_SECRET},dst=/root/.gitconfig" \
  --mount "src=secret,ref=${GITHUB_SSH_KEY_SECRET},dst=/root/.ssh/id_ed25519" \
  --secret-env AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY \
  --secret-env WANDB_API_KEY=jacobm_WANDB_API_KEY \
  -- bash -lc "mkdir -p /root/.ssh ${PROJECT_DIR} && chmod 700 /root/.ssh && chmod 600 /root/.ssh/id_ed25519 || true && ssh-keyscan github.com >> /root/.ssh/known_hosts 2>/dev/null || true && cd ${PROJECT_DIR} && sleep infinity" \
  | tee "${create_log}"

SESSION_ID="$(sed -n 's/^Starting session \([^ ]*\) .*/\1/p' "${create_log}" | tail -n 1)"
rm -f "${create_log}"

if [[ -z "${SESSION_ID}" ]]; then
  echo "Could not parse session ID from beaker output" >&2
  exit 1
fi

echo "Bootstrapping ${SESSION_ID} on ${HAMMOND_HOST}..."

ssh -o BatchMode=yes "${SSH_USER}@${HAMMOND_HOST}" \
  "beaker session exec ${SESSION_ID} -- bash -s" <<REMOTE_BOOTSTRAP
set -euo pipefail

export WORKSPACE="${WORKSPACE}"
export PROJECT_DIR="${PROJECT_DIR}"
export REPO_DIR="${REPO_DIR}"
export BRANCH="${BRANCH}"
export CODEX_VERSION="${CODEX_VERSION}"
export CODEX_TARGET="${CODEX_TARGET}"
export CODEX_AUTH_SECRET="${CODEX_AUTH_SECRET}"
export CODEX_CONFIG_SECRET="${CODEX_CONFIG_SECRET}"
export UV_LINK_MODE=copy

mkdir -p /root/.codex /root/.local/bin "\${PROJECT_DIR}"
beaker secret read -w "\${WORKSPACE}" "\${CODEX_AUTH_SECRET}" > /root/.codex/auth.json
beaker secret read -w "\${WORKSPACE}" "\${CODEX_CONFIG_SECRET}" > /root/.codex/config.toml
chmod 600 /root/.codex/auth.json /root/.codex/config.toml

if ! command -v rg >/dev/null 2>&1; then
  apt-get update
  apt-get install -y ripgrep
fi

tmp_dir="\$(mktemp -d)"
curl -fL -o "\${tmp_dir}/codex.tar.gz" \
  "https://github.com/openai/codex/releases/download/rust-v\${CODEX_VERSION}/codex-\${CODEX_TARGET}.tar.gz"
tar -xzf "\${tmp_dir}/codex.tar.gz" -C "\${tmp_dir}"

release_dir="/root/.codex/packages/standalone/releases/\${CODEX_VERSION}-\${CODEX_TARGET}"
mkdir -p "\${release_dir}/bin" /root/.codex/packages/standalone
install -m 0755 "\${tmp_dir}/codex-\${CODEX_TARGET}" "\${release_dir}/codex"
ln -sf ../codex "\${release_dir}/bin/codex"
cat > "\${release_dir}/codex-package.json" <<CODEX_PACKAGE
{
  "layoutVersion": 1,
  "version": "\${CODEX_VERSION}",
  "target": "\${CODEX_TARGET}",
  "variant": "codex",
  "entrypoint": "bin/codex",
  "resourcesDir": "codex-resources",
  "pathDir": "codex-path"
}
CODEX_PACKAGE
ln -sfn "\${release_dir}" /root/.codex/packages/standalone/current
ln -sfn /root/.codex/packages/standalone/current/bin/codex /root/.local/bin/codex
rm -rf "\${tmp_dir}"

cd "\${PROJECT_DIR}"
if [[ ! -d OLMo-core/.git ]]; then
  git clone git@github.com:allenai/OLMo-core.git
fi
cd "\${REPO_DIR}"
git fetch origin "\${BRANCH}"
git switch "\${BRANCH}" 2>/dev/null || git switch -c "\${BRANCH}" "origin/\${BRANCH}"
git pull --ff-only
uv run --extra dev --extra beaker python -c "import sys; print(sys.version)"

/root/.local/bin/codex doctor --json >/tmp/codex-doctor.json
/root/.local/bin/codex remote-control start --json
REMOTE_BOOTSTRAP

cat <<EOF

Hammond control session is ready.
Session ID: ${SESSION_ID}
Repo: ${REPO_DIR}

Useful commands:
  ssh ${SSH_USER}@${HAMMOND_HOST}
  ssh ${SSH_USER}@${HAMMOND_HOST} "beaker session exec ${SESSION_ID} -- bash -l"
  ssh ${SSH_USER}@${HAMMOND_HOST} "beaker session exec ${SESSION_ID} -- bash -lc 'cd ${REPO_DIR} && /root/.local/bin/codex remote-control start --json'"
EOF
