#!/usr/bin/env bash
# Reproduce the OLMo-core / IT-suite dev workflow on a new machine (e.g. hermione).
# Idempotent: re-runnable; skips what's already present.
#
# Prereqs you provide (auth — can't be scripted, they're your secrets):
#   * GitHub access to the private allenai repos (gh auth login, or an SSH key / PAT).
#   * Your Beaker user token (from https://beaker.org/user -> settings).
#   * A Python 3.12 env active (recommended): `mamba create -n olmoenv python=3.12 && mamba activate olmoenv`
#     (or any venv); this script pip-installs into the CURRENTLY ACTIVE python.
#
# Run from where you want the repos to live (they're cloned as siblings of OLMo-core), e.g.:
#   cd ~/dev && git clone -b prasann/landmark https://github.com/allenai/OLMo-core.git
#   cd OLMo-core && scripts/setup_dev_env.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && git rev-parse --show-toplevel)"
BASE="$(dirname "${REPO_ROOT}")"     # parent dir -> siblings cloned here
PY="$(command -v python || command -v python3)"
echo "[*] repos under: ${BASE}    python: ${PY} ($("${PY}" --version 2>&1))"

# ── 1. Beaker CLI (standalone Go binary) ──────────────────────────────────────
if command -v beaker >/dev/null 2>&1; then
  echo "[ok] beaker CLI: $(beaker --version 2>&1 | head -1)"
else
  echo "[!] beaker CLI not found. Install the Go binary from https://github.com/allenai/beaker/releases"
  echo "    (download the linux build, put it on PATH), then: beaker configure   # paste your user_token"
fi

# ── 2. Python tooling (into the active env) ───────────────────────────────────
echo "[*] pip installing the toolchain into the active python ..."
"${PY}" -m pip install -q --upgrade \
  "beaker-gantry==3.7.0" "beaker-py==2.7.0" \
  transformers datasets wandb "huggingface-hub>=0.34" jinja2 numpy tqdm
echo "[ok] gantry: $("${PY}" -m gantry --version 2>/dev/null || echo '(installed)')"

# ── 3. Clone the repos as siblings (private allenai -> needs GitHub auth) ──────
clone_branch() {  # $1=url $2=branch $3=dir
  local url="$1" branch="$2" dir="${BASE}/$3"
  if [ -d "${dir}/.git" ]; then
    echo "[ok] ${3} present ($(git -C "${dir}" branch --show-current))"
  else
    echo "[*] cloning ${3} @ ${branch}"
    git clone -b "${branch}" "${url}" "${dir}"
  fi
}
clone_branch https://github.com/allenai/oe-eval-internal.git prasann/longctx-eval oe-eval-internal
clone_branch https://github.com/allenai/olmo-cookbook.git     main                 olmo-cookbook
# Optional (only to *generate* new suite data): corpus-reasoning
# clone_branch https://github.com/PrasannS/corpus-reasoning.git main corpus-reasoning

# ── 4. Editable installs (this repo + olmo-cookbook -> the olmo-cookbook-eval CLI) ──
echo "[*] pip install -e OLMo-core (+ extras) and olmo-cookbook ..."
"${PY}" -m pip install -q -e "${REPO_ROOT}[all]" || "${PY}" -m pip install -q -e "${REPO_ROOT}"
[ -d "${BASE}/olmo-cookbook" ] && "${PY}" -m pip install -q -e "${BASE}/olmo-cookbook"
echo "[ok] olmo-cookbook-eval: $(command -v olmo-cookbook-eval || echo 'not on PATH (check the env bin dir)')"

# ── 5. Auth + sanity ─────────────────────────────────────────────────────────
cat <<EOF

── REMAINING MANUAL STEPS (your secrets) ─────────────────────────────────────
  1. Beaker:  beaker configure        # paste your user_token; set default workspace ai2/flex2
              beaker account whoami    # should print 'prasanns'
  2. GitHub:  gh auth login            # or have an SSH key / PAT for the private allenai repos
  3. wandb (optional):  wandb login    # entity for these scripts is prasanns-allen-institute-for-ai

── SANITY CHECKS ─────────────────────────────────────────────────────────────
  python -c "import olmo_core; print('olmo_core ok')"
  beaker account whoami
  olmo-cookbook-eval --help | head -1
  # data is on weka (mounted in jobs) at: /weka/oe-training-default/ai2-llm/checkpoints/prasanns/cr_suite_data

You can now drive the whole pipeline from here (gantry/launch jobs run on Beaker; nothing
GPU-local needed). See instruction-tuning-setup.md for the pipeline + weka pointers.
EOF
