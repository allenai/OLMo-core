#!/bin/bash
# Deploy the rendered viz site to Cloudflare Pages.
#
# The site is a single self-contained index.html, so we stage a clean dir holding
# only that file and `wrangler pages deploy` it. Gives a public *.pages.dev URL.
#
#   bash viz/run.sh                  # build outputs/index.html first
#   bash viz/deploy_cloudflare.sh    # deploy it
#   # or in one step:  bash viz/run.sh --deploy
#
# One-time account setup (see viz/README.md for detail):
#   1. Create a free Cloudflare account.
#   2. Auth, either:
#        export CLOUDFLARE_API_TOKEN=...   export CLOUDFLARE_ACCOUNT_ID=...   (CI/headless)
#      or once interactively:  npx wrangler login
#   3. (optional) export CF_PAGES_PROJECT=corpus-reasoning-hub   # project name
#
# Env overrides:
#   CF_PAGES_PROJECT   Pages project name (default: corpus-reasoning-hub)
#                      NOTE: 'corpus-reasoning-viz' is reserved for the mixing-tables
#                      site (guarded below); this build defaults to a separate project.
#   CF_PAGES_BRANCH    deploy branch (default: main = production)
#   VIZ_OUT_DIR        where index.html lives (default: viz/outputs)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${VIZ_OUT_DIR:-$HERE/outputs}"
# This is the OLMo-core `main` viz (Overview/Data/Experiments + the results-hub
# "Results" tab). It deploys to its OWN Pages project so it can never clobber the
# mixing-tables site (the "Headline · 32k matrix" build from the standalone
# corpus-reasoning `prasann/singletask-ladder` checkout), which OWNS
# `corpus-reasoning-viz`. See viz/README.md.
PROJECT="${CF_PAGES_PROJECT:-corpus-reasoning-hub}"
BRANCH="${CF_PAGES_BRANCH:-main}"
SRC_HTML="$OUT/index.html"

# Guard: refuse to publish this (results-hub-mirror) build onto the mixing-tables
# site's project. That overwrite is the "mixing results disappeared" regression.
# If you REALLY mean to, set ALLOW_VIZ_CLOBBER=1.
if [[ "$PROJECT" == "corpus-reasoning-viz" && "${ALLOW_VIZ_CLOBBER:-0}" != "1" ]]; then
  echo "ERROR: this build must not deploy to 'corpus-reasoning-viz' — that project serves the" >&2
  echo "       mixing-tables site (build+deploy it from /scratch/users/prasann/corpus-reasoning" >&2
  echo "       on branch prasann/singletask-ladder). Deploy this one to 'corpus-reasoning-hub'," >&2
  echo "       or override with ALLOW_VIZ_CLOBBER=1 if you truly intend to overwrite it." >&2
  exit 1
fi

if [[ ! -f "$SRC_HTML" ]]; then
  echo "ERROR: $SRC_HTML not found — build it first with: bash viz/run.sh" >&2
  exit 1
fi

# Pick a wrangler invocation: prefer an installed binary, else npx.
if command -v wrangler >/dev/null 2>&1; then
  WR=(wrangler)
elif command -v npx >/dev/null 2>&1; then
  WR=(npx --yes wrangler@latest)
else
  echo "ERROR: need 'wrangler' (npm i -g wrangler) or 'npx' on PATH" >&2
  exit 1
fi

# Stage a clean directory containing only the site (don't publish the JSON sources).
STAGE="$OUT/_cf_site"
rm -rf "$STAGE"
mkdir -p "$STAGE"
cp "$SRC_HTML" "$STAGE/index.html"
# include the central results CSV so the Results tab's download link works
[[ -f "$OUT/results.csv" ]] && cp "$OUT/results.csv" "$STAGE/results.csv"

# Ensure the Pages project exists (harmless if it already does).
"${WR[@]}" pages project create "$PROJECT" --production-branch="$BRANCH" 2>/dev/null || true

echo "Deploying $STAGE -> Cloudflare Pages project '$PROJECT' (branch '$BRANCH')..."
"${WR[@]}" pages deploy "$STAGE" --project-name="$PROJECT" --branch="$BRANCH" "$@"
