#!/bin/bash
# Deploy the rendered viz site to Cloudflare Pages.
#
# THIS checkout (branch prasann/singletask-ladder) builds the "Headline · 32k
# matrix" MIXING-RESULTS site and OWNS the project `corpus-reasoning-viz`
# (-> corpus-reasoning-viz.pages.dev). The OLMo-core `main` build (results-hub
# mirror) deploys to a SEPARATE project `corpus-reasoning-hub` and is guarded from
# overwriting this one. Don't point that build here — that was the regression.
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
#   3. (optional) export CF_PAGES_PROJECT=corpus-reasoning-viz   # project name
#
# Env overrides:
#   CF_PAGES_PROJECT   Pages project name (default: corpus-reasoning-viz)
#   CF_PAGES_BRANCH    deploy branch (default: main = production)
#   VIZ_OUT_DIR        where index.html lives (default: viz/outputs)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${VIZ_OUT_DIR:-$HERE/outputs}"
PROJECT="${CF_PAGES_PROJECT:-corpus-reasoning-viz}"
BRANCH="${CF_PAGES_BRANCH:-main}"
SRC_HTML="$OUT/index.html"

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

# Ensure the Pages project exists (harmless if it already does).
"${WR[@]}" pages project create "$PROJECT" --production-branch="$BRANCH" 2>/dev/null || true

echo "Deploying $STAGE -> Cloudflare Pages project '$PROJECT' (branch '$BRANCH')..."
"${WR[@]}" pages deploy "$STAGE" --project-name="$PROJECT" --branch="$BRANCH" "$@"
