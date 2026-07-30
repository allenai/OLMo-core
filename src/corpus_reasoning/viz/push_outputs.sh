#!/bin/bash
# Push the rendered viz outputs (HTML + JSON) to S3 so the site can be shared.
#
# Set the destination once:
#   export VIZ_S3_DEST=s3://<your-bucket>/<prefix>/corpus_reasoning_viz
# (e.g. the ai2 bucket you already use for checkpoints). Then:
#   bash viz/push_outputs.sh                 # sync outputs/ -> S3
#   bash viz/push_outputs.sh --dryrun        # preview first
#
# Uses --delete so files removed locally are removed on S3; only run from a
# machine whose local outputs/ is a superset of the S3 copy. The HTML is the
# shareable artifact (self-contained, opens in any browser). Large/regeneratable
# blobs are excluded for safety, mirroring the EMO push pattern.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${VIZ_OUT_DIR:-$HERE/outputs}"

if [[ -z "${VIZ_S3_DEST:-}" ]]; then
  echo "ERROR: set VIZ_S3_DEST, e.g. export VIZ_S3_DEST=s3://<bucket>/<prefix>/corpus_reasoning_viz" >&2
  exit 1
fi

aws s3 sync --delete "$@" \
    --exclude "*.npy" \
    --exclude "*.tar" \
    --exclude "*.tar.gz" \
    --exclude "*.zip" \
    "$OUT" "$VIZ_S3_DEST"

echo "Pushed $OUT -> $VIZ_S3_DEST"
