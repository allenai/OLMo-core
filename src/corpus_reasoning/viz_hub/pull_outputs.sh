#!/bin/bash
# Pull the rendered viz outputs from S3 to local.
#
# Intentionally does NOT use --delete (a local machine may have extra artifacts
# you don't want wiped). Set VIZ_S3_DEST to the same value used by push_outputs.sh.
#   bash viz/pull_outputs.sh             # sync S3 -> outputs/
#   bash viz/pull_outputs.sh --dryrun    # preview
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${VIZ_OUT_DIR:-$HERE/outputs}"

if [[ -z "${VIZ_S3_DEST:-}" ]]; then
  echo "ERROR: set VIZ_S3_DEST, e.g. export VIZ_S3_DEST=s3://<bucket>/<prefix>/corpus_reasoning_viz" >&2
  exit 1
fi

mkdir -p "$OUT"
aws s3 sync "$@" "$VIZ_S3_DEST" "$OUT"
echo "Pulled $VIZ_S3_DEST -> $OUT"
