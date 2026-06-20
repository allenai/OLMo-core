#!/usr/bin/env bash
# One-command fetch of the corpus-reasoning task-suite data (unified JSONL,
# pre-tokenization). No auth needed -- the bucket is public-read. Files land in
# $OUT (default ./data) next to the manifest.
#
# Usage:
#   bash src/scripts/data/download_suite_data.sh                 # download everything
#   bash src/scripts/data/download_suite_data.sh contradiction qdmatch   # only these tasks
#   TASKS="absence cycle" bash src/scripts/data/download_suite_data.sh    # same, via env
#   bash src/scripts/data/download_suite_data.sh --eval-only     # skip *_train_* files
#   OUT=/weka/.../cr_suite_eval bash src/scripts/data/download_suite_data.sh --eval-only
#
# The manifest (suite_manifest.tsv) maps each file -> task / cot_mode / split /
# bytes, so the tokenize step knows which --task / cot_mode to pass per file.
set -euo pipefail

BASE="https://storage.googleapis.com/corpus-reasoning-olmo-data-prasann/suite"
OUT="${OUT:-data}"
MANIFEST="$OUT/suite_manifest.tsv"

mkdir -p "$OUT"

# Parse args: --eval-only / --train-only flags, remaining = task filter list.
SPLIT_FILTER=""
TASK_FILTER=()
for a in "$@"; do
  case "$a" in
    --eval-only)  SPLIT_FILTER="eval";;
    --train-only) SPLIT_FILTER="train";;
    *)            TASK_FILTER+=("$a");;
  esac
done
[ -n "${TASKS:-}" ] && read -r -a TASK_FILTER <<< "$TASKS"

echo "[*] fetching manifest -> $MANIFEST"
curl -fsSL "$BASE/suite_manifest.tsv" -o "$MANIFEST"

want_task() {  # $1 = task
  [ ${#TASK_FILTER[@]} -eq 0 ] && return 0
  for t in "${TASK_FILTER[@]}"; do [ "$t" = "$1" ] && return 0; done
  return 1
}

n=0; skipped=0
# columns: file  task  cot_mode  split  bytes
while IFS=$'\t' read -r file task cot split bytes; do
  [ "$file" = "file" ] && continue          # header
  want_task "$task" || { skipped=$((skipped+1)); continue; }
  [ -n "$SPLIT_FILTER" ] && [ "$split" != "$SPLIT_FILTER" ] && { skipped=$((skipped+1)); continue; }
  dest="$OUT/$file"
  # skip if already present at the right size
  if [ -f "$dest" ] && [ "$(stat -c%s "$dest" 2>/dev/null || stat -f%z "$dest")" = "$bytes" ]; then
    echo "[=] $file (have)"; n=$((n+1)); continue
  fi
  echo "[+] $file  ($task/$cot/$split, $bytes B)"
  curl -fSL "$BASE/data/$file" -o "$dest"
  got=$(stat -c%s "$dest" 2>/dev/null || stat -f%z "$dest")
  if [ "$got" != "$bytes" ]; then
    echo "[!] SIZE MISMATCH for $file: got $got want $bytes" >&2; exit 1
  fi
  n=$((n+1))
done < "$MANIFEST"

echo "[done] $n files in $OUT/ ($skipped skipped by filter). Manifest: $MANIFEST"
