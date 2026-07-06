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
#   PARALLEL=16 OUT=/weka/.../cr_suite_data bash src/scripts/data/download_suite_data.sh
#
# PARALLEL (default 8) controls how many files download concurrently -- the whole
# suite (~22.7 GB) drops from ~25 min sequential to a few minutes.
#
# The manifest (suite_manifest.tsv) maps each file -> task / cot_mode / split /
# bytes, so the tokenize step knows which --task / cot_mode to pass per file.
set -euo pipefail

BASE="https://storage.googleapis.com/corpus-reasoning-olmo-data-prasann/suite"
OUT="${OUT:-data}"
PARALLEL="${PARALLEL:-8}"
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

file_size() { stat -c%s "$1" 2>/dev/null || stat -f%z "$1"; }

# Pass 1: decide which files to fetch (apply filters + skip already-present-at-right-size).
TO_FETCH=()   # entries: "file<TAB>bytes"
have=0; skipped=0
while IFS=$'\t' read -r file task cot split bytes; do
  [ "$file" = "file" ] && continue          # header
  want_task "$task" || { skipped=$((skipped+1)); continue; }
  [ -n "$SPLIT_FILTER" ] && [ "$split" != "$SPLIT_FILTER" ] && { skipped=$((skipped+1)); continue; }
  dest="$OUT/$file"
  if [ -f "$dest" ] && [ "$(file_size "$dest")" = "$bytes" ]; then
    have=$((have+1)); continue
  fi
  TO_FETCH+=("$file"$'\t'"$bytes")
done < "$MANIFEST"

echo "[*] ${#TO_FETCH[@]} to download, $have already present, $skipped filtered out (parallel=$PARALLEL)"

# Pass 2: download concurrently (xargs -P). Each worker curls one file.
dl_one() { curl -fsSL "$BASE/data/$1" -o "$OUT/$1" && echo "[+] $1"; }
export -f dl_one
export BASE OUT
if [ "${#TO_FETCH[@]}" -gt 0 ]; then
  printf '%s\n' "${TO_FETCH[@]}" | cut -f1 \
    | xargs -P "$PARALLEL" -n1 -I{} bash -c 'dl_one "$1"' _ {}
fi

# Pass 3: verify every fetched file matches its manifest size.
fail=0
for entry in ${TO_FETCH[@]+"${TO_FETCH[@]}"}; do
  file="${entry%%$'\t'*}"; want="${entry##*$'\t'}"
  got="$(file_size "$OUT/$file" 2>/dev/null || echo missing)"
  if [ "$got" != "$want" ]; then
    echo "[!] SIZE MISMATCH for $file: got $got want $want" >&2; fail=1
  fi
done
[ "$fail" = 0 ] || exit 1

echo "[done] $((${#TO_FETCH[@]} + have)) files in $OUT/ (${#TO_FETCH[@]} fetched, $have cached, $skipped filtered). Manifest: $MANIFEST"
