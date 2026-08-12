#!/bin/bash
# Backfill v3/outlier's BASE rungs (2k-32k) by per-file symlink to v2_clean.
#
# WHY THEY WERE MISSING. v3 replaces outlier's xlong rungs only -- the K-freeze came from the generic
# xlong expander (pool="self_nongold"), which never touched the base ladder, so 2k-32k was already
# true scale-K (K=3/7/13/25) and is meant to carry over unchanged. But making v3/outlier a real
# directory to hold the rebuilt 64k-1M files meant the base rungs had nowhere to come from, and the
# assembly job's own inventory showed outlier with 5 files where contra had 9.
#
# Per-FILE symlinks, not a directory symlink: the directory has to stay real to hold the rebuilt
# xlong rungs. Anything already present in v3 is left alone, so this cannot clobber a rebuilt file.
set -uo pipefail
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
cd /accounts/projects/berkeleynlp/prasann/projects/OLMo-core

WORK='
set -uo pipefail
P=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
V2="$P/_eval_bundle_eval500_v2_clean/outlier"
V3="$P/_eval_bundle_eval500_v3/outlier"

[ -d "$V2" ] || { echo "!!! v2_clean/outlier missing"; exit 1; }
[ -d "$V3" ] && [ ! -L "$V3" ] || { echo "!!! v3/outlier must be a real dir"; exit 1; }

echo "=== v2_clean/outlier (source of the base rungs) ==="
ls -la "$V2"/*.jsonl 2>/dev/null | awk "{printf \"  %10s  %s\n\", \$5, \$9}"

echo ""
echo "=== backfilling anything v3 does not already have ==="
for f in "$V2"/*.jsonl; do
  b=$(basename "$f")
  if [ -e "$V3/$b" ]; then
    kind=$([ -L "$V3/$b" ] && echo symlink || echo REBUILT)
    echo "  [skip] $b (already present, $kind)"
    continue
  fi
  ln -sfn "$f" "$V3/$b"
  echo "  [link] $b  ($(wc -l < "$V3/$b") rows)"
done

echo ""
echo "=== final v3/outlier ladder ==="
for f in "$V3"/*.jsonl; do
  b=$(basename "$f")
  kind=$([ -L "$f" ] && echo "v2-base " || echo "REBUILT ")
  printf "  %s %8s rows  %s\n" "$kind" "$(wc -l < "$f")" "$b"
done

echo ""
echo "=== cross-check: contra vs outlier rung counts ==="
for t in contra outlier; do
  printf "  %-8s %s jsonl\n" "$t" "$(ls "$P/_eval_bundle_eval500_v3/$t"/*.jsonl 2>/dev/null | wc -l)"
done
'

gantry run --name contra-v3-outlier-base-backfill -w ai2/flex2 -b ai2/oe-other \
  --cluster 'ai2/*-cirrascale*' --gpus 0 --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --weka oe-training-default:/weka/oe-training-default \
  --no-python --allow-dirty --timeout 0 --yes -- bash -c "$WORK"
