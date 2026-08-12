#!/usr/bin/env bash
# Assemble the self-contained v3 eval bundle on weka.
#
# v3 = v2_clean with TWO tasks replaced:
#   contradiction -> realistic-mode (iid with the training generator), 2k..1M
#   outlier       -> TRUE scale-K rungs (K ~ n/9.5 at every length, gap>=2), 3k..1M
# and THREE tasks identical to v2: nq, rerank, oolong.
#
# The three unchanged tasks are SYMLINKED, not copied. Copying would move several GB (nq's 2M rung
# alone is 4.5G) and would leave "identical to v2" as a claim to be checked; a symlink makes it true
# by construction and costs nothing. It also means a later fix to a v2 file propagates rather than
# silently diverging.
#
# Run as a gantry CPU job with weka mounted.
set -uo pipefail

P=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
V2="$P/_eval_bundle_eval500_v2_clean"
V3="$P/_eval_bundle_eval500_v3"

echo "=== assembling $V3 ==="
[ -d "$V2" ] || { echo "!!! v2_clean missing"; exit 1; }
mkdir -p "$V3"

# ---- 1. the three unchanged tasks: symlink whole directories -------------------------------
for t in nq rerank oolong; do
  if [ -e "$V3/$t" ] && [ ! -L "$V3/$t" ]; then
    echo "  [$t] real directory already present, leaving alone"; continue
  fi
  ln -sfn "$V2/$t" "$V3/$t"
  n=$(ls "$V3/$t"/*.jsonl 2>/dev/null | wc -l)
  echo "  [$t] -> symlink to v2_clean ($n files)"
done

# ---- 2. contradiction + outlier: real dirs, populated by the staging jobs -------------------
for t in contra outlier; do
  [ -L "$V3/$t" ] && rm -f "$V3/$t"
  mkdir -p "$V3/$t"
done

echo ""
echo "=== v3 contents ==="
for t in contra outlier nq rerank oolong; do
  kind=$([ -L "$V3/$t" ] && echo "symlink->v2" || echo "REPLACED")
  n=$(ls "$V3/$t"/*.jsonl 2>/dev/null | wc -l)
  printf "  %-10s %-12s %2s jsonl\n" "$t" "$kind" "$n"
done

echo ""
echo "=== rung inventory (what an eval can actually read) ==="
for t in contra outlier; do
  echo "  [$t]"
  ls "$V3/$t"/*.jsonl 2>/dev/null | while read -r f; do
    printf "    %10s rows  %s\n" "$(wc -l < "$f")" "$(basename "$f")"
  done
done

echo ""
echo "NOTE: v3 contradiction and outlier are NOT comparable to their v2 counterparts --"
echo "  contra changes perturbation mode (realistic vs both), outlier changes K scaling."
echo "  nq / rerank / oolong ARE the same files as v2 and are directly comparable."
