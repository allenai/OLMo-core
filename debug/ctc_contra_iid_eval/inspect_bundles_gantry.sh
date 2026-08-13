#!/usr/bin/env bash
# List what actually exists in the v2_clean and v3 eval bundles on weka, and which entries in v3 are
# real directories vs symlinks back at v2.
#
# "Run v3 where it exists" is only answerable against the filesystem: v3 was assembled as a
# self-contained root in which contra and outlier are REAL rebuilt dirs while nq/rerank/oolong are
# directory symlinks to v2_clean (identical files, so a v3 label on them would assert a measurement
# that was never made). The four OOD ladders are not mentioned in the assembly notes at all, so
# whether they exist under v3 has to be checked rather than assumed.
#
# Usage:  debug/ctc_contra_iid_eval/inspect_bundles_gantry.sh
set -euo pipefail

CLUSTER="${CLUSTER:-ai2/jupiter-cirrascale-2}"
WORKSPACE="${WORKSPACE:-ai2/flex2}"
BUDGET="${BUDGET:-ai2/oe-other}"
WEKA="${WEKA:-oe-training-default}"
PRIORITY="${PRIORITY:-urgent}"
NAME="${NAME:-inspect-eval-bundles}"
IMAGE="${IMAGE:-tylerr/olmo-core-tch291cu128-2025-11-25}"

read -r -d '' REMOTE <<'REMOTE_EOF' || true
set -uo pipefail
P=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
V2=$P/_eval_bundle_eval500_v2_clean
V3=$P/_eval_bundle_eval500_v3

for B in "$V2" "$V3"; do
  echo "================ $B ================"
  if [ ! -d "$B" ]; then echo "  !!! MISSING"; continue; fi
  # -l on the bundle root: a symlinked task dir prints as 'name -> target', which is exactly the
  # distinction that decides whether a v3 run of that task measures anything new.
  ls -l "$B" | sed 's/^/  /'
done

echo
echo "================ per-task file counts ================"
for t in contra nq rerank outlier oolong fiqa scifact outlier_review contra_fever; do
  n2=$(ls "$V2/$t" 2>/dev/null | wc -l | tr -d ' ')
  n3=$(ls "$V3/$t" 2>/dev/null | wc -l | tr -d ' ')
  kind3="absent"
  if [ -L "$V3/$t" ]; then kind3="SYMLINK -> $(readlink "$V3/$t")"
  elif [ -d "$V3/$t" ]; then kind3="real dir"; fi
  printf "  %-16s v2=%-4s v3=%-4s  v3kind=%s\n" "$t" "$n2" "$n3" "$kind3"
done

echo
echo "================ contra/outlier rung filenames (v3) ================"
ls "$V3/contra" 2>/dev/null | head -20 | sed 's/^/  contra: /'
ls "$V3/outlier" 2>/dev/null | head -20 | sed 's/^/  outlier: /'
REMOTE_EOF

gantry run \
  --name "${NAME}" --task-name "${NAME}" \
  --workspace "${WORKSPACE}" --budget "${BUDGET}" \
  --cluster "${CLUSTER}" --priority "${PRIORITY}" \
  --beaker-image "${IMAGE}" --cpus 1 \
  --weka "${WEKA}:/weka/${WEKA}" \
  --python-manager conda --system-python \
  --allow-dirty --yes --show-logs \
  -- bash -c "${REMOTE}"
