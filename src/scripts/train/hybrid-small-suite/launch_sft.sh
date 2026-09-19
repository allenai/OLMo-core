#!/usr/bin/env bash
# SFT a hybrid checkpoint on Beaker. Pins the branch so you cannot train from the wrong one.
#
#   bash launch_sft.sh <run-name> --preset 1.4b_7to1
#   bash launch_sft.sh <run-name> --model /weka/.../step44124/ [--dataset /weka/.../shards]
#
# Architecture is read from the checkpoint's own config.json -- there is nothing to specify beyond
# the checkpoint and the data.
#
# Why the branch check: `prasann/landmark`'s olmo_core has no `scalable_softmax`, so training a
# hybrid checkpoint from there loads its ssmax weights as unexpected, drops them, and trains a
# different model without failing. gantry ships the CURRENT checkout's commit, so being on the
# wrong branch is silently wrong rather than an error.
set -euo pipefail

BRANCH=prasann/ctc-sft-hybridish
CLUSTER="${CLUSTER:-ai2/jupiter-cirrascale-2}"
RUN="${1:?usage: launch_sft.sh <run-name> (--preset ARM | --model PATH) [--dataset DIR] [...]}"
shift

cur=$(git rev-parse --abbrev-ref HEAD)
if [ "$cur" != "$BRANCH" ]; then
  echo "ERROR: on branch '$cur', but SFT must run from '$BRANCH'." >&2
  echo "       Training elsewhere silently drops Scalable-Softmax. Run: git checkout $BRANCH" >&2
  exit 2
fi
if ! git diff-index --quiet HEAD --; then
  echo "WARNING: working tree is dirty; gantry ships the committed state, not your edits." >&2
fi
if ! git merge-base --is-ancestor HEAD "origin/$BRANCH" 2>/dev/null; then
  echo "ERROR: HEAD is not pushed to origin/$BRANCH. gantry clones the REMOTE commit," >&2
  echo "       so unpushed work would not be in the job. Push first." >&2
  exit 2
fi

exec env PYTHONPATH=src python src/scripts/train/hybrid-small-suite/sft_ctc.py \
  launch "$RUN" "$CLUSTER" "$@"
