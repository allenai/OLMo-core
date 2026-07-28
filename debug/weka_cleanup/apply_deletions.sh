#!/bin/bash
# STAGE 2 of 2 -- EXECUTION. Deletes ONLY the exact paths named in a manifest.
#
# Deliberately contains NO globbing, NO find, and NO pattern expansion in the delete path. Every
# target is a literal line read from the manifest produced by plan_deletions.sh, and each one is
# re-validated against the filesystem here -- so even a corrupted or hand-edited manifest cannot
# cause an unintended delete. Every check must pass or the path is refused:
#
#   1. absolute path under .../checkpoints/prasanns/      (namespace containment)
#   2. basename matches ^step[0-9]+$                      (only training-step dirs, ever)
#   3. is a real directory and NOT a symlink              (no following links out of the tree)
#   4. no '..' anywhere in the path                       (no traversal)
#   5. parent run dir has a non-empty model_and_optim/.metadata   (final model survives)
#   6. parent run dir name doesn't match KEEP_REGEX       (protected runs)
#   7. parent run dir untouched for FRESH_MIN minutes     (nothing in flight)
#
#   MANIFEST=/tmp/m.txt bash apply_deletions.sh            # verify only, deletes nothing
#   MANIFEST=/tmp/m.txt APPLY=1 bash apply_deletions.sh    # delete the verified paths
set -uo pipefail

MANIFEST="${MANIFEST:?set MANIFEST to the manifest file from plan_deletions.sh}"
# Which "the weights survive" invariant to enforce. MUST match the mode the manifest was planned
# with, because the two families of run dir protect their final weights differently:
#   modelonly  parent has a non-empty model_and_optim/.metadata (the model-only final save)
#   keepfinal  a strictly higher step<N>/ survives on disk and is NOT itself in the manifest
MODE="${MODE:-modelonly}"
case "$MODE" in modelonly|keepfinal) ;; *) echo "FATAL: bad MODE '$MODE'" >&2; exit 2 ;; esac
APPLY="${APPLY:-0}"
FRESH_MIN="${FRESH_MIN:-90}"
KEEP_REGEX="${KEEP_REGEX:-(bases|shards|_eval_bundle|tokenizer|hpqaret)}"
[ -s "$MANIFEST" ] || { echo "FATAL: manifest missing or empty: $MANIFEST" >&2; exit 2; }

# The one allowed namespace. Overridable ONLY so the guard logic can be exercised against a local
# test fixture -- never set this in a real run; the gantry launcher never does.
ROOT_PREFIX="${ROOT_PREFIX:-/weka/oe-training-default/ai2-llm/checkpoints/prasanns}"
echo "namespace       : $ROOT_PREFIX"

n_ok=0; n_bad=0; n_del=0; total_kb=0

# FRESHNESS PRE-PASS. The in-flight check has to be evaluated for every parent run BEFORE any
# deletion happens: removing a step dir updates its parent's mtime, so checking inline made the
# guard trip on our own writes and refuse every path after the first in each run. Compute each
# parent's verdict once, up front, and reuse it.
declare -A FRESH INMAN SURVIVOR
# Manifest membership, so keepfinal can prove the survivor is not itself scheduled for deletion.
while IFS= read -r d; do [ -n "$d" ] && INMAN["$d"]=1; done < "$MANIFEST"

while IFS= read -r d; do
  [ -z "$d" ] && continue
  run="$(dirname "$d")"
  [ -n "${FRESH[$run]+x}" ] && continue
  if [ -n "$(find "$run" -maxdepth 2 -newermt "-${FRESH_MIN} minutes" -print -quit 2>/dev/null)" ]; then
    FRESH[$run]=busy
  else
    FRESH[$run]=idle
  fi
  if [ "$MODE" = keepfinal ]; then
    # Highest-numbered step dir currently on disk. Resolved before any deletion; since we never
    # delete the survivor it stays valid for the whole run.
    SURVIVOR[$run]="$(find "$run" -maxdepth 1 -type d -name 'step*' -printf '%f\n' 2>/dev/null \
                      | sed -E 's/^step([0-9]+)$/\1 &/' | sort -n | awk 'END{print $2}')"
  fi
done < "$MANIFEST"
echo "freshness pre-pass: $(printf '%s\n' "${FRESH[@]}" | grep -c idle) idle / $(printf '%s\n' "${FRESH[@]}" | grep -c busy) busy parent runs"

while IFS= read -r d; do
  [ -z "$d" ] && continue
  reason=""
  case "$d" in
    "$ROOT_PREFIX"/*) ;;
    *) reason="outside checkpoints/prasanns/" ;;
  esac
  [ -z "$reason" ] && case "$d" in *..*) reason="path traversal" ;; esac

  base="$(basename "$d")"; run="$(dirname "$d")"; rname="$(basename "$run")"
  [ -z "$reason" ] && ! printf '%s' "$base" | grep -Eq '^step[0-9]+$' && reason="not a step<N> dir"
  [ -z "$reason" ] && [ -L "$d" ]   && reason="is a symlink"
  [ -z "$reason" ] && [ ! -d "$d" ] && reason="not a directory / already gone"
  if [ "$MODE" = modelonly ]; then
    [ -z "$reason" ] && [ ! -s "$run/model_and_optim/.metadata" ] && reason="parent has no final model"
  else
    surv="${SURVIVOR[$run]:-}"
    [ -z "$reason" ] && [ -z "$surv" ]               && reason="no surviving step dir found"
    [ -z "$reason" ] && [ ! -d "$run/$surv" ]        && reason="survivor $surv missing on disk"
    [ -z "$reason" ] && [ -n "${INMAN[$run/$surv]+x}" ] && reason="survivor $surv is itself in the manifest"
    [ -z "$reason" ] && [ "$base" = "$surv" ]        && reason="is the survivor -- never delete the final"
    # strictly-lower-numbered than the survivor, compared as integers not strings
    [ -z "$reason" ] && ! [ "${base#step}" -lt "${surv#step}" ] 2>/dev/null \
        && reason="step not strictly below survivor $surv"
  fi
  [ -z "$reason" ] && printf '%s' "$rname" | grep -Eq "$KEEP_REGEX" && reason="parent is protected"
  [ -z "$reason" ] && [ "${FRESH[$run]:-busy}" = busy ] && reason="parent modified <${FRESH_MIN}m ago"

  if [ -n "$reason" ]; then
    printf "REFUSE  [%s]  %s\n" "$reason" "$d"; n_bad=$((n_bad+1)); continue
  fi

  n_ok=$((n_ok+1))
  kb=$(du -sk "$d" 2>/dev/null | cut -f1); total_kb=$((total_kb + ${kb:-0}))
  if [ "$APPLY" = 1 ]; then
    # literal path from the manifest -- no wildcard, no expansion
    if rm -rf -- "$d"; then
      n_del=$((n_del+1)); printf "DELETED %s\n" "$d"
    else
      printf "ERROR failed to delete %s\n" "$d" >&2
    fi
  else
    printf "VERIFIED (would delete) %s\n" "$d"
  fi
done < "$MANIFEST"

echo
echo "=============================================="
echo "manifest        : $MANIFEST ($(wc -l < "$MANIFEST") lines)"
echo "verified OK     : $n_ok"
echo "refused         : $n_bad"
echo "deleted         : $n_del"
echo "space $([ "$APPLY" = 1 ] && echo reclaimed || echo reclaimable) : $(awk -v k=$total_kb 'BEGIN{printf "%.2f TB", k/1024/1024/1024}')"
[ "$APPLY" = 1 ] || echo "(VERIFY ONLY -- nothing deleted. Re-run with APPLY=1.)"
