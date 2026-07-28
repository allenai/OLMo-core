#!/bin/bash
# STAGE 1 of 2 -- DISCOVERY ONLY. Never deletes anything; has no rm in it at all.
#
# Walks a run root and writes a MANIFEST: one exact absolute directory path per line, each a
# step<N>/ dir that is safe to remove. apply_deletions.sh then consumes that literal list -- the
# delete stage does no globbing, so a pattern can never expand onto something unintended.
#
# Layout (verified by the 2026-07-28 inventory):
#   <run>/stepN/            full training ckpt: model + optimizer + train state   (~54G @4B)
#   <run>/model_and_optim/  MODEL-ONLY final save -- what eval actually loads     (~18G @4B)
#
# Modes:
#   MODE=modelonly  list ALL step<N>/ for runs that have a non-empty
#                   model_and_optim/.metadata, i.e. the final model survives regardless.
#   MODE=keepfinal  list all but the HIGHEST step<N>/ (for runs with no model-only save).
#
#   ROOT=... MODE=modelonly OUT=/tmp/manifest.txt bash plan_deletions.sh
set -uo pipefail

ROOT="${ROOT:?set ROOT to the dir whose immediate children are run dirs}"
MODE="${MODE:?set MODE=modelonly|keepfinal}"
OUT="${OUT:?set OUT to the manifest path to write}"
FRESH_MIN="${FRESH_MIN:-90}"
KEEP_REGEX="${KEEP_REGEX:-(bases|shards|_eval_bundle|tokenizer|hpqaret)}"

case "$MODE" in modelonly|keepfinal) ;; *) echo "FATAL: bad MODE '$MODE'" >&2; exit 2 ;; esac
[ -d "$ROOT" ] || { echo "FATAL: ROOT not a directory: $ROOT" >&2; exit 2; }
case "$(readlink -f "$ROOT")/" in
  */checkpoints/prasanns/*) ;;
  *) echo "FATAL: ROOT outside checkpoints/prasanns/ -- refusing. ROOT=$ROOT" >&2; exit 2 ;;
esac

: > "$OUT"
total_kb=0; n_runs=0
n_fresh=0; n_keep=0; n_nomo=0; n_nosteps=0

for run in "$ROOT"/*/; do
  run="${run%/}"; name="$(basename "$run")"

  if printf '%s' "$name" | grep -Eq "$KEEP_REGEX"; then
    echo "SKIP protected      : $name"; n_keep=$((n_keep+1)); continue
  fi
  mapfile -t steps < <(find "$run" -maxdepth 1 -type d -name 'step*' -printf '%f\n' 2>/dev/null \
                       | sed -E 's/^step([0-9]+)$/\1 &/' | sort -n | awk '{print $2}')
  [ "${#steps[@]}" -eq 0 ] && { n_nosteps=$((n_nosteps+1)); continue; }

  if [ -n "$(find "$run" -maxdepth 2 -newermt "-${FRESH_MIN} minutes" -print -quit 2>/dev/null)" ]; then
    echo "SKIP in-flight      : $name"; n_fresh=$((n_fresh+1)); continue
  fi

  if [ "$MODE" = modelonly ]; then
    if [ ! -s "$run/model_and_optim/.metadata" ]; then
      echo "SKIP no final model : $name"; n_nomo=$((n_nomo+1)); continue
    fi
    victims=("${steps[@]}"); keepmsg="model_and_optim/ (final model)"
  else
    victims=("${steps[@]:0:${#steps[@]}-1}"); keepmsg="${steps[-1]}/ (final)"
    [ "${#victims[@]}" -eq 0 ] && { n_nosteps=$((n_nosteps+1)); continue; }
  fi

  n_runs=$((n_runs+1))
  echo "PLAN $name -> keep $keepmsg"
  for s in "${victims[@]}"; do
    d="$run/$s"
    kb=$(du -sk "$d" 2>/dev/null | cut -f1); total_kb=$((total_kb + ${kb:-0}))
    printf '%s\n' "$d" >> "$OUT"          # exact absolute path, one per line
    printf "     + %8s  %s\n" "$(du -sh "$d" 2>/dev/null | cut -f1)" "$d"
  done
done

echo
echo "=============================================="
echo "ROOT              : $ROOT"
echo "MODE              : $MODE"
echo "runs planned      : $n_runs"
echo "paths in manifest : $(wc -l < "$OUT")"
echo "skipped: in-flight=$n_fresh protected=$n_keep no-final-model=$n_nomo no-steps=$n_nosteps"
echo "would free        : $(awk -v k=$total_kb 'BEGIN{printf "%.2f TB", k/1024/1024/1024}')"
echo "manifest          : $OUT"
echo "(DISCOVERY ONLY -- this script contains no delete operation.)"
