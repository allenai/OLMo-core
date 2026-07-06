#!/bin/bash
# combo_lc_evals.sh — pull RULER + HELMET long-context eval results and combine
# them into a single CSV (../results/ruler_and_helmet.csv).
#
# By default this (re)pulls fresh results by calling:
#   ./pull_ruler_results.py   -> ../results/ruler_results.csv
#   ./pull_helmet_results.py  -> ../results/helmet_results.csv
# and then joins them, model-by-model, into:
#   ../results/ruler_and_helmet.csv
# with the format:
#   modelname,,<chattemplate>,,<ruler columns...>,<helmet columns...>
# (two blank spacer columns; chattemplate is "no", or "yes (no thinking)" when
# the model name mentions "nothink").
#
# Models are matched by name, ignoring a leading "amandab_" prefix that the
# HELMET names carry for checkpoints stored under .../amandab/.
#
# Usage:
#   ./combo_lc_evals.sh                # pull fresh ruler + helmet results, then combine
#   ./combo_lc_evals.sh --skip-pull    # skip pulling; just combine the existing CSVs
#   ./combo_lc_evals.sh -h             # help
#
# Env / paths can be overridden with:
#   RULER_CSV (default ../results/ruler_results.csv)
#   HELMET_CSV (default ../results/helmet_results.csv)
#   OUT_CSV (default ../results/ruler_and_helmet.csv)
set -euo pipefail

SKIP_PULL=0
for arg in "$@"; do
  case "$arg" in
    --skip-pull|--no-pull) SKIP_PULL=1 ;;
    -h|--help)
      sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$(cd "$SCRIPT_DIR/../results" && pwd)"
RULER_CSV="${RULER_CSV:-$RESULTS_DIR/ruler_results.csv}"
HELMET_CSV="${HELMET_CSV:-$RESULTS_DIR/helmet_results.csv}"
OUT_CSV="${OUT_CSV:-$RESULTS_DIR/ruler_and_helmet.csv}"

if [ "$SKIP_PULL" -eq 0 ]; then
  echo "Pulling RULER results -> $RULER_CSV" >&2
  python3 "$SCRIPT_DIR/pull_ruler_results.py" -o "$RULER_CSV"
  echo "Pulling HELMET results -> $HELMET_CSV" >&2
  python3 "$SCRIPT_DIR/pull_helmet_results.py" -o "$HELMET_CSV"
else
  echo "Skipping pull; reading existing $RULER_CSV and $HELMET_CSV" >&2
fi

python3 - "$RULER_CSV" "$HELMET_CSV" "$OUT_CSV" <<'PY'
import csv
import sys

ruler_path, helmet_path, out_path = sys.argv[1:4]


def normalize(name: str) -> str:
    """Join key: drop a leading 'amandab_' so HELMET and RULER names line up."""
    return name[len("amandab_"):] if name.startswith("amandab_") else name


def load(path: str):
    """Return (column_labels, {join_key: (display_name, {label: value})})."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        cols = header[1:]  # everything after 'modelname'
        rows = {}
        for row in reader:
            if not row or not row[0]:
                continue
            name = row[0]
            values = dict(zip(cols, row[1:]))
            rows[normalize(name)] = (name, values)
        return cols, rows


ruler_cols, ruler = load(ruler_path)
helmet_cols, helmet = load(helmet_path)

# Model order: RULER models first (file order), then HELMET-only models.
order = list(ruler.keys()) + [k for k in helmet if k not in ruler]

header = ["modelname", "", "chattemplate", ""] + ruler_cols + helmet_cols
out = [header]
for key in order:
    display = ruler.get(key, helmet.get(key))[0]
    chat = "yes (no thinking)" if "nothink" in display.lower() else "no"
    rvals = ruler.get(key, ("", {}))[1]
    hvals = helmet.get(key, ("", {}))[1]
    row = [display, "", chat, ""]
    row += [rvals.get(c, "") for c in ruler_cols]
    row += [hvals.get(c, "") for c in helmet_cols]
    out.append(row)

with open(out_path, "w", newline="") as f:
    csv.writer(f).writerows(out)

print(f"Wrote {len(out) - 1} models to {out_path}", file=sys.stderr)
PY
