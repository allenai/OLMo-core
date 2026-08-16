#!/usr/bin/env bash
# Build the paper and print a deduplicated warning/error report.
#
#   bash paperdraft2/build.sh          # full build (pdflatex, bibtex, pdflatex x2)
#   QUICK=1 bash paperdraft2/build.sh  # single pass, for iterating on a fix
#
# ⚠ RUN FROM paperdraft2/, NOT FROM iclr2026/. The main file uses BOTH
# `\input{iclr2026/sections/...}` (resolves from the parent) and `\input{math_commands.tex}`
# (lives inside iclr2026/), so neither directory works alone -- TEXINPUTS has to carry the second.
# This is why the source tree looks like it should build one level down and does not.
set -uo pipefail
cd "$(dirname "$0")"
export TEXINPUTS=".//:./iclr2026//:"
MAIN=iclr2026/iclr2026_conference

run() { pdflatex -interaction=nonstopmode -file-line-error "$MAIN.tex" > /dev/null 2>&1; }

run
if [ -z "${QUICK:-}" ]; then
  bibtex iclr2026_conference > bibtex.log 2>&1
  run; run
fi

LOG=iclr2026_conference.log
echo "=============== ERRORS ==============="
grep -E '^(\./)?[^ ]*:[0-9]+:|^! ' "$LOG" | sort -u || true
echo
echo "=============== UNDEFINED REFS / CITATIONS ==============="
grep -oE "(Reference|Citation) \`[^']*' on page [0-9]+ undefined" "$LOG" | sed -E 's/ on page [0-9]+//' | sort -u || true
echo
echo "=============== MULTIPLY-DEFINED LABELS ==============="
grep -oE "Label \`[^']*' multiply defined" "$LOG" | sort -u || true
echo
echo "=============== PACKAGE / LATEX WARNINGS ==============="
grep -E '^(LaTeX|Package) .*Warning' "$LOG" | grep -viE 'rerun|Label\(s\) may have changed' | sort | uniq -c | sort -rn || true
echo
echo "=============== OVERFULL/UNDERFULL BOXES ==============="
printf 'Overfull  \\hbox : %s\n' "$(grep -c 'Overfull \\hbox' "$LOG")"
printf 'Underfull \\hbox : %s\n' "$(grep -c 'Underfull \\hbox' "$LOG")"
printf 'Overfull  \\vbox : %s\n' "$(grep -c 'Overfull \\vbox' "$LOG")"
echo "-- overfull hboxes worse than 20pt (the ones that visibly stick into the margin) --"
grep -E 'Overfull \\hbox \([0-9]+\.[0-9]+pt' "$LOG" \
  | sed -E 's/.*Overfull \\hbox \(([0-9]+)\.[0-9]+pt too wide.*/\1/' \
  | awk '$1>20' | sort -rn | head -20 | tr '\n' ' '
echo
echo
echo "=============== BIBTEX ==============="
grep -iE "warning|error" bibtex.log 2>/dev/null | sort | uniq -c | sort -rn | head -20 || true
echo
# ⚠ THE OUTPUT PDF IS ./iclr2026_conference.pdf, NOT $MAIN.pdf. pdflatex writes to the CURRENT
# directory, so the build lands beside this script while iclr2026/iclr2026_conference.pdf is the
# stale copy that shipped in the zip (7 pages, vs 31 for a real build). Pointing a page count or a
# viewer at the latter silently reports the old paper.
echo "PDF: ./iclr2026_conference.pdf -- $(pdfinfo iclr2026_conference.pdf 2>/dev/null | grep -E '^Pages' || echo 'NOT BUILT')"
