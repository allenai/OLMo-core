#!/usr/bin/env bash
# Runs INSIDE a weka-mounted gantry job. Head-samples the compressive landmark RULER gate logs for the
# pre-SFT (base) and post-SFT checkpoints and writes compact gate-set dumps to $OUT (default /results),
# which gantry captures as a beaker dataset. Pull the dumps and plot locally with plot_gate_jaccard.py.
#
# Extraction is stdlib-only (no numpy/matplotlib needed on-cluster). See in_progress_gate_distribution.md
# for the gate-log schema and weka layout.
set -uo pipefail

BASE="${GATE_BASE:-/weka-mount/oe-training-default/ai2-llm/checkpoints/amandab/gate_scores}"
[ -d "$BASE" ] || BASE="/weka/oe-training-default/ai2-llm/checkpoints/amandab/gate_scores"
OUT="${OUT:-/results}"
MODE="${MODE:-head}"            # head (Q1/Q2/Q5) or balanced (Q3/Q4, all subtasks)
PER_FILE="${PER_FILE:-500}"    # head mode: records per file
PER_KEY="${PER_KEY:-80}"       # balanced mode: records per subtask per file
LENGTHS="${LENGTHS:-8 16 32 64}"
PY="${PY:-python3}"

PRE="q4b-base-fastcomplm-s2385"   # compressive landmark, base (pre-SFT)
POST="q4b-comp-5task-s8550"       # compressive landmark, 5-task SFT (post-SFT)

DUMPS="$OUT/dumps"; mkdir -p "$DUMPS"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "=== BASE=$BASE  OUT=$OUT  PER_FILE=$PER_FILE ==="
ls -la "$BASE" 2>&1 | head
for label in "$PRE" "$POST"; do
  echo "--- $label/ruler ---"; ls -la "$BASE/$label/ruler" 2>&1 | head -20
done

echo "=== schema peek (first record, $PRE 8k) ==="
head -n 1 "$BASE/$PRE/ruler/"gate.ruler8k.* 2>/dev/null | $PY -c '
import sys, json
ln = sys.stdin.readline()
if not ln.strip():
    print("  (no record read)"); sys.exit()
d = json.loads(ln)
layers = d.get("layers", {})
l0 = next(iter(layers.values())) if layers else {}
h0 = next(iter(l0.values())) if l0 else {}
print("  top keys:", list(d))
print("  subtask=", repr(d.get("subtask")), " doc_id=", d.get("doc_id"), " tok=", d.get("decoded_token_num"))
print("  n_layers=", len(layers), " n_heads=", len(l0),
      " len(all_scores)=", len(h0.get("all_scores", [])), " len(blocks)=", len(h0.get("blocks", [])))
' || echo "  peek failed"

for K in $LENGTHS; do
  for label in "$PRE" "$POST"; do
    files=$(ls "$BASE/$label/ruler/"gate.ruler${K}k.* 2>/dev/null)
    if [ -z "$files" ]; then echo "MISSING: $label ruler${K}k"; continue; fi
    echo "=== extract $label ${K}k (mode=$MODE) ==="
    $PY "$HERE/extract_gate_sets.py" $files --len $((K * 1024)) \
        --mode "$MODE" --per-file "$PER_FILE" --per-key "$PER_KEY" \
        --out "$DUMPS/${label}_ruler_${K}k.jsonl"
  done
done

echo "=== dumps written ==="; ls -la "$DUMPS"
# quick coverage report: distinct docs / subtasks / tokens per dump
$PY - "$DUMPS" <<'PYEOF'
import glob, json, os, sys
for p in sorted(glob.glob(os.path.join(sys.argv[1], "*.jsonl"))):
    docs, subs, toks, n = set(), set(), set(), None
    for line in open(p):
        d = json.loads(line); docs.add(d["doc"]); subs.add(d["sub"]); toks.add(d["tok"]); n = d.get("n")
    print(f"{os.path.basename(p)}: recs~ docs={len(docs)} toks={sorted(toks)[:8]} subs={sorted(subs)[:6]} n={n}")
PYEOF
