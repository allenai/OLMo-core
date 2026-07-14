#!/bin/bash
# Compact slurm job + eval status. Designed to print under 30 lines so
# Claude can ingest it cheaply.
#
# Usage:
#   bash scripts/lib/jobstatus.sh              # active + recent (24h)
#   bash scripts/lib/jobstatus.sh -j JOBID     # one job, with last log lines
#   bash scripts/lib/jobstatus.sh -n 6         # change recent-completions window (hours)

set -uo pipefail

LOG_DIR="outputs/logs/batch_logs"
EVAL_DIR="outputs/eval_results"
HOURS=24
ONE_JOB=""

while [ $# -gt 0 ]; do
    case "$1" in
        -j) ONE_JOB="$2"; shift 2 ;;
        -n) HOURS="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [-j JOBID] [-n HOURS]"
            exit 0 ;;
        *) shift ;;
    esac
done

# ── helpers ────────────────────────────────────────────────────────────────

# Find newest log file matching a pattern that contains the job id.
_log_for_job() {
    local jid="$1"
    ls -t "$LOG_DIR"/*"_${jid}".log 2>/dev/null | head -1
}

# Pull a one-line progress hint from a log: the last non-empty line, with
# tqdm carriage-return spam collapsed to the final tqdm tick.
_progress() {
    local f="$1"
    [ -z "$f" ] || [ ! -f "$f" ] && return
    # Last 4KB → CR→LF → strip empties → last line.
    tail -c 4096 "$f" 2>/dev/null \
        | tr '\r' '\n' \
        | awk 'NF' \
        | tail -1 \
        | cut -c1-110
}

# ── single-job mode ────────────────────────────────────────────────────────

if [ -n "$ONE_JOB" ]; then
    squeue -j "$ONE_JOB" -o "%.10i %.30j %.20q %.8T %.10M %.10L %R" 2>/dev/null
    echo "--- log tail ($(_log_for_job "$ONE_JOB")) ---"
    log=$(_log_for_job "$ONE_JOB")
    if [ -n "$log" ]; then
        tail -c 8192 "$log" | tr '\r' '\n' | awk 'NF' | tail -10
    else
        echo "(no log file found)"
    fi
    exit 0
fi

# ── default: active jobs + recent completions ─────────────────────────────

echo "=== ACTIVE JOBS ==="
mapfile -t ACTIVE < <(squeue -u "$USER" -h -o "%i|%j|%T|%M|%L|%R")
if [ ${#ACTIVE[@]} -eq 0 ]; then
    echo "(none)"
else
    for row in "${ACTIVE[@]}"; do
        IFS='|' read -r jid name state runtime left reason <<< "$row"
        # Truncate the job name to keep one line readable.
        printf "%-9s %-30s %-9s %s/%s  %s\n" \
            "$jid" "${name:0:30}" "$state" "$runtime" "$left" "$reason"
        if [ "$state" = "RUNNING" ]; then
            log=$(_log_for_job "$jid")
            prog=$(_progress "$log")
            [ -n "$prog" ] && printf "  └─ %s\n" "$prog"
        fi
    done
fi

echo ""
echo "=== RECENT COMPLETIONS (last ${HOURS}h) ==="
# sacct: terminal states; skip steps (".batch", ".extern"); skip RUNNING.
COMPLETED_OUT=$(sacct -u "$USER" -X --starttime="now-${HOURS}hours" \
      -o "JobID,JobName%30,State,Elapsed,ExitCode" -n -P 2>/dev/null \
    | awk -F'|' '
        $3 !~ /RUNNING|PENDING|CONFIGURING/ && $1 !~ /\./ {
            printf "%-9s %-30s %-12s %s  exit=%s\n", $1, $2, $3, $4, $5
        }' \
    | tail -15)
if [ -z "$COMPLETED_OUT" ]; then
    echo "(none)"
else
    echo "$COMPLETED_OUT"
fi

echo ""
echo "=== EVAL RESULTS (modified <${HOURS}h) ==="
if [ -d "$EVAL_DIR" ]; then
    EVAL_FILES=$(find "$EVAL_DIR" -maxdepth 1 -name "*.json" -mmin "-$((HOURS * 60))" 2>/dev/null | head -10)
    if [ -z "$EVAL_FILES" ]; then
        echo "(none)"
    else
        while IFS= read -r f; do
            line=$(python3 - "$f" <<'PY' 2>/dev/null
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    ds, v = next(iter(d.get("results", {}).items()))
    m = v.get("metrics", {})
    print(f"P={m.get('precision', 0):.3f} R={m.get('recall', 0):.3f} "
          f"F1={m.get('f1', 0):.3f} EM={m.get('exact_match', 0):.3f} "
          f"parse={m.get('parse_rate', 0):.3f}")
except Exception:
    print("(unparsable)")
PY
)
            printf "%-65s %s\n" "$(basename "$f")" "$line"
        done <<< "$EVAL_FILES"
    fi
else
    echo "(no $EVAL_DIR)"
fi

exit 0
