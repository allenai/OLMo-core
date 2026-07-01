"""Print a compact, human-readable sample of an eval generations sidecar to stdout.

Used by the on-node eval runner so a few real (prompt tail, generation, gold/pred/metric) triples
land in the Beaker job log for quick error inspection without pulling the sidecar off weka.

Usage: python print_gen_sample.py <generations.jsonl> [N]
"""
import json
import sys

path = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 6

print("=== SAMPLE GENERATIONS (first %d) from %s ===" % (n, path))
with open(path) as fh:
    for i, line in enumerate(fh):
        if i >= n:
            break
        r = json.loads(line)
        d = r.get("detail") or {}
        metric = {k: d.get(k) for k in ("f1", "exact_match", "recall") if k in d}
        print("--- [%s@%s #%s] metric=%s gold=%s pred=%s ---"
              % (r.get("task"), r.get("rung"), r.get("idx"), metric,
                 d.get("gold_ids"), d.get("predicted_ids")))
        tail = (r.get("prompt_tail") or "")[-220:].replace("\n", " ")
        print("  PROMPT_TAIL: ...%s" % tail)
        print("  GENERATION : %r" % (r.get("generation") or "")[:400])
