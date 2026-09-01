"""Average the measured ladders by CTC class (high vs low corpus-traversal cost).

Class assignment follows suite_table.json: O_T(N) = low, O_T(NM) / O_T(N^2)+ = high.
Each task contributes its LARGEST measured budget, so the average answers "at the most data any of
these tasks got, how does each class stand?" -- not a matched-budget comparison, since the budgets
differ per task by design (a task's pool cap sets its ceiling).
"""
import json
import sys

CLASS = {
    "outlier": "high", "qdmatch_nq": "high", "contradiction": "high", "xabsence": "high",
    "grouping": "high", "reorder": "high", "textgroups": "high",
    "nq": "low", "oolong": "low", "absence": "low",
}
RUNGS = ["2k", "4k", "8k", "16k", "32k"]

data = json.load(open(sys.argv[1] if len(sys.argv) > 1
                     else "debug/taskscale_lengthmix/points.json"))

for variant in ("dense", "sparse"):
    print(f"\n=== {variant}  (each task at its largest measured budget)")
    print(f"{'rung':>5s}  {'high-CTC':>22s}  {'low-CTC':>22s}")
    for rung in RUNGS:
        rows = {"high": [], "low": []}
        for task, variants in data.items():
            cls = CLASS.get(task)
            pts = variants.get(variant, {}).get(rung)
            if not cls or not pts:
                continue
            rows[cls].append((task, pts[max(pts, key=lambda b: float(b))]))
        cells = []
        for cls in ("high", "low"):
            v = [s for _, s in rows[cls]]
            cells.append(f"{sum(v) / len(v):.3f} (n={len(v)})" if v else "--")
        if any(c != "--" for c in cells):
            print(f"{rung:>5s}  {cells[0]:>22s}  {cells[1]:>22s}"
                  f"   high={[t for t, _ in rows['high']]} low={[t for t, _ in rows['low']]}")

print("\n=== sparse / dense ratio, same tasks, same rung (only tasks with BOTH measured)")
for rung in RUNGS:
    rows = {"high": [], "low": []}
    for task, variants in data.items():
        cls = CLASS.get(task)
        d = variants.get("dense", {}).get(rung)
        s = variants.get("sparse", {}).get(rung)
        if not cls or not d or not s:
            continue
        dv = d[max(d, key=lambda b: float(b))]
        sv = s[max(s, key=lambda b: float(b))]
        if dv > 0.02:
            rows[cls].append((task, sv / dv))
    cells = []
    for cls in ("high", "low"):
        v = [r for _, r in rows[cls]]
        cells.append(f"{sum(v) / len(v):.2f} (n={len(v)})" if v else "--")
    if any(c != "--" for c in cells):
        print(f"{rung:>5s}  high {cells[0]:>14s}   low {cells[1]:>14s}"
              f"   high={[(t, round(r, 2)) for t, r in rows['high']]}"
              f" low={[(t, round(r, 2)) for t, r in rows['low']]}")
