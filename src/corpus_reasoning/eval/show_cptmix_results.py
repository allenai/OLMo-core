"""Print a comparison table of all CPT-mix sweep eval results vs the CPT baseline.
Usage: python scripts/eval/show_cptmix_results.py [outputs/eval_results]"""
import glob
import json
import os
import sys

D = sys.argv[1] if len(sys.argv) > 1 else "outputs/eval_results"
rows = []
for f in sorted(glob.glob(os.path.join(D, "*_lc.json"))):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    name = os.path.basename(f).replace("_lc.json", "")
    rul = d.get("ruler_avg_recall")
    c = d.get("contradiction") or {}
    nq = d.get("nq") or {}
    rows.append((name, rul, c.get("f1"), c.get("exact_match"), nq.get("f1")))

# per-model base: 4B rows compare to q4b-cpt-base, 0.6B rows to q06b-cpt-base
def model_key(n):
    return "q06b" if n.startswith("q06b") else "q4b"
bases = {model_key(r[0]): r[1] for r in rows if "cpt-base" in r[0]}
print(f"{'config':<34} {'RULER':>7} {'dRULER':>7} {'contra_f1':>10} {'contra_em':>10} {'nq_f1':>7}")
print("-" * 80)
for name, rul, f1, em, nqf1 in rows:
    b_rul = bases.get(model_key(name))
    drul = f"{rul-b_rul:+.3f}" if (rul is not None and b_rul is not None) else "  -  "
    rul_s = f"{rul:.3f}" if rul is not None else "  -  "
    f1_s = f"{f1:.3f}" if f1 is not None else "  -  "
    em_s = f"{em:.3f}" if em is not None else "  -  "
    nq_s = f"{nqf1:.3f}" if nqf1 is not None else "  -  "
    print(f"{name:<34} {rul_s:>7} {drul:>7} {f1_s:>10} {em_s:>10} {nq_s:>7}")
print("\nGoal: contra_f1 + nq_f1 UP while dRULER >= ~0 (no degradation).")
