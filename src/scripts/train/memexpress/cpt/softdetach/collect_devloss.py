"""Assemble the soft-detach CPT table from Beaker logs (weka is not mounted on the login node):
per run -> training PF (flop_meter, last logged) + dev loss (eval SUMMARY line). Writes
softdetach_devloss.csv next to the ledgers and prints arm x budget -> CE / PF.

    python src/scripts/train/memexpress/cpt/softdetach/collect_devloss.py
"""
import csv, os, re, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))


def logs(exp):
    r = subprocess.run(["timeout", "90", "beaker", "experiment", "logs", exp], capture_output=True, text=True)
    return r.stdout


def train_rows():
    out = {}
    for row in csv.DictReader(open(f"{HERE}/LAUNCH_LEDGER.tsv"), delimiter="\t"):
        if row["state"] != "LAUNCHED":
            continue
        exp = row["note"].split()[0] if row["note"] else ""
        if not re.fullmatch(r"[A-Z0-9]{26}", exp):
            continue
        out[row["run"]] = dict(arm=row["arm"], budget=row["budget"], train_exp=exp)
    return out


def main():
    runs = train_rows()
    for run, r in runs.items():
        L = logs(r["train_exp"])
        # the FLOP meter's run totals are in the wandb summary block at the end of a FINISHED log
        # (same regex as debug/ds64/collect_ds64.py); the console's throughput/total petaflops is
        # dense-priced and truncated, never use it (records/ds64-handoff.md, flop-meter memory)
        pf = re.findall(r"wandb:\s+flop_meter/dense_pflops\s+([\d.]+)", L)
        act = re.findall(r"wandb:\s+flop_meter/actual_pflops\s+([\d.]+)", L)
        dense_eq = re.findall(r"wandb:\s+flop_meter/actual_over_dense\s+([\d.]+)", L)
        steps = re.findall(r"step=(\d+)/(\d+)", L)
        r.update(dense_pf=float(pf[-1]) if pf else None, actual_pf=float(act[-1]) if act else None,
                 actual_over_dense=float(dense_eq[-1]) if dense_eq else None,
                 steps=f"{steps[-1][0]}/{steps[-1][1]}" if steps else None)
    if os.path.exists(f"{HERE}/EVAL_LEDGER.tsv"):
        for line in open(f"{HERE}/EVAL_LEDGER.tsv"):
            f = line.rstrip("\n").split("\t")
            if len(f) < 4 or f[0] != "EVAL" or f[1] not in runs:
                continue
            L = logs(f[3])
            m = re.search(r"SUMMARY \w+: (.*)", L)
            if m:
                runs[f[1]].update({k: float(v) for k, v in re.findall(r"(\w+)=([0-9.]+)", m.group(1))}, eval_exp=f[3])
    keys = ["run", "arm", "budget", "steps", "dense_pf", "actual_pf", "actual_over_dense", "full_ce", "tail20_ce", "own_ce", "own_compaction", "train_exp", "eval_exp"]
    with open(f"{HERE}/softdetach_devloss.csv", "w") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); w.writeheader()
        for run in sorted(runs):
            w.writerow(dict(run=run, **runs[run]))
    print(f"{'run':30} {'steps':>8} {'densePF':>8} {'actPF':>8} {'full':>7} {'tail20':>7} {'own':>7} {'x':>5}")
    for run in sorted(runs, key=lambda k: (runs[k]['arm'], int(runs[k]['budget'][:-1]))):
        r = runs[run]
        g = lambda k, f="{:7.3f}": (f.format(r[k]) if r.get(k) is not None else f"{'—':>7}")
        print(f"{run:30} {r.get('steps') or '—':>8} {g('dense_pf','{:8.1f}')} {g('actual_pf','{:8.1f}')} {g('full_ce')} {g('tail20_ce')} {g('own_ce')} {g('own_compaction','{:5.2f}')}")


if __name__ == "__main__":
    main()
