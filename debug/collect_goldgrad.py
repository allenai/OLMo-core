"""Per-arm CE trajectories for the goldgrad n=100 run (arms are appended to one slurm log)."""
import glob, re

path = sorted(glob.glob("/scratch/users/prasann/attn_explore_logs/goldgrad_gg-n100_*.log"))[-1]
mode, ce, extra = None, {}, {}
for line in open(path, errors="ignore"):
    m = re.search(r"=== \[(\w+)\]", line)
    if m:
        mode = m.group(1)
        ce.setdefault(mode, [])
    m2 = re.search(r"train/CE loss=([0-9.]+)", line)
    if m2 and mode:
        ce[mode].append(float(m2.group(1)))
    m3 = re.search(r"\[goldgrad\](.*)", line)
    if m3 and mode:
        extra.setdefault(mode, []).append(m3.group(1).strip())

print(f"log: {path}\n")
for k, v in ce.items():
    if not v:
        continue
    pts = " ".join(f"{v[int(i * len(v) / 6)]:.3f}" for i in range(6))
    print(f"  {k:18s} n={len(v):3d} | {pts}  FINAL={v[-1]:.4f}")
    for e in extra.get(k, [])[:3]:
        print(f"      {e}")
