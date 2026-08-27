"""List memory-networks runs with throughput, to compare sparse-landmark vs dense wall-clock."""
import sys
import wandb

api = wandb.Api(timeout=60)
runs = api.runs("prasann-uc-berkeley-electrical-engineering-computer-sciences/memory-networks", order="-created_at", per_page=200)
rows = []
for i, r in enumerate(runs):
    if i >= 400:
        break
    name = r.name or ""
    s = r.summary
    tps = None
    for key in ("throughput/device/tokens_per_second", "throughput (device)/tokens per second",
                "throughput/device/TPS"):
        if key in s:
            tps = s[key]
            break
    # fall back: scan summary keys
    if tps is None:
        for k in s.keys():
            if "tokens_per_second" in k or ("throughput" in k and "second" in k):
                tps = s[k]
                break
    bps = None
    for k in s.keys():
        if "batches_per_second" in k or "steps_per_second" in k:
            bps = s[k]
            break
    rows.append((r.created_at, name, r.state, tps, bps, r.id))

for row in rows:
    print("\t".join(str(x) for x in row))
