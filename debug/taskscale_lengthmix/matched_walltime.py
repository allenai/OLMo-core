"""Matched-wall-clock comparison for outlier, averaging over training seeds.

Sparse trains 1.66x faster per token, so a sparse arm of B tokens costs the same node-hours as a
DENSE arm of only B * (0.00332 / 0.00552) tokens. Dense is interpolated LINEARLY in budget between
its two bracketing measured points -- extrapolating past the largest measured dense budget would
invent the very regime the comparison is trying to test, so bracketed pairs only.

Seed handling: `dense`/`sparse` hold seed 1, `dense_seed3`/`sparse_seed3` hold the replicate. Where
both exist the mean is used and the seed spread is printed, because that spread turns out to be the
same size as the binomial eval noise -- neither can be ignored when calling a difference real.
"""
import json
from math import sqrt

H_DENSE, H_SPARSE = 0.00552, 0.00332       # node-hours per Mtok, measured on H100 (jupiter)
EVAL_SIZE = 600
pts = json.load(open("points.json"))["outlier"]


def series(arm, rung):
    """Merge seed 1 and seed 3 for one arm/rung -> {budget: (mean, spread, n_seeds)}."""
    base = pts.get(arm, {}).get(rung, {})
    rep = pts.get(f"{arm}_seed3", {}).get(rung, {})
    out = {}
    for b in set(base) | set(rep):
        vals = [v[b] for v in (base, rep) if b in v]
        out[int(b)] = (sum(vals) / len(vals), (max(vals) - min(vals)) if len(vals) > 1 else 0.0, len(vals))
    return out


def interp(sd, budget):
    """Linear interpolation of a {budget: (mean, ...)} series; None outside the measured range."""
    bs = sorted(sd)
    if budget < bs[0] or budget > bs[-1]:
        return None
    for lo, hi in zip(bs, bs[1:]):
        if lo <= budget <= hi:
            t = (budget - lo) / (hi - lo)
            return sd[lo][0] + t * (sd[hi][0] - sd[lo][0])
    return sd[bs[-1]][0]


se = sqrt(0.25 / EVAL_SIZE)                # worst-case binomial SE at f1 = 0.5
print(f"2 SE at eval_size={EVAL_SIZE}: +-{2 * se:.3f} per arm, "
      f"+-{2 * se * sqrt(2):.3f} on a difference\n")
for rung in ("8k", "16k", "32k"):
    dense, sparse = series("dense", rung), series("sparse", rung)
    if not dense or not sparse:
        continue
    print(f"== {rung}")
    for b in sorted(sparse):
        hours = b / 1e6 * H_SPARSE
        b_dense = hours / H_DENSE * 1e6
        dv = interp(dense, b_dense)
        if dv is None:
            continue
        sv, spread, nseed = sparse[b]
        diff = sv - dv
        seeds = f" [{nseed} seeds, spread {spread:.3f}]" if nseed > 1 else ""
        lead = "sparse" if diff > 0 else "dense "
        sig = "" if abs(diff) < 2 * se * sqrt(2) else "  <-- outside 2 SE"
        print(f"  {hours:5.2f}h  sparse@{b/1e6:.0f}M {sv:.3f}  vs  dense@{b_dense/1e6:.0f}M {dv:.3f}"
              f"   {lead} +{abs(diff):.3f}{sig}{seeds}")
    print()
