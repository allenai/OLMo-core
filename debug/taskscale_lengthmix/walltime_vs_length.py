"""Where does sparse win on TRAINING TIME, as a function of the training context length?

Our arms all trained at a token-weighted mean length of roughly 4k-9k, and there the measured
speedup is a flat 1.66x -- too small a lever to overturn sparse's data disadvantage on most tasks.
But that ratio is not a constant: dense pays attention proportional to L per token while sparse
pays ~L/64, so the advantage grows with the training mix's length. This extrapolates that.

RATE MODEL, anchored on measurement rather than assumed:

    h_per_Mtok(variant, L) = c_params + c_attn * L * f        f = 1 dense, 1/64 sparse

  Two measured numbers pin both constants at the length we actually ran (L0 = 6659):
    dense  0.00552 h/Mtok, sparse 0.00332 h/Mtok  (medians of per-task job-duration fits)
  The gap is the dense attention term, since sparse's is 1/64 of it:
    c_attn = (0.00552 - 0.00332) / (L0 * (1 - 1/64))      c_params = 0.00552 - c_attn * L0

⚠ The LENGTH dependence is a model, not a measurement. Every arm in this campaign trained at
4k-9k, so the data pins the rate at one length and says nothing about its slope; the L-scaling
comes from the standard quadratic-attention argument. It also assumes the 1/64 landmark ratio holds
at long context and ignores memory pressure, which at 1M context is a heroic assumption. Treat the
long-L columns as the shape of the argument.

THE ACTUAL QUESTION. Sparse needs some factor R more tokens than dense to reach the same score --
measured per task, and the campaign's central negative result. Sparse is cheaper in TIME exactly
when speedup(L) > R. So each task has a break-even training length, and tasks where sparse never
reaches the target have none at any length.
"""
import json

L0 = 6659.0
H_DENSE_L0, H_SPARSE_L0 = 0.00552, 0.00332
SPARSE_FRAC = 1.0 / 64.0
C_ATTN = (H_DENSE_L0 - H_SPARSE_L0) / (L0 * (1 - SPARSE_FRAC))
C_PARAMS = H_DENSE_L0 - C_ATTN * L0


def rate(variant, L):
    f = 1.0 if variant == "dense" else SPARSE_FRAC
    return C_PARAMS + C_ATTN * L * f


def speedup(L):
    return rate("dense", L) / rate("sparse", L)


def token_ratio(task, rung, points):
    """Measured tokens sparse needs / tokens dense needs, at the best score BOTH variants reach."""
    d = points[task].get("dense", {}).get(rung)
    s = points[task].get("sparse", {}).get(rung)
    if not d or not s or len(d) < 2 or len(s) < 2:
        return None, None
    dv = {float(b): v for b, v in d.items()}
    sv = {float(b): v for b, v in s.items()}
    target = min(max(dv.values()), max(sv.values()))
    # A "ratio" between two floor-level scores is noise dressed as a measurement.
    if target < 0.15:
        return None, None

    def budget_at(pts, t):
        xs = sorted(pts)
        # If the variant was already above the target at its SMALLEST budget, we never observed it
        # below the target and its cost is unmeasured -- the same guard the length-law fitter uses.
        # Without it, absence (dense pinned at .98+) produced negative token ratios.
        if pts[xs[0]] >= t:
            return None
        for i in range(1, len(xs)):
            if pts[xs[i]] >= t:
                x0, x1, y0, y1 = xs[i - 1], xs[i], pts[xs[i - 1]], pts[xs[i]]
                return x0 + (t - y0) / max(y1 - y0, 1e-9) * (x1 - x0) if y1 > y0 else xs[i]
        return None
    bd, bs = budget_at(dv, target), budget_at(sv, target)
    if not bd or not bs:
        return None, None
    return bs / bd, target


def main():
    pts = json.load(open("debug/taskscale_lengthmix/points.json"))
    print(f"rate model: c_params={C_PARAMS:.6f} h/Mtok, c_attn={C_ATTN:.3e} h/Mtok per token of L")
    print(f"anchored at L={L0:.0f}: dense {rate('dense', L0):.5f}, sparse {rate('sparse', L0):.5f}, "
          f"speedup {speedup(L0):.2f}x\n")
    print("speedup vs training context length (MODEL, anchored at one measured length):")
    for L in (2048, 8192, 32768, 131072, 262144, 1048576):
        print(f"   L={L:>8d}   dense {rate('dense', L):8.4f}   sparse {rate('sparse', L):8.4f}   "
              f"speedup {speedup(L):6.1f}x")

    print("\nper task: tokens sparse needs / dense needs at the best shared score, and the training")
    print("length at which sparse's speed advantage would pay for it")
    print(f"\n{'task':14s} {'rung':>4s} {'score':>6s} {'R (token ratio)':>16s}  break-even L")
    for task in sorted(pts):
        for rung in sorted(pts[task].get("dense", {}), key=lambda r: int(r.rstrip("k"))):
            R, target = token_ratio(task, rung, pts)
            if R is None:
                continue
            need = None
            for L in range(1024, 4_000_000, 1024):
                if speedup(L) >= R:
                    need = L
                    break
            note = (f"{need / 1024:.0f}k" if need else ">4M (never)")
            if R <= speedup(L0):
                note += "  <- already cheaper at our training length"
            print(f"{task:14s} {rung:>4s} {target:6.3f} {R:16.2f}  {note}")


if __name__ == "__main__":
    main()
