"""Scale-resolved WALL-CLOCK ceiling for head gating + static FFN deletion + block skip.

`ceiling.py` prices the routers in FLOPs. FLOPs are not the constraint and they are not even the
right ordering: measured end to end at 4B/16k, adding the KV router to the winning stack RAISES the
FLOP cut (10.4x -> 11.7x) and LOWERS the speed (5.590x -> 5.387x). This script builds the ceiling
in step seconds, in the order the hardware actually rewards:

    head gating  ->  static deletion of always-null FFN layers  ->  compacted block skip

with the KV router carried as the measured NEGATIVE increment it is.

MEASURED INPUTS
  1. `bench_step_partition_{scale}*.json` -- cumulative-ablation partition of a real fwd+bwd step
     into head-waste / FFN / attention / GDN / floor, summing to 100% by construction.
     0.8B / 4B / 9B / 27B, 2k-128k, on both the plain and the activation-checkpointed basis.
  2. Three end-to-end router matrices, one per anchor cell -- `router_matrix_4B_16384_v3.json`
     (plain basis), `router_matrix_9B_8192_ac_v2.json` and `router_matrix_27B_8192_ac.json`
     (activation-checkpointed basis). Each carries "drops nothing" arms at keep=1.0 that isolate
     pure machinery cost, and the stacked arms this model has to reproduce.
  3. `rungs_v1*-t0*.json` -- what the trained routers actually do. They LAYER-PRUNE: 13 of 36
     layers send >99% of tokens to the null rung at target c=0.05 and 22 of 36 at c=0.02. Those
     layers get deleted, not routed, which is why the recommended stack carries no FFN-router
     overhead at all.

THE MODEL
  Component times come from FLOPs / effective-throughput, calibrated per scale against the
  measured partition and interpolated in log-context between anchors (held flat outside the
  measured range). Two components are not FLOP-like and are fitted affine in T instead: `floor`
  (embeddings, residual adds, final norm, zero_grad, inter-kernel gaps, and the surviving head --
  untouchable, so `1 / floor_share` is the hard cap on this whole line of work) and `head_waste`
  (measured nearly constant in T, because the fused-linear-CE backward touches a full
  `d_model x vocab` fp32 accumulator whether one row survives or a million).

  Block skip is modelled with the attention SCORE term falling as `skip**2` and everything else as
  `skip`, because the compacted path runs attention over a genuinely shorter sequence rather than a
  masked one. `validate()` checks the whole composition against six measured arms per anchor. It
  comes out 8-17% CONSERVATIVE, so the table is a floor on the achievable number.

    python research/kv_route/flexcompute_40x/scale_ceiling.py
"""

import glob
import json
import math
import os
from typing import Dict, List, Tuple

from olmo_core.nn.attention import Attention
from olmo_core.nn.transformer import TransformerConfig

OUT = os.path.dirname(os.path.abspath(__file__))
VOCAB = 248320
SCALES = {
    "0.8B": TransformerConfig.qwen3_5_0_8B,
    "4B": TransformerConfig.qwen3_5_4B,
    "9B": TransformerConfig.qwen3_5_9B,
    "27B": TransformerConfig.qwen3_5_27B,
}
GRID = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
FLOP_COMPONENTS = ("ffn", "attn", "gdn")
PARTS = ("head_waste", "ffn", "attn", "gdn", "floor")

# (file, scale, context, activation-checkpointed). The 4B anchor is the only one on the plain
# basis; the other two only fit under checkpointing, which is why their numbers are not directly
# comparable to 4B's -- see `basis_artefact` below.
ANCHORS = [
    ("router_matrix_4B_16384_v4.json", "4B", 16384, False),
    ("router_matrix_9B_8192_ac_v3.json", "9B", 8192, True),
    ("router_matrix_27B_8192_ac_v2.json", "27B", 8192, True),
]
# Same model, same context, same arms -- but the sequence packed into 32 documents. Packing already
# makes attention cheap, so it roughly halves the prize. Anyone planning a real run needs this
# number, not the unpacked one.
PACKED = ("router_matrix_4B_16384_v3_packed32.json", "4B", 16384)
BEST_ARM = "staticFFN+bskipK-0.25+headgate"
WITHKV_ARM = "staticFFN+kv0.25+bskipC-0.25+headgate"
SKIP_KEEP = 0.25  # compacted block-skip survival fraction, the setting every anchor measured


def buckets(model, L: int) -> Dict[str, float]:
    """Per-token TRAINING FLOPs (fwd+bwd) split into the buckets the ablation can see."""
    ffn = gdn = proj = score = 0.0
    for b in model.blocks.values():
        ffn += b.feed_forward.num_flops_per_token(L)
        a = b.attention
        if type(a) is Attention:
            p = a.num_flops_per_token(0)
            proj += p
            score += a.num_flops_per_token(L) - p
        else:
            gdn += a.num_flops_per_token(L)
    head = float(model.lm_head.num_flops_per_token(L))
    return dict(
        ffn=ffn,
        gdn=gdn,
        proj=proj,
        score=score,
        attn=proj + score,
        head=head,
        total=ffn + gdn + proj + score + head,
    )


def raw_partitions() -> Dict[Tuple[str, bool], Dict[int, dict]]:
    """Every measured partition row, keyed by (scale, activation-checkpointed) then context."""
    out: Dict[Tuple[str, bool], Dict[int, dict]] = {}
    for path in sorted(glob.glob(os.path.join(OUT, "bench_step_partition_*.json"))):
        for r in json.load(open(path)):
            if "error" in r or "full_s" not in r:
                continue
            out.setdefault((r["scale"], bool(r.get("ac", False))), {})[r["tokens"]] = r
    return out


def ac_correction(raw) -> Dict[str, float]:
    """MEASURED per-component plain/AC time ratio, median over every cell carrying both."""
    ratios: Dict[str, List[float]] = {k: [] for k in PARTS}
    for scale in {s for s, _ in raw}:
        ac, plain = raw.get((scale, True), {}), raw.get((scale, False), {})
        for T in sorted(set(ac) & set(plain)):
            for k in PARTS:
                if ac[T][f"{k}_s"] > 0:
                    ratios[k].append(plain[T][f"{k}_s"] / ac[T][f"{k}_s"])
    return {k: (sorted(v)[len(v) // 2] if v else 1.0) for k, v in ratios.items()}


def partition_rows(scale: str, raw, corr, ac_basis: bool = False) -> List[dict]:
    """Rows on one basis: measured where they exist, converted from the other where they do not."""
    want, other = (True, False) if ac_basis else (False, True)
    inv = {k: (1 / v if ac_basis else v) for k, v in corr.items()}
    rows: Dict[int, dict] = {}
    for T, r in raw.get((scale, other), {}).items():
        conv = {f"{k}_s": r[f"{k}_s"] * inv[k] for k in PARTS}
        step = sum(conv.values())
        rows[T] = dict(
            tokens=T,
            scale=scale,
            full_s=step,
            basis="converted",
            peak_gib=r.get("peak_gib"),
            **conv,
            **{f"{k}_pct": 100 * conv[f"{k}_s"] / step for k in PARTS},
        )
    for T, r in raw.get((scale, want), {}).items():
        rows[T] = dict(r, basis="measured")
    return [rows[t] for t in sorted(rows)]


def affine_fit(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    """Least-squares ``y = a + b*x``; with one point, a flat line through it."""
    n = len(xs)
    if n == 1:
        return ys[0], 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    b = 0.0 if den == 0 else sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return my - b * mx, b


def interp_log(anchors: List[Tuple[float, float]], x: float) -> float:
    """Piecewise-linear in ``log2(x)``, HELD FLAT outside the measured range.

    Extrapolating an efficiency curve past where it was measured is how a modelled cell turns into
    a fabricated one; clamping keeps the error in the conservative direction for long context.
    """
    pts = sorted(anchors)
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            w = (math.log2(x) - math.log2(x0)) / (math.log2(x1) - math.log2(x0))
            return y0 + w * (y1 - y0)
    return pts[-1][1]


def arm_table(path: str) -> Dict[str, float]:
    return {r["arm"]: r["wall_speedup"] for r in json.load(open(path))["rows"]}


def coefficients(anchor_shares: Dict[str, Dict[str, float]], arms: Dict[str, Dict[str, float]]):
    """Derive the three stack coefficients from the anchors, MEASURED, one value per anchor.

    * ``p_static`` -- the fraction of the FFN component that static layer deletion actually
      removes, backed out of the ``ffn-static`` arm rather than assumed from the rung histogram.
    * ``ov_skip_compact`` -- what the compacted gather/scatter costs, from ``bskipC-1.00``, which
      removes no FLOPs at all.
    * ``kv_factor`` -- the KV router's effect when ADDED to the winning stack. It is below 1 at
      every anchor: it raises the FLOP cut and lowers the speed.
    """
    per: Dict[str, Dict[str, float]] = {}
    for scale, sh in anchor_shares.items():
        a = arms[scale]
        block = sh["ffn"] + sh["attn"] + sh["gdn"]
        best, withkv = a[BEST_ARM], a[WITHKV_ARM]
        per[scale] = {
            "p_static": (1 - 1 / a["ffn-static"]) / sh["ffn"],
            "ov_skip_compact": (1 / a["bskipC-1.00"] - 1) / block,
            "kv_factor": withkv / best,
        }

    def med(key):
        vals = [v[key] for v in per.values()]
        return sorted(vals)[len(vals) // 2]

    # An earlier revision of this file had to exclude 27B's `ffn-static` arm, which measured
    # exactly 1.000x. That was a real bug, not a bad measurement: under activation checkpointing
    # the model's own `CheckpointWrapper` forwards attribute READS to the wrapped block but not
    # WRITES, so assigning to `wrapper.feed_forward` registered a new submodule on the wrapper and
    # the inner block kept running its real FFN. The static-FFN arm therefore measured nothing at
    # 9B and 27B. Fixed upstream; all three anchors now carry a real value. (This is the same trap
    # `AcBlock` in `bench_step_partition.py` was written to avoid -- see its docstring.)
    return {
        "p_static": med("p_static"),
        "ov_skip_compact": med("ov_skip_compact"),
        "kv_factor": med("kv_factor"),
        "per_anchor": per,
    }


def stack(sh: Dict[str, float], score_frac: float, co: Dict[str, float], skip: float = SKIP_KEEP):
    """Cumulative stack in the order the hardware rewards. Returns TIMES, not speedups."""
    score = sh["attn"] * score_frac
    proj = sh["attn"] - score
    block = sh["ffn"] + sh["attn"] + sh["gdn"]
    t0 = 1.0
    t1 = t0 - sh["head_waste"]  # head gating: exact, zero overhead
    t2 = t1 - sh["ffn"] * co["p_static"]  # static FFN deletion: no router, no prologue
    nonscore = sh["ffn"] * (1 - co["p_static"]) + proj + sh["gdn"]
    # Compaction runs attention over a genuinely shorter sequence, so the quadratic score term
    # falls as skip**2 while everything else falls as skip.
    t3 = sh["floor"] + skip * nonscore + skip**2 * score + co["ov_skip_compact"] * block
    t4 = t3 / co["kv_factor"]  # KV router: measured NEGATIVE increment
    return {
        "dense": t0,
        "head": t1,
        "head_static": t2,
        "head_static_skip": t3,
        "head_static_skip_kv": t4,
    }


def validate(anchor_shares, arms, score_fracs, co):
    """Reproduce the measured arms at every anchor, on that anchor's own basis."""
    out: Dict[str, Dict[str, dict]] = {}
    for scale, sh in anchor_shares.items():
        a, sf = arms[scale], score_fracs[scale]
        score, proj = sh["attn"] * sf, sh["attn"] * (1 - sf)
        block = sh["ffn"] + sh["attn"] + sh["gdn"]
        ov, p = co["ov_skip_compact"], co["p_static"]

        def sk(keep, ffn_share, head):
            t = sh["floor"] + keep * (ffn_share + proj + sh["gdn"]) + keep**2 * score + ov * block
            return 1 / (t + (0.0 if head else sh["head_waste"]))

        pred = {
            "headgate": 1 / (1 - sh["head_waste"]),
            "ffn-static": 1 / (1 - sh["ffn"] * p),
            "bskipC-0.25": sk(0.25, sh["ffn"], False),
            "staticFFN+bskipC-0.25": sk(0.25, sh["ffn"] * (1 - p), False),
            BEST_ARM: sk(0.25, sh["ffn"] * (1 - p), True),
        }
        pred[WITHKV_ARM] = pred[BEST_ARM] * co["kv_factor"]
        out[scale] = {
            k: {
                "predicted": v,
                "measured": a.get(k),
                "rel_err": (v / a[k] - 1) if a.get(k) else None,
            }
            for k, v in pred.items()
        }
    return out


def calibrate(model, rows) -> dict:
    """Effective throughput per FLOP-like component, plus the two affine fits."""
    measured_T = [r["tokens"] for r in rows]
    tf: Dict[str, List[Tuple[float, float]]] = {c: [] for c in FLOP_COMPONENTS}
    for r in rows:
        b = buckets(model, r["tokens"])
        for c in FLOP_COMPONENTS:
            tf[c].append((r["tokens"], b[c] * r["tokens"] / r[f"{c}_s"] / 1e12))
    return {
        "tf": tf,
        "head_waste": affine_fit(measured_T, [r["head_waste_s"] for r in rows]),
        "floor": affine_fit(measured_T, [r["floor_s"] for r in rows]),
    }


def predict(model, cal: dict, T: int) -> Dict[str, float]:
    """Component times of one step at context ``T`` under a calibration."""
    b = buckets(model, T)
    t = {c: b[c] * T / (interp_log(cal["tf"][c], T) * 1e12) for c in FLOP_COMPONENTS}
    for k in ("head_waste", "floor"):
        a, slope = cal[k]
        t[k] = max(a + slope * T, 0.0)
    return t


def leave_one_out(model, rows) -> Dict[int, dict]:
    """Refit without each measured cell and predict it -- the only honest check on modelled cells."""
    res = {}
    for i, r in enumerate(rows):
        rest = rows[:i] + rows[i + 1 :]
        if len(rest) < 2:
            continue
        pred = sum(predict(model, calibrate(model, rest), r["tokens"]).values())
        res[r["tokens"]] = {
            "interior": 0 < i < len(rows) - 1,
            "measured_step_s": r["full_s"],
            "predicted_step_s": pred,
            "step_rel_err": pred / r["full_s"] - 1,
        }
    return res


def main():
    models = {k: f(vocab_size=VOCAB).build(init_device="meta") for k, f in SCALES.items()}
    raw = raw_partitions()
    corr = ac_correction(raw)
    plain = {s: partition_rows(s, raw, corr, ac_basis=False) for s in SCALES}
    acbas = {s: partition_rows(s, raw, corr, ac_basis=True) for s in SCALES}

    arms, anchor_shares, score_fracs = {}, {}, {}
    for fname, scale, ctx, is_ac in ANCHORS:
        arms[scale] = arm_table(os.path.join(OUT, fname))
        row = raw[(scale, is_ac)][ctx]
        anchor_shares[scale] = {k: row[f"{k}_pct"] / 100 for k in PARTS}
        b = buckets(models[scale], ctx)
        score_fracs[scale] = b["score"] / b["attn"]
    co = coefficients(anchor_shares, arms)
    val = validate(anchor_shares, arms, score_fracs, co)

    out: Dict[str, dict] = {
        "_ac_correction": corr,
        "_coefficients": {k: v for k, v in co.items() if k != "per_anchor"},
        "_coefficients_per_anchor": co["per_anchor"],
        "_anchors": {
            s: {"ctx": c, "ac_basis": a, "file": f, "arms": arms[s]} for f, s, c, a in ANCHORS
        },
        "_validation": val,
        "_packed": {
            "file": PACKED[0],
            "scale": PACKED[1],
            "ctx": PACKED[2],
            "arms": arm_table(os.path.join(OUT, PACKED[0])),
        },
        "_launch_bound": {
            # Reported by the router-matrix runs: past ~5x the step stops being GPU-bound.
            "gpu_idle_dense_pct": 0.9,
            "gpu_idle_best_arm_pct": 12.6,
            "cuda_graph_headroom": 1.14,
        },
    }

    basis_artefact = {}
    for scale, model in models.items():
        rows = plain[scale]
        if len(rows) < 2:
            continue
        cal = {"plain": calibrate(model, rows), "ac": calibrate(model, acbas[scale])}
        basis = {r["tokens"]: r["basis"] for r in rows}
        ac_basis = {r["tokens"]: r["basis"] for r in acbas[scale]}

        per_len = {}
        for T in GRID:
            b = buckets(model, T)
            sf = b["score"] / b["attn"] if b["attn"] else 0.0
            cell = {}
            for name in ("plain", "ac"):
                t = predict(model, cal[name], T)
                step = sum(t.values())
                sh = {k: v / step for k, v in t.items()}
                st = stack(sh, sf, co)
                cell[name] = {
                    "step_s": step,
                    "wall_shares": sh,
                    "amdahl_cap": 1 / sh["floor"],
                    "stages": {k: 1 / v for k, v in st.items() if k != "dense"},
                    "marginal": {
                        "head": st["dense"] / st["head"],
                        "static_ffn": st["head"] / st["head_static"],
                        "block_skip": st["head_static"] / st["head_static_skip"],
                        "kv": st["head_static_skip"] / st["head_static_skip_kv"],
                    },
                }
            per_len[T] = {
                **cell["plain"],
                "ac": cell["ac"],
                "basis": basis.get(T, "modelled"),
                "ac_basis": ac_basis.get(T, "modelled"),
                "flop_shares": {k: b[k] / b["total"] for k in ("ffn", "gdn", "proj", "score")},
            }
        basis_artefact[scale] = {
            T: per_len[T]["ac"]["stages"]["head_static_skip"]
            / per_len[T]["stages"]["head_static_skip"]
            for T in GRID
        }
        out[scale] = {
            "params_b": sum(p.numel() for p in model.parameters()) / 1e9,
            "basis_by_context": basis,
            "peak_gib": {r["tokens"]: r.get("peak_gib") for r in acbas[scale]},
            "leave_one_out": leave_one_out(model, rows),
            "by_context": per_len,
        }
    out["_basis_artefact"] = basis_artefact

    with open(os.path.join(OUT, "scale_ceiling.json"), "w") as fh:
        json.dump(out, fh, indent=1)

    print(
        "# plain/AC per-component correction (measured): "
        + ", ".join(f"{k} {v:.3f}" for k, v in corr.items())
    )
    print(
        f"# coefficients (measured): static-FFN removes {100*co['p_static']:.1f}% of the FFN "
        f"component, compacted block skip costs {100*co['ov_skip_compact']:.1f}% of block work, "
        f"KV added to the winning stack multiplies it by {co['kv_factor']:.3f}"
    )
    for s, v in co["per_anchor"].items():
        print(f"    {s:>4}: " + "  ".join(f"{k}={x:.4f}" for k, x in v.items()))
    print("\n# composition vs measured arms, each on its own basis:")
    for scale, d in val.items():
        print(f"  --- {scale}")
        for k, v in d.items():
            if v["measured"]:
                print(
                    f"    {k:>40}  measured {v['measured']:.3f}x  model {v['predicted']:.3f}x"
                    f"  {100*v['rel_err']:+.1f}%"
                )

    for scale in SCALES:
        if scale not in out:
            continue
        d = out[scale]
        print(f"\n### {scale}  ({d['params_b']:.2f}B params)   [plain basis]")
        print(
            f"\n{'ctx':>8} {'basis':>10} {'step s':>8} {'floor%':>7} {'hw%':>6} {'ffn%':>6} "
            f"| {'head':>7} {'+static':>8} {'+skip':>8} {'+KV':>8} | {'cap':>7} {'AC art':>7}"
        )
        for T in GRID:
            r = d["by_context"][T]
            w, st = r["wall_shares"], r["stages"]
            print(
                f"{T:>8} {r['basis']:>10} {r['step_s']:>8.3f} {100*w['floor']:>7.1f} "
                f"{100*w['head_waste']:>6.1f} {100*w['ffn']:>6.1f} | {st['head']:>6.2f}x "
                f"{st['head_static']:>7.2f}x {st['head_static_skip']:>7.2f}x "
                f"{st['head_static_skip_kv']:>7.2f}x | {r['amdahl_cap']:>6.2f}x "
                f"{basis_artefact[scale][T]:>6.2f}x"
            )
        loo = d["leave_one_out"]
        if loo:
            errs = [abs(v["step_rel_err"]) for v in loo.values()]
            inner = [abs(v["step_rel_err"]) for v in loo.values() if v["interior"]]
            print(
                f"  leave-one-out on {len(loo)} cells: median "
                f"{100*sorted(errs)[len(errs)//2]:.1f}%, worst {100*max(errs):.1f}%"
                + (f", interior-only worst {100*max(inner):.1f}%" if inner else "")
            )

    print("\n# wrote scale_ceiling.json")
    print("\n\n<!-- markdown for README.md -->\n")
    print(
        "| model | ctx | basis | step s | floor | head gate | +static FFN | +block skip "
        "| +KV router | cap |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for scale in SCALES:
        if scale not in out:
            continue
        for T in GRID:
            r = out[scale]["by_context"][T]
            st = r["stages"]
            tag = {"measured": "**measured**", "converted": "AC-converted", "modelled": "modelled"}[
                r["basis"]
            ]
            print(
                f"| {scale} | {T//1024}k | {tag} | {r['step_s']:.2f} | "
                f"{100*r['wall_shares']['floor']:.1f}% | {st['head']:.2f}x | "
                f"{st['head_static']:.2f}x | **{st['head_static_skip']:.2f}x** | "
                f"{st['head_static_skip_kv']:.2f}x | {r['amdahl_cap']:.1f}x |"
            )


if __name__ == "__main__":
    main()
