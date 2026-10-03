"""FAST layer skipping: device-side fixed-capacity gathers (no per-layer host sync) + CUDA graphs.

    python debug/learned_router/layerskip/fastskip.py --task nq --ckpt ... --tok v6aL_s0_rho0.1339:0.5 \\
        --ls weights/nq/fr_low_L0.5_k0.5.pt --rung 2k --batch 1 --rows 8

Per-layer CAPACITY routing (MoD-style): at layer l the block runs on the always-active columns plus the
top-k_l eligible columns by router logit (``torch.topk`` -> sorted gather indices -> ``gather`` /
``scatter``), all on device with static shapes, so the whole compacted forward is CUDA-graph
capturable. For the timing, k_l is set per row to the number of eligible columns the threshold rule
(logit + c* > 0) keeps at layer l on that row (one reference pass, untimed), so the fast path makes the
SAME decisions as the reference and its answer CE must equal it (checked per row). A deployment would
use calibrated (bucketed) capacities instead.

Timed (CUDA events, median of --reps after warmup), forward to the answer logits only, batch B copies:
  full (eager / graph), token-compacted (eager / graph), token + layer skip: per-layer gather with host
  sync (``wallclock.forward_gather``, eager), fixed-capacity gather (eager / graph).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import frontier as FR  # noqa: E402
import layerskip_lib as LS  # noqa: E402
import train_layerskip as TLS  # noqa: E402
import wallclock as WC  # noqa: E402

RL, TR, G = FR.RL, FR.TR, FR.G
log = TLS.log


class Plan:
    """Static per-row plan: always-active positions, eligible positions, per-layer capacities."""

    def __init__(self, elig: torch.Tensor, caps, B: int):
        e = elig[0]
        self.ei = torch.nonzero(e).flatten()  # (E,)
        self.nf = torch.nonzero(~e).flatten()  # (Nf,)
        self.E, self.Nf, self.caps, self.B = int(self.ei.numel()), int(self.nf.numel()), list(caps), B
        self.nf_b = self.nf[None].expand(B, -1)


def prepare(model, x, pos, cols):
    model._pooled_keep_holder = None
    kw = {} if pos is None else {"position_ids": pos}
    ids, _, abk, pbk, lmk = model._prepare_inputs(x, None, logits_to_keep=cols, **kw)
    return ids, pbk, lmk


def plain_forward(model, ids, pos, pbk, lmk):
    h = LS._embed(model, ids)
    for key, block in model.blocks.items():
        h = block(h, position_ids=pos, **pbk.get(int(key), {}))
    return model.lm_head(h, **lmk)


def capacity_forward(model, ids, pos, pbk, lmk, plan: Plan, router, c: float):
    """Fixed-capacity layer skipping, no host sync: per layer top-k_l eligible by router logit."""
    h = LS._embed(model, ids)
    B = ids.shape[0]
    for key, block in model.blocks.items():
        layer = int(key)
        k = plan.caps[layer]
        if k >= plan.E:
            h = block(h, position_ids=pos, **pbk.get(layer, {}))
            continue
        if k > 0:
            z = router.logits(layer, h[:, plan.ei])  # (B, E)
            top = torch.topk(z, k, dim=1).indices  # (B, k)
            idx = torch.cat([plan.nf_b, plan.ei[top]], 1)
        else:
            idx = plan.nf_b
        idx = torch.sort(idx, dim=1).values  # causal order
        hi = torch.gather(h, 1, idx[..., None].expand(-1, -1, h.shape[-1]))
        out = block(hi, position_ids=torch.gather(pos, 1, idx), **pbk.get(layer, {}))
        h = h.scatter(1, idx[..., None].expand(-1, -1, h.shape[-1]), out)
    return model.lm_head(h, **lmk)


def graphed(fn, warmup=3):
    """Capture fn() (static inputs in its closure) into a CUDA graph; returns (replay, output)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    return g.replay, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="nq")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt-format", default="hf")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--rung", default="2k")
    ap.add_argument("--tok", required=True)
    ap.add_argument("--ls", required=True)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--tokenizer", default=os.environ.get("DEVLOSS_TOKENIZER", G.TOKENIZER_BY_FAMILY[G.FAMILY]))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    from olmo_core.nn.attention.fla_autotune import freeze_fla_length_autotune

    ids = G.RESERVED_IDS[G.FAMILY]
    vocab = G.VOCAB_BY_FAMILY[G.FAMILY]
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rows = TR.load_split(a.data_root, a.task, [a.rung], tok, ids, a.rows)
    all_ids = np.concatenate([np.asarray(r["ids"], dtype=np.int64) for r in rows])
    pieces = tok.convert_ids_to_tokens(list(range(vocab)))
    stop_ids, _, tables = G.build_tables(all_ids, vocab, pieces, ids, lambda t: tok.decode([int(t)]))
    model = G.load_model(a.ckpt, a.ckpt_format, vocab, ids, 42, stop_ids, "flash_2")
    for p_ in model.parameters():
        p_.requires_grad_(False)
    freeze_fla_length_autotune()
    dev = next(model.parameters()).device
    TR.configure(model, stop_ids, tables.to(dev))
    model.eval()
    L = len(model.blocks)
    wname, f = a.tok.rsplit(":", 1)
    tokr = RL.LinearRouter.from_state(torch.load(os.path.join(os.path.dirname(HERE), "weights", a.task, f"{wname}.pt"), map_location="cpu")).to(dev)
    st = torch.load(os.path.join(HERE, a.ls), map_location="cpu")
    lr = LS.LayerRouter.from_state(st).to(dev)
    c = float(st["c_star"])
    B = a.batch
    res = []
    for j, r in enumerate(rows):
        with torch.no_grad():
            p = FR.base_prep(model, r, ids, dev)
            it = FR.token_item(p, tokr, float(f), ids)
            ce_cap_g = None
            xf, posf, colf = p["x"].expand(B, -1).contiguous(), None, p["pred"][None].expand(B, -1).contiguous()
            xc, posc, colc = it["x"].expand(B, -1).contiguous(), it["pos"].expand(B, -1).contiguous(), it["pred"][None].expand(B, -1).contiguous()
            # reference decisions (threshold rule) -> per-layer capacities; reference CE (soft path, binary gates)
            lg_g, sk = WC.forward_gather(model, it["x"], it["pos"], it["elig"], lr, c, it["pred"][None])
            E = int(it["elig"].sum())
            caps = [E - s for s in sk]
            plan = Plan(it["elig"], caps, B)
            lg_ref = LS.forward_layerskip(model, it["x"], logits_to_keep=it["pred"][None], position_ids=it["pos"],
                                          gate=LS.Gater(it["elig"], "det", router=lr, c=c))
            ce_ref = float(F.cross_entropy(lg_ref[0].float(), it["tgt"]))
            ce_gather = float(F.cross_entropy(lg_g[0].float(), it["tgt"]))
            idsf, pbkf, lmkf = prepare(model, xf, posf, colf)
            idsc, pbkc, lmkc = prepare(model, xc, posc, colc)
            fns = {
                "full": lambda: plain_forward(model, idsf, posf, pbkf, lmkf),
                "tok": lambda: plain_forward(model, idsc, posc, pbkc, lmkc),
                "ls_sync": lambda: WC.forward_gather(model, xc, posc, it["elig"], lr, c, colc)[0],
                "ls_cap": lambda: capacity_forward(model, idsc, posc, pbkc, lmkc, plan, lr, c),
            }
            lg_cap = fns["ls_cap"]()
            ce_cap = float(F.cross_entropy(lg_cap[0].float(), it["tgt"]))
            t = {k: WC.timeit(fn, a.warmup, a.reps) for k, fn in fns.items()}
            graph_err = None
            if not a.no_graph:
                for k in ("full", "tok", "ls_cap"):
                    try:
                        rep, out = graphed(fns[k])
                        if k == "ls_cap":
                            rep()
                            torch.cuda.synchronize()
                            ce_cap_g = float(F.cross_entropy(out[0].float(), it["tgt"]))
                        t[k + "_graph"] = WC.timeit(rep, a.warmup, a.reps)
                        del rep
                    except Exception as ex:  # noqa: BLE001
                        graph_err = f"{k}: {ex!r}"[:300]
                        log(f"row {j}: CUDA graph capture of {k} failed: {graph_err}")
                        torch.cuda.synchronize()
            rec = {"T": p["T"], "T2": it["T2"], "E": E, "caps": caps, "ms": t, "ce_ref": ce_ref, "ce_gather": ce_gather, "ce_cap": ce_cap,
                   "ce_cap_graph": ce_cap_g, "graph_err": graph_err,
                   "flops_tok": LS.flops_ratio(p["T"], it["T2"], [0] * L), "flops_ls": LS.flops_ratio(p["T"], it["T2"], sk)}
            res.append(rec)
            log(f"row {j}: T {p['T']} -> T2 {it['T2']} (E {E}); CE ref {ce_ref:.5f} gather {ce_gather:.5f} cap {ce_cap:.5f} cap-graph "
                f"{rec['ce_cap_graph']}; " + " ".join(f"{k} {v:.2f}ms" for k, v in t.items()))
    agg = {k: float(np.mean([r_["ms"][k] for r_ in res if k in r_["ms"]])) for k in res[0]["ms"]}
    sp = {k: agg["full"] / v for k, v in agg.items()}
    if "full_graph" in agg:
        sp.update({k + "_vs_full_graph": agg["full_graph"] / v for k, v in agg.items() if k.endswith("_graph")})
    summary = {"ms": agg, "speedup_vs_full_eager": sp, "flops_tok": float(np.mean([r_["flops_tok"] for r_ in res])),
               "flops_ls": float(np.mean([r_["flops_ls"] for r_ in res])), "rows": len(res), "batch": B, "rung": a.rung,
               "max_abs_ce_diff_cap_vs_ref": float(max(abs(r_["ce_cap"] - r_["ce_ref"]) for r_ in res))}
    log("MEAN: " + json.dumps(summary))
    json.dump({"summary": summary, "rows": res}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
