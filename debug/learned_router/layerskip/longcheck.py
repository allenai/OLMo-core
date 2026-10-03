"""Quality at long context of a 2k-trained (token router, layer-skip router) pair, e.g. the timed nq setting.

    python debug/learned_router/layerskip/longcheck.py --task nq --rung 32k --data-root .../testlong \\
        --tok v6aL_s0_rho0.1339:0.5 --tok-ref v6avg_s0_rhobar:1.0 --ls weights/nq/fr_low_L0.5_k0.5.pt

Per row: full CE; the bar (gold_fl20p8_noslot, heuristic_real); each token router at f x the bar's per-row
budget (@pair x f); + the layer-skip router at its val-calibrated cutoff (exact gather path,
``wallclock.forward_gather``). Paired dCE vs full and vs the bar, FLOPs vs full (``flops_ratio``).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="nq")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt-format", default="hf")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--rung", default="32k")
    ap.add_argument("--rows", type=int, default=56)
    ap.add_argument("--tok", required=True, help="weights:f of the token router the layer router sits on")
    ap.add_argument("--tok-ref", default="", help="comma list of extra token routers weights:f (token-only)")
    ap.add_argument("--ls", required=True)
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

    def load_tok(spec):
        w, f = spec.rsplit(":", 1)
        return RL.LinearRouter.from_state(torch.load(os.path.join(os.path.dirname(HERE), "weights", a.task, f"{w}.pt"), map_location="cpu")).to(dev), float(f)

    toks = {a.tok: load_tok(a.tok)} | {s: load_tok(s) for s in a.tok_ref.split(",") if s}
    st = torch.load(os.path.join(HERE, a.ls), map_location="cpu")
    lr = LS.LayerRouter.from_state(st).to(dev)
    c = float(st["c_star"])
    res = {k: {"ce": [], "flops": []} for k in ["full", "bar"] + [f"tok {s}" for s in toks] + ["tok+ls"]}
    t0 = time.time()
    for j, r in enumerate(rows):
        with torch.no_grad():
            p = FR.base_prep(model, r, ids, dev)
            res["full"]["ce"].append(p["ce_full"])
            res["full"]["flops"].append(1.0)
            bar = FR.compact(p, p["real_bar"], torch.zeros(p["T"], dtype=torch.bool))
            res["bar"]["ce"].append(FR.flash_ce(model, bar))
            res["bar"]["flops"].append(LS.flops_ratio(p["T"], bar["T2"], [0] * L))
            for s, (tr_, f) in toks.items():
                it = FR.token_item(p, tr_, f, ids)
                res[f"tok {s}"]["ce"].append(FR.flash_ce(model, it))
                res[f"tok {s}"]["flops"].append(LS.flops_ratio(p["T"], it["T2"], [0] * L))
                if s == a.tok:
                    lg, sk = WC.forward_gather(model, it["x"], it["pos"], it["elig"], lr, c, it["pred"][None])
                    res["tok+ls"]["ce"].append(float(F.cross_entropy(lg[0].float(), it["tgt"])))
                    res["tok+ls"]["flops"].append(LS.flops_ratio(p["T"], it["T2"], sk))
        if j + 1 in (1, 2, 5, 10) or (j + 1) % 10 == 0 or j + 1 == len(rows):
            el = time.time() - t0
            log(f"row {j + 1}/{len(rows)} T {p['T']} ({el:.0f}s, ETA {el / (j + 1) * (len(rows) - j - 1):.0f}s)")
    full, bar = np.array(res["full"]["ce"]), np.array(res["bar"]["ce"])
    summ = {}
    for k, v in res.items():
        ce = np.array(v["ce"])
        d, db = ce - full, ce - bar
        summ[k] = {"dce": float(d.mean()), "dce_se": float(d.std(ddof=1) / math.sqrt(len(d))), "vs_bar": float(db.mean()),
                   "vs_bar_se": float(db.std(ddof=1) / math.sqrt(len(db))), "flops": float(np.mean(v["flops"])),
                   "n_bad_vs_bar": int((db > 0.1).sum()), "eval_size": len(d)}
        log(f"{k:40s} dCE vs full {summ[k]['dce']:+.4f} ± {summ[k]['dce_se']:.4f} | vs bar {summ[k]['vs_bar']:+.4f} ± {summ[k]['vs_bar_se']:.4f} "
            f"({summ[k]['n_bad_vs_bar']} rows >0.1) | FLOPs x{summ[k]['flops']:.4f}")
    json.dump({"summary": summ, "per_row": res, "argv": sys.argv, "rung": a.rung}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
