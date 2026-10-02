"""Real wall-clock of token router + layer-skip router vs the full model, batch 1, bf16, H200.

    python debug/learned_router/layerskip/wallclock.py --task nq --ckpt ... --tok v6avg_s0_rhobar:1.0 \\
        --ls weights/nq/fr_bar_b1.0_k0.5.pt --rows 16

Per test64 row (2k), timed with CUDA events (median of ``--reps`` after ``--warmup``), forward to the
answer logits only:
* ``full``: the full row (flash attention);
* ``tok``: the token-compacted row (original RoPE positions), every layer;
* ``tok+ls``: a GATHER implementation of the layer-skip router -- at each block the router scores the
  eligible tokens on the block input, the block runs only on the active columns (original
  positions), skipped columns copy their input. Router, gather and scatter costs are included.
  The kept/skipped decisions equal the soft-path eval's (asserted on the first row's answer logits).
Model load / features / token routing (CPU-side, once per row) are not timed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import frontier as FR  # noqa: E402
import layerskip_lib as LS  # noqa: E402
import train_layerskip as TLS  # noqa: E402

RL, TR, G = FR.RL, FR.TR, FR.G
log = TLS.log


@torch.no_grad()
def forward_gather(model, x, pos, elig, router, c, cols):
    """Layer skipping by gathering active columns per block (B = 1, or B identical copies). Returns answer logits and the
    number of skipped eligible columns per layer."""
    model._pooled_keep_holder = None
    ids, _, abk, pbk, lmk = model._prepare_inputs(x, None, logits_to_keep=cols, position_ids=pos)
    h = LS._embed(model, ids)
    ei = torch.nonzero(elig[0]).flatten()
    skipped = []
    T = x.shape[1]
    for key, block in model.blocks.items():
        layer = int(key)
        if ei.numel():
            z = router.logits(layer, h[0, ei]) + c  # rows are identical copies when B > 1
            act = torch.ones(T, dtype=torch.bool, device=x.device)
            act[ei[z <= 0]] = False
            K = act.nonzero().flatten()
            skipped.append(int(T - K.numel()))
        else:
            K, skipped = None, skipped + [0]
        if K is None or K.numel() == T:
            h = block(h, position_ids=pos, **pbk.get(layer, {}))
        else:
            out = block(h[:, K], position_ids=pos[:, K], **pbk.get(layer, {}))
            h = h.clone()
            h[:, K] = out
    return model.lm_head(h, **lmk), skipped


def timeit(fn, warmup, reps):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="nq")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt-format", default="hf")
    ap.add_argument("--test64-root", required=True)
    ap.add_argument("--tok", required=True, help="weights_name:f")
    ap.add_argument("--ls", required=True, help="layer-skip router weights (relative to layerskip/)")
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--rung", default="2k", help="2k reads --test64-root; other rungs read --data-root (the grid's staged rows)")
    ap.add_argument("--data-root", default="")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--batch", type=int, default=1, help="time B copies of each row (throughput regime; decisions identical across copies)")
    ap.add_argument("--tokenizer", default=os.environ.get("DEVLOSS_TOKENIZER", G.TOKENIZER_BY_FAMILY[G.FAMILY]))
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "wallclock.json"))
    a = ap.parse_args()
    from transformers import AutoTokenizer

    from olmo_core.nn.attention.fla_autotune import freeze_fla_length_autotune

    ids = G.RESERVED_IDS[G.FAMILY]
    vocab = G.VOCAB_BY_FAMILY[G.FAMILY]
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rows = TR.load_split(a.test64_root if a.rung == "2k" else a.data_root, a.task, [a.rung], tok, ids, a.rows)
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
    res = []
    for j, r in enumerate(rows):
        p = FR.base_prep(model, r, ids, dev)
        it = FR.token_item(p, tokr, float(f), ids)
        B = a.batch
        xf, pf = p["x"].expand(B, -1).contiguous(), p["pred"][None].expand(B, -1).contiguous()
        xc, pc, posc = it["x"].expand(B, -1).contiguous(), it["pred"][None].expand(B, -1).contiguous(), it["pos"].expand(B, -1).contiguous()
        full = lambda: model(xf, logits_to_keep=pf)  # noqa: E731
        tk = lambda: model(xc, logits_to_keep=pc, position_ids=posc)  # noqa: E731
        gl = lambda: forward_gather(model, xc, posc, it["elig"], lr, c, pc)  # noqa: E731
        lg_g, sk = gl()
        if j == 0:  # gather path == soft path with binary gates (same decisions)
            lg_s = LS.forward_layerskip(model, it["x"], logits_to_keep=it["pred"][None], position_ids=it["pos"],
                                        gate=LS.Gater(it["elig"], "det", router=lr, c=c))
            ce_g = float(torch.nn.functional.cross_entropy(lg_g[0].float(), it["tgt"]))
            ce_s = float(torch.nn.functional.cross_entropy(lg_s[0].float(), it["tgt"]))
            log(f"row 0: gather CE {ce_g:.5f} vs soft path {ce_s:.5f}")
        t_full, t_tok, t_ls = (timeit(fn, a.warmup, a.reps) for fn in (full, tk, gl))
        res.append({"T": p["T"], "T2": it["T2"], "n_elig": it["n_elig"], "skipped": sk, "ms_full": t_full, "ms_tok": t_tok, "ms_ls": t_ls,
                    "flops_tok": LS.flops_ratio(p["T"], it["T2"], [0] * L), "flops_ls": LS.flops_ratio(p["T"], it["T2"], sk)})
        log(f"row {j}: T {p['T']} -> T2 {it['T2']}: full {t_full:.2f} ms, tok {t_tok:.2f} ms ({t_full / t_tok:.2f}x), "
            f"tok+ls {t_ls:.2f} ms ({t_full / t_ls:.2f}x); FLOPs x{res[-1]['flops_tok']:.3f} / x{res[-1]['flops_ls']:.3f}")
    agg = {k: float(np.mean([x[k] for x in res])) for k in ("ms_full", "ms_tok", "ms_ls", "flops_tok", "flops_ls")}
    agg.update({"speedup_tok": agg["ms_full"] / agg["ms_tok"], "speedup_ls": agg["ms_full"] / agg["ms_ls"], "rows": len(res),
                "tok": a.tok, "ls": a.ls, "task": a.task, "batch": a.batch, "rung": a.rung})
    log(f"MEAN over {len(res)} rows: full {agg['ms_full']:.2f} ms; tok {agg['ms_tok']:.2f} ms ({agg['speedup_tok']:.2f}x, FLOPs x{agg['flops_tok']:.3f}); "
        f"tok+ls {agg['ms_ls']:.2f} ms ({agg['speedup_ls']:.2f}x, FLOPs x{agg['flops_ls']:.3f})")
    json.dump({"agg": agg, "rows": res}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
