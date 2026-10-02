"""CPU tests of the per-(token, layer) skip relaxation on an ATTENTION-ONLY tiny model (fla's GDN
kernels need a GPU: the hybrid half is ``test_layerskip_gpu.py``).

    HF_HUB_OFFLINE=1 python debug/learned_router/layerskip/test_layerskip_cpu.py

1. backwards compatibility, BIT-identical: ``forward_layerskip`` without per-layer gates, or with an
   all-ones gate, equals ``model(x, soft_keep=g)`` (and ``model(x)`` without ``g``) exactly;
2. binary per-layer gates == the EXPLICIT per-layer removal reference (``forward_reference``), on the
   full row with a binary token gate g and on the token-compacted row, several skip rates;
3. a token skipped at EVERY layer == the token dropped by the token router (hard compaction);
4. float64 finite-difference gradcheck of CE w.r.t. interior per-layer gates (and the token gate);
5. the router gets a nonzero gradient through hard-concrete gates.
"""
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import layerskip_lib as LS  # noqa: E402
import router_lib as RL  # noqa: E402

sys.path.insert(0, os.path.dirname(HERE))  # learned_router/smoke_cpu.py, not devloss_grid's
sys.modules.pop("smoke_cpu", None)
import smoke_cpu as SM  # noqa: E402
import train_router as TR  # noqa: E402

G = TR.G


def build_tiny():
    from transformers import AutoTokenizer

    from olmo_core.config import DType
    from olmo_core.nn.transformer import TransformerConfig

    tok = AutoTokenizer.from_pretrained(SM.TOK)
    ids = G.RESERVED_IDS[G.FAMILY]
    row = G.ROSTER["ctc_nq"]
    ex = G.load_examples(SM.DATA, row, "2k", 1)[0]
    r, m, _ = G.render_ctc_row(tok, ex, row["seg_task"], ids)
    specials = [ids.doc_start, ids.doc_end, ids.eos, ids.landmark, ids.pad]
    uniq = sorted(set(r) - set(specials))
    remap = {t: i for i, t in enumerate(uniq)}
    U = len(uniq)
    for j, s in enumerate(specials):
        remap[s] = U + j
    V = ((U + 5 + 7) // 8) * 8
    sid = types.SimpleNamespace(doc_start=U, doc_end=U + 1, eos=U + 2, landmark=U + 3, pad=U + 4)
    rs = [remap[t] for t in r]
    gold = G.gold_docs(row["spec"], ex)
    torch.manual_seed(0)
    cfg = TransformerConfig.olmo2_190M(vocab_size=V, n_layers=3, fused_ops=False, dtype=DType.float32)
    model = cfg.build(init_device="cpu")
    gi = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, prm in model.named_parameters():
            if prm.dim() == 1:
                prm.fill_(0.0 if name.endswith("bias") else 1.0)
            else:
                prm.copy_(0.05 * torch.randn(prm.shape, generator=gi))
    G.attach_soft_tokens(model, sid, 42, [sid.doc_start, sid.doc_end, sid.eos, sid.landmark, sid.pad])
    for p_ in model.parameters():
        p_.requires_grad_(False)
    stop_ids, _, tables = G.build_tables(np.asarray(rs), V, [f"t{i}" for i in range(V)], sid, str)
    TR.configure(model, stop_ids, tables)
    p = TR.prep_row(model, {"ids": rs, "mask": m, "gold": gold, "rung": "2k"}, sid, "cpu")
    model.eval()
    return cfg, model, p, sid, stop_ids, tables


def main():
    cfg, model, p, sid, stop_ids, tables = build_tiny()
    f = p["feats"]
    N, T, L = int(f["idx"].numel()), p["T"], len(model.blocks)
    x, pred, tgt = p["x"], p["pred"][None], p["tgt"]
    gen = torch.Generator().manual_seed(5)
    print(f"[ls-cpu] tiny attention-only model: {L} layers, T={T}, routed={N}, answer tokens={int(tgt.numel())}")

    # ---- 1. backwards compatibility (bit-identical)
    with torch.no_grad():
        ref0 = model(x, logits_to_keep=pred)
        got0 = LS.forward_layerskip(model, x, logits_to_keep=pred)
        assert torch.equal(ref0, got0), "no gates: not bit-identical to model(x)"
        tk = torch.rand(N, generator=gen) < 0.6
        g = RL.soft_keep_row(f, tk.float(), T)[None]
        gs = RL.soft_keep_row(f, torch.rand(N, generator=gen), T)[None]  # interior token gate too
        for gg, nm in ((g, "binary"), (gs, "interior")):
            ref = model(x, logits_to_keep=pred, soft_keep=gg)
            a = LS.forward_layerskip(model, x, logits_to_keep=pred, tok_keep=gg)
            b = LS.forward_layerskip(model, x, logits_to_keep=pred, tok_keep=gg,
                                     gate=LS.Gater(torch.zeros(1, T, dtype=torch.bool), "ones"))
            assert torch.equal(ref, a) and torch.equal(ref, b), f"(B,T) soft_keep ({nm}) not bit-identical"
    print("[ls-cpu] 1. (B,T) soft_keep alone / + all-ones layer gate: BIT-identical to model(x, soft_keep=g) (binary and interior g)")

    # ---- 2. binary per-layer gates == explicit per-layer removal
    elig_full = torch.zeros(1, T, dtype=torch.bool)
    elig_full[0, f["idx"][tk]] = True
    kept = torch.ones(T, dtype=torch.bool)
    kept[f["idx"][~tk]] = False
    kidx = kept.nonzero().flatten()
    cols = torch.searchsorted(kidx, p["pred"])[None]
    worst = 0.0
    with torch.no_grad():
        for rate in (0.9, 0.5, 0.2, 0.0):
            av = (torch.rand(L, 1, T, generator=gen) < rate) | ~elig_full[None]
            keep_layers = (av[:, 0] & kept[None])  # (L, T)
            lr_ = LS.forward_reference(model, x, keep_layers, logits_to_keep=pred)
            ls_ = LS.forward_layerskip(model, x, logits_to_keep=pred, tok_keep=g,
                                       gate=LS.Gater(elig_full, "fixed", values=av.float()))
            # token-compacted row, per-layer soft gates only
            lc_ = LS.forward_layerskip(model, x[:, kidx], logits_to_keep=cols, position_ids=kidx[None],
                                       gate=LS.Gater(elig_full[:, kidx], "fixed", values=av[:, :, kidx].float()))
            scale = float(lr_.abs().max())
            e1, e2 = float((ls_ - lr_).abs().max()) / scale, float((lc_ - lr_).abs().max()) / scale
            ce = [float(F.cross_entropy(z[0], tgt)) for z in (lr_, ls_, lc_)]
            worst = max(worst, e1, e2)
            print(f"[ls-cpu] 2. layer keep {rate:.1f}: CE ref {ce[0]:.6f} soft(full row) {ce[1]:.6f} soft(compacted) {ce[2]:.6f}; "
                  f"max rel logit diff {e1:.1e} / {e2:.1e}")
    assert worst < 1e-5, worst

    # ---- 3. skipped at every layer == dropped by the token router (hard compaction path)
    with torch.no_grad():
        drop = torch.rand(N, generator=gen) < 0.5
        el = torch.zeros(1, T, dtype=torch.bool)
        el[0, f["idx"][drop]] = True
        lg = LS.forward_layerskip(model, x, logits_to_keep=pred, gate=LS.Gater(el, "fixed", values=torch.zeros(L, 1, T)))
        ce_skip = float(F.cross_entropy(lg[0], tgt))
        ce_hard = float(TR.ce_masked(model, p, RL.keep_mask_from(f, ~drop, T)[None])[0][0])
    print(f"[ls-cpu] 3. skip-at-every-layer CE {ce_skip:.6f} == token-dropped hard compaction {ce_hard:.6f} (|d| {abs(ce_skip - ce_hard):.1e})")
    assert abs(ce_skip - ce_hard) < 1e-4

    # ---- 4. float64 gradcheck w.r.t. interior per-layer gates and the token gate
    md = cfg.build(init_device="cpu")
    md.load_state_dict(model.state_dict(), strict=False)
    G.attach_soft_tokens(md, sid, 42, stop_ids)
    TR.configure(md, stop_ids, tables)
    md = md.double().eval()
    for p_ in md.parameters():
        p_.requires_grad_(False)
    el = torch.zeros(1, T, dtype=torch.bool)
    el[0, f["idx"]] = True
    base = torch.full((L, 1, T), 0.7, dtype=torch.float64)
    pos = f["idx"][torch.tensor([0, N // 3, N // 2, N - 3])]
    sel = [(0, int(pos[0])), (1, int(pos[1])), (2, int(pos[2])), (1, int(pos[3])), (0, int(pos[3]))]
    gsel = int(f["idx"][N // 4])

    def fn(av, gv):
        vals = base.clone()
        for j, (l_, t_) in enumerate(sel):
            vals = vals.index_put((torch.tensor([l_]), torch.tensor([0]), torch.tensor([t_])), av[j:j + 1])
        tkv = torch.ones(1, T, dtype=torch.float64).index_put((torch.tensor([0]), torch.tensor([gsel])), gv)
        lg = LS.forward_layerskip(md, x, logits_to_keep=pred, tok_keep=tkv, gate=LS.Gater(el, "fixed", values=vals))
        return F.cross_entropy(lg[0], tgt)

    av = torch.tensor([0.2, 0.5, 0.9, 0.35, 0.6], dtype=torch.float64, requires_grad=True)
    gv = torch.tensor([0.4], dtype=torch.float64, requires_grad=True)
    # eps 1e-3: RoPE runs in float32 even in a float64 model (see test_relax_cpu.py)
    ok = torch.autograd.gradcheck(fn, (av, gv), eps=1e-3, atol=2e-4, rtol=0.05)
    ga, gg_ = torch.autograd.grad(fn(av, gv), (av, gv))
    print(f"[ls-cpu] 4. gradcheck (float64, finite differences) OK={ok}; dCE/da {[round(v, 5) for v in ga.tolist()]} dCE/dg {gg_.tolist()}")

    # ---- 5. router gradient through hard-concrete gates
    router = LS.LayerRouter(L, int(model.embeddings.weight.shape[1]))
    with torch.no_grad():
        router.w.normal_(0, 1, generator=torch.Generator().manual_seed(1))
    gt = LS.Gater(el, "sample", router=router, c=0.5, beta=0.5, seed=3)
    lg = LS.forward_layerskip(model, x, logits_to_keep=pred, gate=gt)
    F.cross_entropy(lg[0].float(), tgt).backward()
    gb, gw = float(router.b.grad.norm()), float(router.w.grad.norm())
    zfrac = float(torch.cat([gt.gates[l_] for l_ in range(L)]).gt(0).float().mean())
    print(f"[ls-cpu] 5. router grad |db| {gb:.2e} |dw| {gw:.2e} (gates > 0 on {zfrac:.2f} of eligible pairs)")
    assert gb > 0 and gw > 0
    print("[ls-cpu] ALL PASSED")


if __name__ == "__main__":
    main()
