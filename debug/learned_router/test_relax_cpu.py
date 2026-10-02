"""CPU tests of the differentiable-removal path (``soft_keep``) on an ATTENTION-ONLY tiny model
(fla's GatedDeltaNet kernels need a GPU -- the GDN half is ``test_relax_gpu.py``).

    HF_HUB_OFFLINE=1 python debug/learned_router/test_relax_cpu.py

1. no ``soft_keep`` / ``soft_keep = 1`` == the plain forward (CE);
2. binary ``soft_keep`` (p in {0, 1}) == the EXACT hard compaction (``_compact_pooled_soft_tokens``,
   custom mask, keep none, drop_slots, markers kept) -- attention-only, so the relaxation is exact;
3. ``torch.autograd.gradcheck`` (float64, finite differences) of CE w.r.t. p on a handful of routed
   tokens at interior p;
4. the hard-concrete gate: range [0, 1], exact zeros/ones occur, straight-through is binary;
5. the removal-aware GDN short conv: binary p == conv of the compacted row (exact), p = 1 == plain
   conv, float64 gradcheck.
"""
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import router_lib as RL  # noqa: E402
import smoke_cpu as SM  # noqa: E402  (reuses its data/tokenizer constants)
import train_router as TR  # noqa: E402

G = TR.G


def main():
    from transformers import AutoTokenizer

    from olmo_core.config import DType
    from olmo_core.nn.transformer import TransformerConfig

    tok = AutoTokenizer.from_pretrained(SM.TOK)
    ids = G.RESERVED_IDS[G.FAMILY]
    row = G.ROSTER["ctc_nq"]
    ex = G.load_examples(SM.DATA, row, "2k", 1)[0]
    r, m, _ = G.render_ctc_row(tok, ex, row["seg_task"], ids)
    # remap onto a tiny vocab (as smoke_cpu.py)
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
    cfg = TransformerConfig.olmo2_190M(vocab_size=V, n_layers=2, fused_ops=False, dtype=DType.float32)
    model = cfg.build(init_device="cpu")
    gi = torch.Generator().manual_seed(0)  # build() leaves parameters uninitialised: init explicitly
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
    f = p["feats"]
    N, S = int(f["idx"].numel()), p["T"]

    def relaxed_ce(z, mdl=model):
        mdl.eval()
        mdl._pooled_keep_holder = None
        sk = RL.soft_keep_row(f, z, S)
        lg = mdl(p["x"], logits_to_keep=p["pred"][None], soft_keep=sk[None])[0]
        return F.cross_entropy(lg.float(), p["tgt"])

    # 1. p = 1 == plain forward
    cef = TR.ce_full(model, p)
    ce1 = float(relaxed_ce(torch.ones(N)))
    assert abs(ce1 - cef) < 1e-5, (ce1, cef)
    print(f"[relax-cpu] p=1: relaxed CE {ce1:.6f} == full {cef:.6f}")

    # 2. binary p == exact hard compaction (attention-only: no conv, so exact)
    gen = torch.Generator().manual_seed(3)
    worst = 0.0
    for rate in (0.9, 0.5, 0.1, 0.0):
        keep = torch.rand(N, generator=gen) < rate
        hard = float(TR.ce_masked(model, p, RL.keep_mask_from(f, keep, S)[None])[0][0])
        rel = float(relaxed_ce(keep.float()))
        worst = max(worst, abs(hard - rel))
        print(f"[relax-cpu] keep {rate:.1f}: hard {hard:.6f} relaxed {rel:.6f} |diff| {abs(hard - rel):.2e}")
    assert worst < 1e-4, worst

    # 3. finite-difference gradient check (float64) on 6 routed tokens at interior p
    md = cfg.build(init_device="cpu")
    md.load_state_dict(model.state_dict(), strict=False)
    G.attach_soft_tokens(md, sid, 42, stop_ids)
    TR.configure(md, stop_ids, tables)
    md = md.double()
    for p_ in md.parameters():
        p_.requires_grad_(False)
    base = torch.full((N,), 0.7, dtype=torch.float64)
    sel = torch.tensor([0, 5, N // 3, N // 2, N - 7, N - 1])

    def fn(pv):
        z = base.index_put((sel,), pv)
        md.eval()
        md._pooled_keep_holder = None
        lg = md(p["x"], logits_to_keep=p["pred"][None], soft_keep=RL.soft_keep_row(f, z, S)[None])[0]
        return F.cross_entropy(lg, p["tgt"])

    pv = torch.tensor([0.2, 0.5, 0.9, 0.35, 0.6, 0.05], dtype=torch.float64, requires_grad=True)
    # eps 1e-3: RoPE runs in float32 even in a float64 model (full-precision rotary), so a 1e-6 step
    # drowns in rounding; second-order error at 1e-3 is far below the tolerance
    ok = torch.autograd.gradcheck(fn, (pv,), eps=1e-3, atol=2e-4, rtol=0.05)
    g = torch.autograd.grad(fn(pv), pv)[0]
    print(f"[relax-cpu] gradcheck (float64, finite differences) OK={ok}; dCE/dp on 6 tokens {g.tolist()}")

    # 4. hard-concrete gate
    la = torch.linspace(-4, 4, 4001)
    z = RL.hard_concrete(la, 2 / 3, torch.Generator().manual_seed(0))
    assert float(z.min()) == 0.0 and float(z.max()) == 1.0 and bool(((z >= 0) & (z <= 1)).all())
    zst = RL.hard_concrete(la.clone().requires_grad_(True), 0.1, torch.Generator().manual_seed(0), straight_through=True)
    zd = zst.detach()
    assert bool(((zd.abs() < 1e-6) | ((zd - 1).abs() < 1e-6)).all()), "straight-through forward not binary"
    pz = RL.hard_concrete_p_nonzero(la, 2 / 3)
    emp = float((RL.hard_concrete(la.repeat(50), 2 / 3, torch.Generator().manual_seed(1)) > 0).float().mean())
    print(f"[relax-cpu] hard-concrete: exact 0/1 present, ST binary; P(z>0) analytic {float(pz.mean()):.3f} vs empirical {emp:.3f}")
    assert abs(float(pz.mean()) - emp) < 0.01
    # 4a. routed markers: binary gates over body + markers == hard compaction in "mask" marker mode
    pr = TR.prep_row(model, {"ids": rs, "mask": m, "gold": gold, "rung": "2k"}, sid, "cpu", route_markers=True)
    fr = pr["feats"]
    assert int(fr["is_marker"].sum()) == 2 * int(RL.torch.as_tensor(0) + (pr["feats"]["doc"].max() + 1)), "two markers per doc"
    pst = model._pooled_soft_tokens
    pst["keep_token_mask_markers"] = "mask"
    for rate in (0.7, 0.3):
        keep = torch.rand(int(fr["idx"].numel()), generator=torch.Generator().manual_seed(int(rate * 10))) < rate
        hard = float(TR.ce_masked(model, pr, RL.keep_mask_from(fr, keep, pr["T"])[None])[0][0])
        model.eval()
        model._pooled_keep_holder = None
        lg = model(pr["x"], logits_to_keep=pr["pred"][None], soft_keep=RL.soft_keep_row(fr, keep.float(), pr["T"])[None])[0]
        rel = float(F.cross_entropy(lg.float(), pr["tgt"]))
        print(f"[relax-cpu] routed markers keep {rate}: hard {hard:.6f} relaxed {rel:.6f} |diff| {abs(hard - rel):.1e}")
        assert abs(hard - rel) < 1e-4
    pst["keep_token_mask_markers"] = True

    # 4b. --hard-drop-zero (train_e2e_router.py): physically dropping z == 0 tokens == full-row soft
    #     path, loss and router gradient (plain clamp), at a router with ~half the gates exactly 0
    import train_e2e_router as TE

    router = RL.LinearRouter(int(model.embeddings.weight.shape[1]), "full")
    with torch.no_grad():
        router.b.fill_(-0.5)
        router.w_gold.fill_(3.0)
        router.w_emb.copy_(torch.randn(router.w_emb.shape, generator=torch.Generator().manual_seed(2)))
    res = {}
    for mode in (False, True):
        router.zero_grad(set_to_none=True)
        la = router.logits(f, p["e"].float())
        z = RL.hard_concrete(la, 0.3, torch.Generator().manual_seed(9), st_clamp=False)
        loss = F.cross_entropy(TE.relaxed_logits(model, p, z, drop_zero=mode), p["tgt"])
        loss.backward()
        res[mode] = (float(loss), torch.cat([q.grad.flatten() for q in router.parameters()]), float((z > 0).float().mean()))
    rl = abs(res[True][0] - res[False][0]) / abs(res[False][0])
    rg = float((res[True][1] - res[False][1]).norm() / res[False][1].norm())
    print(f"[relax-cpu] hard-drop-zero (z>0 on {res[False][2]:.2f} of routed tokens): loss rel diff {rl:.1e}, router grad rel diff {rg:.1e}")
    assert rl < 1e-4 and rg < 1e-4, (rl, rg)

    # 5. removal-aware short conv (GatedDeltaNet under soft_keep): binary p == conv of the compacted
    #    row at kept positions; p = 1 == plain conv; float64 gradcheck w.r.t. p
    from olmo_core.nn.convolution import CausalConv1d

    for T, C, chunk in ((300, 24, 64), (257, 8, 128), (40, 4, 16)):
        conv = CausalConv1d(hidden_size=C, kernel_size=4, bias=True, dtype=torch.float64)
        torch.nn.init.normal_(conv.weight)
        torch.nn.init.normal_(conv.bias)
        x = torch.randn(1, T, C, dtype=torch.float64)

        def ref(xx):
            y = F.conv1d(xx.transpose(1, 2), conv.weight, conv.bias, padding=3, groups=C)[..., : xx.shape[1]]
            return F.silu(y).transpose(1, 2)

        y1 = conv.forward_soft_keep(x, torch.ones(1, T, dtype=torch.float64))
        scale = float(ref(x).abs().max())
        e1 = float((y1 - ref(x)).abs().max()) / scale
        keep = torch.rand(T, generator=torch.Generator().manual_seed(T)) < 0.4
        keep[:3] = True
        keep[50:90] = False  # a long dropped run
        yk = conv.forward_soft_keep(x, keep[None].double())
        yc = ref(x[:, keep])
        eb = float((yk[:, keep] - yc).abs().max()) / scale
        print(f"[relax-cpu] soft conv T={T} C={C} chunk={chunk}: p=1 vs plain {e1:.1e}; binary vs compacted {eb:.1e} (max rel; 1e-6 carry-over clamp)")
        assert e1 < 1e-4 and eb < 1e-4, (e1, eb)
    xs = torch.randn(1, 37, 3, dtype=torch.float64)
    conv = CausalConv1d(hidden_size=3, kernel_size=4, bias=True, dtype=torch.float64)
    pv = (0.1 + 0.8 * torch.rand(1, 37, dtype=torch.float64)).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda q: conv.forward_soft_keep(xs, q, chunk=8), (pv,), eps=1e-6, atol=1e-6, rtol=1e-4)
    print("[relax-cpu] soft conv gradcheck (float64) OK")
    print("[relax-cpu] OK")


if __name__ == "__main__":
    main()
