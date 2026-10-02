"""GPU tests of the per-(token, layer) skip relaxation on a tiny random Qwen3.5-style HYBRID model
(GatedDeltaNet + gated attention, fla kernels, fp32; the same model as test_relax_gpu.py):

1. (B,T) soft_keep alone / + all-ones layer gate == ``model(x, soft_keep=g)`` (bit-identical check);
2. binary per-layer gates == the EXPLICIT per-layer removal reference (GDN runs on the sub-sequence),
   full row with a binary token gate and the token-compacted row;
3. directional finite differences vs autograd w.r.t. interior per-layer gates.

    python debug/learned_router/layerskip/test_layerskip_gpu.py   (one GPU; ~1 min)
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import layerskip_lib as LS  # noqa: E402

sys.path.insert(0, os.path.dirname(HERE))
import router_lib as RL  # noqa: E402
import train_router as TR  # noqa: E402

G = TR.G


def main():
    import types

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from olmo_core.nn.attention.fla_autotune import freeze_fla_length_autotune

    rng = np.random.default_rng(0)
    V = 512
    sid = types.SimpleNamespace(doc_start=V - 5, doc_end=V - 4, eos=V - 3, landmark=V - 2, pad=V - 1)
    row = list(rng.integers(0, V - 8, 20))
    for _ in range(8):
        row += [sid.doc_start] + list(rng.integers(0, V - 8, int(rng.integers(40, 90)))) + [sid.doc_end]
    row += list(rng.integers(0, V - 8, 30)) + [sid.eos]
    mask = [False] * (len(row) - 7) + [True] * 6 + [False]
    from olmo_core.config import DType
    from olmo_core.nn.attention import AttentionBackendName
    from olmo_core.nn.transformer import TransformerConfig

    # test_relax_gpu.build() with 8 layers (gdn,gdn,gdn,attn x 2)
    cfg = TransformerConfig.qwen3_5_like(d_model=256, vocab_size=V, n_layers=8, n_heads=4, n_kv_heads=2, head_dim=64,
                                         intermediate_size=512, linear_num_key_heads=2, linear_num_value_heads=4,
                                         linear_key_head_dim=64, linear_value_head_dim=64, attn_backend=AttentionBackendName.torch)
    cfg.dtype = DType.float32
    model = cfg.build(init_device="cpu")
    gi = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, prm in model.named_parameters():
            if name.endswith("A_log"):
                prm.copy_(torch.rand(prm.shape, generator=gi) * 2.7)
            elif name.endswith("dt_bias"):
                prm.copy_(-2 + 2 * torch.rand(prm.shape, generator=gi))
            elif "conv1d" in name:
                prm.copy_(0.3 * torch.randn(prm.shape, generator=gi))
            elif prm.dim() == 1:
                prm.fill_(0.0 if name.endswith("bias") else 1.0)
            else:
                prm.copy_(0.02 * torch.randn(prm.shape, generator=gi))
    G.attach_soft_tokens(model, sid, 42, [sid.doc_start, sid.doc_end, sid.eos, sid.landmark, sid.pad])
    model = model.cuda().float().eval()
    kinds = [type(b.attention).__name__ for b in model.blocks.values()]
    print(f"[ls-gpu] froze {freeze_fla_length_autotune()} FLA kernels; blocks {kinds}", flush=True)
    for p_ in model.parameters():
        p_.requires_grad_(False)
    stop_ids, _, tables = G.build_tables(np.asarray(row), V, [f"t{i}" for i in range(V)], sid, str)
    TR.configure(model, stop_ids, tables.to("cuda"))
    p = TR.prep_row(model, {"ids": row, "mask": mask, "gold": {2}, "rung": "2k"}, sid, "cuda")
    model.eval()
    f = p["feats"]
    N, T, L = int(f["idx"].numel()), p["T"], len(model.blocks)
    x, pred, tgt = p["x"], p["pred"][None], p["tgt"]
    gen = torch.Generator().manual_seed(5)

    # 1. backwards compatibility
    with torch.no_grad():
        tk = (torch.rand(N, generator=gen) < 0.6).cuda()
        g = RL.soft_keep_row(f, tk.float(), T)[None]
        ref = model(x, logits_to_keep=pred, soft_keep=g)
        a = LS.forward_layerskip(model, x, logits_to_keep=pred, tok_keep=g)
        b = LS.forward_layerskip(model, x, logits_to_keep=pred, tok_keep=g,
                                 gate=LS.Gater(torch.zeros(1, T, dtype=torch.bool, device="cuda"), "ones"))
        ea, eb = float((a - ref).abs().max()), float((b - ref).abs().max())
        print(f"[ls-gpu] 1. vs model(x, soft_keep=g): no gate max|d| {ea:.1e} (bit-identical {torch.equal(a, ref)}), "
              f"all-ones gate {eb:.1e} (bit-identical {torch.equal(b, ref)})", flush=True)
        assert ea < 1e-5 and eb < 1e-5

    # 2. binary per-layer gates == explicit per-layer removal (GDN on the sub-sequence)
    elig = torch.zeros(1, T, dtype=torch.bool, device="cuda")
    elig[0, f["idx"][tk]] = True
    kept = torch.ones(T, dtype=torch.bool, device="cuda")
    kept[f["idx"][~tk]] = False
    kidx = kept.nonzero().flatten()
    cols = torch.searchsorted(kidx, p["pred"])[None]
    worst = 0.0
    with torch.no_grad():
        for rate in (0.9, 0.5, 0.2, 0.0):
            av = (torch.rand(L, 1, T, generator=gen).cuda() < rate) | ~elig[None]
            lr_ = LS.forward_reference(model, x, av[:, 0] & kept[None], logits_to_keep=pred)
            ls_ = LS.forward_layerskip(model, x, logits_to_keep=pred, tok_keep=g, gate=LS.Gater(elig, "fixed", values=av.float()))
            lc_ = LS.forward_layerskip(model, x[:, kidx], logits_to_keep=cols, position_ids=kidx[None],
                                       gate=LS.Gater(elig[:, kidx], "fixed", values=av[:, :, kidx].float()))
            ce = [float(F.cross_entropy(z[0].float(), tgt)) for z in (lr_, ls_, lc_)]
            gap = max(abs(ce[1] - ce[0]), abs(ce[2] - ce[0]))
            worst = max(worst, gap)
            print(f"[ls-gpu] 2. layer keep {rate:.1f}: CE ref {ce[0]:.6f} soft(full row) {ce[1]:.6f} soft(compacted) {ce[2]:.6f} |gap| {gap:.1e}", flush=True)
    assert worst < 1e-3, worst

    # 3. directional finite differences vs autograd (interior per-layer gates on eligible pairs)
    el = torch.zeros(1, T, dtype=torch.bool, device="cuda")
    el[0, f["idx"]] = True
    gd = torch.Generator().manual_seed(7)
    base = (0.3 + 0.4 * torch.rand(L, 1, T, generator=gd)).cuda()

    def ce_at(vals):
        lg = LS.forward_layerskip(model, x, logits_to_keep=pred, gate=LS.Gater(el, "fixed", values=vals))
        return F.cross_entropy(lg[0].float(), tgt)

    vb = base.clone().requires_grad_(True)
    g_auto = torch.autograd.grad(ce_at(vb), vb)[0]

    def cd(d, h):
        with torch.no_grad():
            return (float(ce_at(base + h * d)) - float(ce_at(base - h * d))) / (2 * h)

    res = []
    for trial in range(4):
        d = (2 * torch.rand(L, 1, T, generator=gd) - 1).cuda() * el[None]
        if trial == 3:
            d[: L // 2] = 0  # upper layers only
        fd = (4 * cd(d, 0.02) - cd(d, 0.04)) / 3
        ad = float((g_auto * d).sum())
        res.append((ad, fd))
        print(f"[ls-gpu] 3. direction {trial}: autograd {ad:+.6f} finite-diff {fd:+.6f} |err| {abs(ad - fd):.2e}", flush=True)
    scale = max(abs(fd) for _, fd in res)
    worst = max(abs(ad - fd) for ad, fd in res) / scale
    print(f"[ls-gpu] max |autograd - FD| = {worst:.3f} x max|FD| ({scale:.2e})", flush=True)
    assert worst < 0.03
    print("[ls-gpu] OK", flush=True)


if __name__ == "__main__":
    main()
