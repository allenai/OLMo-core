"""GPU tests of the differentiable-removal path on a tiny random Qwen3.5-style HYBRID model
(GatedDeltaNet + gated attention, fla kernels), fp32:

1. ``soft_keep = 1`` == the plain forward;
2. central finite differences vs autograd for directional derivatives of CE w.r.t. p (interior p);
3. binary ``soft_keep`` vs the exact hard compaction: equal up to kernel numerics (attention bias is
   exact at p in {0,1}; GDN scales beta/g and runs the removal-aware short conv, exact for binary p).

    python debug/learned_router/test_relax_gpu.py   (one GPU; ~1 min)
"""
import math
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import router_lib as RL  # noqa: E402
import train_router as TR  # noqa: E402

G = TR.G


def build(V, mix="hybrid"):
    from olmo_core.config import DType
    from olmo_core.nn.attention import AttentionBackendName
    from olmo_core.nn.transformer import TransformerConfig

    kw = dict(d_model=256, vocab_size=V, n_layers=4, n_heads=4, n_kv_heads=2, head_dim=64,
              intermediate_size=512, linear_num_key_heads=2, linear_num_value_heads=4,
              linear_key_head_dim=64, linear_value_head_dim=64, attn_backend=AttentionBackendName.torch)
    cfg = TransformerConfig.qwen3_5_like(**kw)
    cfg.dtype = DType.float32
    return cfg


def main():
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from olmo_core.nn.attention.fla_autotune import freeze_fla_length_autotune

    rng = np.random.default_rng(0)
    V = 512
    sid = types.SimpleNamespace(doc_start=V - 5, doc_end=V - 4, eos=V - 3, landmark=V - 2, pad=V - 1)
    # synthetic row: prompt, 8 documents of 40-90 tokens, query, answer (last 6 tokens scored)
    row = list(rng.integers(0, V - 8, 20))
    for _ in range(8):
        row += [sid.doc_start] + list(rng.integers(0, V - 8, int(rng.integers(40, 90)))) + [sid.doc_end]
    row += list(rng.integers(0, V - 8, 30)) + [sid.eos]
    mask = [False] * (len(row) - 7) + [True] * 6 + [False]
    cfg = build(V)
    model = cfg.build(init_device="cpu")
    # build() leaves parameters uninitialised (real runs load a checkpoint): initialise explicitly and
    # deterministically, with a strong short conv so the conv path matters for the binary-gap check
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
    model = model.cuda().float()
    print(f"froze {freeze_fla_length_autotune()} FLA kernels; blocks: "
          f"{sorted({type(b.attention).__name__ for b in model.blocks.values()})}", flush=True)
    for p_ in model.parameters():
        p_.requires_grad_(False)
    stop_ids, _, tables = G.build_tables(np.asarray(row), V, [f"t{i}" for i in range(V)], sid, str)
    TR.configure(model, stop_ids, tables.to("cuda"))
    p = TR.prep_row(model, {"ids": row, "mask": mask, "gold": {2}, "rung": "2k"}, sid, "cuda")
    f = p["feats"]
    N, S = int(f["idx"].numel()), p["T"]

    def relaxed_ce(z):
        model.eval()
        model._pooled_keep_holder = None
        lg = model(p["x"], logits_to_keep=p["pred"][None], soft_keep=RL.soft_keep_row(f, z, S)[None])[0]
        return F.cross_entropy(lg.float(), p["tgt"])

    cef = TR.ce_full(model, p)
    ce1 = float(relaxed_ce(torch.ones(N, device="cuda")))
    print(f"[relax-gpu] p=1: relaxed {ce1:.6f} vs full {cef:.6f} |diff| {abs(ce1 - cef):.2e}", flush=True)
    assert math.isfinite(cef) and abs(ce1 - cef) < 1e-3, (ce1, cef)

    # finite differences vs autograd along random DIRECTIONS over all routed tokens (a per-token
    # derivative is ~1e-4 here, below fp32/TF32 kernel noise; a directional derivative aggregates
    # hundreds of tokens). Base p ~ U(0.3, 0.7), direction entries in [-1, 1] (never clamped),
    # Richardson-extrapolated central differences (h = 0.04, 0.02) to cancel the O(h^2) term.
    gd = torch.Generator().manual_seed(7)
    base = (0.3 + 0.4 * torch.rand(N, generator=gd)).cuda()
    zb = base.clone().requires_grad_(True)
    g_auto = torch.autograd.grad(relaxed_ce(zb), zb)[0]

    def cd(d, h):
        with torch.no_grad():
            return (float(relaxed_ce(base + h * d)) - float(relaxed_ce(base - h * d))) / (2 * h)

    res = []
    for trial in range(4):
        d = (2 * torch.rand(N, generator=gd) - 1).cuda()
        if trial == 3:  # perturb only the first half of the row
            d[N // 2:] = 0
        fd = (4 * cd(d, 0.02) - cd(d, 0.04)) / 3
        ad = float((g_auto * d).sum())
        res.append((ad, fd))
        print(f"[relax-gpu] direction {trial}: autograd {ad:+.6f} finite-diff {fd:+.6f} |err| {abs(ad - fd):.2e}", flush=True)
    # fla's kernels are TF32-accurate even in fp32, so a finite difference has an absolute noise floor
    # (~1e-4 here); judge the error against the scale of the largest directional derivative
    scale = max(abs(fd) for _, fd in res)
    worst = max(abs(ad - fd) for ad, fd in res) / scale
    print(f"[relax-gpu] max |autograd - FD| = {worst:.3f} x max|FD| ({scale:.2e})", flush=True)
    assert worst < 0.03, "autograd != finite differences"

    # binary p vs exact hard compaction
    gen = torch.Generator().manual_seed(5)
    worst_gap = 0.0
    for rate in (0.9, 0.5, 0.2):
        keep = torch.rand(N, generator=gen) < rate
        hard = float(TR.ce_masked(model, p, RL.keep_mask_from(f, keep.cuda(), S)[None])[0][0])
        rl = float(relaxed_ce(keep.float().cuda()))
        worst_gap = max(worst_gap, abs(hard - rl))
        print(f"[relax-gpu] keep {rate}: hard dCE {hard - cef:+.5f} relaxed dCE {rl - cef:+.5f} |gap| {abs(hard - rl):.5f}", flush=True)
    assert worst_gap < 2e-3, f"binary relaxed != hard compaction ({worst_gap})"
    # --hard-drop-zero (train_e2e_router.relaxed_logits): compacting away z == 0 tokens == the full-row
    # soft path, loss and gradient w.r.t. the gates (plain clamp), fp32
    import train_e2e_router as TE

    lg_ = torch.Generator().manual_seed(4)
    la = (torch.randn(N, generator=lg_) * 1.5 - 0.3).cuda().requires_grad_(True)
    res = {}
    for mode in (False, True):
        la.grad = None
        z = RL.hard_concrete(la, 0.3, torch.Generator().manual_seed(11), st_clamp=False)
        loss = F.cross_entropy(TE.relaxed_logits(model, p, z, drop_zero=mode), p["tgt"])
        loss.backward()
        res[mode] = (float(loss), la.grad.clone(), float((z > 0).float().mean()))
    rl = abs(res[True][0] - res[False][0]) / abs(res[False][0])
    rg = float((res[True][1] - res[False][1]).norm() / res[False][1].norm())
    print(f"[relax-gpu] hard-drop-zero (z>0 on {res[False][2]:.2f}): loss rel diff {rl:.1e}, gate-logit grad rel diff {rg:.1e}", flush=True)
    assert rl < 1e-3 and rg < 1e-2, (rl, rg)
    print("[relax-gpu] OK", flush=True)


if __name__ == "__main__":
    main()
