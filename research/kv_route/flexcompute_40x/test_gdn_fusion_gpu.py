"""Exactness + speed for the fused GDN q/k/v path (GPU; fla kernels are CUDA-only).

Checks, in order:
  1. a SPLIT checkpoint loads into the fused module via the state-dict hook, with weights that
     concatenate exactly;
  2. fused and split modules produce the same output and the same input gradient;
  3. the fused block is faster, measured with the arms INTERLEAVED on one process, because this
     node showed 24% run-to-run drift and the win being chased is smaller than that.

    python research/kv_route/flexcompute_40x/test_gdn_fusion_gpu.py --tokens 8192
"""

import argparse
import time

import torch

from olmo_core.nn.attention.recurrent import GatedDeltaNet

D_MODEL, N_HEADS, N_V_HEADS, HEAD_DIM = 2560, 16, 32, 128


def build(fuse: bool, device):
    torch.manual_seed(0)
    m = GatedDeltaNet(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_v_heads=N_V_HEADS,
        head_dim=HEAD_DIM,
        fuse_qkv=fuse,
        dtype=torch.bfloat16,
        init_device=str(device),
    )
    return m.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    dev = torch.device("cuda")
    T = args.tokens

    split = build(False, dev)
    fused = build(True, dev)

    # 1/2. move the split module's weights into the fused one THROUGH the load hook
    sd = split.state_dict()
    missing, unexpected = fused.load_state_dict(sd, strict=False)
    unexpected = [
        u
        for u in unexpected
        if not u.startswith(("w_q", "w_k", "w_v", "q_conv1d", "k_conv1d", "v_conv1d"))
    ]
    print(f"load via hook: missing={list(missing)} unexpected={unexpected}")
    exp_w = torch.cat([split.w_q.weight, split.w_k.weight, split.w_v.weight], 0)
    exp_c = torch.cat([split.q_conv1d.weight, split.k_conv1d.weight, split.v_conv1d.weight], 0)
    print(f"  w_qkv exact:      {torch.equal(fused.w_qkv.weight, exp_w)}")
    print(f"  qkv_conv1d exact: {torch.equal(fused.qkv_conv1d.weight, exp_c)}")

    x = torch.randn(1, T, D_MODEL, device=dev, dtype=torch.bfloat16)
    outs, grads = {}, {}
    for name, mod in (("split", split), ("fused", fused)):
        xi = x.clone().requires_grad_(True)
        o = mod(xi)
        o.sum().backward()
        outs[name] = o.detach().float()
        grads[name] = xi.grad.detach().float()
    do = (outs["fused"] - outs["split"]).abs().max().item()
    dg = (grads["fused"] - grads["split"]).abs().max().item()
    oscale = outs["split"].abs().max().item()
    gscale = grads["split"].abs().max().item()
    grms = grads["split"].pow(2).mean().sqrt().item()
    print(f"\nmax |dout|  = {do:.3e}   out  max {oscale:.3e}  -> rel {do/max(oscale,1e-9):.2e}")
    print(
        f"max |dgrad| = {dg:.3e}   grad max {gscale:.3e}  rms {grms:.3e}  "
        f"-> rel-to-max {dg/max(gscale,1e-9):.2e}, rel-to-rms {dg/max(grms,1e-9):.2e}"
    )
    # bf16 has ~3 decimal digits; a difference at the 1e-3 relative level is rounding, not a bug.
    print(
        f"   bf16 eps ~ 7.8e-03; grad diff is {dg/max(gscale,1e-9)/7.8e-3:.2f}x eps relative to max"
    )

    # 3. interleaved A/B
    def run(mod):
        def f():
            mod.zero_grad(set_to_none=True)
            xi = x.detach().requires_grad_(True)
            mod(xi).sum().backward()

        return f

    fs, gs = run(fused), run(split)
    for _ in range(5):
        fs()
        gs()
    torch.cuda.synchronize()
    ts = {"split": [], "fused": []}
    for _ in range(args.iters):
        for name, fn in (("split", gs), ("fused", fs)):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts[name].append(time.perf_counter() - t0)
    med = {k: sorted(v)[len(v) // 2] * 1e3 for k, v in ts.items()}
    p10 = {k: sorted(v)[len(v) // 10] * 1e3 for k, v in ts.items()}
    p90 = {k: sorted(v)[9 * len(v) // 10] * 1e3 for k, v in ts.items()}
    print(f"\nsplit  {med['split']:7.3f} ms  [p10 {p10['split']:.3f}, p90 {p90['split']:.3f}]")
    print(f"fused  {med['fused']:7.3f} ms  [p10 {p10['fused']:.3f}, p90 {p90['fused']:.3f}]")
    print(f"-> {med['split']/med['fused']:.3f}x")


if __name__ == "__main__":
    main()
