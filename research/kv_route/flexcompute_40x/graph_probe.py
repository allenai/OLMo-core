"""What exactly stops a training step from being captured in a CUDA graph?

Past ~5x the routed step is launch-bound (GPU idle 0.9% dense -> 12.6% at the best arm), so the
remaining wall-clock is kernel-launch engineering, and CUDA graphs are the obvious tool. Every
whole-step capture attempt in ``bench_joint_sweep.py`` dies with the same useless message --
``cudaErrorStreamCaptureInvalidated: operation failed due to a previous error during capture`` --
which names the symptom, not the cause.

This bisects it. Each ``--case`` adds one more piece of a real step, and a capture is attempted
around exactly that piece. The first case that fails is the blocker.

**One case per process.** A failed capture leaves the CUDA context in a state where every later
capture reports the same "previous error", so cases run in separate processes (see the sbatch);
running them in a loop inside one process would blame the first failure for everything after it.

    python research/kv_route/flexcompute_40x/graph_probe.py --case fwd_bwd_fusedce
"""

import argparse
import json
import os
import time
from typing import Any, Dict

import torch

from olmo_core.nn.lm_head import LMLossImplementation, LMOutputWithLoss
from olmo_core.nn.transformer import TransformerConfig

VOCAB = 248320
OUT = os.path.dirname(os.path.abspath(__file__))
CASES = [
    "matmul",  # sanity: does capture work at all in this process/driver?
    "block_fwd",  # one transformer block, forward only
    "block_fwd_bwd",  # one transformer block, forward + backward
    "model_fwd_nohead",  # whole model forward, no LM head
    "model_fwd_bwd_nohead",  # + backward, loss = h.sum()
    "model_fwd_bwd_nohead_nobudget",  # + budget_attach off
    "model_fwd_bwd_defaultce",  # whole step, unfused cross entropy
    "model_fwd_bwd_fusedce",  # whole step as the trainer runs it (fused-linear CE)
    "model_fwd_bwd_gatedhead",  # + head gating as shipped (data-dependent gather shape)
    "model_fwd_bwd_statichead",  # + head gating with a FIXED row budget (static shape)
]


class NoHead(torch.nn.Module):
    def forward(self, h, **kwargs):
        loss = h.float().sum()
        return LMOutputWithLoss(logits=None, loss=loss, ce_loss=loss.detach(), z_loss=None)


class StaticGateHead(torch.nn.Module):
    """Head gating with a FIXED row budget, so the gather has a static shape.

    Shipped head gating (``LMHead._sparse_loss_positions``) does ``x[labels != ignore]``, whose
    output shape depends on tensor VALUES. That needs a host sync to learn the shape, which is
    exactly what a CUDA graph cannot capture -- and it is the only thing in a training step that
    blocks capture (see the case ladder above). Selecting a fixed ``capacity`` rows by ``topk``
    instead keeps the shape static.

    **Exact, not an approximation**, as long as ``capacity >=`` the number of supervised rows:
    ``topk`` over the 0/1 supervision mask returns all supervised rows first, and any surplus rows
    it pads with carry ``label == ignore_index``, which contributes zero loss and zero gradient
    under ``loss_reduction="sum"``. Same argument as fixed-capacity block skip.
    """

    def __init__(self, head: torch.nn.Module, capacity: int):
        super().__init__()
        self.head = head
        self.capacity = int(capacity)
        head._sparse_loss_positions = False  # this module does the gating instead
        # The fused-linear CE kernel is the ONE thing in a training step that cannot be captured
        # (case ladder: default CE captures at 2.09x, fused CE does not). Below a few thousand
        # rows it is also the wrong tool -- it exists to avoid a seq_len x vocab logits tensor
        # that a gathered head never builds -- so switch to the plain implementation, which is
        # exactly the rule LMHead already applies internally via sparse_loss_unfused_max_rows.
        if self.capacity <= getattr(head, "_sparse_loss_unfused_max_rows", 4096):
            head._loss_implementation = LMLossImplementation.default

    def forward(self, x, *, labels=None, ignore_index: int = -100, **kwargs):
        if labels is None:
            return self.head(x, labels=labels, ignore_index=ignore_index, **kwargs)
        flat = x.reshape(-1, x.shape[-1])
        lf = labels.reshape(-1)
        idx = torch.topk(
            (lf != ignore_index).to(torch.float32), self.capacity, sorted=False
        ).indices
        return self.head(
            flat.index_select(0, idx).unsqueeze(0),
            labels=lf.index_select(0, idx).unsqueeze(0),
            ignore_index=ignore_index,
            **kwargs,
        )


def try_capture(fn, warmup: int = 3) -> Dict[str, Any]:
    """Warm up on a side stream (the documented prerequisite), then capture and replay ``fn``."""
    out: Dict[str, Any] = {}
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    ts = []
    for _ in range(5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    out["eager_median_s"] = ts[len(ts) // 2]

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(10):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    out["graph_median_s"] = ts[len(ts) // 2]
    out["speedup"] = out["eager_median_s"] / out["graph_median_s"]
    out["captured"] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=CASES)
    ap.add_argument("--scale", default="0.8B")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument(
        "--head-capacity",
        type=int,
        default=256,
        help="fixed row budget for model_fwd_bwd_statichead; must be >= the supervised rows",
    )
    ap.add_argument("--ac", action="store_true")
    ap.add_argument(
        "--bskip-nosync",
        action="store_true",
        help="prototype: drop the per-block keep.sum().item() host sync (see the code note)",
    )
    ap.add_argument("--loss-density", type=float, default=0.0064)
    ap.add_argument(
        "--routed",
        default="",
        help="install the three routers and pin them to this bench_joint_sweep arm name",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = torch.device("cuda")
    T = args.ctx
    print(
        f"# {torch.cuda.get_device_name(0)} torch {torch.__version__} case={args.case}", flush=True
    )

    result: Dict[str, Any] = {"case": args.case, "scale": args.scale, "ctx": T}
    result["routed"] = args.routed
    try:
        if args.routed:
            # The real question the ladder feeds into: what does a graph buy ON TOP of the
            # routing win? Reuse the sweep's rig so the arm is the same object the wall-clock
            # matrix measured, rather than a second implementation of the same operating point.
            from types import SimpleNamespace

            from bench_joint_sweep import Rig, build_arms

            rig = Rig(
                SimpleNamespace(
                    scale=args.scale,
                    ctx=T,
                    batch=1,
                    attn_backend="flash_2",
                    ffn_start_layer=0,
                    divisors="1,2.4,4.9,10,32",
                    ac=args.ac,
                    docs=0,
                    loss_density=args.loss_density,
                )
            )
            arm = {a.name: a for a in build_arms("all")}[args.routed]
            rig.apply(arm)
            if args.bskip_nosync:
                # PROTOTYPE, measurement only. BlockSkipHolder.accumulate does
                # `float(keep.sum().item())` (block_skip.py:206) once per routed block, purely to
                # accumulate a budget statistic. That host sync is the last thing standing between
                # a routed step and a CUDA graph. Under FIXED CAPACITY the kept count is known
                # from the capacity fraction without looking at the tensor, so the same number can
                # be had sync-free -- which is what this patches in, to measure what removing the
                # blocker would be worth before anyone writes the real device-side version.
                holder = rig.model._block_skip["holder"]
                orig_acc = holder.accumulate

                def acc(exp_keep, keep, layer_idx, *a, _o=orig_acc, _h=holder, **kw):
                    n = keep.numel()
                    kept = float(min(n, max(1, round((_h.capacity_frac or 1.0) * keep.shape[-1]))))
                    kept *= n / keep.shape[-1]
                    _h._hard_kept[layer_idx] = kept
                    _h._n_tokens[layer_idx] = int(n)
                    _h.cum_kept += kept
                    _h.cum_tokens += int(n)
                    if _h.collect_loss:
                        _h._exp_keep.append(exp_keep)
                        _h._exp_layers.append(layer_idx)

                holder.accumulate = acc
            if args.case == "model_fwd_bwd_statichead":
                rig.model.lm_head = StaticGateHead(rig.real_head, args.head_capacity).to(dev)
            elif args.case == "model_fwd_bwd_nohead":
                rig.model.lm_head = NoHead().to(dev)
            rig.use_zero_grad = False
            rig.model.zero_grad(set_to_none=False)
            result["n_sup"] = rig.n_sup
            result.update(try_capture(rig.step))
        elif args.case == "matmul":
            a = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
            b = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
            c = torch.empty(4096, 4096, device=dev, dtype=torch.bfloat16)

            def fn():
                torch.mm(a, b, out=c)

            result.update(try_capture(fn))
        else:
            builder = {
                "0.8B": TransformerConfig.qwen3_5_0_8B,
                "4B": TransformerConfig.qwen3_5_4B,
            }[args.scale]
            cfg = builder(vocab_size=VOCAB)
            if args.case == "model_fwd_bwd_defaultce":
                cfg.lm_head.loss_implementation = LMLossImplementation.default
            else:
                cfg.lm_head.loss_implementation = LMLossImplementation.fused_linear
            torch.manual_seed(0)
            model = cfg.build(init_device="cpu").to(dev)
            model.init_weights(max_seq_len=T, device=dev)
            model = model.to(torch.bfloat16)
            model.train()
            d_model = model.embeddings.weight.shape[-1]

            if args.case.startswith("block_"):
                block = next(iter(model.blocks.values()))
                x = torch.randn(1, T, d_model, device=dev, dtype=torch.bfloat16)
                if args.case == "block_fwd":
                    with torch.no_grad():

                        def fn():
                            block(x)

                        result.update(try_capture(fn))
                else:
                    x = x.requires_grad_(True)

                    def fn():
                        model.zero_grad(set_to_none=False)
                        block(x).float().sum().backward()

                    result.update(try_capture(fn))
            else:
                ids = torch.randint(0, VOCAB, (1, T), device=dev)
                n_sup = max(1, T // 156)
                labels = torch.full((1, T), -100, dtype=torch.long, device=dev)
                labels[:, -n_sup:] = torch.randint(0, VOCAB, (1, n_sup), device=dev)
                if "nohead" in args.case:
                    model.lm_head = NoHead().to(dev)
                if args.case == "model_fwd_bwd_gatedhead":
                    model.lm_head._sparse_loss_positions = True
                if args.case == "model_fwd_bwd_statichead":
                    model.lm_head = StaticGateHead(model.lm_head, args.head_capacity).to(dev)
                if args.case == "model_fwd_bwd_nohead_nobudget":
                    model.budget_attach = False
                if args.case == "model_fwd_nohead":
                    with torch.no_grad():

                        def fn():
                            model(ids, labels=labels, loss_reduction="sum", return_logits=False)

                        result.update(try_capture(fn))
                else:

                    def fn():
                        model.zero_grad(set_to_none=False)
                        out = model(ids, labels=labels, loss_reduction="sum", return_logits=False)
                        out.loss.backward()

                    result.update(try_capture(fn))
    except Exception as exc:
        import traceback

        result["captured"] = False
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)[:800]
        traceback.print_exc()

    print("RESULT " + json.dumps(result), flush=True)
    tag = f"_{args.routed}" if args.routed else ""
    tag += "_nosync" if args.bskip_nosync else ""
    out = args.out or os.path.join(
        OUT, f"graph_probe_{args.scale}_{T}_{args.case}{tag}.json".replace("/", "-")
    )
    with open(out, "w") as fh:
        json.dump(result, fh, indent=1)


if __name__ == "__main__":
    main()
