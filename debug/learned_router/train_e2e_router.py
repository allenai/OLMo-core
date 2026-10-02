"""END-TO-END gradient training of the linear token router (router_lib.LinearRouter: embedding,
position, gold features) against a TARGET KEEP RATE -- no rule initialisation or distillation.

    python debug/learned_router/train_e2e_router.py --task nq --ckpt ... --train-root .../router_train_e2e \\
        --val-root .../router_val_e2e --rhos 0.1

Relaxation: the exact ``soft_keep`` path (attention +log p key bias, GDN beta/g scaling and the
removal-aware short conv; binary gates reproduce the hard compaction) with hard-concrete gates
(log alpha = router logit; temperature beta0 -> beta1 over ``--anneal-frac`` of training,
straight-through binary forward from ``--st-frac``).

Objective (CoFi / L0-pruning style target-sparsity Lagrangian):

    L = task + lambda1 * (keep - rho_t) + lambda2 * (keep - rho_t)^2
    task = answer CE(relaxed) + w_kl * KL(p_full || p_relaxed)   (answer positions)

``keep`` = expected L0 keep fraction (mean P(z > 0) over the step's routed tokens); rho_t anneals
linearly 1 -> rho over the first ``--warm-frac`` of steps, then holds; lambda1/lambda2 by gradient
ASCENT (SGD, ``--lam-lr``). Init: bias = logit(0.95), every other weight 0.

The saved router is the FINAL state (end of the straight-through phase). Per-epoch VAL metrics use
the deterministic gate (logit > 0) with the exact hard removal; ``select_e2e.py`` picks among the
rho runs AFTER training.
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
import router_lib as RL  # noqa: E402
import train_diff_router as TD  # noqa: E402
import train_router as TR  # noqa: E402

G = TR.G
log = TR.log


class _Swap:
    """--xtask-offload: one frozen per-task model resident on the GPU at a time (the others wait on CPU).
    Rows are visited in task blocks, so models move only at block boundaries."""

    enabled = False
    resident = None
    n_swaps = 0
    sec = 0.0


def swap_in(m):
    if not _Swap.enabled or m is None or m is _Swap.resident:
        return
    t = time.time()
    if _Swap.resident is not None:
        _Swap.resident.to("cpu")
    m.to("cuda")
    _Swap.resident = m
    _Swap.n_swaps += 1
    _Swap.sec += time.time() - t


def _swapping(fn):
    def wrapped(model, *args, **kw):
        swap_in(model)
        return fn(model, *args, **kw)

    return wrapped


def relaxed_logits(model, p, z, drop_zero: bool = False):
    """Answer logits under the relaxed removal ``z``. ``drop_zero``: physically compact away the
    routed tokens whose gate is exactly 0 (original RoPE positions kept) and apply ``soft_keep`` to
    the survivors only. Forward-identical to the full row (a z = 0 token is already removed there:
    attention bias log(1e-8), identity GDN step, skipped by the removal-aware conv) and, with the
    plain [0, 1] clamp, gradient-identical too (a clamped gate has zero task gradient)."""
    model.eval()
    model._pooled_keep_holder = None
    sk = RL.soft_keep_row(p["feats"], z, p["T"])
    if not drop_zero:
        return model(p["x"], logits_to_keep=p["pred"][None], soft_keep=sk[None])[0].float()
    keep = torch.ones(p["T"], dtype=torch.bool, device=p["x"].device)
    keep[p["feats"]["idx"][z.detach() <= 0]] = False
    kept = keep.nonzero().flatten()
    cols = torch.searchsorted(kept, p["pred"])
    assert bool((kept[cols] == p["pred"]).all()), "answer prediction position dropped"
    return model(p["x"][:, kept], logits_to_keep=cols[None], soft_keep=sk[kept][None], position_ids=kept[None])[0].float()


@torch.no_grad()
def calibrate_offset(la: torch.Tensor, target: float, beta: float) -> float:
    """Scalar c with mean_i P(z_i > 0 | log_alpha_i + c) == target (bisection; P is monotone in c)."""
    shift = beta * math.log(-RL.HC_GAMMA / RL.HC_ZETA)
    # bracket from the logit range: a FIXED [-60, 60] bracket silently saturated once |logits| grew past it
    # (b drifts under calibration), leaving the budget unenforced (keep 0.97 at rho_t 0.40). Fixed 2026-09-30.
    lo, hi = -float(la.max()) - 60.0, -float(la.min()) + 60.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if float(torch.sigmoid(la + mid - shift).mean()) > target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def topk_keep(la: torch.Tensor, rho: float) -> torch.Tensor:
    """Deterministic eval rule under budget calibration: keep the top ceil(rho * N) routed tokens."""
    N = int(la.numel())
    k = min(N, max(0, int(math.ceil(rho * N))))
    keep = torch.zeros(N, dtype=torch.bool, device=la.device)
    if k:
        keep[torch.topk(la, k).indices] = True
    return keep


@torch.no_grad()
def global_offset(router, rows, rho):
    """Offset c such that the deterministic gate (logit + c > 0) keeps a fraction rho of all routed
    tokens pooled over ``rows`` (the global-threshold eval rule of batch-scope calibration)."""
    z = torch.cat([router.logits(p["feats"], p["e"].float() if router.use_emb else None) for p in rows])
    k = min(int(z.numel()), max(1, int(math.ceil(rho * z.numel()))))
    return -float(torch.topk(z, k).values[-1]) + 1e-4


@torch.no_grad()
def val_threshold(model, router, rows, cef, c):
    dce, comp, keep = [], [], []
    for i, p in enumerate(rows):
        k = router.logits(p["feats"], p["e"].float() if router.use_emb else None) + c > 0
        ce, cc = TR.ce_masked(p.get("_m", model), p, RL.keep_mask_from(p["feats"], k, p["T"])[None])
        dce.append(float(ce[0]) - cef[i])
        comp.append(float(cc[0]))
        keep.append(float(k.float().mean()) if k.numel() else 1.0)
    return {"dce": float(np.mean(dce)), "dce_median": float(np.median(dce)), "dce_max": float(np.max(dce)),
            "dce_se": float(np.std(dce, ddof=1) / math.sqrt(len(dce))) if len(dce) > 1 else float("nan"),
            "comp": float(np.mean(comp)), "keep": float(np.mean(keep)), "n": len(dce), "offset": c, "per_row_dce": dce}


@torch.no_grad()
def val_pair(model, router, rows, cef):
    """Hard-deletion dCE with the PAIRED budget: each row keeps exactly ``p["k_pair"]`` routed tokens (the
    count gold_fl20p8_noslot keeps on that row), top-k by router score -- the eval rule of ``router_<name>@pair``."""
    dce, comp = [], []
    for i, p in enumerate(rows):
        z = router.logits(p["feats"], p["e"].float() if router.use_emb else None)
        k = RL.topk_keep(z, p["feats"], int(p["k_pair"]), getattr(router, "span", 0))
        ce, cc = TR.ce_masked(p.get("_m", model), p, RL.keep_mask_from(p["feats"], k, p["T"])[None])
        dce.append(float(ce[0]) - cef[i])
        comp.append(float(cc[0]))
    return {"dce": float(np.mean(dce)), "dce_median": float(np.median(dce)), "comp": float(np.mean(comp)), "n": len(dce),
            "dce_se": float(np.std(dce, ddof=1) / math.sqrt(len(dce))) if len(dce) > 1 else float("nan"), "per_row_dce": dce}


@torch.no_grad()
def match_comp_offset(model, router, rows, target):
    """Offset whose deterministic gate gives mean T2/T == target over ``rows`` (bisection; no labels used)."""
    zs = [router.logits(p["feats"], p["e"].float() if router.use_emb else None) for p in rows]
    zall = torch.cat(zs)
    lo, hi = -float(zall.max()) - 1.0, -float(zall.min()) + 1.0  # bracket from the logit range (see calibrate_offset)
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if getattr(router, "marker_follow", False):
            comp = np.mean([RL.follow_comp(z + mid > 0, p["feats"], p["T"], "whole" if router.marker_follow == "whole" else "any") for p, z in zip(rows, zs)])
        else:
            comp = np.mean([((T := p["T"]) - int(z.numel()) + int((z + mid > 0).sum())) / T for p, z in zip(rows, zs)])
        if comp > target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


@torch.no_grad()
def val_topk(model, router, rows, cef, rho):
    dce, comp, keep = [], [], []
    for i, p in enumerate(rows):
        k = topk_keep(router.logits(p["feats"], p["e"].float() if router.use_emb else None), rho)
        ce, c = TR.ce_masked(p.get("_m", model), p, RL.keep_mask_from(p["feats"], k, p["T"])[None])
        dce.append(float(ce[0]) - cef[i])
        comp.append(float(c[0]))
        keep.append(float(k.float().mean()) if k.numel() else 1.0)
    return {"dce": float(np.mean(dce)), "dce_median": float(np.median(dce)), "dce_max": float(np.max(dce)),
            "dce_se": float(np.std(dce, ddof=1) / math.sqrt(len(dce))) if len(dce) > 1 else float("nan"),
            "comp": float(np.mean(comp)), "keep": float(np.mean(keep)), "n": len(dce)}


def verify_batch(model, rows, lpf, a, n: int = 4):
    """Batched (one right-padded forward) vs per-row relaxed loss and router gradient, fixed gates."""
    dev = next(model.parameters()).device
    router = RL.LinearRouter(int(model.embeddings.weight.shape[1]), a.variant).to(dev)
    with torch.no_grad():
        router.w_gold.fill_(2.0)
        router.w_emb.copy_(torch.randn(router.w_emb.shape, generator=torch.Generator().manual_seed(3)).to(dev))
    ps = rows[:n]
    res = {}
    for mode in ("per-row", "batched"):
        router.zero_grad(set_to_none=True)
        zs = [RL.hard_concrete(router.logits(p["feats"], p["e"].float()), 0.5, torch.Generator().manual_seed(7 + j)) for j, p in enumerate(ps)]
        torch.cuda.synchronize()
        t0 = time.time()
        lgs = relaxed_logits_batch(model, ps, zs, a.pad_id) if mode == "batched" else [relaxed_logits(model, p, z) for p, z in zip(ps, zs)]
        loss = 0.0
        for j, (p, lg) in enumerate(zip(ps, lgs)):
            lp = F.log_softmax(lg, -1)
            lpf_j = lpf[j].to(dev).float()
            loss = loss + (F.nll_loss(lp, p["tgt"]) + a.w_kl * (lpf_j.exp() * (lpf_j - lp)).sum(-1).mean()) / n
        loss.backward()
        torch.cuda.synchronize()
        res[mode] = (float(loss), torch.cat([q.grad.flatten() for q in router.parameters()]), time.time() - t0)
    rl = abs(res["batched"][0] - res["per-row"][0]) / max(abs(res["per-row"][0]), 1e-12)
    rg = float((res["batched"][1] - res["per-row"][1]).norm() / res["per-row"][1].norm())
    log(f"batch verify ({n} rows): loss {res['per-row'][0]:.6f} vs {res['batched'][0]:.6f} (rel {rl:.1e}), router grad rel diff "
        f"{rg:.1e}; fwd+bwd {res['per-row'][2]:.2f}s -> {res['batched'][2]:.2f}s ({res['per-row'][2] / res['batched'][2]:.1f}x)")
    return {"rel_loss": rl, "rel_grad": rg, "sec_per_row": res["per-row"][2], "sec_batched": res["batched"][2]}


def verify_hard_drop(model, rows, lpf, a, n: int = 3):
    """Full-row soft path vs --hard-drop-zero: loss and router gradient on a few rows (fixed seed,
    plain clamp, a router at an intermediate keep), plus fwd+bwd step time of each path."""
    dev = next(model.parameters()).device
    out = []
    for bias in (0.0, -1.5):
        router = RL.LinearRouter(int(model.embeddings.weight.shape[1]), a.variant).to(dev)
        with torch.no_grad():
            router.b.fill_(bias)
            router.w_gold.fill_(3.0)
            router.w_pos.copy_(0.5 * torch.randn(router.w_pos.shape, generator=torch.Generator().manual_seed(1)).to(dev))
        # verify on n rows of EACH length (the train list is 2k rows first, then 8k)
        by_len = sorted(rows, key=lambda q: q["T"])
        pick = by_len[:n] + by_len[-n:]
        for i, p in enumerate(pick):
            res = {}
            for mode in (False, True):
                router.zero_grad(set_to_none=True)
                gen = torch.Generator().manual_seed(123 + i)
                la = router.logits(p["feats"], p["e"].float())
                z = RL.hard_concrete(la, 0.3, gen, st_clamp=False)
                torch.cuda.synchronize()
                t0 = time.time()
                lp = F.log_softmax(relaxed_logits(model, p, z, drop_zero=mode), -1)
                lpf_i = lpf[next(j for j, q in enumerate(rows) if q is p)].to(dev).float()
                loss = F.nll_loss(lp, p["tgt"]) + a.w_kl * (lpf_i.exp() * (lpf_i - lp)).sum(-1).mean()
                loss.backward()
                torch.cuda.synchronize()
                g = torch.cat([q.grad.flatten() for q in (router.b, router.w_pos, router.w_gold, router.w_emb)])
                res[mode] = (float(loss), g.clone(), time.time() - t0, float((z > 0).float().mean()))
            (l0, g0, t0_, k0), (l1, g1, t1_, _) = res[False], res[True]
            rel_l = abs(l1 - l0) / max(abs(l0), 1e-8)
            rel_g = float((g1 - g0).norm() / g0.norm().clamp(min=1e-12))
            out.append({"bias": bias, "row": i, "T": p["T"], "z_pos_frac": k0, "loss_full": l0, "loss_drop": l1, "rel_loss": rel_l,
                        "rel_grad": rel_g, "sec_full": t0_, "sec_drop": t1_})
            log(f"hard-drop verify bias {bias:+.1f} row {i} T={p['T']} z>0 {k0:.2f}: loss {l0:.6f} vs {l1:.6f} (rel {rel_l:.1e}), "
                f"router grad rel diff {rel_g:.1e}; fwd+bwd {t0_:.2f}s -> {t1_:.2f}s ({t0_ / max(t1_, 1e-6):.2f}x)")
    return out


def relaxed_logits_batch(model, ps, zs, pad_id: int):
    """One right-padded forward for several rows (the soft_keep of pads is 1; causal attention and the
    recurrent mixers never let a pad influence an earlier position). Returns per-row answer logits."""
    model.eval()
    model._pooled_keep_holder = None
    dev = ps[0]["x"].device
    T = max(p["T"] for p in ps)
    A = max(int(p["pred"].numel()) for p in ps)
    x = torch.full((len(ps), T), pad_id, dtype=ps[0]["x"].dtype, device=dev)
    sk = torch.ones(len(ps), T, dtype=torch.float32, device=dev)
    cols = torch.zeros(len(ps), A, dtype=torch.long, device=dev)
    rows_sk = []
    for b, (p, z) in enumerate(zip(ps, zs)):
        x[b, : p["T"]] = p["x"][0]
        rows_sk.append(F.pad(RL.soft_keep_row(p["feats"], z, p["T"]), (0, T - p["T"]), value=1.0))
        n = int(p["pred"].numel())
        cols[b, :n] = p["pred"]
        cols[b, n:] = p["pred"][-1]
    sk = torch.stack(rows_sk)
    lg = model(x, logits_to_keep=cols, soft_keep=sk).float()
    return [lg[b, : int(p["pred"].numel())] for b, p in enumerate(ps)]


@torch.no_grad()
def full_logprobs(model, p):
    model.eval()
    model._pooled_keep_holder = None
    lg = model(p["x"], logits_to_keep=p["pred"][None])[0].float()
    return F.log_softmax(lg, -1).to(torch.bfloat16).cpu()


def keep_dropout(z: torch.Tensor, p: float, gen) -> torch.Tensor:
    """Train-time keep-dropout: additionally remove a random fraction ``p`` of the routed tokens (resampled
    every step) after the gates are sampled, so the router must keep enough margin that the answer survives
    random extra loss (a saturated objective -- CE_full ~ 0 -- otherwise gives no reason to keep a span whole).
    Eval is unchanged."""
    if p <= 0:
        return z
    m = (torch.rand(z.shape, generator=gen) >= p).to(z.device, z.dtype)
    return z * m


def unit_gate(la, beta, gen, st, st_clamp, feats, span: int, kd: float, kd_token: bool = False):
    """Hard-concrete gates (+ keep-dropout). Span routers sample ONE gate per unit (span / marker) and
    broadcast it to the unit's tokens, so a span is kept or removed whole at train time too."""
    if not span:
        return keep_dropout(RL.hard_concrete(la, beta, gen, straight_through=st, st_clamp=st_clamp), kd, gen)
    u = feats[f"unit{span}"].to(la.device)
    zu = RL.hard_concrete(RL.unit_mean(la, u), beta, gen, straight_through=st, st_clamp=st_clamp)
    if kd_token:  # keep-dropout on TOKENS after broadcasting the unit gate (as F0/F2 get it)
        return keep_dropout(zu[u], kd, gen)
    return keep_dropout(zu, kd, gen)[u]


def train_rho(model, rho, tr, va, cef_tr, lpf_tr, cef_va, a, wb=None):
    dev = tr[0]["x"].device  # rows live on the GPU; the model may be offloaded (--xtask-offload)
    torch.manual_seed(1000 + a.seed)
    gen = torch.Generator().manual_seed(2000 + a.seed)
    router = RL.LinearRouter(int(model.embeddings.weight.shape[1]), a.variant).to(dev)
    router.route_markers = a.route_markers
    router.marker_follow = ("whole" if a.marker_follow_whole else True) if a.marker_follow_doc else False
    with torch.no_grad():
        router.b.fill_(math.log(a.init_p / (1 - a.init_p)))
        if a.route_markers and a.marker_init_p is not None:
            # markers start near-kept (still learned end to end): dropping ~5% of document boundaries at
            # random from step 1 merges documents before the router can learn anything
            router.w_marker.fill_(math.log(a.marker_init_p / (1 - a.marker_init_p)) - math.log(a.init_p / (1 - a.init_p)))
    lr_gb = a.lr_gold_bias if a.lr_gold_bias is not None else a.lr  # separate lr so w_gold can outrun the bias
    opt = torch.optim.Adam([{"params": [router.w_pos, router.w_marker, router.w_rel, router.w_span], "lr": a.lr},
                            {"params": [router.b, router.w_gold], "lr": lr_gb},
                            {"params": [router.w_emb], "lr": a.emb_lr}], betas=(0.9, a.adam_beta2))
    # structure-regularised oracle (--residual-l2 > 0): train-time logit = router(features) + r_i, a free
    # per-ROW residual with an L2 penalty; eval uses the router alone. lambda -> inf is the plain e2e router,
    # lambda -> 0 the free per-row oracle.
    resid = [None] * len(tr)
    if a.residual_l2 > 0:
        resid = [torch.zeros(int(p["feats"]["idx"].numel()), device=dev, requires_grad=True) for p in tr]
        opt.add_param_group({"params": resid, "lr": a.residual_lr})

    def rlog(i):
        z = router.logits(tr[i]["feats"], tr[i]["e"].float() if router.use_emb else None)
        return z if resid[i] is None else z + resid[i]

    lam1, lam2 = 0.0, 0.0
    if a.xtask_offload or a.xtasks:  # task blocks: batches never straddle two tasks (one model swap per block)
        blocks = {}
        for i, p in enumerate(tr):
            blocks.setdefault(p["task"], []).append(i)
        n_steps = sum(math.ceil(len(v) / a.rows_per_step) for v in blocks.values()) * a.epochs
    else:
        n_steps = math.ceil(len(tr) / a.rows_per_step) * a.epochs
    steps, epochs = [], []
    v0 = TD.val_hard(model, router, va, cef_va)
    epochs.append({"epoch": 0, "val": {k: v for k, v in v0.items() if k != "per_row_dce"}})
    log(f"[rho{rho}] {len(tr)} train rows, {n_steps} steps; epoch 0 val dCE {v0['dce']:+.4f} T2/T {v0['comp']:.2f}")
    t0 = time.time()
    step = 0
    for ep in range(1, a.epochs + 1):
        order = torch.randperm(len(tr), generator=gen).tolist()
        if a.xtask_offload or a.xtasks:
            tnames = list(blocks)
            batches = []
            for ti in torch.randperm(len(tnames), generator=gen).tolist():
                bl = [blocks[tnames[ti]][j] for j in torch.randperm(len(blocks[tnames[ti]]), generator=gen).tolist()]
                batches += [bl[k : k + a.rows_per_step] for k in range(0, len(bl), a.rows_per_step)]
        else:
            batches = [order[s0 : s0 + a.rows_per_step] for s0 in range(0, len(order), a.rows_per_step)]
        acc = {"ce": [], "kl": [], "dce": [], "keep": [], "keep_det": []}
        for batch in batches:
            prog = step / max(1, n_steps - 1)
            rho_b_ = tr[batch[0]].get("rho_task", rho)  # cross-task: the batch's own task budget
            rho_t = 1.0 - (1.0 - rho_b_) * min(1.0, prog / a.warm_frac)
            beta = a.beta0 * (a.beta1 / a.beta0) ** min(1.0, prog / a.anneal_frac)
            st = prog >= a.st_frac
            opt.zero_grad(set_to_none=True)
            keeps = []
            prof = a.profile and step == 3
            tm = {}

            def tick(k, t_prev=[None]):  # cuda-synced phase timer for the profiled step
                if prof:
                    torch.cuda.synchronize()
                    now = time.time()
                    if t_prev[0] is not None:
                        tm[k] = tm.get(k, 0.0) + now - t_prev[0]
                    t_prev[0] = now

            tick("start")
            if a.batch_rows and not a.hard_drop_zero:
                zs, las = [], []
                for i in batch:
                    p = tr[i]
                    las.append(rlog(i))
                if a.objective == "calibrated" and a.calib_scope == "batch":
                    # ONE offset for the step's pooled routed tokens: rows may keep different fractions
                    c_b = calibrate_offset(torch.cat([l.detach() for l in las]), rho_t, beta)
                    las = [l + c_b for l in las]
                elif a.objective == "calibrated":
                    las = [l + calibrate_offset(l.detach(), rho_t, beta) for l in las]
                for i, la in zip(batch, las):
                    zs.append(unit_gate(la, beta, gen, st, True, tr[i]["feats"], router.span, a.keep_dropout, a.kd_token))
                    acc["keep_det"].append(float((la.detach() > 0).float().mean()))
                tick("router+gates")
                lgs = relaxed_logits_batch(model, [tr[i] for i in batch], zs, a.pad_id)
                tick("forward")
                loss = 0.0
                for i, lg in zip(batch, lgs):
                    p = tr[i]
                    lp = F.log_softmax(lg, -1)
                    ce = F.nll_loss(lp, p["tgt"])
                    lpf = lpf_tr[i].to(dev).float()
                    kl = (lpf.exp() * (lpf - lp)).sum(-1).mean()
                    loss = loss + (ce + a.w_kl * kl) / len(batch)
                    acc["ce"].append(float(ce))
                    acc["kl"].append(float(kl))
                    acc["dce"].append(float(ce) - cef_tr[i])
                tick("loss")
                loss.backward()
                tick("backward")
            for i in (batch if not (a.batch_rows and not a.hard_drop_zero) else []):
                p = tr[i]
                e = p["e"].float() if router.use_emb else None
                la = rlog(i)
                if a.objective == "calibrated":
                    # budget calibration: the offset puts the row's expected keep at exactly rho_t, so the
                    # router only learns the RANKING (no multiplier); c carries no gradient
                    la = la + calibrate_offset(la.detach(), rho_t, beta)
                z = unit_gate(la, beta, gen, st, not a.hard_drop_zero, p["feats"], router.span, a.keep_dropout, a.kd_token)
                tick("router+gates")
                # hard-drop-zero: compact away z == 0 tokens (exact, see relaxed_logits); in the final
                # straight-through phase the full-row path is kept, so dropped tokens there still get
                # their exact straight-through task gradient
                lg = relaxed_logits(p.get("_m", model), p, z, drop_zero=a.hard_drop_zero and (not st or a.objective == "calibrated"))
                tick("forward")
                lp = F.log_softmax(lg, -1)
                ce = F.nll_loss(lp, p["tgt"])
                lpf = lpf_tr[i].to(dev).float()
                kl = (lpf.exp() * (lpf - lp)).sum(-1).mean()
                tick("loss")
                ((ce + a.w_kl * kl) / len(batch)).backward()
                tick("backward")
                acc["ce"].append(float(ce))
                acc["kl"].append(float(kl))
                acc["dce"].append(float(ce) - cef_tr[i])
                acc["keep_det"].append(float((la.detach() > 0).float().mean()))
            # the per-row backward freed the router graph: recompute the (cheap) router logits for the
            # batch-level keep (Lagrangian term, or just the logged value under calibration)
            las_k = [rlog(i) for i in batch]
            if a.objective == "calibrated" and a.calib_scope == "batch":
                c_b = calibrate_offset(torch.cat([l.detach() for l in las_k]), rho_t, beta)
                las_k = [l + c_b for l in las_k]
            elif a.objective == "calibrated":
                las_k = [l + calibrate_offset(l.detach(), rho_t, beta) for l in las_k]
            keep = RL.hard_concrete_p_nonzero(torch.cat(las_k), beta).mean() if a.calib_scope == "batch" else \
                torch.stack([RL.hard_concrete_p_nonzero(l, beta).mean() for l in las_k]).mean()
            gap = keep - rho_t
            if a.objective == "lagrangian":
                (lam1 * gap + lam2 * gap * gap).backward()
            if a.residual_l2 > 0:
                (a.residual_l2 * sum(resid[i].pow(2).mean() for i in batch) / len(batch)).backward()
            if a.wpos_l2 > 0:  # small L2 on the position weights (16-row runs overfit single offsets)
                (a.wpos_l2 * router.w_pos.pow(2).sum()).backward()
            opt.step()
            if a.emb_cap > 0:  # cap the embedding term: token identity can only nudge the score
                with torch.no_grad():
                    nrm = float(router.w_emb.norm())
                    if nrm > a.emb_cap:
                        router.w_emb.mul_(a.emb_cap / nrm)
            tick("lagrange+opt")
            if prof:
                tot = sum(tm.values())
                log(f"PROFILE step {step + 1} ({len(batch)} rows, T={[tr[i]['T'] for i in batch]}): total {tot:.2f}s | "
                    + "  ".join(f"{k} {v:.3f}s ({100 * v / tot:.0f}%)" for k, v in tm.items()))
            if a.objective == "lagrangian":
                g = float(gap.detach())
                lam1 += a.lam_lr * g  # gradient ascent on the multipliers
                lam2 += a.lam_lr * g * g
            acc["keep"].append(float(keep))
            step += 1
            rec = {"step": step, "epoch": ep, "rho_t": rho_t, "beta": beta, "st": st, "keep": float(keep),
                   "ce": float(np.mean(acc["ce"][-len(batch):])), "kl": float(np.mean(acc["kl"][-len(batch):])),
                   "lam1": lam1, "lam2": lam2, "w_gold": float(router.w_gold), "w_marker": float(router.w_marker), "b": float(router.b),
                   "sec": time.time() - t0}
            steps.append(rec)
            if wb is not None:
                wb.log({f"train/{k}": v for k, v in rec.items() if k not in ("step", "st")} | {"train/st": int(st)}, step=step + getattr(a, "_step_off", 0))
            if step in (1, 2, 5) or step % 10 == 0:
                el = time.time() - t0
                log(f"[rho{rho}] step {step}/{n_steps} ep {ep}: keep {rec['keep']:.3f} (rho_t {rho_t:.3f}) CE {rec['ce']:.4f} KL {rec['kl']:.4f} "
                    f"lam1 {lam1:+.2f} lam2 {lam2:.2f} beta {beta:.2f}{' ST' if st else ''} w_gold {rec['w_gold']:+.2f} w_marker {rec['w_marker']:+.2f} b {rec['b']:+.2f} "
                    f"| {el:.0f}s, ETA {el / step * (n_steps - step):.0f}s")
        do_val = a.objective == "lagrangian" or ep in (max(1, a.epochs // 2), a.epochs)
        if not do_val:
            trm = {k: float(np.mean(x)) for k, x in acc.items()}
            epochs.append({"epoch": ep, "train": trm, "rho_t": rho_t, "beta": beta, "st": st,
                           "w_gold": float(router.w_gold), "b": float(router.b)})
            if ep in (1, 2, 5) or ep % 10 == 0:
                log(f"[rho{rho}] epoch {ep}: train CE {trm['ce']:.4f} (dCE {trm['dce']:+.4f}) KL {trm['kl']:.4f} keepL0 {trm['keep']:.3f} "
                    f"rho_t {rho_t:.3f} beta {beta:.2f}{' ST' if st else ''} w_gold {float(router.w_gold):+.2f}")
            continue
        if a.objective == "lagrangian":
            v = TD.val_hard(model, router, va, cef_va)
        elif a.calib_scope == "batch":
            c_star = global_offset(router, tr, rho)
            v = val_threshold(model, router, va, cef_va, c_star)
        else:
            v = val_topk(model, router, va, cef_va, rho)
        trm = {k: float(np.mean(x)) for k, x in acc.items()}
        if a.objective == "calibrated" and ep == a.epochs:
            trm["hard_topk"] = val_threshold(model, router, tr, cef_tr, c_star) if a.calib_scope == "batch" else val_topk(model, router, tr, cef_tr, rho)
        if wb is not None:
            wb.log({"val/dce": v["dce"], "val/dce_median": v["dce_median"], "val/compaction": v["comp"], "val/keep": v["keep"],
                    "epoch": ep} | {f"epoch_train/{k}": x for k, x in trm.items()}, step=step + getattr(a, "_step_off", 0))
        epochs.append({"epoch": ep, "train": trm, "val": {k: x for k, x in v.items() if k != "per_row_dce"}, "val_rule": "topk" if a.objective == "calibrated" else "p>0.5",
                       "rho_t": rho_t, "beta": beta, "st": st, "lam1": lam1, "lam2": lam2,
                       "w_gold": float(router.w_gold), "b": float(router.b)})
        log(f"[rho{rho}] epoch {ep}: train CE {trm['ce']:.4f} (dCE {trm['dce']:+.4f}) KL {trm['kl']:.4f} keepL0 {trm['keep']:.3f} "
            f"keep_det {trm['keep_det']:.3f} rho_t {rho_t:.3f} | val HARD dCE {v['dce']:+.4f} (median {v['dce_median']:+.4f}) "
            f"T2/T {v['comp']:.3f} keep {v['keep']:.3f} | lam1 {lam1:+.2f} lam2 {lam2:.2f}"
            + (f" | TRAIN HARD top-k dCE {trm['hard_topk']['dce']:+.4f} T2/T {trm['hard_topk']['comp']:.3f} (train-val gap "
               f"{v['dce'] - trm['hard_topk']['dce']:+.4f})" if "hard_topk" in trm else ""))
    state = router.state()
    if a.objective == "calibrated" and a.calib_scope == "batch":
        # global threshold folded into the bias: the driver's default p > 0.5 rule applies
        state["b"] = state["b"] + global_offset(router, tr, rho)
        state.update({"keep_rule": "p>0.5", "rho": rho, "calib": "batch"})
    elif a.objective == "calibrated":
        state.update({"keep_rule": "topk", "rho": rho})
    return state, {"rho": rho, "objective": a.objective, "steps": steps, "epochs": epochs, "final_val": epochs[-1]["val"],
                            "train_sec": time.time() - t0}


def heuristic_mask(p, ids) -> torch.Tensor:
    """(S,) bool: gold_fl20p8_noslot's REAL set for one prepared row (the model's own marking calls; gold
    documents whole; gold-blind fl20p8 when the task has no gold set). Diagnostic only."""
    from check_hand_fl import heuristic_real
    from olmo_core.nn.attention.chunked_mask import build_chunk_ids_from_tokens

    x = p["x"][0].cpu()
    cid = build_chunk_ids_from_tokens(x[None], doc_start_id=ids.doc_start, doc_end_id=ids.doc_end, eos_id=ids.eos, mode="chunked")[0]
    return heuristic_real(x, cid, p["gold"], ids).to(p["x"].device)


def heuristic_route_keep(p, hr) -> torch.Tensor:
    """The heuristic's decision on every ROUTED token of the row (markers included when routed)."""
    return hr[p["feats"]["idx"]]


@torch.no_grad()
def heuristic_eval(model, rows, cef, ids):
    """Exact gold_fl20p8_noslot on prepared rows through the router's compaction path: body tokens by
    the heuristic, markers freed only for fully kept bodies (the heuristic's marker semantics)."""
    dce, comp, keep = [], [], []
    for i, p in enumerate(rows):
        m = p.get("_m", model)
        swap_in(m)
        pst = m._pooled_soft_tokens
        old = pst.get("keep_token_mask_markers")
        hr = heuristic_mask(p, ids)
        body = torch.zeros_like(hr)
        f = p["feats"]
        body[f["idx"][f["is_marker"] == 0]] = True
        pst["keep_token_mask_markers"] = False
        try:
            ce, cc = TR.ce_masked(m, p, (hr & body)[None])
        finally:
            pst["keep_token_mask_markers"] = old
        dce.append(float(ce[0]) - cef[i])
        comp.append(float(cc[0]))
        keep.append(float(heuristic_route_keep(p, hr).float().mean()))
    return {"dce": float(np.mean(dce)), "dce_median": float(np.median(dce)),
            "dce_se": float(np.std(dce, ddof=1) / math.sqrt(len(dce))) if len(dce) > 1 else float("nan"),
            "comp": float(np.mean(comp)), "keep": float(np.mean(keep)), "n": len(dce), "per_row_dce": dce}


@torch.no_grad()
def diag_vs_heuristic(model, router, tr, va, cef_tr, cef_va, ids, tag=""):
    """Optimisation vs generalisation: router vs the exact heuristic on TRAIN and VAL rows, the router's
    threshold matched (label-free) to the heuristic's mean T2/T on the same rows. Also the gold-token
    keep fraction of both."""
    out = {}
    for split, rows, cef in (("train", tr, cef_tr), ("val", va, cef_va)):
        h = heuristic_eval(model, rows, cef, ids)
        c = match_comp_offset(model, router, rows, h["comp"])
        r = val_threshold(model, router, rows, cef, c)
        gk_r, gk_h = [], []
        for p in rows:
            f = p["feats"]
            g = (f["gold"] > 0) & (f["is_marker"] == 0)
            if bool(g.any()):
                k = router.logits(f, p["e"].float() if router.use_emb else None) + c > 0
                gk_r.append(float(k[g].float().mean()))
                gk_h.append(float(heuristic_route_keep(p, heuristic_mask(p, ids))[g].float().mean()))
        pr = [a_ - b_ for a_, b_ in zip(r.get("per_row_dce", []), h["per_row_dce"])]
        out[split] = {"heuristic": {k: v for k, v in h.items() if k != "per_row_dce"},
                      "router_at_heur_comp": {k: v for k, v in r.items() if k != "per_row_dce"},
                      "paired": float(np.mean(pr)) if pr else None,
                      "paired_se": float(np.std(pr, ddof=1) / math.sqrt(len(pr))) if len(pr) > 1 else None,
                      "gold_keep_router": float(np.mean(gk_r)) if gk_r else None, "gold_keep_heuristic": float(np.mean(gk_h)) if gk_h else None}
        o = out[split]
        log(f"[diag{tag}] {split}: heuristic dCE {h['dce']:+.4f} T2/T {h['comp']:.3f} | router @ same T2/T dCE {r['dce']:+.4f} "
            f"(T2/T {r['comp']:.3f}) paired {o['paired'] if o['paired'] is not None else float('nan'):+.4f} +- "
            f"{o['paired_se'] if o['paired_se'] is not None else float('nan'):.4f} | gold keep router "
            f"{o['gold_keep_router'] if o['gold_keep_router'] is not None else float('nan'):.3f} heuristic "
            f"{o['gold_keep_heuristic'] if o['gold_keep_heuristic'] is not None else float('nan'):.3f}")
    tp, vp = out["train"]["paired"], out["val"]["paired"]
    if tp is not None and vp is not None:
        se = max(out["train"]["paired_se"] or 0.0, 0.005)
        out["diagnosis"] = ("optimisation" if tp > se else ("generalisation" if vp > max(out["val"]["paired_se"] or 0.0, 0.005) else "parity"))
        log(f"[diag{tag}] diagnosis: {out['diagnosis']} (train paired {tp:+.4f}, val paired {vp:+.4f})")
    return out


POLISH_SHIFTS = (-10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 40.0)
POLISH_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)


@torch.no_grad()
def polish(model, state, rows, cef, rho, dev, passes: int = 2, fine: bool = False, comp_target=None,
           fine_shifts=(-10.0, -3.0, 3.0, 10.0), pair: bool = False):
    """Zeroth-order HARD-loss polish of a few scalar directions the relaxed gradient can miss (e.g. keeping
    a document WHOLE is worth more than the sum of its tokens' marginal values): additive shifts of w_gold
    and w_marker, multiplicative scales of the w_pos / w_rel / w_emb groups. Every candidate is scored by the
    exact hard-deletion train dCE with the threshold re-set so the pooled routed keep stays rho (same T2/T
    budget). Coordinate search, ``passes`` sweeps. Generic: no task-specific structure."""
    base = RL.LinearRouter.from_state(state).to(dev)
    cur = {"gold": 0.0, "marker": 0.0, "pos": 1.0, "rel": 1.0, "emb": 1.0}
    grids = {"gold": POLISH_SHIFTS, "marker": POLISH_SHIFTS, "pos": POLISH_SCALES, "rel": POLISH_SCALES, "emb": POLISH_SCALES}

    def build(c):
        r = RL.LinearRouter.from_state(state).to(dev)
        r.w_gold.add_(c["gold"])
        r.w_marker.add_(c["marker"])
        r.w_pos.mul_(c["pos"])
        r.w_rel.mul_(c["rel"])
        r.w_emb.mul_(c["emb"])
        return r

    def offset_for(r):
        # hold the realised T2/T fixed at the target when one is given (holding only the pooled routed keep
        # let the search buy dCE with T2/T drift: grouping 0.29 -> 0.35), else the pooled keep rho
        return match_comp_offset(model, r, rows, comp_target) if comp_target is not None else global_offset(r, rows, rho)

    def score(c):
        r = build(c)
        if pair:  # paired budget per row: no threshold to set
            v = val_pair(model, r, rows, cef)
            return v["dce"], 0.0, v["comp"]
        off = offset_for(r)
        v = val_threshold(model, r, rows, cef, off)
        return v["dce"], off, v["comp"]

    t_pol = time.time()
    best, trace = score(cur), []
    log(f"[polish] start: train hard dCE {best[0]:+.4f} (T2/T {best[2]:.3f}) on {len(rows)} rows")
    for ps in range(passes):
        for k, grid in grids.items():
            for g in grid:
                if g == cur[k]:
                    continue
                c = dict(cur, **{k: g})
                sc = score(c)
                trace.append({"pass": ps, "coord": k, "value": g, "dce": sc[0], "comp": sc[2]})
                if sc[0] < best[0] - 1e-4:
                    best, cur = sc, c
            log(f"[polish] pass {ps} {k}: best {cur[k]} -> train hard dCE {best[0]:+.4f} (T2/T {best[2]:.3f}) | {time.time() - t_pol:.0f}s")
    r = build(cur)
    if fine:
        # per-feature additive shifts of the position profile: start/end offset one-hots 0..15 and the 20
        # start/end deciles, one coordinate pass, same hard-loss objective at fixed pooled keep
        coords = [("pos", i) for i in list(range(16)) + list(range(RL.N_OFF, RL.N_OFF + 16))] + [("rel", i) for i in range(2 * RL.N_DEC)]

        def score_r(rr):
            if pair:
                v = val_pair(model, rr, rows, cef)
                return v["dce"], 0.0, v["comp"]
            off = offset_for(rr)
            v = val_threshold(model, rr, rows, cef, off)
            return v["dce"], off, v["comp"]

        n_imp = 0
        for grp, i in coords:
            w = r.w_pos if grp == "pos" else r.w_rel
            w0 = float(w[i])
            for d in fine_shifts:
                w[i] = w0 + d
                sc = score_r(r)
                trace.append({"pass": "fine", "coord": f"{grp}{i}", "value": d, "dce": sc[0], "comp": sc[2]})
                if sc[0] < best[0] - 1e-4:
                    best, w0 = sc, w0 + d
                    n_imp += 1
            w[i] = w0
        log(f"[polish] fine: {n_imp} improving shifts -> train hard dCE {best[0]:+.4f} (T2/T {best[2]:.3f}) | {time.time() - t_pol:.0f}s")
    st = r.state()
    st["b"] = st["b"] + best[1]
    st.update({k: v for k, v in state.items() if k not in st and k != "b"})
    st["polish"] = cur
    return st, {"coords": cur, "train_dce": best[0], "train_comp": best[2], "trace": trace}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt-format", default="hf", choices=["hf", "distcp"])
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--val-root", required=True)
    ap.add_argument("--rungs", default="2k,8k")
    ap.add_argument("--n-train", type=int, default=96, help="per rung")
    ap.add_argument("--n-val", type=int, default=32, help="per rung")
    ap.add_argument("--rhos", default="0.1")
    ap.add_argument("--variant", default="full", choices=["full", "nogold", "noemb", "nocontpos", "relpos", "relpos_noemb", "doc_only", "coarse_pos", "span8", "span4"])
    ap.add_argument("--rho-match-bar", action="store_true",
                    help="train at the evaluation budget: rho = gold_fl20p8_noslot's routed-keep fraction pooled over the train rows")
    ap.add_argument("--diag-heuristic", action="store_true",
                    help="train/val diagnosis vs the exact gold_fl20p8_noslot keep set at equal T2/T (optimisation vs generalisation)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--rows-per-step", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--lr-gold-bias", type=float, default=None, help="separate lr for w_gold and b (default: --lr)")
    ap.add_argument("--residual-l2", type=float, default=0.0, help="per-row free residual on the router logit, L2 penalty (0 = off)")
    ap.add_argument("--residual-lr", type=float, default=0.3)
    ap.add_argument("--emb-cap", type=float, default=0.0, help="project ||w_emb|| <= cap after every step (0 = no cap)")
    ap.add_argument("--polish-on", default="train", choices=["train", "val"], help="score polish candidates on train or val rows")
    ap.add_argument("--kd-token", action="store_true", help="span routers: keep-dropout on tokens after broadcasting (default: per unit)")
    ap.add_argument("--keep-dropout", type=float, default=0.0, help="train-time extra random removal of routed tokens (0 = off)")
    ap.add_argument("--restart-avg", default="none", choices=["none", "all", "top2"],
                    help="average the restarts' polished weights (all, or the best 2 by val) instead of picking the best")
    ap.add_argument("--restarts", type=int, default=1, help="independent trainings per rho, the best by val hard dCE is kept")
    ap.add_argument("--polish-fast", action="store_true", help="1 coarse pass; fine shifts +-5 only")
    ap.add_argument("--polish-passes", type=int, default=0, help="coarse polish passes (0 = 1 if --polish-fast else 2)")
    ap.add_argument("--polish-rows", type=int, default=0, help="score polish candidates on the first N train rows (0 = all)")
    ap.add_argument("--polish-fine", action="store_true", help="after --polish: per-feature shifts of the position profile (hard loss)")
    ap.add_argument("--polish", action="store_true", help="hard-loss coordinate search on w_gold/w_marker shifts and w_pos/w_rel/w_emb scales after training")
    ap.add_argument("--emb-lr", type=float, default=0.01)
    ap.add_argument("--adam-beta2", type=float, default=0.8)
    ap.add_argument("--init-p", type=float, default=0.95)
    ap.add_argument("--beta0", type=float, default=2 / 3)
    ap.add_argument("--beta1", type=float, default=0.1)
    ap.add_argument("--warm-frac", type=float, default=0.5)
    ap.add_argument("--anneal-frac", type=float, default=0.7)
    ap.add_argument("--st-frac", type=float, default=0.85)
    ap.add_argument("--lam-lr", type=float, default=1.0)
    ap.add_argument("--w-kl", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--objective", default="calibrated", choices=["calibrated", "lagrangian"])
    ap.add_argument("--profile", action="store_true", help="phase-time step 3 (cuda-synced)")
    ap.add_argument("--marker-follow-doc", action="store_true",
                    help="markers are not routed: a document's markers are kept iff >= 1 of its body tokens is kept")
    ap.add_argument("--stg-tasks", default="", help="comma list of tasks whose train/val rows come from router_*_stg")
    ap.add_argument("--pair-eval", action="store_true",
                    help="polish + restart selection scored with the PAIRED budget (each row keeps the bar's own count)")
    ap.add_argument("--marker-follow-whole", action="store_true",
                    help="with --marker-follow-doc: markers kept only when the whole body is kept (the bar's rule)")
    ap.add_argument("--route-markers", action="store_true",
                    help="markers are routed too (own weight); a dropped doc can vanish entirely incl. its markers")
    ap.add_argument("--wpos-l2", type=float, default=0.0)
    ap.add_argument("--marker-init-p", type=float, default=None, help="routed markers: initial keep probability")
    ap.add_argument("--calib-scope", default="batch", choices=["batch", "row"],
                    help="budget calibration offset per step (pooled rows; eval = global threshold) or per row (eval = top-k)")
    ap.add_argument("--match-comps", default="", help="also save routers whose val mean T2/T equals each target (label-free)")
    ap.add_argument("--xtasks", default="", help="CROSS-TASK router: comma list of tasks trained jointly (one model each)")
    ap.add_argument("--xtask-name", default="xtask")
    ap.add_argument("--xtask-offload", action="store_true",
                    help="keep ONE per-task model on the GPU (others on CPU), visit rows in task blocks (pooled many-task runs)")
    ap.add_argument("--calib-root", default="", help="label-free cutoff calibration on unlabeled rows of --calib-rung")
    ap.add_argument("--calib-rung", default="8k")
    ap.add_argument("--match-keep-2k", action="store_true", help="also calibrate to the 2k routed-token keep fraction")
    ap.add_argument("--calib-skip", type=int, default=0)
    ap.add_argument("--calib-n", type=int, default=32)
    ap.add_argument("--match-weights", default="", help="eval-only: existing weight names to re-threshold at --match-comps")
    ap.add_argument("--eval-only", action="store_true", help="skip training; score existing weights on the test rows")
    ap.add_argument("--eval-rungs", default="", help="test-row eval IN this process after training (grid driver's run_cell)")
    ap.add_argument("--eval-rows", default="16")
    ap.add_argument("--eval-schemes", default="")
    ap.add_argument("--eval-data-root", default="")
    ap.add_argument("--eval-out", default="")
    ap.add_argument("--no-batch-rows", dest="batch_rows", action="store_false",
                    help="one forward per row instead of one right-padded forward per step")
    ap.add_argument("--val-extra-from-train", type=int, default=0,
                    help="per rung: train-split rows [n_train, n_train+K) added to VAL (disjoint from the train rows)")
    ap.add_argument("--hard-drop-zero", action="store_true", help="physically drop z == 0 tokens each step (exact; plain clamp)")
    ap.add_argument("--verify-batch", action="store_true")
    ap.add_argument("--verify-hard-drop", type=int, default=0, help="rows to verify full-row vs hard-drop-zero on, then continue")
    ap.add_argument("--wandb-group", default=None)
    ap.add_argument("--wandb-project", default="memory-networks")
    ap.add_argument("--wandb-entity", default="prasann-uc-berkeley-electrical-engineering-computer-sciences")
    ap.add_argument("--name", default="e2e_rho{rho}")
    ap.add_argument("--tokenizer", default=os.environ.get("DEVLOSS_TOKENIZER", G.TOKENIZER_BY_FAMILY[G.FAMILY]))
    ap.add_argument("--out-dir", default=HERE)
    a = ap.parse_args()
    t_start = time.time()
    log(f"start e2e-router task={a.task} rhos={a.rhos} epochs={a.epochs} rungs={a.rungs} w_kl={a.w_kl}")
    from transformers import AutoTokenizer

    from olmo_core.nn.attention.fla_autotune import freeze_fla_length_autotune

    ids = G.RESERVED_IDS[G.FAMILY]
    vocab = G.VOCAB_BY_FAMILY[G.FAMILY]
    a.pad_id = ids.eos
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rungs = a.rungs.split(",")
    xt = [t for t in a.xtasks.split(",") if t] if a.xtasks else [a.task]
    if a.xtasks:
        a.batch_rows = False  # rows of different tasks go through different frozen models
        man = json.load(open(os.path.join(TR.REPO, "debug", "devloss_grid", "manifest.json")))["tasks"]
    pieces = tok.convert_ids_to_tokens(list(range(vocab)))
    tr, va, model = [], [], None
    for t in xt:
        # --stg-tasks: these tasks read the matched-regime staged train/val rows (router_*_stg, 2026-09-30)
        stg = t in set(a.stg_tasks.split(",")) if a.stg_tasks else False
        troot = a.train_root.replace("router_train_e2e", "router_train_stg") if stg else a.train_root
        vroot = a.val_root.replace("router_val_e2e", "router_val_stg") if stg else a.val_root
        tr_rows = TR.load_split(troot, t, rungs, tok, ids, a.n_train)
        va_rows = TR.load_split(vroot, t, rungs, tok, ids, a.n_val)
        if a.val_extra_from_train and not stg:
            extra = TR.load_split(troot, t, rungs, tok, ids, a.n_train + a.val_extra_from_train)
            seen = {tuple(r["ids"]) for r in tr_rows}
            va_rows += [r for r in extra if tuple(r["ids"]) not in seen]
        all_ids = np.concatenate([np.asarray(r["ids"], dtype=np.int64) for r in tr_rows])
        stop_ids, _, tables = G.build_tables(all_ids, vocab, pieces, ids, lambda q: tok.decode([int(q)]))
        if a.xtasks:
            ck, fmt = man[t]["ckpt"], ("distcp" if man[t]["ckpt_format"] == "distcp" else "hf")
        else:
            ck, fmt = a.ckpt, a.ckpt_format
        m_t = G.load_model(ck, fmt, vocab, ids, 42, stop_ids, "flash_2")
        for p_ in m_t.parameters():
            p_.requires_grad_(False)
        assert not any(p_.requires_grad for p_ in m_t.parameters()), "model params must be frozen"
        freeze_fla_length_autotune()
        dev = next(m_t.parameters()).device
        TR.configure(m_t, stop_ids, tables.to(dev))
        max_pos = max(len(r["ids"]) for r in tr_rows + va_rows) + 8
        for mod in m_t.modules():
            rope = getattr(mod, "rope", None)
            if rope is not None and hasattr(rope, "warmup_cache"):
                rope.warmup_cache(max_pos, dev)
        if a.eval_only and not (a.diag_heuristic or a.polish):
            tr_rows = tr_rows[:1]  # val rows are kept: label-free threshold calibration (--match-weights)
        if a.route_markers:
            m_t._pooled_soft_tokens["keep_token_mask_markers"] = "mask"
        elif a.marker_follow_doc:
            # markers follow their document: any kept body token ("doc"), or the whole body kept (False = default rule)
            m_t._pooled_soft_tokens["keep_token_mask_markers"] = False if a.marker_follow_whole else "doc"
        tr_t = [TR.prep_row(m_t, r, ids, dev, route_markers=a.route_markers) for r in tr_rows]
        va_t = [TR.prep_row(m_t, r, ids, dev, route_markers=a.route_markers) for r in va_rows]
        for q in tr_t + va_t:
            q["_m"], q["task"] = m_t, t
            if a.marker_follow_doc:
                q["feats"]["mk_follow"] = "whole" if a.marker_follow_whole else True
        tr += tr_t
        va += va_t
        model = model or m_t
        log(f"task {t}: {len(tr_t)} train / {len(va_t)} val rows, model {os.path.basename(ck)}")
        if a.xtask_offload:
            # full-model targets now, while this model is resident; then park it on CPU
            for q in tr_t:
                q["_lpf"] = full_logprobs(m_t, q)
            for q in va_t:
                q["_cef"] = TR.ce_full(m_t, q)
            m_t.to("cpu")
            torch.cuda.empty_cache()
            log(f"task {t}: targets done, model parked on CPU (GPU mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB)")
    if a.xtask_offload:
        for q in tr + va:
            if not getattr(q["_m"], "_swap_hooked", False):
                q["_m"].register_forward_pre_hook(lambda mod, args: swap_in(mod))
                q["_m"]._swap_hooked = True
        TR.ce_masked = _swapping(TR.ce_masked)  # compacts through model internals before its forward
        TR.vocab_scores = _swapping(TR.vocab_scores)
        _Swap.enabled = True
    t0 = time.time()
    cef_tr, lpf_tr = [], []
    for j, p in enumerate(tr):
        lpf = p.pop("_lpf") if "_lpf" in p else full_logprobs(p["_m"], p)
        lpf_tr.append(lpf)
        cef_tr.append(float(F.nll_loss(lpf.float(), p["tgt"].cpu())))
        if j + 1 in (1, 2, 5, 10) or (j + 1) % 50 == 0 or j + 1 == len(tr):
            log(f"full-model targets {j + 1}/{len(tr)} ({time.time() - t0:.0f}s)")
    cef_va = [p.pop("_cef") if "_cef" in p else TR.ce_full(p["_m"], p) for p in va]
    log(f"[time] setup (model load, rows, full-model targets) {time.time() - t_start:.0f}s")
    log(f"train {len(tr)} rows (CE_full {np.mean(cef_tr):.4f}), val {len(va)} rows (CE_full {np.mean(cef_va):.4f}); "
        f"routed tokens/row median {int(np.median([int(p['feats']['idx'].numel()) for p in tr]))}; "
        f"answer tokens median {int(np.median([int(p['tgt'].numel()) for p in tr]))}; mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    if a.xtasks:
        a.task = a.xtask_name  # weights/runs of the shared router live under weights/<xtask_name>/
    os.makedirs(os.path.join(a.out_dir, "runs", a.task), exist_ok=True)
    os.makedirs(os.path.join(a.out_dir, "weights", a.task), exist_ok=True)
    if a.verify_batch:
        vb = verify_batch(model, tr, lpf_tr, a)
        json.dump(vb, open(os.path.join(a.out_dir, "runs", a.task, "e2e_batch_verify.json"), "w"), indent=1)
    if a.verify_hard_drop:
        ver = verify_hard_drop(model, tr, lpf_tr, a, a.verify_hard_drop)
        json.dump(ver, open(os.path.join(a.out_dir, "runs", a.task, "e2e_hard_drop_verify.json"), "w"), indent=1)
    if a.rho_match_bar and not a.eval_only:
        with torch.no_grad():
            kk = [heuristic_route_keep(p, heuristic_mask(p, ids)) for p in tr]
        rho_b = float(torch.cat(kk).float().mean())
        for p, k_ in zip(tr, kk):
            p["k_pair"] = int(k_.sum())
        with torch.no_grad():
            for p in va:
                p["k_pair"] = int(heuristic_route_keep(p, heuristic_mask(p, ids)).sum())
        if a.xtasks:
            # cross-task: every task trains at ITS OWN matched budget (rho per task, used per task-block batch)
            by_t = {}
            for p, k_ in zip(tr, kk):
                by_t.setdefault(p["task"], []).append(k_)
            rho_task = {t: float(torch.cat(v).float().mean()) for t, v in by_t.items()}
            for p in tr:
                p["rho_task"] = rho_task[p["task"]]
            log("per-task rho: " + ", ".join(f"{t} {r:.3f}" for t, r in rho_task.items()))
        hb = heuristic_eval(model, tr, cef_tr, ids)
        log(f"rho matched to gold_fl20p8_noslot: routed keep {rho_b:.4f} pooled over {len(tr)} train rows "
            f"(heuristic train T2/T {hb['comp']:.3f}, dCE {hb['dce']:+.4f})")
        a.rhos = f"{rho_b:.4f}"
        a._comp_target = hb["comp"]  # the target T2/T on the train rows (polish holds it fixed)
    for rho in ([] if a.eval_only else [float(r) for r in a.rhos.split(",")]):
        name = a.name.format(rho="bar" if a.rho_match_bar else f"{rho:g}")  # --rho-match-bar: rho known only at run time
        wb = None
        if a.wandb_group:
            try:
                import wandb

                wb = wandb.init(entity=a.wandb_entity, project=a.wandb_project, group=a.wandb_group, reinit=True,
                                name=f"{a.task}-{name}{'-hd0' if a.hard_drop_zero else ''}-{os.environ.get('SLURM_JOB_ID', 'local')}",
                                config={**vars(a), "rho": rho}, settings=wandb.Settings(init_timeout=90))
                log(f"wandb run {wb.url} (group https://wandb.ai/{a.wandb_entity}/{a.wandb_project}/groups/{a.wandb_group})")
            except Exception as ex:  # noqa: BLE001
                log(f"wandb init failed ({ex!r}); continuing without wandb")
                wb = None
        seed0, cands = a.seed, []
        # --polish-rows N: score polish candidates on the first N train rows only (speed)
        if a.polish_on == "val":
            # polish scored on the VAL rows at the val target T2/T (the final test rows stay disjoint)
            tr_pol, cef_pol = (va[: a.polish_rows], cef_va[: a.polish_rows]) if a.polish_rows else (va, cef_va)
        else:
            if a.polish_rows and a.xtasks:  # stratified across tasks (train rows are ordered by task)
                tn = sorted({p["task"] for p in tr})
                per = max(1, -(-a.polish_rows // len(tn)))
                sel = [i for t in tn for i in [j for j, p in enumerate(tr) if p["task"] == t][:per]]
                tr_pol, cef_pol = [tr[i] for i in sel], [cef_tr[i] for i in sel]
            else:
                tr_pol, cef_pol = (tr[: a.polish_rows], cef_tr[: a.polish_rows]) if a.polish_rows else (tr, cef_tr)
        sel_target = None
        if (a.restarts > 1 or a.polish_on == "val") and a.rho_match_bar:
            # select restarts at the target T2/T on the val rows (label-free threshold), not at pooled keep rho
            sel_target = heuristic_eval(model, va, cef_va, ids)["comp"]
        for rs in range(a.restarts):
            # --restarts K: K independent trainings (seeds seed..seed+K-1), each polished; the one with the
            # lowest VAL hard dCE at the same pooled keep rho is kept (the training is chaotic run to run)
            a.seed = seed0 + rs
            a._step_off = rs * 100000  # keep wandb steps monotonic across restarts
            t_rs = time.time()
            state, res = train_rho(model, rho, tr, va, cef_tr, lpf_tr, cef_va, a, wb)
            if a.polish:  # with restarts, the (slower) fine pass runs only on the selected one
                state, res["polish"] = polish(model, state, tr_pol, cef_pol, rho, dev, passes=a.polish_passes if a.polish_passes else (1 if a.polish_fast else 2),
                                              fine=a.polish_fine and a.restarts == 1, comp_target=sel_target if a.polish_on == "val" else getattr(a, "_comp_target", None),
                                              fine_shifts=(-5.0, 5.0) if a.polish_fast else (-10.0, -3.0, 3.0, 10.0), pair=a.pair_eval)
            if a.restarts > 1:
                rr = RL.LinearRouter.from_state(state).to(dev)
                if a.pair_eval:
                    vsel = val_pair(model, rr, va, cef_va)
                else:
                    off = match_comp_offset(model, rr, va, sel_target) if sel_target is not None else global_offset(rr, va, rho)
                    vsel = val_threshold(model, rr, va, cef_va, off)
                log(f"[restart {rs} seed {a.seed}] val hard dCE {vsel['dce']:+.4f} (T2/T {vsel['comp']:.3f}) | [time] restart {time.time() - t_rs:.0f}s")
                cands.append((vsel["dce"], rs, state, res))
        if a.restarts > 1 and a.restart_avg != "none":
            # variance reduction: AVERAGE the restarts' (polished) weights instead of picking one; the offset
            # is re-set afterwards (global threshold at the target keep), so only the ranking is averaged
            pool = sorted(cands, key=lambda c: c[0])[: (2 if a.restart_avg == "top2" else len(cands))]
            keys = ["b", "w_pos", "w_rel", "w_gold", "w_emb", "w_marker", "w_span"]
            avg = dict(pool[0][2])
            for k in keys:
                if k in avg:
                    avg[k] = sum(c[2][k] for c in pool) / len(pool)
            ra = RL.LinearRouter.from_state(avg).to(dev)
            off = match_comp_offset(model, ra, tr, a._comp_target) if getattr(a, "_comp_target", None) else global_offset(ra, tr, rho)
            avg["b"] = avg["b"] + off
            cands.append((None, f"avg_{a.restart_avg}", avg, dict(pool[0][3])))
            res = cands[-1][3]
            state = avg
            res["restarts"] = [{"restart": c[1], "seed": seed0 + c[1] if isinstance(c[1], int) else None, "val_dce": c[0]} for c in cands]
            res["restart_selected"] = f"avg_{a.restart_avg} of {len(pool)}"
            log(f"[restarts] AVERAGED the weights of {len(pool)} restart(s) ({a.restart_avg}); offset re-set ({off:+.3f})")
            rs_best = None
            # the best single restart from the SAME job, fine-polished the same way, saved as <name>_best for a
            # like-for-like comparison (training is nondeterministic across jobs)
            best_state = pool[0][2]
            if a.polish and a.polish_fine:
                best_state, _ = polish(model, best_state, tr_pol, cef_pol, rho, dev, passes=0, fine=True,
                                       comp_target=sel_target if a.polish_on == "val" else getattr(a, "_comp_target", None),
                                       fine_shifts=(-5.0, 5.0) if a.polish_fast else (-10.0, -3.0, 3.0, 10.0), pair=a.pair_eval)
            torch.save(best_state, os.path.join(a.out_dir, "weights", a.task, f"{name}_best.pt"))
            log(f"[restarts] best single restart saved as {name}_best.pt")
        if a.restarts > 1 and a.restart_avg == "none":
            _, rs_best, state, res = min(cands, key=lambda c: c[0])
            res["restarts"] = [{"restart": c[1], "seed": seed0 + c[1], "val_dce": c[0]} for c in cands]
            res["restart_selected"] = rs_best
            log(f"[restarts] selected restart {rs_best} (seed {seed0 + rs_best})")
        if a.restarts > 1:
            if a.polish and a.polish_fine:
                state, res["polish_fine"] = polish(model, state, tr_pol, cef_pol, rho, dev, passes=0, fine=True,
                                                   comp_target=sel_target if a.polish_on == "val" else getattr(a, "_comp_target", None),
                                                   fine_shifts=(-5.0, 5.0) if a.polish_fast else (-10.0, -3.0, 3.0, 10.0), pair=a.pair_eval)
        a.seed = seed0
        wpath = os.path.join(a.out_dir, "weights", a.task, f"{name}.pt")
        torch.save(state, wpath + ".part")
        os.replace(wpath + ".part", wpath)
        router = RL.LinearRouter.from_state(state).to(dev)
        res.update({"task": a.task, "config": name, "variant": a.variant, "ckpt": a.ckpt, "argv": sys.argv,
                    "git_commit": G.git_commit(), "train_size": len(tr), "val_size": len(va),
                    "ce_full_train_mean": float(np.mean(cef_tr)), "ce_full_val_mean": float(np.mean(cef_va)),
                    "weights": TR.weights_summary(state), "vocab_scores": TR.vocab_scores(model, router, tr, tok),
                    "weights_path": os.path.relpath(wpath, TR.REPO),
                    "hparams": {k: getattr(a, k) for k in ("epochs", "rows_per_step", "lr", "emb_lr", "adam_beta2", "init_p",
                                                            "beta0", "beta1", "warm_frac", "anneal_frac", "st_frac", "lam_lr",
                                                            "w_kl", "seed", "lr_gold_bias", "residual_l2", "residual_lr", "keep_dropout", "restarts", "polish", "polish_fine", "polish_fast", "polish_rows", "polish_on", "emb_cap", "marker_follow_doc", "marker_follow_whole", "rungs", "hard_drop_zero", "objective", "val_extra_from_train", "batch_rows", "calib_scope", "route_markers", "wpos_l2")}})
        rpath = os.path.join(a.out_dir, "runs", a.task, f"{name}.json")
        json.dump(res, open(rpath + ".part", "w"), indent=1)
        os.replace(rpath + ".part", rpath)
        fv = res["final_val"]
        if wb is not None:
            wb.summary.update({"final_val_dce": res["final_val"]["dce"], "final_val_compaction": res["final_val"]["comp"]})
            wb.finish()
        router_b = RL.LinearRouter.from_state(state).to(dev)
        for tc in [float(x) for x in a.match_comps.split(",") if x]:
            c = match_comp_offset(model, router_b, va, tc)
            st2 = dict(state)
            st2["b"] = state["b"] + c
            st2["match_comp"] = tc
            torch.save(st2, os.path.join(a.out_dir, "weights", a.task, f"{name}_c{tc:g}.pt"))
            vm = val_threshold(model, router_b, va, cef_va, c)
            res.setdefault("match_comp_val", {})[f"{tc:g}"] = vm
            json.dump(res, open(rpath, "w"), indent=1)
            log(f"[rho{rho}] matched T2/T {tc:g}: val dCE {vm['dce']:+.4f} (median {vm['dce_median']:+.4f}, SE {vm['dce_se']:.4f}) "
                f"realised T2/T {vm['comp']:.3f} -> weights {name}_c{tc:g}.pt")
        if a.diag_heuristic:
            res["diag"] = diag_vs_heuristic(model, router_b, tr, va, cef_tr, cef_va, ids)
            json.dump(res, open(rpath, "w"), indent=1)
        if a.xtask_offload:
            log(f"[rho{rho}] model swaps: {_Swap.n_swaps} ({_Swap.sec:.0f}s total)")
        log(f"[rho{rho}] DONE final val HARD dCE {fv['dce']:+.4f} (median {fv['dce_median']:+.4f}) T2/T {fv['comp']:.3f} "
            f"keep {fv['keep']:.3f} w_gold {float(state['w_gold']):+.2f} -> {rpath}")
    calib = va
    if a.calib_root:
        # label-free per-LENGTH calibration: unlabeled rows of the target rung (answers never used)
        cr = TR.load_split(a.calib_root, a.task, [a.calib_rung], tok, ids, a.calib_skip + a.calib_n)[a.calib_skip:]
        calib = [TR.prep_row(model, r, ids, dev, route_markers=a.route_markers) for r in cr]
        for q in calib:
            if a.marker_follow_doc:
                q["feats"]["mk_follow"] = "whole" if a.marker_follow_whole else True
        for q in calib:
            q["_m"] = model
        log(f"calibration rows: {len(calib)} unlabeled {a.calib_rung} rows from {a.calib_root} (skip {a.calib_skip})")
    if a.match_weights:
        # existing routers -> label-free global thresholds hitting each target mean T2/T on the calibration rows
        for nm in a.match_weights.split(","):
            st0 = torch.load(os.path.join(a.out_dir, "weights", a.task, f"{nm}.pt"), map_location="cpu")
            if a.eval_only and a.polish:
                # re-polish an existing router (hard-loss coordinate search on the train rows at its rho)
                st0, pol = polish(model, st0, tr, cef_tr, float(st0["rho"]), dev, fine=a.polish_fine)
                nm = f"{nm}_pol{'f' if a.polish_fine else ''}"
                torch.save(st0, os.path.join(a.out_dir, "weights", a.task, f"{nm}.pt"))
                log(f"[match] re-polished -> {nm}.pt (train hard dCE {pol['train_dce']:+.4f})")
            rb = RL.LinearRouter.from_state(st0).to(dev)
            if a.diag_heuristic:
                dg = diag_vs_heuristic(model, rb, tr, va, cef_tr, cef_va, ids, tag=f" {nm}")
                os.makedirs(os.path.join(a.out_dir, "runs", a.task), exist_ok=True)
                json.dump(dg, open(os.path.join(a.out_dir, "runs", a.task, f"diag_{nm}.json"), "w"), indent=1)
            if a.calib_root and a.match_keep_2k:
                # (i) keep the SAME fraction of routed tokens as the router keeps on the 2k val rows
                with torch.no_grad():
                    k2 = float(torch.cat([(rb.logits(q["feats"], q["e"].float() if rb.use_emb else None) > 0).float() for q in va]).mean())
                c = global_offset(rb, calib, k2)
                st2 = dict(st0)
                st2["b"] = st0["b"] + c
                st2["match_keep"] = k2
                torch.save(st2, os.path.join(a.out_dir, "weights", a.task, f"{nm}_k2k_{a.calib_rung}.pt"))
                log(f"[match] {nm} -> 2k routed keep {k2:.3f} on unlabeled {a.calib_rung} rows (offset {c:+.3f}) -> {nm}_k2k_{a.calib_rung}.pt")
            for tc in [float(x) for x in a.match_comps.split(",") if x]:
                c = match_comp_offset(model, rb, calib, tc)
                st2 = dict(st0)
                st2["b"] = st0["b"] + c
                st2["match_comp"] = tc
                sfx = f"_c{tc:g}" + (f"_{a.calib_rung}" if a.calib_root else "")
                torch.save(st2, os.path.join(a.out_dir, "weights", a.task, f"{nm}{sfx}.pt"))
                if a.calib_root:
                    log(f"[match] {nm} -> T2/T {tc:g} on unlabeled {a.calib_rung} rows (offset {c:+.3f}) -> {nm}{sfx}.pt")
                    continue
                vm = val_threshold(model, rb, va, cef_va, c)
                log(f"[match] {nm} -> T2/T {tc:g}: val dCE {vm['dce']:+.4f} (median {vm['dce_median']:+.4f}, SE {vm['dce_se']:.4f}) "
                    f"realised {vm['comp']:.3f} -> {nm}{sfx}.pt")
    if a.eval_rungs:
        # test rows, scored IN this process with the already-loaded model (the grid driver's cell logic)
        import types as _types

        te = time.time()
        schemes = {}
        for sname in a.eval_schemes.split(","):
            if not sname:
                continue
            if sname in G.SCHEMES:
                schemes[sname] = G.SCHEMES[sname]
            elif sname.startswith("router_"):
                nm, _, spec = sname[len("router_"):].partition("@")
                extra = {"topk_rho": float(spec[1:])} if spec.startswith("k") else ({"topk_comp": float(spec[1:])} if spec.startswith("c") else ({"topk_pair": True} if spec == "pair" else {}))
                schemes[sname] = dict(G.SCHEMES["router_l0.2"], router=nm, **extra)
        schemes = {"full": None, **schemes}
        rungs_e = a.eval_rungs.split(",")
        nrows = [int(x) for x in a.eval_rows.split(",")]
        nrows = nrows * len(rungs_e) if len(nrows) == 1 else nrows
        pieces = tok.convert_ids_to_tokens(list(range(vocab)))
        for rg, nr in zip(rungs_e, nrows):
            ea = _types.SimpleNamespace(task=f"ctc_{a.task}", rung=rg, rows=nr, ckpt=a.ckpt, ckpt_format=a.ckpt_format,
                                        data_root=a.eval_data_root, cpt_source=None, seed=42, cpt_block=512, attn_backend="flash_2",
                                        tokenizer=a.tokenizer)
            G.run_cell(ea, model, tok, pieces, lambda t: tok.decode([int(t)]), ids, vocab, schemes, te, a.eval_out.replace("{rung}", rg))
        log(f"in-process test eval of {a.eval_rungs} took {time.time() - te:.0f}s")
    log(f"all done in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
