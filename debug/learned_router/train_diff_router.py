"""Train the linear token router (router_lib.LinearRouter, same features as train_router.py) by a
DIFFERENTIABLE relaxation of token removal on the frozen dense checkpoint, under a task-relative
error tolerance, then save per-tau weights + curves.

    python debug/learned_router/train_diff_router.py --task nq --ckpt ... --train-root ... --val-root ... \\
        --taus 0.05,0.1,0.2

Relaxation (train time only; eval is the exact hard-removal path, ``sel="router"`` in the grid
driver). Each routed token carries a hard-concrete gate z in [0, 1] (log_alpha = router logit),
applied WITHOUT compacting the row through ``model(..., soft_keep=...)``:
  * attention layers: additive key bias log(z + 1e-8) on top of the causal mask (exact at z in {0,1});
  * GatedDeltaNet: q/k/v conv inputs, beta and g (log decay) scaled by z (z = 0 -> identity state
    step, exact for the recurrence; the short conv is approximate).
Temperature anneals beta0 -> beta1 over the first ``--anneal-frac`` of epochs; from ``--st-frac`` on
the forward uses the binary straight-through gate, so training ends on the eval-time semantics.

Objective (task-relative tolerance): minimise E[keep_frac] subject to mean dCE <= eps, where
eps = max(tau * mean CE_full(train rows), --floor). Lagrangian keep_L0 + mu * (dCE / eps - 1), dual
ascent on mu >= 0 once per epoch on the deployed (deterministic, exact hard removal) policy's mean
train dCE. Router init: bias = logit(--init-p) (p ~ 0.95), gold
and embedding weights 0 -- the model trades DOWN from keep-all.

Selection (val rows only): each epoch the deterministic gate (logit > 0) is scored with the exact
hard compaction on the val rows; the saved checkpoint is the most compact epoch whose val mean dCE
<= eps_val = max(tau * mean CE_full(val), floor). Epoch 0 (keep-all) always qualifies.
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
import train_router as TR  # noqa: E402

G = TR.G
log = TR.log


def relaxed_ce(model, p, z: torch.Tensor) -> torch.Tensor:
    """Answer CE (with grad to ``z``) of one row under the relaxed removal ``z`` on routed tokens."""
    model.eval()
    model._pooled_keep_holder = None
    sk = RL.soft_keep_row(p["feats"], z, p["T"])
    lg = model(p["x"], logits_to_keep=p["pred"][None], soft_keep=sk[None])[0]
    return F.cross_entropy(lg.float(), p["tgt"])


@torch.no_grad()
def measure_gap(model, rows, cef, cef_rel, rates=(0.8, 0.5, 0.2), seed=11):
    """Relaxed CE with a BINARY gate vs the exact hard-compaction CE on the same masks."""
    gen = torch.Generator().manual_seed(seed)
    out = []
    for i, p in enumerate(rows):
        N = int(p["feats"]["idx"].numel())
        for rate in rates:
            keep = (torch.rand(N, generator=gen) < rate).to(p["x"].device)
            hard = float(TR.ce_masked(model, p, RL.keep_mask_from(p["feats"], keep, p["T"])[None])[0][0])
            rel = float(relaxed_ce(model, p, keep.float()))
            out.append({"row": i, "rung": p["rung"], "T": p["T"], "rate": rate, "dce_hard": hard - cef[i],
                        "dce_relaxed": rel - cef_rel[i], "gap": rel - cef_rel[i] - (hard - cef[i])})
        log(f"gap row {i + 1}/{len(rows)} T={p['T']}: " + "  ".join(
            f"keep{o['rate']}: hard {o['dce_hard']:+.3f} relaxed {o['dce_relaxed']:+.3f}" for o in out[-len(rates):]))
    gaps = np.array([o["gap"] for o in out])
    hard = np.array([o["dce_hard"] for o in out])
    rel = np.array([o["dce_relaxed"] for o in out])
    summ = {"n": len(out), "mean_gap": float(gaps.mean()), "mean_abs_gap": float(np.abs(gaps).mean()),
            "median_abs_gap": float(np.median(np.abs(gaps))), "mean_abs_dce_hard": float(np.abs(hard).mean()),
            "corr_hard_relaxed": float(np.corrcoef(hard, rel)[0, 1]) if len(out) > 2 else None,
            "p1_path_diff_mean_abs": float(np.mean(np.abs(np.array(cef_rel[: len(rows)]) - np.array(cef[: len(rows)])))),
            "by_rung": {}}
    for rg in sorted({o["rung"] for o in out}):
        g = np.array([o["gap"] for o in out if o["rung"] == rg])
        h = np.array([o["dce_hard"] for o in out if o["rung"] == rg])
        summ["by_rung"][rg] = {"mean_abs_gap": float(np.abs(g).mean()), "mean_gap": float(g.mean()), "mean_abs_dce_hard": float(np.abs(h).mean())}
    return summ, out


@torch.no_grad()
def val_hard(model, router, rows, cef):
    dce, comp, keep = [], [], []
    for i, p in enumerate(rows):
        e = p["e"].float() if router.use_emb else None
        k = router.logits(p["feats"], e) > 0
        ce, c = TR.ce_masked(p.get("_m", model), p, RL.keep_mask_from(p["feats"], k, p["T"])[None])
        dce.append(float(ce[0]) - cef[i])
        comp.append(float(c[0]))
        keep.append(float(k.float().mean()) if k.numel() else 1.0)
    return {"dce": float(np.mean(dce)), "dce_median": float(np.median(dce)), "dce_max": float(np.max(dce)),
            "comp": float(np.mean(comp)), "keep": float(np.mean(keep)), "per_row_dce": dce}


def train_tau(model, tau, tr, va, cef_tr, cef_rel_tr, cef_va, a):
    dev = next(model.parameters()).device
    torch.manual_seed(1000 + a.seed)
    gen = torch.Generator().manual_seed(2000 + a.seed)
    router = RL.LinearRouter(int(model.embeddings.weight.shape[1]), a.variant).to(dev)
    with torch.no_grad():
        router.b.fill_(math.log(a.init_p / (1 - a.init_p)))
    # beta2 0.8, not 0.999: the constraint term (x mu/eps, eps ~ 0.005) gives gradients ~100x the L0
    # term; with a long second-moment memory the L0 term then barely moved the bias once mu hit 0
    opt = torch.optim.Adam([{"params": [router.b, router.w_pos, router.w_gold], "lr": a.lr},
                            {"params": [router.w_emb], "lr": a.emb_lr}], betas=(0.9, a.adam_beta2))
    eps_tr = max(tau * float(np.mean(cef_tr)), a.floor)
    eps_va = max(tau * float(np.mean(cef_va)), a.floor)
    mu, ema = a.mu0, None
    steps, epochs = [], []
    v0 = val_hard(model, router, va, cef_va)
    epochs.append({"epoch": 0, "val": v0, "train": None, "mu": mu, "beta": None, "st": False,
                   "w_gold": 0.0, "b": float(router.b), "ok": v0["dce"] <= eps_va})
    best = {"epoch": 0, "comp": v0["comp"], "dce": v0["dce"], "state": router.state()}
    log(f"[tau{tau}] eps_train {eps_tr:.4f} (CE_full {np.mean(cef_tr):.4f}) eps_val {eps_va:.4f}; epoch 0 val dCE {v0['dce']:+.4f} T2/T {v0['comp']:.2f}")
    t0 = time.time()
    n_steps = math.ceil(len(tr) / a.rows_per_step) * a.epochs
    step = 0
    for ep in range(1, a.epochs + 1):
        frac = (ep - 1) / max(1, a.epochs - 1)
        beta = a.beta0 * (a.beta1 / a.beta0) ** min(1.0, frac / a.anneal_frac)
        st = frac >= a.st_frac
        order = torch.randperm(len(tr), generator=gen).tolist()
        acc = {"dce": [], "keep": [], "keep_hard": []}
        for s0 in range(0, len(order), a.rows_per_step):
            opt.zero_grad(set_to_none=True)
            bd, bk, bh = [], [], []
            batch = order[s0 : s0 + a.rows_per_step]
            for i in batch:
                p = tr[i]
                e = p["e"].float() if router.use_emb else None
                la = router.logits(p["feats"], e)
                z = RL.hard_concrete(la, beta, gen, straight_through=st)
                dce = relaxed_ce(model, p, z) - cef_rel_tr[i]
                keep_l0 = RL.hard_concrete_p_nonzero(la, beta).mean()
                loss = keep_l0 + mu * (dce / eps_tr - 1.0)
                (loss / len(batch)).backward()
                bd.append(float(dce))
                bk.append(float(keep_l0))
                bh.append(float((la.detach() > 0).float().mean()))
            gn = float(torch.sqrt(sum((q.grad.float() ** 2).sum() for q in router.parameters() if q.grad is not None)))
            opt.step()
            step += 1
            bdm = float(np.mean(bd))
            rec = {"step": step, "epoch": ep, "dce": bdm, "keep_l0": float(np.mean(bk)), "keep_det": float(np.mean(bh)),
                   "mu": mu, "beta": beta, "st": st, "w_gold": float(router.w_gold), "b": float(router.b),
                   "grad_norm": gn, "sec": time.time() - t0}
            steps.append(rec)
            acc["dce"] += bd
            acc["keep"] += bk
            acc["keep_hard"] += bh
            if step in (1, 2, 5, 10) or step % 25 == 0:
                el = time.time() - t0
                log(f"[tau{tau}] step {step}/{n_steps} ep {ep}: dCE {bdm:+.4f} (eps {eps_tr:.4f}) keepL0 {rec['keep_l0']:.2f} "
                    f"keep_det {rec['keep_det']:.2f} mu {mu:.2f} beta {beta:.2f}{' ST' if st else ''} w_gold {rec['w_gold']:+.2f} "
                    f"b {rec['b']:+.2f} | {el:.0f}s, ETA {el / step * (n_steps - step):.0f}s")
        # dual ascent ONCE PER EPOCH on the DEPLOYED policy's train dCE: the deterministic gate scored
        # with the exact hard removal on every train row (the constraint is on the mean over the train
        # rows, for the router that is evaluated). A per-step update on the sampled relaxed dCE made mu
        # spike (heavy-tailed rows) and over-tighten (sampled masks drop tokens the deterministic gate
        # keeps: outlier +0.020 sampled vs -0.001 deterministic), sending the router back to keep-all.
        vt = val_hard(model, router, tr, cef_tr)
        mu = max(0.0, mu + a.mu_lr * float(np.clip(vt["dce"] / eps_tr - 1.0, -1.0, a.mu_clip)))
        v = val_hard(model, router, va, cef_va)
        ok = v["dce"] <= eps_va
        trm = {k: float(np.mean(x)) for k, x in acc.items()}
        trm.update({"det_dce": vt["dce"], "det_comp": vt["comp"], "det_keep": vt["keep"]})
        epochs.append({"epoch": ep, "train": trm, "val": v, "mu": mu, "beta": beta, "st": st,
                       "w_gold": float(router.w_gold), "b": float(router.b), "ok": ok})
        if ok and (v["comp"] < best["comp"] - 1e-4 or (abs(v["comp"] - best["comp"]) <= 1e-4 and v["dce"] < best["dce"])):
            best = {"epoch": ep, "comp": v["comp"], "dce": v["dce"], "state": router.state()}
        log(f"[tau{tau}] epoch {ep}: train sampled dCE {trm['dce']:+.4f} keepL0 {trm['keep']:.2f} | train HARD dCE {vt['dce']:+.4f} "
            f"(eps {eps_tr:.4f}) T2/T {vt['comp']:.2f} | "
            f"val HARD dCE {v['dce']:+.4f} (eps {eps_va:.4f}{' OK' if ok else ' --'}) T2/T {v['comp']:.2f} keep {v['keep']:.2f} "
            f"| mu {mu:.2f} beta {beta:.2f}{' ST' if st else ''}{'  *best' if best['epoch'] == ep else ''}")
    return best, {"tau": tau, "eps_train": eps_tr, "eps_val": eps_va, "floor": a.floor, "best_epoch": best["epoch"],
                  "best_val": {"comp": best["comp"], "dce": best["dce"]}, "steps": steps, "epochs": epochs,
                  "train_sec": time.time() - t0, "final_state": router.state()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt-format", default="hf", choices=["hf", "distcp"])
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--val-root", required=True)
    ap.add_argument("--rungs", default="2k,8k")
    ap.add_argument("--n-train", type=int, default=16)
    ap.add_argument("--n-val", type=int, default=8)
    ap.add_argument("--taus", default="0.05,0.1,0.2")
    ap.add_argument("--floor", type=float, default=0.005, help="absolute floor on the tolerance (nats)")
    ap.add_argument("--variant", default="full", choices=["full", "nogold", "noemb"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--rows-per-step", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--emb-lr", type=float, default=0.01)
    ap.add_argument("--init-p", type=float, default=0.95)
    ap.add_argument("--beta0", type=float, default=2 / 3)
    ap.add_argument("--beta1", type=float, default=0.1)
    ap.add_argument("--anneal-frac", type=float, default=0.6)
    ap.add_argument("--st-frac", type=float, default=0.75)
    ap.add_argument("--mu0", type=float, default=0.5)
    ap.add_argument("--adam-beta2", type=float, default=0.8)
    ap.add_argument("--mu-lr", type=float, default=0.2, help="dual step per epoch")
    ap.add_argument("--mu-clip", type=float, default=2.0, help="upper clip on the normalised violation dCE/eps - 1")
    ap.add_argument("--ema", type=float, default=0.7)
    ap.add_argument("--gap-rows", type=int, default=3, help="per rung; 0 = skip the relaxation-gap measurement")
    ap.add_argument("--gap-rungs", default="2k,8k", help="rungs of the train split the gap is measured on")
    ap.add_argument("--gap-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="diff_tau{tau}", help="weights/runs name template")
    ap.add_argument("--tokenizer", default=os.environ.get("DEVLOSS_TOKENIZER", G.TOKENIZER_BY_FAMILY[G.FAMILY]))
    ap.add_argument("--out-dir", default=HERE)
    a = ap.parse_args()
    t_start = time.time()
    log(f"start diff-router task={a.task} taus={a.taus} floor={a.floor} epochs={a.epochs}")
    from transformers import AutoTokenizer

    from olmo_core.nn.attention.fla_autotune import freeze_fla_length_autotune

    ids = G.RESERVED_IDS[G.FAMILY]
    vocab = G.VOCAB_BY_FAMILY[G.FAMILY]
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rungs = a.rungs.split(",")
    tr_rows = TR.load_split(a.train_root, a.task, rungs, tok, ids, a.n_train)
    va_rows = TR.load_split(a.val_root, a.task, rungs, tok, ids, a.n_val)
    all_ids = np.concatenate([np.asarray(r["ids"], dtype=np.int64) for r in tr_rows])
    pieces = tok.convert_ids_to_tokens(list(range(vocab)))
    stop_ids, _, tables = G.build_tables(all_ids, vocab, pieces, ids, lambda t: tok.decode([int(t)]))
    # masked-SDPA (torch) attention for the relaxed path; flash_2 is what the hard eval uses and
    # what CE_full comes from -- the p=1 path difference is measured and reported (gap json).
    model = G.load_model(a.ckpt, a.ckpt_format, vocab, ids, 42, stop_ids, "flash_2")
    for p_ in model.parameters():
        p_.requires_grad_(False)
    log(f"froze {freeze_fla_length_autotune()} FLA kernels")
    dev = next(model.parameters()).device
    TR.configure(model, stop_ids, tables.to(dev))
    max_pos = max(len(r["ids"]) for r in tr_rows + va_rows) + 8
    for mod in model.modules():
        rope = getattr(mod, "rope", None)
        if rope is not None and hasattr(rope, "warmup_cache"):
            rope.warmup_cache(max_pos, dev)
    tr = [TR.prep_row(model, r, ids, dev) for r in tr_rows]
    va = [TR.prep_row(model, r, ids, dev) for r in va_rows]
    t0 = time.time()
    cef_tr, cef_rel_tr, cef_va = [], [], []
    for j, p in enumerate(tr):
        cef_tr.append(TR.ce_full(model, p))
        with torch.no_grad():
            cef_rel_tr.append(float(relaxed_ce(model, p, torch.ones(int(p["feats"]["idx"].numel()), device=dev))))
        if j + 1 in (1, 2, 5, 10) or j + 1 == len(tr):
            log(f"CE_full {j + 1}/{len(tr)} ({time.time() - t0:.0f}s)")
    for p in va:
        cef_va.append(TR.ce_full(model, p))
    log(f"CE_full train {np.mean(cef_tr):.4f} (relaxed-path p=1: {np.mean(cef_rel_tr):.4f}, mean |diff| "
        f"{np.mean(np.abs(np.array(cef_tr) - np.array(cef_rel_tr))):.5f}) val {np.mean(cef_va):.4f}; "
        f"max mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    os.makedirs(os.path.join(a.out_dir, "runs", a.task), exist_ok=True)
    os.makedirs(os.path.join(a.out_dir, "weights", a.task), exist_ok=True)

    # memory/time probe of one relaxed forward+backward at the longest train row
    pl = max(tr, key=lambda q: q["T"])
    torch.cuda.reset_peak_memory_stats()
    tt = time.time()
    zz = torch.full((int(pl["feats"]["idx"].numel()),), 0.9, device=dev, requires_grad=True)
    relaxed_ce(model, pl, zz).backward()
    torch.cuda.synchronize()
    log(f"relaxed fwd+bwd at T={pl['T']}: {time.time() - tt:.1f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, "
        f"|grad| {float(zz.grad.norm()):.3e}")

    if a.gap_rows > 0:
        gap_rows = [TR.prep_row(model, r, ids, dev) for r in TR.load_split(a.train_root, a.task, a.gap_rungs.split(","), tok, ids, a.gap_rows)]
        g_cef = [TR.ce_full(model, q) for q in gap_rows]
        with torch.no_grad():
            g_rel = [float(relaxed_ce(model, q, torch.ones(int(q["feats"]["idx"].numel()), device=dev))) for q in gap_rows]
        summ, recs = measure_gap(model, gap_rows, g_cef, g_rel)
        log(f"RELAXATION GAP (binary gate, relaxed - hard dCE): mean {summ['mean_gap']:+.4f} mean|.| {summ['mean_abs_gap']:.4f} "
            f"median|.| {summ['median_abs_gap']:.4f} vs mean|hard dCE| {summ['mean_abs_dce_hard']:.4f}; corr {summ['corr_hard_relaxed']}; "
            f"by rung {summ['by_rung']}")
        json.dump({"task": a.task, "summary": summ, "rows": recs}, open(os.path.join(a.out_dir, "runs", a.task, "diff_gap.json"), "w"), indent=1)
    if a.gap_only:
        return

    done = {}  # (eps_train, eps_val) -> (name, best, res): taus whose tolerance hits the same floor
    for tau in [float(t) for t in a.taus.split(",")]:
        name = a.name.format(tau=f"{tau:g}")
        key = (round(max(tau * float(np.mean(cef_tr)), a.floor), 9), round(max(tau * float(np.mean(cef_va)), a.floor), 9))
        if key in done:
            src, best, res0 = done[key]
            res = {k: v for k, v in res0.items() if k not in ("task", "config", "argv", "weights_path")}
            res.update({"tau": tau, "dedup_of": src, "final_state": res0.get("_final_state")})
            log(f"[tau{tau}] tolerance identical to {src} (eps {key[0]:.4f}, floor) -> reusing its router")
        else:
            best, res = train_tau(model, tau, tr, va, cef_tr, cef_rel_tr, cef_va, a)
            res["_final_state"] = res["final_state"]
            done[key] = (name, best, res)
        wpath = os.path.join(a.out_dir, "weights", a.task, f"{name}.pt")
        torch.save(best["state"], wpath + ".part")
        os.replace(wpath + ".part", wpath)
        final_state = res.pop("final_state")
        res = {k: v for k, v in res.items() if k != "_final_state"}
        router = RL.LinearRouter.from_state(best["state"]).to(dev)
        res.update({"task": a.task, "config": name, "variant": a.variant, "ckpt": a.ckpt, "argv": sys.argv,
                    "git_commit": G.git_commit(), "train_size": len(tr), "val_size": len(va),
                    "ce_full_train_mean": float(np.mean(cef_tr)), "ce_full_val_mean": float(np.mean(cef_va)),
                    "weights_best": TR.weights_summary(best["state"]), "weights_final": TR.weights_summary(final_state),
                    "vocab_scores_best": TR.vocab_scores(model, router, tr, tok), "weights_path": os.path.relpath(wpath, TR.REPO),
                    "hparams": {k: getattr(a, k) for k in ("epochs", "rows_per_step", "lr", "emb_lr", "init_p", "beta0", "beta1",
                                                            "anneal_frac", "st_frac", "mu0", "mu_lr", "mu_clip", "floor", "seed", "adam_beta2")}})
        rpath = os.path.join(a.out_dir, "runs", a.task, f"{name}.json")
        json.dump(res, open(rpath + ".part", "w"), indent=1)
        os.replace(rpath + ".part", rpath)
        log(f"[tau{tau}] DONE best epoch {best['epoch']} val T2/T {best['comp']:.3f} dCE {best['dce']:+.4f} "
            f"(eps_val {res['eps_val']:.4f}) w_gold {float(best['state']['w_gold']):+.2f} -> {rpath}")
    log(f"all done in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
