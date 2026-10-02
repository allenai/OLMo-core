"""Stage 2: an end-to-end learned per-(token, layer) SKIP router on top of the FROZEN token router.

    python debug/learned_router/layerskip/train_layerskip.py --task nq --ckpt ... \\
        --tok-router debug/learned_router/weights/nq/e2ecal2_rho0.1_c0.13.pt --targets 0.75,0.5,0.25

Rows: the token router's 2k split (16 train, 32 val + 32 extra train-split rows as val = 64) and the
grid's 16 test rows. Every row is first compacted by the frozen token router (exact hard semantics:
dropped routed tokens removed, original RoPE positions); the ELIGIBLE tokens are the routed body
tokens it kept. Everything else (prompt, query, answer, markers) always gets every layer.

Relaxation (``layerskip_lib``): per-layer gate a[t, l]; ``h_out = h_in + a (block(h_in) - h_in)``
and ``soft_keep = a[:, l]`` for the mixer (attention +log a key bias, GDN beta/decay x a,
removal-aware conv). Hard-concrete gates on ``logit + c`` (beta0 -> beta1 over ``--anneal-frac``,
straight-through binary from ``--st-frac``).

Size control (batch-level budget calibration, as the token router): each step solves the offset c
so that the mean P(z > 0) over the step's eligible (token, layer) pairs equals rho_t (annealed
1 -> target over ``--warm-frac``). The logits depend on c (a layer's input depends on the earlier
layers' gates), so c is solved by ``--calib-iters`` no-grad fixed-point prepasses with the step's
noise before the gradient pass. Loss = answer CE + w_kl KL(p_full || p) (full model = no token
drop, no layer skip).

Eval: deterministic gate ``logit + c* > 0`` with ONE global c* bisected on the val rows so that the
fraction of eligible pairs kept equals the target; scored on the soft path with binary gates (exact).
Compute: ``layerskip_lib.flops_ratio`` (collect_grid's FLOP model, per-layer active columns).
Baselines at the same eligible-keep: token router only, + uniform random layer skip (seeded), and
the fixed rules skip-every-attention-layer (keep 0.75), skip-odd-layers (0.5), skip-every-GDN (0.25).
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
import layerskip_lib as LS  # noqa: E402

sys.path.insert(0, os.path.dirname(HERE))
import router_lib as RL  # noqa: E402
import train_e2e_router as TE  # noqa: E402
import train_router as TR  # noqa: E402

G = TR.G


def log(m: str) -> None:
    print(f"[layerskip] {m}", flush=True)


# ------------------------------------------------------------------------------------------------
# rows
# ------------------------------------------------------------------------------------------------
def prep(model, r, ids, dev, tokr):
    """Token-router-compacted row: x (1, T2), pos (1, T2) original positions, elig (1, T2), pred
    columns, tgt, per-eligible-token features; plus the full-row CE / answer log-probs."""
    p = TR.prep_row(model, r, ids, dev)
    f = p["feats"]
    with torch.no_grad():
        k = tokr.logits(f, p["e"].float()) > 0
    keep = torch.ones(p["T"], dtype=torch.bool, device=dev)
    keep[f["idx"][~k]] = False
    kept = keep.nonzero().flatten()
    elig_full = torch.zeros(p["T"], dtype=torch.bool, device=dev)
    elig_full[f["idx"][k]] = True
    cols = torch.searchsorted(kept, p["pred"])
    assert bool((kept[cols] == p["pred"]).all()), "answer prediction position dropped"
    with torch.no_grad():
        lg = model(p["x"], logits_to_keep=p["pred"][None])[0].float()
    lpf = F.log_softmax(lg, -1)
    return {"x": p["x"][:, kept], "pos": kept[None], "elig": elig_full[kept][None], "pred": cols, "tgt": p["tgt"],
            "tokfeat": {"pos": f["pos"][k], "gold": f["gold"][k], "e": p["e"][k]},
            "lpf": lpf.to(torch.bfloat16), "ce_full": float(F.nll_loss(lpf, p["tgt"])),
            "T": p["T"], "T2": int(kept.numel()), "n_elig": int(k.sum()), "n_routed": int(k.numel()), "rung": p["rung"]}


def collate(items, pad_id: int):
    dev = items[0]["x"].device
    Tm = max(it["T2"] for it in items)
    A = max(int(it["pred"].numel()) for it in items)
    B = len(items)
    x = torch.full((B, Tm), pad_id, dtype=items[0]["x"].dtype, device=dev)
    pos = torch.zeros(B, Tm, dtype=torch.long, device=dev)
    elig = torch.zeros(B, Tm, dtype=torch.bool, device=dev)
    cols = torch.zeros(B, A, dtype=torch.long, device=dev)
    for b, it in enumerate(items):
        n = it["T2"]
        x[b, :n] = it["x"][0]
        pos[b, :n] = it["pos"][0]
        pos[b, n:] = it["pos"][0, -1] + 1 + torch.arange(Tm - n, device=dev)
        elig[b, :n] = it["elig"][0]
        a = int(it["pred"].numel())
        cols[b, :a] = it["pred"]
        cols[b, a:] = it["pred"][-1]
    tf = {k: torch.cat([it["tokfeat"][k] for it in items]) for k in items[0]["tokfeat"]}
    return x, pos, elig, cols, tf


def run_batch(model, items, pad_id, gater_fn):
    """-> list of per-row answer logits (fp32), the Gater used."""
    x, pos, elig, cols, tf = collate(items, pad_id)
    gt = gater_fn(elig, tf)
    lg = LS.forward_layerskip(model, x, logits_to_keep=cols, position_ids=pos, gate=gt)
    return [lg[b, : int(it["pred"].numel())].float() for b, it in enumerate(items)], gt


def per_row_layer_skips(gt, items, L):
    """(rows, L) number of eligible columns skipped per layer, from a Gater's recorded binary gates."""
    counts = torch.tensor([it["n_elig"] for it in items])
    bounds = torch.cumsum(counts, 0) - counts
    out = np.zeros((len(items), L))
    for layer in range(L):
        z = gt.gates[layer].float().cpu()
        for b in range(len(items)):
            out[b, layer] = float((z[bounds[b]: bounds[b] + counts[b]] <= 0.5).sum())
    return out


@torch.no_grad()
def evaluate(model, items, pad_id, gater_fn, L, bs: int = 16):
    """Per-row CE, answer log-probs KL to the full model, per-layer skip counts, FLOP ratio."""
    ce, kl, skips = [], [], []
    for s in range(0, len(items), bs):
        chunk = items[s: s + bs]
        lgs, gt = run_batch(model, chunk, pad_id, gater_fn)
        for it, lg in zip(chunk, lgs):
            lp = F.log_softmax(lg, -1)
            ce.append(float(F.nll_loss(lp, it["tgt"])))
            lpf = it["lpf"].float()
            kl.append(float((lpf.exp() * (lpf - lp)).sum(-1).mean()))
        skips.append(per_row_layer_skips(gt, chunk, L))
    skips = np.concatenate(skips)
    ne = np.array([it["n_elig"] for it in items], dtype=float)
    kept_pairs = float((ne[:, None] * L - skips.sum(1, keepdims=True)).sum() / max(1.0, ne.sum() * L))
    flops = [LS.flops_ratio(it["T"], it["T2"], skips[i]) for i, it in enumerate(items)]
    return {"ce": ce, "kl": kl, "keep_pairs": kept_pairs, "flops": flops, "skip_per_layer": skips.sum(0).tolist(),
            "elig_total": float(ne.sum())}


def calibrate_eval_offset(model, items, pad_id, router, target, L, iters: int = 22):
    """Global c* s.t. the deterministic gate keeps ``target`` of the eligible (token, layer) pairs over
    ``items`` (bisection; each probe is a full forward since a layer's input depends on the gates)."""
    lo, hi = -40.0, 40.0

    def frac(c):
        return evaluate(model, items, pad_id, lambda el, tf: LS.Gater(el, "det", router=router, c=c, tokfeat=tf), L)["keep_pairs"]

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if frac(mid) > target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def summarize(res, base_ce, ref_ce=None):
    d = np.array(res["ce"]) - np.array(base_ce)
    out = {"dce": float(d.mean()), "dce_se": float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan"),
           "dce_median": float(np.median(d)), "dce_max": float(d.max()), "flops": float(np.mean(res["flops"])),
           "keep_pairs": res["keep_pairs"], "kl": float(np.mean(res["kl"])), "eval_size": len(d)}
    if ref_ce is not None:
        d2 = np.array(res["ce"]) - np.array(ref_ce)
        out.update({"dce_vs_tok": float(d2.mean()), "dce_vs_tok_se": float(d2.std(ddof=1) / math.sqrt(len(d2)))})
    return out


# ------------------------------------------------------------------------------------------------
# training
# ------------------------------------------------------------------------------------------------
def train_one(model, variant, target, tr, va, a, L, d, wb=None, init_state=None):
    dev = next(model.parameters()).device
    torch.manual_seed(1000 + a.seed)
    gen = torch.Generator().manual_seed(2000 + a.seed)
    router = LS.LayerRouter(L, d, variant, LS.attn_layers()).to(dev)
    if init_state is not None:  # in-memory warm start (frontier.py)
        router = LS.LayerRouter.from_state(init_state).to(dev)
        log(f"[{variant} rho{target}] warm start from an in-memory router (its target {init_state.get('target')})")
    elif getattr(a, "init_router", None):
        st0 = torch.load(a.init_router.format(variant=variant), map_location="cpu")
        assert st0["variant"] == variant, (st0["variant"], variant)
        router = LS.LayerRouter.from_state(st0).to(dev)
        log(f"[{variant} rho{target}] warm start from {a.init_router.format(variant=variant)} (its target {st0.get('target')})")
    opt = torch.optim.Adam(router.param_groups(a.lr, a.w_lr), betas=(0.9, a.adam_beta2))
    n_steps = math.ceil(len(tr) / a.rows_per_step) * a.epochs
    shift_fn = lambda beta: beta * math.log(-RL.HC_GAMMA / RL.HC_ZETA)  # noqa: E731
    c = 60.0
    steps, epochs = [], []
    t0 = time.time()
    step = 0
    log(f"[{variant} rho{target}] router {router.n_params()} params, {len(tr)} train rows, {n_steps} steps")
    for ep in range(1, a.epochs + 1):
        order = torch.randperm(len(tr), generator=gen).tolist()
        for s0 in range(0, len(order), a.rows_per_step):
            prog = step / max(1, n_steps - 1)
            rho_t = getattr(a, "rho_start", 1.0) - (getattr(a, "rho_start", 1.0) - target) * min(1.0, prog / a.warm_frac)
            beta = a.beta0 * (a.beta1 / a.beta0) ** min(1.0, prog / a.anneal_frac)
            st = prog >= a.st_frac
            items = [tr[i] for i in order[s0: s0 + a.rows_per_step]]
            seed = 7919 * (step + 1) + a.seed

            def mk(el, tf, cc):
                return LS.Gater(el, "sample", router=router, c=cc, beta=beta, seed=seed, st=st, tokfeat=tf)

            # budget calibration: fixed-point prepasses (same noise), then the gradient pass
            c_hist = []
            for _ in range(a.calib_iters):
                with torch.no_grad():
                    _, gt = run_batch(model, items, a.pad_id, lambda el, tf: mk(el, tf, c))
                raw = torch.cat([gt.raw[l_] for l_ in range(L) if l_ in gt.raw]) if gt.raw else None
                if raw is not None and raw.numel():
                    c = TE.calibrate_offset(raw, rho_t, beta)
                c_hist.append(c)
            opt.zero_grad(set_to_none=True)
            lgs, gt = run_batch(model, items, a.pad_id, lambda el, tf: mk(el, tf, c))
            loss = 0.0
            ces, kls = [], []
            for it, lg in zip(items, lgs):
                lp = F.log_softmax(lg, -1)
                ce = F.nll_loss(lp, it["tgt"])
                lpf = it["lpf"].float()
                kl = (lpf.exp() * (lpf - lp)).sum(-1).mean()
                loss = loss + (ce + a.w_kl * kl) / len(items)
                ces.append(float(ce) - it["ce_full"])
                kls.append(float(kl))
            loss.backward()
            opt.step()
            raw = torch.cat([gt.raw[l_] for l_ in range(L)]) if gt.raw else torch.zeros(0)
            keep_l0 = float(torch.sigmoid(raw + c - shift_fn(beta)).mean()) if raw.numel() else 1.0
            keep_z = float(torch.cat([gt.gates[l_] for l_ in range(L)]).gt(0).float().mean()) if raw.numel() else 1.0
            step += 1
            rec = {"step": step, "epoch": ep, "rho_t": rho_t, "beta": beta, "st": int(st), "c": c, "c_drift": c_hist[-1] - c_hist[0] if c_hist else 0.0,
                   "keep_l0": keep_l0, "keep_z": keep_z, "dce": float(np.mean(ces)), "kl": float(np.mean(kls)), "sec": time.time() - t0}
            steps.append(rec)
            if wb is not None:
                wb.log({f"train/{k}": v for k, v in rec.items() if k != "step"}, step=step)
            if step in (1, 2, 5) or step % 10 == 0:
                el_ = time.time() - t0
                log(f"[{variant} rho{target}] step {step}/{n_steps} ep {ep}: keepL0 {keep_l0:.3f} (rho_t {rho_t:.3f}) z>0 {keep_z:.3f} "
                    f"dCE {rec['dce']:+.4f} KL {rec['kl']:.4f} c {c:+.2f} beta {beta:.2f}{' ST' if st else ''} | {el_:.0f}s, "
                    f"ETA {el_ / step * (n_steps - step):.0f}s")
        if ep in (max(1, a.epochs // 2), a.epochs) or (a.val_every and ep % a.val_every == 0):
            cs = calibrate_eval_offset(model, va, a.pad_id, router, target, L)
            v = evaluate(model, va, a.pad_id, lambda el, tf: LS.Gater(el, "det", router=router, c=cs, tokfeat=tf), L)
            vs = summarize(v, [it["ce_full"] for it in va])
            epochs.append({"epoch": ep, "c_star": cs, "val": vs})
            if wb is not None:
                wb.log({"val/dce": vs["dce"], "val/flops": vs["flops"], "val/keep_pairs": vs["keep_pairs"], "epoch": ep}, step=step)
            log(f"[{variant} rho{target}] epoch {ep}: VAL (c*={cs:+.3f}) dCE {vs['dce']:+.4f} +- {vs['dce_se']:.4f} "
                f"(median {vs['dce_median']:+.4f}) keep_pairs {vs['keep_pairs']:.3f} FLOPs x{vs['flops']:.4f}")
    return router, {"steps": steps, "epochs": epochs, "train_sec": time.time() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="nq")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt-format", default="hf", choices=["hf", "distcp"])
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--val-root", required=True)
    ap.add_argument("--eval-data-root", required=True)
    ap.add_argument("--tok-router", required=True)
    ap.add_argument("--rungs", default="2k")
    ap.add_argument("--n-train", type=int, default=16)
    ap.add_argument("--n-val", type=int, default=32)
    ap.add_argument("--val-extra-from-train", type=int, default=32)
    ap.add_argument("--eval-rows", type=int, default=16)
    ap.add_argument("--targets", default="0.75,0.5,0.25")
    ap.add_argument("--variants", default="perlayer")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--rows-per-step", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--w-lr", type=float, default=0.01)
    ap.add_argument("--adam-beta2", type=float, default=0.8)
    ap.add_argument("--beta0", type=float, default=2 / 3)
    ap.add_argument("--beta1", type=float, default=0.1)
    ap.add_argument("--warm-frac", type=float, default=0.5)
    ap.add_argument("--anneal-frac", type=float, default=0.7)
    ap.add_argument("--st-frac", type=float, default=0.85)
    ap.add_argument("--w-kl", type=float, default=1.0)
    ap.add_argument("--calib-iters", type=int, default=2)
    ap.add_argument("--val-every", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rand-seeds", type=int, default=3)
    ap.add_argument("--init-router", default=None,
                    help="warm start: weights path, may contain {variant} (e.g. weights/outlier/ls_all_{variant}_k0.75.pt)")
    ap.add_argument("--rho-start", type=float, default=1.0, help="keep target at step 0 (anneals linearly to the target)")
    ap.add_argument("--eval-only", action="store_true",
                    help="no training: re-score saved weights/<task>/<name>_<variant>_k<target>.pt -> runs/<task>/<name>_evalonly.json")
    ap.add_argument("--name", default="ls")
    ap.add_argument("--wandb-group", default=None)
    ap.add_argument("--wandb-project", default="memory-networks")
    ap.add_argument("--wandb-entity", default="prasann-uc-berkeley-electrical-engineering-computer-sciences")
    ap.add_argument("--tokenizer", default=os.environ.get("DEVLOSS_TOKENIZER", G.TOKENIZER_BY_FAMILY[G.FAMILY]))
    ap.add_argument("--out-dir", default=HERE)
    a = ap.parse_args()
    t_start = time.time()
    log(f"start task={a.task} variants={a.variants} targets={a.targets} epochs={a.epochs} tok_router={a.tok_router}")
    from transformers import AutoTokenizer

    from olmo_core.nn.attention.fla_autotune import freeze_fla_length_autotune

    ids = G.RESERVED_IDS[G.FAMILY]
    vocab = G.VOCAB_BY_FAMILY[G.FAMILY]
    a.pad_id = ids.eos
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rungs = a.rungs.split(",")
    tr_rows = TR.load_split(a.train_root, a.task, rungs, tok, ids, a.n_train)
    va_rows = TR.load_split(a.val_root, a.task, rungs, tok, ids, a.n_val)
    if a.val_extra_from_train:
        extra = TR.load_split(a.train_root, a.task, rungs, tok, ids, a.n_train + a.val_extra_from_train)
        seen = {tuple(r["ids"]) for r in tr_rows}
        va_rows += [r for r in extra if tuple(r["ids"]) not in seen]
    te_rows = TR.load_split(a.eval_data_root, a.task, rungs, tok, ids, a.eval_rows)
    all_ids = np.concatenate([np.asarray(r["ids"], dtype=np.int64) for r in tr_rows])
    pieces = tok.convert_ids_to_tokens(list(range(vocab)))
    stop_ids, _, tables = G.build_tables(all_ids, vocab, pieces, ids, lambda t: tok.decode([int(t)]))
    model = G.load_model(a.ckpt, a.ckpt_format, vocab, ids, 42, stop_ids, "flash_2")
    for p_ in model.parameters():
        p_.requires_grad_(False)
    freeze_fla_length_autotune()
    dev = next(model.parameters()).device
    TR.configure(model, stop_ids, tables.to(dev))
    max_pos = 2 * max(len(r["ids"]) for r in tr_rows + va_rows + te_rows) + 8
    for mod in model.modules():
        rope = getattr(mod, "rope", None)
        if rope is not None and hasattr(rope, "warmup_cache"):
            rope.warmup_cache(max_pos, dev)
    model.eval()
    L, d = len(model.blocks), int(model.embeddings.weight.shape[1])
    tst = torch.load(a.tok_router, map_location="cpu")
    tokr = RL.LinearRouter.from_state(tst).to(dev)
    assert not tokr.route_markers
    log(f"token router {a.tok_router}: variant {tokr.variant}, keep_rule {tst.get('keep_rule')}, match_comp {tst.get('match_comp')}")

    t0 = time.time()
    splits = {}
    for nm, rows in (("train", tr_rows), ("val", va_rows), ("test", te_rows)):
        out = []
        for j, r in enumerate(rows):
            out.append(prep(model, r, ids, dev, tokr))
            if j + 1 in (1, 2, 5, 10) or (j + 1) % 25 == 0 or j + 1 == len(rows):
                log(f"prep {nm} {j + 1}/{len(rows)} ({time.time() - t0:.0f}s)")
        splits[nm] = out
    tr, va, te = splits["train"], splits["val"], splits["test"]
    for nm, it in splits.items():
        log(f"{nm}: {len(it)} rows, T {np.mean([x['T'] for x in it]):.0f} -> T2 {np.mean([x['T2'] for x in it]):.0f} "
            f"(x{np.mean([x['T2'] / x['T'] for x in it]):.3f}), eligible/row {np.mean([x['n_elig'] for x in it]):.0f} "
            f"({np.mean([x['n_elig'] / x['T2'] for x in it]):.2f} of T2), CE_full {np.mean([x['ce_full'] for x in it]):.4f}")

    # token-router-only references: flash (no soft_keep; == the grid's hard compaction) and the soft
    # path with all-ones gates (the kernels every layer-skip policy runs on)
    ones_fn = lambda el, tf: LS.Gater(el, "ones")  # noqa: E731
    ref = {}
    for nm, it in splits.items():
        with torch.no_grad():
            flash = [float(F.cross_entropy(model(x_["x"], logits_to_keep=x_["pred"][None], position_ids=x_["pos"])[0].float(), x_["tgt"]))
                     for x_ in it]
        soft = evaluate(model, it, a.pad_id, ones_fn, L)
        ref[nm] = {"full": [x_["ce_full"] for x_ in it], "tok_flash": flash, "tok": soft["ce"], "tok_flops": soft["flops"]}
        log(f"{nm}: token-router only dCE (flash) {np.mean(flash) - np.mean(ref[nm]['full']):+.4f}, (soft ones) "
            f"{np.mean(soft['ce']) - np.mean(ref[nm]['full']):+.4f}; FLOPs x{np.mean(soft['flops']):.4f}")
    out = {"task": a.task, "argv": sys.argv, "git_commit": G.git_commit(), "tok_router": a.tok_router,
           "train_size": len(tr), "val_size": len(va), "eval_size": len(te),
           "rows": {nm: [{"T": x_["T"], "T2": x_["T2"], "n_elig": x_["n_elig"]} for x_ in it] for nm, it in splits.items()},
           "ref": ref, "hparams": {k: v for k, v in vars(a).items() if k not in ("pad_id",)}, "points": {}, "baselines": {}}
    os.makedirs(os.path.join(a.out_dir, "runs", a.task), exist_ok=True)
    os.makedirs(os.path.join(a.out_dir, "weights", a.task), exist_ok=True)
    opath = os.path.join(a.out_dir, "runs", a.task, f"{a.name}{'_evalonly' if a.eval_only else ''}.json")

    if os.path.exists(opath) and not a.eval_only:
        # preempted + requeued: keep the finished points (their weights are saved) and skip them below
        prev = json.load(open(opath))
        out["points"] = prev.get("points", {})
        if out["points"]:
            log(f"resuming: {len(out['points'])} finished point(s) in {opath}: {sorted(out['points'])}")

    def dump():
        json.dump(out, open(opath + ".part", "w"), indent=1)
        os.replace(opath + ".part", opath)

    # fixed-rule and random baselines on the test (and val) rows
    attn = set(LS.attn_layers())
    rules = {"skip_attn": [l_ for l_ in range(L) if l_ in attn], "skip_odd": list(range(1, L, 2)),
             "skip_gdn": [l_ for l_ in range(L) if l_ not in attn]}

    def fixed_fn(skip_layers):
        def fn(el, tf):
            vals = torch.ones(L, *el.shape, device=el.device)
            vals[skip_layers] = 0.0
            return LS.Gater(el, "fixed", values=vals)
        return fn

    def rand_fn(keep, seed):
        def fn(el, tf):
            g_ = torch.Generator().manual_seed(seed)
            return LS.Gater(el, "fixed", values=(torch.rand(L, *el.shape, generator=g_) < keep).float().to(el.device))
        return fn

    for nm, sk in rules.items():
        for split in ("val", "test"):
            it = splits[split]
            r_ = evaluate(model, it, a.pad_id, fixed_fn(sk), L)
            out["baselines"].setdefault(nm, {})[split] = {**summarize(r_, ref[split]["full"], ref[split]["tok"]), "ce": r_["ce"]}
        s_ = out["baselines"][nm]["test"]
        log(f"baseline {nm}: TEST dCE {s_['dce']:+.4f} +- {s_['dce_se']:.4f} (vs tok {s_['dce_vs_tok']:+.4f} +- {s_['dce_vs_tok_se']:.4f}) "
            f"keep_pairs {s_['keep_pairs']:.3f} FLOPs x{s_['flops']:.4f}")
    targets = [float(t) for t in a.targets.split(",")]
    for tgt_ in targets:
        for split in ("val", "test"):
            it = splits[split]
            ces, fl, kp = [], [], []
            for sd in range(a.rand_seeds):
                r_ = evaluate(model, it, a.pad_id, rand_fn(tgt_, 100 + sd), L)
                ces.append(r_["ce"])
                fl.append(r_["flops"])
                kp.append(r_["keep_pairs"])
            avg = {"ce": np.mean(ces, 0).tolist(), "flops": np.mean(fl, 0).tolist(), "keep_pairs": float(np.mean(kp)), "kl": [0.0]}
            s_ = summarize(avg, ref[split]["full"], ref[split]["tok"])
            s_["per_seed_dce"] = [float(np.mean(np.array(c_) - np.array(ref[split]["full"]))) for c_ in ces]
            out["baselines"].setdefault(f"random_{tgt_:g}", {})[split] = {**s_, "ce": avg["ce"]}
        s_ = out["baselines"][f"random_{tgt_:g}"]["test"]
        log(f"baseline random keep {tgt_:g} ({a.rand_seeds} seeds): TEST dCE {s_['dce']:+.4f} +- {s_['dce_se']:.4f} "
            f"(vs tok {s_['dce_vs_tok']:+.4f}) per-seed {[round(v, 4) for v in s_['per_seed_dce']]} FLOPs x{s_['flops']:.4f}")
    dump()

    for variant in a.variants.split(","):
        for tgt_ in targets:
            name = f"{a.name}_{variant}_k{tgt_:g}"
            if name in out["points"] and not a.eval_only:
                log(f"[{name}] already done (resumed); skipping")
                continue
            wb = None
            if a.wandb_group and not a.eval_only:
                try:
                    import wandb

                    wb = wandb.init(entity=a.wandb_entity, project=a.wandb_project, group=a.wandb_group, reinit=True,
                                    name=f"{a.task}-{name}-{os.environ.get('SLURM_JOB_ID', 'local')}",
                                    config={**{k: v for k, v in vars(a).items()}, "variant": variant, "target": tgt_},
                                    settings=wandb.Settings(init_timeout=90))
                    log(f"wandb run {wb.url} (group https://wandb.ai/{a.wandb_entity}/{a.wandb_project}/groups/{a.wandb_group})")
                except Exception as ex:  # noqa: BLE001
                    log(f"wandb init failed ({ex!r}); continuing without wandb")
                    wb = None
            if a.eval_only:
                st0 = torch.load(os.path.join(a.out_dir, "weights", a.task, f"{name}.pt"), map_location="cpu")
                router, hist = LS.LayerRouter.from_state(st0).to(dev), {"eval_only": True, "epochs": [{"c_star": st0["c_star"]}]}
            else:
                router, hist = train_one(model, variant, tgt_, tr, va, a, L, d, wb)
            cs = hist["epochs"][-1]["c_star"]
            det = lambda el, tf, r_=router, c_=cs: LS.Gater(el, "det", router=r_, c=c_, tokfeat=tf)  # noqa: E731
            pt = {"variant": variant, "target": tgt_, "c_star": cs, "n_params": router.n_params(), "history": hist}
            for split in ("train", "val", "test"):
                r_ = evaluate(model, splits[split], a.pad_id, det, L)
                s_ = summarize(r_, ref[split]["full"], ref[split]["tok"])
                ne = sum(x_["n_elig"] for x_ in splits[split])
                s_["skip_rate_per_layer"] = [v / max(1, ne) for v in r_["skip_per_layer"]]
                pt[split] = {**s_, "ce": r_["ce"]}
            te_s, tr_s = pt["test"], pt["train"]
            pt["train_val_gap"] = pt["val"]["dce"] - tr_s["dce"]
            # token-independent control at MATCHED budget: the round((1 - target) L) layers this router
            # skips most on VAL, skipped for EVERY eligible token (is the policy more than a layer mask?)
            vrate = np.array(pt["val"]["skip_rate_per_layer"])
            n_static = int(round((1.0 - tgt_) * L))
            static = sorted(int(v) for v in np.argsort(-vrate, kind="stable")[:n_static])
            r_ = evaluate(model, te, a.pad_id, fixed_fn(static), L)
            pt["static_test"] = {**summarize(r_, ref["test"]["full"], ref["test"]["tok"]), "layers": static, "ce": r_["ce"]}
            ss = pt["static_test"]
            log(f"[{name}] static-mask control (skip layers {static} for every eligible token): TEST dCE {ss['dce']:+.4f} +- {ss['dce_se']:.4f} "
                f"(vs tok {ss['dce_vs_tok']:+.4f} +- {ss['dce_vs_tok_se']:.4f}) keep_pairs {ss['keep_pairs']:.3f} FLOPs x{ss['flops']:.4f}")
            state = router.state()
            state.update({"c_star": cs, "target": tgt_, "tok_router": a.tok_router})
            if not a.eval_only:
                torch.save(state, os.path.join(a.out_dir, "weights", a.task, f"{name}.pt"))
            out["points"][name] = pt
            dump()
            sr = np.array(te_s["skip_rate_per_layer"])
            att_idx = sorted(attn)
            gdn_idx = [l_ for l_ in range(L) if l_ not in attn]
            log(f"[{name}] TEST dCE {te_s['dce']:+.4f} +- {te_s['dce_se']:.4f} (median {te_s['dce_median']:+.4f}; vs tok {te_s['dce_vs_tok']:+.4f} "
                f"+- {te_s['dce_vs_tok_se']:.4f}) keep_pairs {te_s['keep_pairs']:.3f} FLOPs x{te_s['flops']:.4f} "
                f"(tok-only x{np.mean(ref['test']['tok_flops']):.4f}) | train dCE {tr_s['dce']:+.4f} val {pt['val']['dce']:+.4f} "
                f"| skip rate attn {sr[att_idx].mean():.2f} gdn {sr[gdn_idx].mean():.2f} early(0-15) {sr[:16].mean():.2f} late(16-31) {sr[16:].mean():.2f}")
            log(f"[{name}] per-layer skip rate: {' '.join(f'{v:.2f}' for v in sr)}")
            if wb is not None:
                wb.summary.update({"test_dce": te_s["dce"], "test_flops": te_s["flops"], "val_dce": pt["val"]["dce"], "train_dce": tr_s["dce"]})
                wb.finish()
    log(f"all done in {time.time() - t_start:.0f}s -> {opath}")


if __name__ == "__main__":
    main()
