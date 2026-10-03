"""Per-example OVERFIT ceiling on compute: free per-example token and (token, layer) keep logits, trained
to minimise DIFFERENTIABLE expected FLOPs s.t. dCE <= tau on that one example (uses the answer, so a
non-deployable LOWER BOUND on FLOPs -- it measures the headroom the learned routers leave).

    python debug/learned_router/layerskip/oracle_ls.py --task nq --ckpt ... --rows 0-5 --taus 0.02,0.05

Per row, four configurations are trained in ONE batched forward (B = 4):
    (i) token-only @ tau 0.02, (i) @ 0.05, (ii) token + layer @ 0.02, (ii) @ 0.05.
* Parameters: theta_t per ROUTED token (document body + markers); phi_{t,l} per document BODY token and
  layer (arm ii; markers, prompt, question and answer always run every layer). Init p = 0.95.
* Gates: hard-concrete on theta / phi (beta 2/3 -> 0.1 over 70%, straight-through from 85%). Exact soft
  relaxation (``layerskip_lib.forward_layerskip``): token gate = soft_keep, layer gate = residual mix +
  per-layer soft_keep.
* Expected FLOPs (differentiable): per layer, n_l = #always-active + sum_markers P(g>0) + sum_body
  P(g>0) P(a_l>0); cost = linear(n_l) [+ ATTN_QUAD n_l^2 at attention layers]; / full-row FLOPs
  (``collect_grid`` constants -- the same model as every other number here).
* Lagrangian F + mu (dCE - tau) / tau, dual ascent on an EMA of the sampled relaxed dCE.
* ``--force-id-prefix K`` (fair variant for doc-id answers): every document's markers and first K body tokens
  are always kept AND run every layer (as the bar keeps them), so the oracle cannot leak the label by
  selecting only the answer documents' ids.
* HARD check: deterministic gates (logit > 0), exact deletion (grid semantics, routed markers) +
  per-layer skip; if dCE > tau, add back the highest-logit dropped items (tokens, then layer pairs) in
  chunks until <= tau; then greedily remove the lowest-logit kept items (tokens, then layer pairs) in
  halving chunks while dCE stays <= tau.
References on the same rows: the bar (gold_fl20p8_noslot) and the frontier pick per tier
(``frontier.json`` + ``runs/<task>/fr_*.json``, test64 per-row numbers).
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
import frontier as FR  # noqa: E402
import layerskip_lib as LS  # noqa: E402
import train_layerskip as TLS  # noqa: E402

RL, TR, G = FR.RL, FR.TR, FR.G
log = TLS.log


def flop_consts():
    import collect_grid as CG

    L = CG.N_LAYERS
    lin = torch.tensor([CG.ATTN_LIN if l in CG.ATTN_LAYERS else CG.GDN_LIN for l in range(L)], dtype=torch.float64)
    quad = torch.tensor([CG.ATTN_QUAD if l in CG.ATTN_LAYERS else 0.0 for l in range(L)], dtype=torch.float64)
    return CG, lin, quad


def expected_flops(n_fixed, p_tok_mk, p_tok_body, q_body, lin, quad, full):
    """n_l = n_fixed + sum(p markers) + sum_t p_t q_{t,l}  ->  sum_l lin_l n_l + quad_l n_l^2, / full."""
    n = n_fixed + p_tok_mk.sum() + (p_tok_body[None, :] * q_body).sum(1)  # (L,)
    n = n.double()
    return ((lin.to(n.device) * n + quad.to(n.device) * n * n).sum() / full).float()


class Row:
    """A test row with everything needed for soft training and hard evaluation."""

    def __init__(self, model, p, ids, dev, L, force_prefix: int = 0):
        self.p, self.ids, self.L = p, ids, L
        fe = p["feats"]
        self.T = p["T"]
        self.idx = fe["idx"]
        self.mk = fe["is_marker"] > 0.5
        self.body_pos = self.idx[~self.mk]
        self.N, self.Nb = int(self.idx.numel()), int((~self.mk).sum())
        self.n_fixed = self.T - self.N
        self.elig = torch.zeros(1, self.T, dtype=torch.bool, device=dev)
        self.elig[0, self.body_pos] = True
        bi = torch.full((self.T,), -1, dtype=torch.long, device=dev)
        bi[self.body_pos] = torch.arange(self.Nb, device=dev)
        self.body_index = bi  # position -> body slot
        # --force-id-prefix K: every document's markers and first K body tokens are always kept (as the bar
        # keeps them), removing the "keep only the gold doc's id" selection shortcut on id-answer tasks
        self.forced = torch.zeros(self.N, dtype=torch.bool, device=dev)
        if force_prefix > 0:
            self.forced = self.mk | (fe["j"].to(dev) < force_prefix)
        # forced body tokens also run EVERY layer (as in the bar): skipping a token at all layers would remove it
        self.forced_body = self.forced[~self.mk]  # (Nb,)

    @torch.no_grad()
    def hard(self, model, keep_tok, pairs):
        """keep_tok (N,) bool over routed tokens; pairs (L, Nb) bool (layer kept). -> (ce, flops, info)."""
        p = self.p
        mask = RL.keep_mask_from(p["feats"], keep_tok, self.T)
        real = FR.router_real(p["x"][0].cpu(), p["cid"], mask.cpu(), self.ids).to(p["x"].device)
        kept = real.nonzero().flatten()
        cols = torch.searchsorted(kept, p["pred"])
        x = p["x"][:, kept]
        bodyk = self.body_index[kept]  # body slot or -1
        el = (bodyk >= 0)
        vals = torch.ones(self.L, 1, int(kept.numel()), device=x.device)
        if el.any():
            vals[:, 0, el] = pairs[:, bodyk[el]].float()
        gate = LS.Gater(el[None], "fixed", values=vals)
        lg = LS.forward_layerskip(model, x, logits_to_keep=cols[None], position_ids=kept[None], gate=gate)
        ce = float(F.cross_entropy(lg[0].float(), p["tgt"]))
        n_skip = [float((vals[l, 0, el] < 0.5).sum()) for l in range(self.L)]
        return ce, LS.flops_ratio(self.T, int(kept.numel()), n_skip), int(kept.numel())


def greedy(row, model, keep_tok, pairs, ce_full, tau, theta, phi, layer_arm, max_evals=60):
    """Repair (add back) then prune (remove) with exact hard evals. Returns final masks + stats."""
    n_eval = 0

    def ev(kt, pr):
        nonlocal n_eval
        n_eval += 1
        ce, fl, t2 = row.hard(model, kt, pr)
        return ce - ce_full, fl

    keep_tok, pairs = keep_tok.clone(), pairs.clone()
    body_of_tok = torch.zeros(row.N, dtype=torch.long, device=keep_tok.device) - 1
    body_of_tok[~row.mk] = torch.arange(row.Nb, device=keep_tok.device)
    d, fl = ev(keep_tok, pairs)
    d0, fl0 = d, fl
    # ---- add back while over tau: dropped tokens by theta desc, then dropped pairs by phi desc
    while d > tau:
        drop_t = (~keep_tok & ~row.forced).nonzero().flatten()
        if drop_t.numel():
            order = drop_t[torch.argsort(theta[drop_t], descending=True)]
            k = max(1, int(math.ceil(0.1 * order.numel())))
            keep_tok[order[:k]] = True
        elif layer_arm and (~pairs).any():
            kb = keep_tok[~row.mk]  # body tokens kept
            cand = (~pairs) & kb[None, :]
            ids_ = cand.flatten().nonzero().flatten()
            if not ids_.numel():
                break
            sc = phi.flatten()[ids_]
            order = ids_[torch.argsort(sc, descending=True)]
            k = max(1, int(math.ceil(0.1 * order.numel())))
            pairs.view(-1)[order[:k]] = True
        else:
            break
        d, fl = ev(keep_tok, pairs)
        if n_eval > max_evals:
            break
    # ---- prune while under tau: kept pairs by phi asc (layer arm), then kept tokens by theta asc
    phases = ["tokens"] + (["pairs"] if layer_arm else [])
    for ph in phases:
        if ph == "pairs":
            kb = keep_tok[~row.mk] & ~row.forced_body
            cand = pairs & kb[None, :]
            ids_ = cand.flatten().nonzero().flatten()
            order = ids_[torch.argsort(phi.flatten()[ids_])]
        else:
            ids_ = (keep_tok & ~row.forced).nonzero().flatten()
            order = ids_[torch.argsort(theta[ids_])]
        pos, c = 0, max(1, int(0.1 * order.numel()))
        budget = n_eval + max_evals // len(phases)  # each phase gets its share of hard evals
        while pos < order.numel() and n_eval < budget and d <= tau:
            chunk = order[pos: pos + c]
            kt2, pr2 = keep_tok.clone(), pairs.clone()
            if ph == "pairs":
                pr2.view(-1)[chunk] = False
            else:
                kt2[chunk] = False
            d2, fl2 = ev(kt2, pr2)
            if d2 <= tau:
                keep_tok, pairs, d, fl = kt2, pr2, d2, fl2
                pos += c
            elif c == 1:
                pos += 1  # this single item is needed: skip it
            else:
                c //= 2
    return keep_tok, pairs, d, fl, {"evals": n_eval, "soft_det_dce": d0, "soft_det_flops": fl0}


def dump_kept(row, keep_tok, tok, max_docs=40):
    """Exactly what the oracle keeps: per document with any kept routed token, the decoded kept body text,
    whether the doc is gold, how many of its tokens are kept, and whether the markers are kept."""
    fe = row.p["feats"]
    x = row.p["x"][0]
    out = []
    docs = sorted(set(int(v) for v in fe["doc"][keep_tok].tolist()))
    for dd in docs[:max_docs]:
        m = (fe["doc"] == dd)
        kb = m & keep_tok & ~row.mk
        pos = fe["idx"][kb]
        out.append({"doc": dd, "gold": bool((fe["gold"][m] > 0.5).any()), "n_body": int((m & ~row.mk).sum()),
                    "n_kept_body": int(kb.sum()), "markers_kept": int((m & keep_tok & row.mk).sum()),
                    "kept_j": fe["j"][kb].tolist()[:64], "text": tok.decode(x[pos].tolist())[:400]})
    return {"n_docs_touched": len(docs), "docs": out}


def keep_stats(row, keep_tok, pairs, L):
    fe = row.p["feats"]
    gold = fe["gold"] > 0.5
    body = ~row.mk
    idr = body & (fe["j"] < 8) & ~gold
    attn = [l for l in range(L) if (l + 1) % 4 == 0]
    gdn = [l for l in range(L) if (l + 1) % 4 != 0]
    kb = keep_tok[body]

    def frac(m):
        return float(keep_tok[m].float().mean()) if bool(m.any()) else float("nan")

    pr = pairs[:, kb] if bool(kb.any()) else pairs[:, :0]
    sk = (1.0 - pr.float().mean(1)).tolist() if pr.shape[1] else [float("nan")] * L
    return {"gold_body": frac(body & gold), "nongold_body": frac(body & ~gold), "nongold_id_region": frac(idr),
            "markers": frac(row.mk), "skip_rate_per_layer": sk,
            "skip_attn": float(np.mean([sk[l] for l in attn])), "skip_gdn": float(np.mean([sk[l] for l in gdn])),
            "skip_early": float(np.mean(sk[:16])), "skip_late": float(np.mean(sk[16:]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt-format", default="hf")
    ap.add_argument("--data-root", required=True, help="test64 root (2k) or the grid's staged root (other rungs)")
    ap.add_argument("--rung", default="2k")
    ap.add_argument("--rows", default="0-5")
    ap.add_argument("--taus", default="0.02,0.05")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--mu0", type=float, default=0.1)
    ap.add_argument("--mu-lr", type=float, default=0.02)
    ap.add_argument("--init-p", type=float, default=0.95)
    ap.add_argument("--max-evals", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force-id-prefix", type=int, default=0, help="always keep every doc's markers + first K body tokens")
    ap.add_argument("--tokenizer", default=os.environ.get("DEVLOSS_TOKENIZER", G.TOKENIZER_BY_FAMILY[G.FAMILY]))
    ap.add_argument("--out", default="")
    ap.add_argument("--wandb-group", default=None)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    from olmo_core.nn.attention.fla_autotune import freeze_fla_length_autotune

    lo, hi = (int(v) for v in a.rows.split("-"))
    taus = [float(t) for t in a.taus.split(",")]
    ids = G.RESERVED_IDS[G.FAMILY]
    vocab = G.VOCAB_BY_FAMILY[G.FAMILY]
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rows = TR.load_split(a.data_root, a.task, [a.rung], tok, ids, hi + 1)[lo: hi + 1]
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
    CG, lin, quad = flop_consts()
    sfx = f"_fid{a.force_id_prefix}" if a.force_id_prefix else ""
    out_path = a.out or os.path.join(HERE, "runs", a.task, f"oracle{sfx}_{a.rung}_r{lo}-{hi}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    res = {"task": a.task, "rung": a.rung, "argv": sys.argv, "git_commit": G.git_commit(), "rows": []}
    if os.path.exists(out_path):
        res["rows"] = json.load(open(out_path)).get("rows", [])
    done = {r["row"] for r in res["rows"]}
    wb = None
    if a.wandb_group:
        try:
            import wandb

            wb = wandb.init(entity="prasann-uc-berkeley-electrical-engineering-computer-sciences", project="memory-networks",
                            group=a.wandb_group, name=f"{a.task}-oracle-{a.rung}-r{lo}-{hi}-{os.environ.get('SLURM_JOB_ID', 'local')}",
                            config=vars(a), settings=wandb.Settings(init_timeout=90))
        except Exception as ex:  # noqa: BLE001
            log(f"wandb init failed ({ex!r})")
    for ri, r in enumerate(rows):
        row_id = lo + ri
        if row_id in done:
            log(f"row {row_id} done (resumed)")
            continue
        t_row = time.time()
        p = FR.base_prep(model, r, ids, dev)
        R = Row(model, p, ids, dev, L, a.force_id_prefix)
        full = CG.forward_flops(R.T)
        ce_full = p["ce_full"]
        # bar on this row
        bar_it = FR.compact(p, p["real_bar"], torch.zeros(R.T, dtype=torch.bool))
        ce_bar = FR.flash_ce(model, bar_it)
        fl_bar = LS.flops_ratio(R.T, bar_it["T2"], [0] * L)
        cfgs = [(arm, tau) for arm in ("tok", "tok+layer") for tau in taus]
        B = len(cfgs)
        torch.manual_seed(a.seed + row_id)
        gen = torch.Generator().manual_seed(1000 + a.seed + row_id)
        th0 = math.log(a.init_p / (1 - a.init_p))
        theta = (th0 + 0.01 * torch.randn(B, R.N, generator=gen)).to(dev).requires_grad_(True)
        phi = (th0 + 0.01 * torch.randn(B, L, R.Nb, generator=gen)).to(dev).requires_grad_(True)
        opt = torch.optim.Adam([theta, phi], lr=a.lr, betas=(0.9, 0.8))
        mu = [a.mu0] * B
        ema = [0.0] * B
        is_layer = torch.tensor([arm == "tok+layer" for arm, _ in cfgs], device=dev)
        x = p["x"].expand(B, -1).contiguous()
        cols = p["pred"][None].expand(B, -1).contiguous()
        elig = R.elig.expand(B, -1).contiguous()
        log(f"[{a.task} row {row_id}] T {R.T}, routed {R.N} (body {R.Nb}), CE_full {ce_full:.4f}; bar dCE {ce_bar - ce_full:+.4f} FLOPs x{fl_bar:.3f}")
        t0 = time.time()
        hist = []
        for step in range(a.steps):
            prog = step / max(1, a.steps - 1)
            beta = (2 / 3) * (0.1 / (2 / 3)) ** min(1.0, prog / 0.7)
            st = prog >= 0.85
            zt = RL.hard_concrete(theta, beta, gen, straight_through=st, st_clamp=True)  # (B, N)
            zt = torch.where(R.forced[None], torch.ones_like(zt), zt)
            zl = RL.hard_concrete(phi, beta, gen, straight_through=st, st_clamp=True)  # (B, L, Nb)
            zl = torch.where(is_layer[:, None, None] & ~R.forced_body[None, None], zl, torch.ones_like(zl))
            tk = torch.ones(B, R.T, device=dev).scatter(1, R.idx[None].expand(B, -1), zt)
            vals = torch.ones(L, B, R.T, device=dev).scatter(2, R.body_pos[None, None].expand(L, B, -1), zl.permute(1, 0, 2).contiguous())
            gate = LS.Gater(elig, "fixed", values=vals)
            lg = LS.forward_layerskip(model, x, logits_to_keep=cols, tok_keep=tk, gate=gate).float()
            ce = F.cross_entropy(lg.flatten(0, 1), p["tgt"][None].expand(B, -1).flatten(), reduction="none").view(B, -1).mean(1)
            dce = ce - ce_full
            pt = torch.where(R.forced[None], torch.ones_like(theta), RL.hard_concrete_p_nonzero(theta, beta))
            pl = torch.where(is_layer[:, None, None] & ~R.forced_body[None, None], RL.hard_concrete_p_nonzero(phi, beta), torch.ones_like(phi))
            loss = 0.0
            fls = []
            for b, (arm, tau) in enumerate(cfgs):
                fl = expected_flops(R.n_fixed, pt[b][R.mk], pt[b][~R.mk], pl[b], lin, quad, full)
                fls.append(float(fl))
                loss = loss + fl + mu[b] * dce[b] / tau
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            for b, (arm, tau) in enumerate(cfgs):
                ema[b] = float(dce[b]) if step == 0 else 0.9 * ema[b] + 0.1 * float(dce[b])
                mu[b] = max(0.0, mu[b] + a.mu_lr * float(np.clip((ema[b] - tau) / tau, -1, 3)))
            if step in (0, 1, 4, 9) or (step + 1) % 50 == 0:
                el = time.time() - t0
                hist.append({"step": step + 1, "dce": [float(v) for v in dce], "eflops": fls, "mu": list(mu)})
                log(f"[{a.task} r{row_id}] step {step + 1}/{a.steps}: " + " | ".join(
                    f"{arm}@{tau}: dCE {float(dce[b]):+.3f} E[FL] {fls[b]:.3f} mu {mu[b]:.2f}" for b, (arm, tau) in enumerate(cfgs))
                    + f" | {el:.0f}s, ETA {el / (step + 1) * (a.steps - step - 1):.0f}s")
        t_soft = time.time() - t0
        out = {"row": row_id, "T": R.T, "n_routed": R.N, "n_body": R.Nb, "ce_full": ce_full,
               "bar": {"dce": ce_bar - ce_full, "flops": fl_bar}, "train_sec": t_soft, "hist": hist, "arms": {}}
        for b, (arm, tau) in enumerate(cfgs):
            tg = time.time()
            th, ph = theta[b].detach(), phi[b].detach()
            kt = (th > 0) | R.forced
            pr = ((ph > 0) | R.forced_body[None]) if arm == "tok+layer" else torch.ones_like(ph, dtype=torch.bool)
            kt, pr, d, fl, info = greedy(R, model, kt, pr, ce_full, tau, th, ph, arm == "tok+layer", a.max_evals)
            out["arms"][f"{arm}@{tau:g}"] = {"dce": d, "flops": fl, "meets": d <= tau, **info, "greedy_sec": time.time() - tg,
                                             "kept": dump_kept(R, kt, tok),
                                             "keeps": keep_stats(R, kt, pr, L)}
            o = out["arms"][f"{arm}@{tau:g}"]
            log(f"[{a.task} r{row_id}] {arm}@{tau:g}: soft-det dCE {info['soft_det_dce']:+.4f} FL x{info['soft_det_flops']:.3f} -> HARD "
                f"dCE {d:+.4f} FLOPs x{fl:.3f} ({info['evals']} evals); gold {o['keeps']['gold_body']:.2f} non-gold {o['keeps']['nongold_body']:.2f} "
                f"ids {o['keeps']['nongold_id_region']:.2f} markers {o['keeps']['markers']:.2f}; skip attn {o['keeps']['skip_attn']:.2f} gdn {o['keeps']['skip_gdn']:.2f}")
        out["row_sec"] = time.time() - t_row
        res["rows"].append(out)
        json.dump(res, open(out_path + ".part", "w"), indent=1)
        os.replace(out_path + ".part", out_path)
        log(f"[{a.task} r{row_id}] done in {out['row_sec']:.0f}s (soft {t_soft:.0f}s)")
        if wb is not None:
            wb.log({f"{k}/flops": v["flops"] for k, v in out["arms"].items()} | {"row": row_id})
    if wb is not None:
        wb.finish()
    log(f"all rows done -> {out_path}")


if __name__ == "__main__":
    main()
