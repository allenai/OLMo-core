"""Train the learned linear token router (router_lib.LinearRouter) for ONE task on a FROZEN dense
checkpoint with REINFORCE + RLOO, then save per-config weights + a results JSON with full curves.

    python debug/learned_router/train_router.py --task nq --ckpt /data/.../ctc-4b-nq-full \\
        --train-root /data/prasann/devloss_grid/data/router_train \\
        --val-root /data/prasann/devloss_grid/data/router_val \\
        --configs l0.05,l0.2,l0.5,nogold_l0.2,noemb_l0.2 --out-dir debug/learned_router

Objective per training row (CE = mean answer-token CE in nats, teacher forced, exactly the grid
driver's construction: ``Transformer._compact_pooled_soft_tokens`` with ``keep_token_rule="custom"``,
doc-level keep = none, ``drop_slots=True``, markers kept):

    R(m) = -(CE(m) - CE_full) - lambda * keep_frac(m),  keep_frac = kept routed / routed tokens.

Gradient: the CE term by REINFORCE with a leave-one-out baseline over K Bernoulli masks per row
(``A_k = R_k - mean_{j != k} R_j``, ``grad = -mean_k A_k grad log pi(m_k)``); the keep_frac term
analytically (``E[keep_frac] = mean_i p_i``) -- the same objective's gradient with less variance. The
masks are re-drawn every time a row is visited (every epoch). Adam; bias = 0 (p = 0.5), every weight
0 at init. Each config early-stops on the held-out VAL reward (sampled policy), and the best epoch's
weights are saved. Rows never come from the grid's test rows (fetch_train_rows.py).

Config names: ``[nogold_|noemb_]l<lambda>[_s<seed>]`` (variant, lambda, seed).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "debug", "devloss_grid"))
sys.path.insert(0, HERE)

import ctc_devloss_grid as G  # noqa: E402
from router_lib import (  # noqa: E402
    POS_NAMES,
    LinearRouter,
    keep_mask_from,
    rms_embed,
    routed_features,
)

IGN = -100


def log(m: str) -> None:
    print(f"[router] {m}", flush=True)


def parse_config(name: str):
    m = re.fullmatch(r"(?:(nogold|noemb)_)?l([0-9.]+)(?:_s(\d+))?", name)
    if not m:
        raise SystemExit(f"bad config name {name!r}")
    return (m.group(1) or "full"), float(m.group(2)), int(m.group(3) or 0)


# ------------------------------------------------------------------------------------------------
def load_split(root: str, task_key: str, rungs, tok, ids, n_per_rung: int):
    row = G.ROSTER[f"ctc_{task_key}"]
    out = []
    for rung in rungs:
        exs = G.load_examples(root, row, rung, n_per_rung)
        for ex in exs:
            r_ids, r_mask, n_spans = G.render_ctc_row(tok, ex, row["seg_task"], ids)
            if G.SEG_CFG[row["seg_task"]]["chunk_by"] == "document" and n_spans != len(ex.get("documents") or []):
                log(f"WARNING {rung}: span/doc mismatch, row dropped")
                continue
            out.append({"ids": r_ids, "mask": r_mask, "gold": G.gold_docs(row["spec"], ex), "rung": rung,
                        "src": ex.get("_router_src")})
    return out


def prep_row(model, r, ids, dev, route_markers: bool = False):
    from olmo_core.nn.attention.chunked_mask import build_chunk_ids_from_tokens

    x = torch.tensor(np.asarray(r["ids"])[None], device=dev)
    ans = torch.tensor(np.nonzero(np.asarray(r["mask"]))[0], device=dev)
    cid0 = build_chunk_ids_from_tokens(x.cpu(), doc_start_id=ids.doc_start, doc_end_id=ids.doc_end,
                                       eos_id=ids.eos, mode="chunked")
    n_docs = int(cid0.max()) + 1
    feats = routed_features(x[0], cid0[0], ids.doc_start, ids.doc_end, n_docs, sorted(r["gold"]) if r["gold"] else None,
                            route_markers=route_markers)
    e = rms_embed(model.embeddings.weight, feats["tok"]).to(torch.bfloat16)
    return {"x": x, "pred": ans - 1, "tgt": x[0, ans], "n_docs": n_docs, "feats": feats, "e": e,
            "T": int(x.shape[1]), "rung": r["rung"], "gold": r["gold"]}


@torch.no_grad()
def ce_full(model, p) -> float:
    model.eval()
    model._pooled_keep_holder = None
    lg = model(p["x"], logits_to_keep=p["pred"][None])[0].float()
    return float(F.cross_entropy(lg, p["tgt"]))


@torch.no_grad()
def ce_masked(model, p, masks: torch.Tensor, max_tokens: int = 65536):
    """Answer CE for each of the K keep masks (K, S) of one row, batched through the compaction path.
    Returns (ce (K,), t2_over_t (K,))."""
    from olmo_core.nn.attention.pooled_doc_kv import PooledDocKeepHolder

    pst = model._pooled_soft_tokens
    K, S = masks.shape
    # rough per-sample compacted length, to split K into token-budgeted sub-batches
    est = (masks.sum(1) + (S - int(p["feats"]["idx"].numel()))).tolist()
    ces, comps = [], []
    k0 = 0
    while k0 < K:
        k1 = k0 + 1
        while k1 < K and sum(est[k0 : k1 + 1]) <= max_tokens:
            k1 += 1
        B = k1 - k0
        x = p["x"].expand(B, -1).contiguous()
        pst["keep_token_mask"] = masks[k0:k1]
        model.train()
        model._pooled_keep_holder = PooledDocKeepHolder(keep_docs=torch.zeros(B, p["n_docs"], dtype=torch.bool))
        cb = model._compact_pooled_soft_tokens(x, None, IGN)[0]
        pos = cb.position_ids  # (B, T2), ascending real positions, pads (= T-1) at the end
        cols = torch.searchsorted(pos.contiguous(), p["pred"][None].expand(B, -1).contiguous())
        got = pos.gather(1, cols)
        assert bool((got == p["pred"][None]).all()), "answer prediction position was compacted away"
        lens = (pos < p["T"] - 1).sum(1) + 1  # real columns (the EOS at T-1 is real and kept)
        lg = model(x, logits_to_keep=cols)  # (B, n_ans, V)
        model.eval()
        model._pooled_keep_holder = None
        tgt = p["tgt"][None].expand(B, -1)
        ce = F.cross_entropy(lg.float().flatten(0, 1), tgt.flatten(), reduction="none").view(B, -1).mean(1)
        ces.append(ce)
        comps.append(lens.float() / p["T"])
        k0 = k1
    return torch.cat(ces), torch.cat(comps)


def configure(model, stop_ids, tables):
    pst = model._pooled_soft_tokens
    G.configure_scheme(model, pst, dict(keep="none", slot="cent_cmean", rule="custom", k=0.0, noslot=True,
                                        keep_markers=True), tables, stop_ids)


@torch.no_grad()
def evaluate(model, router, rows, cef, lam, n_samp: int, seed: int):
    """Deterministic (p > 0.5) and sampled policy on ``rows``: mean dCE, keep_frac, T2/T, reward."""
    out = {"det": {"dce": [], "keep": [], "comp": []}, "samp": {"dce": [], "keep": [], "comp": []}}
    gen = torch.Generator().manual_seed(seed)
    for i, p in enumerate(rows):
        e = p["e"].float() if router.use_emb else None
        pr = torch.sigmoid(router.logits(p["feats"], e))
        dec = [pr > 0.5] + [torch.rand(pr.shape, generator=gen).to(pr.device) < pr for _ in range(n_samp)]
        masks = torch.stack([keep_mask_from(p["feats"], d, p["T"]) for d in dec])
        ce, comp = ce_masked(model, p, masks)
        for j, d in enumerate(dec):
            key = "det" if j == 0 else "samp"
            out[key]["dce"].append(float(ce[j]) - cef[i])
            out[key]["keep"].append(float(d.float().mean()) if d.numel() else 1.0)
            out[key]["comp"].append(float(comp[j]))
    res = {}
    for key, v in out.items():
        dce, keep, comp = (np.mean(v[k]) for k in ("dce", "keep", "comp"))
        res[key] = {"dce": float(dce), "dce_median": float(np.median(v["dce"])), "keep": float(keep), "comp": float(comp),
                    "reward": float(-dce - lam * keep)}
    return res


def vocab_scores(model, router, rows, tok, top: int = 40):
    """w.e_rms/sqrt(d) for every vocabulary id that occurs in a routed position of the training rows."""
    if not router.use_emb:
        return None
    from collections import Counter

    cnt = Counter()
    for p in rows:
        cnt.update(p["feats"]["tok"].tolist())
    ids_ = torch.tensor(sorted(cnt), device=router.w_emb.device)
    e = rms_embed(model.embeddings.weight, ids_)
    s = ((e @ router.w_emb.float()) / math.sqrt(router.d_emb)).tolist()
    recs = sorted(zip(s, ids_.tolist()), key=lambda t: t[0])
    freq = [(sv, i) for sv, i in recs if cnt[i] >= 3]

    def fmt(lst):
        return [{"s": round(sv, 4), "id": i, "piece": tok.decode([i]), "count": cnt[i]} for sv, i in lst]

    digit = [sv for sv, i in recs if any(ch.isdigit() for ch in tok.decode([i]))]
    other = [sv for sv, i in recs if not any(ch.isdigit() for ch in tok.decode([i]))]
    return {
        "n_types": len(recs), "n_types_count_ge3": len(freq),
        "s_std": float(np.std([sv for sv, _ in recs])), "s_std_count_ge3": float(np.std([sv for sv, _ in freq])) if freq else None,
        "top_count_ge3": fmt(freq[::-1][:top]), "bottom_count_ge3": fmt(freq[:top]),
        "digit_mean": float(np.mean(digit)) if digit else None, "nondigit_mean": float(np.mean(other)) if other else None,
    }


def train_one(model, name, tr, va, cef_tr, cef_va, a, tok, stop_ids, tables):
    variant, lam, seed = parse_config(name)
    torch.manual_seed(1000 + seed)
    gen = torch.Generator().manual_seed(2000 + seed)
    dev = next(model.parameters()).device
    router = LinearRouter(int(model.embeddings.weight.shape[1]), variant).to(dev)
    opt = torch.optim.Adam([
        {"params": [router.b, router.w_pos, router.w_gold], "lr": a.lr},
        {"params": [router.w_emb], "lr": a.emb_lr},
    ])
    steps, epochs = [], []
    best = {"reward": -math.inf, "epoch": -1, "state": router.state()}
    t0 = time.time()
    n_steps_per_epoch = math.ceil(len(tr) / a.rows_per_step)
    total_steps = n_steps_per_epoch * a.epochs
    step = 0
    ev0 = evaluate(model, router, va, cef_va, lam, a.val_samples, seed=7)
    epochs.append({"epoch": 0, "val": ev0, "train": None, "w_gold": 0.0, "b": 0.0})
    log(f"[{name}] epoch 0 (init p=0.5) val samp dCE={ev0['samp']['dce']:+.3f} keep={ev0['samp']['keep']:.2f} "
        f"T2/T={ev0['samp']['comp']:.2f} R={ev0['samp']['reward']:+.3f} | det keep={ev0['det']['keep']:.2f}")
    bad = 0
    for ep in range(1, a.epochs + 1):
        order = torch.randperm(len(tr), generator=gen).tolist()
        ep_acc = {"R": [], "dce": [], "keep": [], "comp": []}
        for s0 in range(0, len(order), a.rows_per_step):
            opt.zero_grad(set_to_none=True)
            st = {"R": [], "dce": [], "keep": [], "comp": []}
            batch = order[s0 : s0 + a.rows_per_step]
            for i in batch:
                p = tr[i]
                e = p["e"].float() if router.use_emb else None
                z = router.logits(p["feats"], e)
                pr = torch.sigmoid(z)
                with torch.no_grad():
                    m = torch.rand((a.K,) + tuple(pr.shape), generator=gen).to(dev) < pr.detach()[None]
                    masks = torch.stack([keep_mask_from(p["feats"], m[k], p["T"]) for k in range(a.K)])
                    ce, comp = ce_masked(model, p, masks)
                    dce = ce - cef_tr[i]
                    keep = m.float().mean(1) if m.shape[1] else torch.ones(a.K, device=dev)
                    Rce = -dce
                    adv = Rce - (Rce.sum() - Rce) / (a.K - 1)  # leave-one-out baseline
                logp = -F.binary_cross_entropy_with_logits(z[None].expand(a.K, -1), m.float(), reduction="none").sum(1)
                loss = -(adv.detach() * logp).mean() + lam * pr.mean()
                (loss / len(batch)).backward()
                R = (-dce - lam * keep)
                for k_, v_ in (("R", R), ("dce", dce), ("keep", keep), ("comp", comp)):
                    st[k_].append(float(v_.mean()))
            gn = float(torch.sqrt(sum((q.grad.float() ** 2).sum() for q in router.parameters() if q.grad is not None)))
            opt.step()
            step += 1
            rec = {"step": step, "epoch": ep, **{k: float(np.mean(v)) for k, v in st.items()},
                   "w_gold": float(router.w_gold), "b": float(router.b), "grad_norm": gn,
                   "w_emb_norm": float(router.w_emb.norm()), "sec": time.time() - t0}
            steps.append(rec)
            for k_ in ep_acc:
                ep_acc[k_] += st[k_]
            if step in (1, 2, 5, 10) or step % 25 == 0:
                el = time.time() - t0
                log(f"[{name}] step {step}/{total_steps} ep {ep}: R={rec['R']:+.3f} dCE={rec['dce']:+.3f} keep={rec['keep']:.2f} "
                    f"T2/T={rec['comp']:.2f} w_gold={rec['w_gold']:+.2f} b={rec['b']:+.2f} |w_emb|={rec['w_emb_norm']:.2f} "
                    f"| {el:.0f}s, ETA<= {el / step * (total_steps - step):.0f}s")
        ev = evaluate(model, router, va, cef_va, lam, a.val_samples, seed=7)
        trm = {k: float(np.mean(v)) for k, v in ep_acc.items()}
        epochs.append({"epoch": ep, "train": trm, "val": ev, "w_gold": float(router.w_gold), "b": float(router.b)})
        improved = ev["samp"]["reward"] > best["reward"] + 1e-4
        if improved:
            best = {"reward": ev["samp"]["reward"], "epoch": ep, "state": router.state()}
            bad = 0
        else:
            bad += 1
        log(f"[{name}] epoch {ep}: train R={trm['R']:+.3f} dCE={trm['dce']:+.3f} keep={trm['keep']:.2f} | val samp "
            f"R={ev['samp']['reward']:+.3f} dCE={ev['samp']['dce']:+.3f} keep={ev['samp']['keep']:.2f} T2/T={ev['samp']['comp']:.2f}"
            f" | val det dCE={ev['det']['dce']:+.3f} keep={ev['det']['keep']:.2f}{'  *best' if improved else ''}")
        if ep >= a.min_epochs and bad >= a.patience:
            log(f"[{name}] early stop at epoch {ep} (best {best['epoch']})")
            break
    final_state = router.state()
    router_best = LinearRouter.from_state(best["state"]).to(dev)
    return router_best, {
        "config": name, "variant": variant, "lambda": lam, "seed": seed,
        "best_epoch": best["epoch"], "best_val_reward": best["reward"],
        "epochs_run": epochs[-1]["epoch"], "steps": steps, "epochs": epochs,
        "train_sec": time.time() - t0,
    }, best["state"], final_state


def weights_summary(st: dict) -> dict:
    return {"b": float(st["b"]), "w_gold": float(st["w_gold"]),
            "w_pos": {n: round(float(v), 4) for n, v in zip(POS_NAMES, st["w_pos"].tolist())},
            "w_emb_norm": float(st["w_emb"].norm())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="manifest key (nq, outlier, ...)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt-format", default="hf", choices=["hf", "distcp"])
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--val-root", required=True)
    ap.add_argument("--rungs", default="2k,8k")
    ap.add_argument("--n-train", type=int, default=16, help="per rung")
    ap.add_argument("--n-val", type=int, default=8, help="per rung")
    ap.add_argument("--configs", default="l0.05,l0.2,l0.5,nogold_l0.2,noemb_l0.2")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--rows-per-step", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--min-epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--emb-lr", type=float, default=0.01)
    ap.add_argument("--val-samples", type=int, default=2)
    ap.add_argument("--tokenizer", default=os.environ.get("DEVLOSS_TOKENIZER", G.TOKENIZER_BY_FAMILY[G.FAMILY]))
    ap.add_argument("--attn-backend", default="flash_2", choices=["torch", "flash_2"])
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--check-batch", action="store_true", help="assert batched == one-at-a-time CE on 2 rows, then continue")
    a = ap.parse_args()
    t_start = time.time()
    log(f"start task={a.task} configs={a.configs} K={a.K} rows/step={a.rows_per_step} epochs<={a.epochs}")

    from transformers import AutoTokenizer

    ids = G.RESERVED_IDS[G.FAMILY]
    vocab = G.VOCAB_BY_FAMILY[G.FAMILY]
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rungs = a.rungs.split(",")
    tr_rows = load_split(a.train_root, a.task, rungs, tok, ids, a.n_train)
    va_rows = load_split(a.val_root, a.task, rungs, tok, ids, a.n_val)
    log(f"rendered train {len(tr_rows)} / val {len(va_rows)} rows ({time.time() - t_start:.0f}s); "
        f"lengths train {[len(r['ids']) for r in tr_rows][:4]}...")
    all_ids = np.concatenate([np.asarray(r["ids"], dtype=np.int64) for r in tr_rows])
    pieces = tok.convert_ids_to_tokens(list(range(vocab)))
    stop_ids, _, tables = G.build_tables(all_ids, vocab, pieces, ids, lambda t: tok.decode([int(t)]))
    model = G.load_model(a.ckpt, a.ckpt_format, vocab, ids, 42, stop_ids, a.attn_backend)
    for p_ in model.parameters():
        p_.requires_grad_(False)
    # every sampled mask gives a new compacted length; FLA would re-autotune per length (~25 s each,
    # memory fla-autotune-per-length-eval-slowdown). Speed only -- results are bit-identical.
    from olmo_core.nn.attention.fla_autotune import freeze_fla_length_autotune

    log(f"froze length-derived autotune keys on {freeze_fla_length_autotune()} FLA kernel(s)")
    dev = next(model.parameters()).device
    configure(model, stop_ids, tables.to(dev))
    max_pos = max(len(r["ids"]) for r in tr_rows + va_rows) + 8
    for mod in model.modules():
        rope = getattr(mod, "rope", None)
        if rope is not None and hasattr(rope, "warmup_cache"):
            rope.warmup_cache(max_pos, dev)

    t0 = time.time()
    tr = [prep_row(model, r, ids, dev) for r in tr_rows]
    va = [prep_row(model, r, ids, dev) for r in va_rows]
    cef_tr, cef_va = [], []
    for j, p in enumerate(tr + va):
        (cef_tr if j < len(tr) else cef_va).append(ce_full(model, p))
        if j + 1 in (1, 2, 5, 10) or (j + 1) % 16 == 0:
            el = time.time() - t0
            log(f"CE_full {j + 1}/{len(tr) + len(va)} ({el:.0f}s, ETA {el / (j + 1) * (len(tr) + len(va) - j - 1):.0f}s)")
    log(f"CE_full train mean {np.mean(cef_tr):.3f} val mean {np.mean(cef_va):.3f}; routed tokens/row median "
        f"{int(np.median([int(p['feats']['idx'].numel()) for p in tr]))}, gold rows {sum(p['gold'] is not None for p in tr)}/{len(tr)}")

    if a.check_batch:
        for p, c in list(zip(tr, cef_tr))[:2]:
            g_ = torch.Generator().manual_seed(0)
            m = torch.rand((4,) + tuple(p["feats"]["idx"].shape), generator=g_).to(dev) < 0.5
            m[0] = True  # all kept == full row
            masks = torch.stack([keep_mask_from(p["feats"], m[k], p["T"]) for k in range(4)])
            ce_b, comp_b = ce_masked(model, p, masks)
            ce_s = torch.cat([ce_masked(model, p, masks[k : k + 1])[0] for k in range(4)])
            log(f"check-batch T={p['T']}: batched {ce_b.tolist()} single {ce_s.tolist()} full {c:.4f} comp {comp_b.tolist()}")
            assert torch.allclose(ce_b, ce_s, atol=2e-2), "batched != single"
            assert abs(float(ce_b[0]) - c) < 2e-2, "all-kept mask != full"

    os.makedirs(os.path.join(a.out_dir, "weights", a.task), exist_ok=True)
    os.makedirs(os.path.join(a.out_dir, "runs", a.task), exist_ok=True)
    for name in a.configs.split(","):
        router, res, best_state, final_state = train_one(model, name, tr, va, cef_tr, cef_va, a, tok, stop_ids, tables)
        wpath = os.path.join(a.out_dir, "weights", a.task, f"{name}.pt")
        torch.save(best_state, wpath + ".part")
        os.replace(wpath + ".part", wpath)
        res.update({
            "task": a.task, "ckpt": a.ckpt, "argv": sys.argv, "git_commit": G.git_commit(),
            "train_size": len(tr), "val_size": len(va), "train_src": [r["src"] for r in tr_rows], "val_src": [r["src"] for r in va_rows],
            "ce_full_train_mean": float(np.mean(cef_tr)), "ce_full_val_mean": float(np.mean(cef_va)),
            "weights_best": weights_summary(best_state), "weights_final": weights_summary(final_state),
            "vocab_scores_best": vocab_scores(model, router, tr, tok), "weights_path": os.path.relpath(wpath, REPO),
        })
        rpath = os.path.join(a.out_dir, "runs", a.task, f"{name}.json")
        json.dump(res, open(rpath + ".part", "w"), indent=1)
        os.replace(rpath + ".part", rpath)
        wb = res["weights_best"]
        log(f"[{name}] DONE best epoch {res['best_epoch']} val R={res['best_val_reward']:+.3f}  w_gold={wb['w_gold']:+.2f} b={wb['b']:+.2f} "
            f"|w_emb|={wb['w_emb_norm']:.2f}  -> {rpath}")
    log(f"all configs done in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
