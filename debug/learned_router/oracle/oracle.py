"""Stage 1 of the oracle-then-classifier pilot: per-row ORACLE keep masks.

For every training / val row, FREE per-token keep logits theta_t (one per routed token, markers
included; no features, no sharing across rows) are optimised through the exact relaxed deletion path
(``soft_keep``: attention log-p bias, GDN beta/g scaling, removal-aware conv) with hard-concrete gates
and PER-ROW budget calibration (an offset puts the row's expected keep at rho_t). Loss = answer CE +
KL(full || routed). rho is swept 0.6 -> 0.45 -> 0.3 -> 0.2 -> 0.1, each stage warm-started from the
previous one. Per seed, the oracle mask is the smallest rho whose HARD-deletion dCE <= tau_row =
max(0.1 CE_full, 0.02 nats); if no stage passes, the highest-theta dropped tokens are added back in
chunks until the hard check passes. Seeds run as one batched forward; the soft label of a token is the
fraction of seeds that kept it.

    TASK=nq SCRIPT=oracle.py ARGS="--split train" sbatch ... debug/learned_router/oracle/run_oracle.sbatch

Writes ``oracle/out/<task>/labels_<split><range>.pt`` (per row: routed positions, soft label, per-seed
choice) and ``summary_<split><range>.json`` (oracle vs gold_fl20p8_noslot on the same rows).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import router_lib as RL  # noqa: E402
import train_e2e_router as E  # noqa: E402

TR, G, log = E.TR, E.G, E.log


def row_key(p) -> str:
    return hashlib.sha1(p["x"][0].cpu().numpy().tobytes()).hexdigest()[:16]


def setup(a, splits):
    """Model + prepared rows (routed markers), exactly as train_e2e_router.py prepares them."""
    from transformers import AutoTokenizer

    from olmo_core.nn.attention.fla_autotune import freeze_fla_length_autotune

    ids = G.RESERVED_IDS[G.FAMILY]
    vocab = G.VOCAB_BY_FAMILY[G.FAMILY]
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    pieces = tok.convert_ids_to_tokens(list(range(vocab)))
    tr_rows = TR.load_split(a.train_root, a.task, ["2k"], tok, ids, a.n_train)
    va_rows = TR.load_split(a.val_root, a.task, ["2k"], tok, ids, a.n_val)
    extra = TR.load_split(a.train_root, a.task, ["2k"], tok, ids, a.n_train + a.val_extra_from_train)
    seen = {tuple(r["ids"]) for r in tr_rows}
    va_rows += [r for r in extra if tuple(r["ids"]) not in seen]
    all_ids = np.concatenate([np.asarray(r["ids"], dtype=np.int64) for r in tr_rows])
    stop_ids, _, tables = G.build_tables(all_ids, vocab, pieces, ids, lambda q: tok.decode([int(q)]))
    model = G.load_model(a.ckpt, a.ckpt_format, vocab, ids, 42, stop_ids, "flash_2")
    for p_ in model.parameters():
        p_.requires_grad_(False)
    freeze_fla_length_autotune()
    dev = next(model.parameters()).device
    TR.configure(model, stop_ids, tables.to(dev))
    model._pooled_soft_tokens["keep_token_mask_markers"] = "mask"  # routed markers: kept iff the mask keeps them
    rows = {"train": tr_rows, "val": va_rows}
    max_pos = max(len(r["ids"]) for s in splits for r in rows[s]) + 8
    for mod in model.modules():
        rope = getattr(mod, "rope", None)
        if rope is not None and hasattr(rope, "warmup_cache"):
            rope.warmup_cache(max_pos, dev)
    prepped = {s: [TR.prep_row(model, r, ids, dev, route_markers=True) for r in rows[s]] for s in splits}
    return model, tok, ids, vocab, prepped, dev


@torch.no_grad()
def sigmoid_offset(theta: torch.Tensor, target: float, beta: float) -> float:
    """c with mean(sigmoid((theta + c) / beta)) == target (bisection)."""
    lo, hi = -float(theta.max()) - 60.0 * beta - 1.0, -float(theta.min()) + 60.0 * beta + 1.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if float(torch.sigmoid((theta + mid) / beta).mean()) > target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def topk_mask(p, theta: torch.Tensor, k: int) -> torch.Tensor:
    keep = torch.zeros_like(theta, dtype=torch.bool)
    if k > 0:
        keep[torch.topk(theta, min(k, theta.numel())).indices] = True
    return keep


def oracle_row(model, p, a, pad_id: int, seed_base: int):
    N = int(p["feats"]["idx"].numel())
    S = a.seeds
    lpf = E.full_logprobs(model, p).to(p["x"].device).float()
    cef = float(F.nll_loss(lpf, p["tgt"]))
    tau = max(a.tau_rel * cef, a.tau_min)
    gens = [torch.Generator().manual_seed(seed_base * 1000 + s) for s in range(S)]
    g0 = torch.Generator().manual_seed(seed_base)
    theta = (math.log(a.init_p / (1 - a.init_p)) + 0.01 * torch.randn(S, N, generator=g0)).to(p["x"].device).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=a.lr, betas=(0.9, 0.8))
    stages, prev = [], 1.0
    for rho in a.rhos:
        for step in range(a.steps):
            fr = step / max(1, a.steps - 1)
            rho_t = prev + (rho - prev) * min(1.0, fr / a.warm_frac)
            beta = a.beta0 * (a.beta1 / a.beta0) ** fr
            st = step >= a.steps - a.st_steps
            zs = []
            for s in range(S):
                if a.gate == "sigmoid":
                    # deterministic relaxation: z = sigmoid((theta + c) / beta), c puts mean z at rho_t; no sampling noise
                    c = sigmoid_offset(theta[s].detach(), rho_t, beta)
                    z = torch.sigmoid((theta[s] + c) / beta)
                    if st:
                        z = (z > 0.5).to(z.dtype) + z - z.detach()
                    zs.append(z)
                else:
                    la = theta[s] + E.calibrate_offset(theta[s].detach(), rho_t, beta)
                    zs.append(RL.hard_concrete(la, beta, gens[s], straight_through=st, st_clamp=True))
            lgs = E.relaxed_logits_batch(model, [p] * S, zs, pad_id)
            loss = 0.0
            for lg in lgs:
                lp = F.log_softmax(lg, -1)
                kl = (lpf.exp() * (lpf - lp)).sum(-1).mean()
                loss = loss + ((kl if a.loss == "kl" else F.nll_loss(lp, p["tgt"]) + kl)) / S
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            k = int(math.ceil(rho * N))
            keeps = [topk_mask(p, theta[s].detach(), k) for s in range(S)]
            ce, comp = TR.ce_masked(model, p, torch.stack([RL.keep_mask_from(p["feats"], kp, p["T"]) for kp in keeps]))
        stages.append({"rho": rho, "k": k, "dce": [float(c) - cef for c in ce], "comp": [float(c) for c in comp],
                       "theta": theta.detach().cpu().clone(), "loss": float(loss)})
        if a.log_stages:
            log(f"    stage rho {rho}: hard dCE " + " ".join(f"{float(c) - cef:+.3f}" for c in ce) + f" (tau {tau:.3f}) T2/T "
                + " ".join(f"{float(c):.3f}" for c in comp) + f" | relaxed loss {float(loss):.4f}")
        prev = rho
    chosen, labels = [], []
    for s in range(S):
        ok = [st_ for st_ in stages if st_["dce"][s] <= tau]
        if ok:
            st_ = min(ok, key=lambda q: q["rho"])
            keep = topk_mask(p, st_["theta"][s].to(p["x"].device), st_["k"])
            chosen.append({"rho": st_["rho"], "k": st_["k"], "dce": st_["dce"][s], "comp": st_["comp"][s], "addback": 0})
        else:
            # add back the highest-theta dropped tokens of the rho = max stage, in chunks, until the hard check passes
            st0 = max(stages, key=lambda q: q["rho"])
            th = st0["theta"][s].to(p["x"].device)
            ks = sorted({min(N, st0["k"] + int(math.ceil(j * a.chunk * N))) for j in range(1, int(math.ceil(1.0 / a.chunk)) + 1)})
            keeps = [topk_mask(p, th, kk) for kk in ks]
            with torch.no_grad():
                ce, comp = TR.ce_masked(model, p, torch.stack([RL.keep_mask_from(p["feats"], kp, p["T"]) for kp in keeps]))
            dces = [float(c) - cef for c in ce]
            j = next((i for i, d in enumerate(dces) if d <= tau), len(ks) - 1)
            keep = keeps[j]
            chosen.append({"rho": None, "k": ks[j], "dce": dces[j], "comp": float(comp[j]), "addback": j + 1})
        labels.append(keep.float().cpu())
    return {"key": row_key(p), "N": N, "T": p["T"], "cef": cef, "tau": tau, "label": torch.stack(labels).mean(0),
            "chosen": chosen, "stages": [{k_: v for k_, v in st_.items() if k_ != "theta"} for st_ in stages],
            "theta_final": stages[-1]["theta"].half()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt-format", default="hf")
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--val-root", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--eval-data-root", default="")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--rows", default="", help="python slice a:b of the split's rows (default all)")
    ap.add_argument("--n-train", type=int, default=32)
    ap.add_argument("--n-val", type=int, default=32)
    ap.add_argument("--val-extra-from-train", type=int, default=32)
    ap.add_argument("--rhos", default="0.6,0.45,0.3,0.2,0.1")
    ap.add_argument("--steps", type=int, default=40, help="optimiser steps per rho stage")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--init-p", type=float, default=0.95)
    ap.add_argument("--beta0", type=float, default=0.5)
    ap.add_argument("--beta1", type=float, default=0.1)
    ap.add_argument("--warm-frac", type=float, default=0.5, help="fraction of a stage over which rho_t moves to the stage's rho")
    ap.add_argument("--st-steps", type=int, default=2, help="straight-through binary gates for the last steps of each stage")
    ap.add_argument("--tau-rel", type=float, default=0.1)
    ap.add_argument("--tau-min", type=float, default=0.02)
    ap.add_argument("--chunk", type=float, default=0.05, help="add-back chunk, fraction of routed tokens")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "out"))
    ap.add_argument("--log-stages", action="store_true", help="log the hard dCE of every seed at every rho stage")
    ap.add_argument("--loss", default="ce_kl", choices=["ce_kl", "kl"], help="kl: faithfulness to the full model only (no answer CE)")
    ap.add_argument("--gate", default="hc", choices=["hc", "sigmoid"], help="hc: hard-concrete samples; sigmoid: deterministic relaxation")
    a = ap.parse_args()
    a.rhos = [float(r) for r in a.rhos.split(",")]
    t0 = time.time()
    model, tok, ids, vocab, prepped, dev = setup(a, [a.split])
    rows = prepped[a.split]
    lo, hi = (int(x) if x else None for x in a.rows.split(":")) if a.rows else (None, None)
    idx = list(range(len(rows)))[lo:hi]
    log(f"oracle {a.task} {a.split} rows {idx[0]}..{idx[-1]} ({len(idx)}), rhos {a.rhos}, {a.steps} steps/stage, {a.seeds} seeds; "
        f"setup {time.time() - t0:.0f}s")
    out_dir = os.path.join(a.out_dir, a.task)
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{a.split}{'' if not a.rows else '_' + a.rows.replace(':', '-')}"
    recs, hrefs = [], []
    t1 = time.time()
    for n_done, i in enumerate(idx, 1):
        p = rows[i]
        tr0 = time.time()
        rec = oracle_row(model, p, a, ids.eos, seed_base=10007 * (1 if a.split == "train" else 2) + i)
        rec.update({"split": a.split, "row": i})
        h = E.heuristic_eval(model, [p], [rec["cef"]], ids)
        rec["heuristic"] = {"dce": h["dce"], "comp": h["comp"], "keep": h["keep"]}
        recs.append(rec)
        ch = rec["chosen"]
        el = time.time() - t1
        log(f"[{a.split} {i}] T {p['T']} N {rec['N']} CE_full {rec['cef']:.4f} tau {rec['tau']:.3f} | oracle T2/T "
            + " ".join(f"{c['comp']:.3f}" for c in ch) + " dCE " + " ".join(f"{c['dce']:+.3f}" for c in ch)
            + " rho " + " ".join(str(c['rho']) for c in ch)
            + f" | label keep {float(rec['label'].mean()):.3f} (agree {float(((rec['label'] == 0) | (rec['label'] == 1)).float().mean()):.2f})"
            + f" | heuristic T2/T {h['comp']:.3f} dCE {h['dce']:+.3f} | {time.time() - tr0:.0f}s"
            + (f" | {n_done}/{len(idx)} ETA {el / n_done * (len(idx) - n_done):.0f}s" if n_done in (1, 2, 5, 10) or n_done % 10 == 0 else ""))
        if n_done % 8 == 0 or n_done == len(idx):
            torch.save(recs, os.path.join(out_dir, f"labels_{tag}.pt.part"))
            os.replace(os.path.join(out_dir, f"labels_{tag}.pt.part"), os.path.join(out_dir, f"labels_{tag}.pt"))
    comp_o = [np.mean([c["comp"] for c in r["chosen"]]) for r in recs]
    dce_o = [np.mean([c["dce"] for c in r["chosen"]]) for r in recs]
    summ = {"task": a.task, "split": a.split, "rows": a.rows, "eval_size": len(recs), "args": {k: v for k, v in vars(a).items()},
            "oracle_comp": float(np.mean(comp_o)), "oracle_dce": float(np.mean(dce_o)),
            "oracle_pass_rate": float(np.mean([c["dce"] <= r["tau"] for r in recs for c in r["chosen"]])),
            "heuristic_comp": float(np.mean([r["heuristic"]["comp"] for r in recs])),
            "heuristic_dce": float(np.mean([r["heuristic"]["dce"] for r in recs])),
            "label_keep": float(np.mean([float(r["label"].mean()) for r in recs])),
            "sec": time.time() - t0, "git_commit": G.git_commit()}
    json.dump(summ, open(os.path.join(out_dir, f"summary_{tag}.json"), "w"), indent=1)
    log(f"SUMMARY {a.task} {tag}: oracle T2/T {summ['oracle_comp']:.3f} dCE {summ['oracle_dce']:+.4f} (pass {summ['oracle_pass_rate']:.2f}) | "
        f"heuristic T2/T {summ['heuristic_comp']:.3f} dCE {summ['heuristic_dce']:+.4f} | {summ['sec']:.0f}s")


if __name__ == "__main__":
    main()
