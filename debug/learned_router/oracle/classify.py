"""Stage 2 of the oracle-then-classifier pilot: fit the linear router to the ORACLE soft labels.

The router (``relpos`` variant, routed markers): bias + 56 offset one-hots + generic start/end deciles +
doc-length buckets + gold flag + is_marker + RMS embedding. It is fit by WEIGHTED logistic regression on
the per-token soft labels from ``oracle.py`` (train rows), weight 1 + (w_keep - 1) * label so false drops
cost more, plus an L2 on w_emb (and a tiny one elsewhere). Convex; a few seconds on the GPU.

Selection (val rows, never test): for each (w_keep, lambda_emb) the label-free global threshold is set on
the val rows so mean T2/T equals gold_fl20p8_noslot's T2/T on the same rows; the config with the lowest
HARD-deletion val dCE wins. Test: the grid's 2k test rows through the grid driver's router path
(in-process ``G.run_cell``), at the val-calibrated threshold for the bar's T2/T and for 0.7x it, paired
against gold_fl20p8_noslot -> ``debug/devloss_grid/results_router_oracle/<task>_2k.json``.

    TASK=nq SCRIPT=classify.py sbatch ... debug/learned_router/oracle/run_oracle.sbatch
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import oracle as O  # noqa: E402
import router_lib as RL  # noqa: E402
import train_e2e_router as E  # noqa: E402

TR, G, log = E.TR, E.G, E.log


def auc(score: torch.Tensor, y: torch.Tensor) -> float:
    """ROC AUC of ``score`` against binary ``y`` (ties averaged)."""
    s, yb = score.double().cpu().numpy(), y.bool().cpu().numpy()
    npos, nneg = int(yb.sum()), int((~yb).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ss = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return float((ranks[yb].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def fit(rows, labels, d_emb, w_keep, lam_emb, lam=1e-4, iters=300, dev="cuda"):
    r = RL.LinearRouter(d_emb, "relpos").to(dev)
    r.route_markers = True
    feats = [p["feats"] for p in rows]
    es = [p["e"].float() for p in rows]
    ys = [labels[i].to(dev) for i in range(len(rows))]
    W = [1.0 + (w_keep - 1.0) * y for y in ys]
    wsum = float(sum(w.sum() for w in W))
    params = [r.b, r.w_pos, r.w_rel, r.w_gold, r.w_marker, r.w_emb]
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = 0.0
        for f, e, y, w in zip(feats, es, ys, W):
            z = r.logits(f, e)
            loss = loss + (w * F.binary_cross_entropy_with_logits(z, y, reduction="none")).sum()
        loss = loss / wsum + lam_emb * r.w_emb.pow(2).sum() + lam * (r.w_pos.pow(2).sum() + r.w_rel.pow(2).sum()
                                                                      + r.w_gold.pow(2).sum() + r.w_marker.pow(2).sum())
        loss.backward()
        return loss

    opt.step(closure)
    return r


def load_labels(task, rows_by_split, out_dir, model=None, ids=None, need_labels=("train",)):
    recs = []
    for f in sorted(glob.glob(os.path.join(out_dir, task, "labels_*.pt"))):
        recs += torch.load(f, map_location="cpu")
    by_key = {(r["split"], r["key"]): r for r in recs}
    out = {}
    for split, rows in rows_by_split.items():
        lab, cef, heur, keep_rows = [], [], [], []
        for p in rows:
            r = by_key.get((split, O.row_key(p)))
            if r is None:
                if split in need_labels:
                    continue
                # val rows need no oracle label for selection: CE_full + the heuristic reference only
                c = TR.ce_full(model, p)
                h = E.heuristic_eval(model, [p], [c], ids)
                r = {"label": None, "cef": c, "heuristic": {"dce": h["dce"], "comp": h["comp"], "keep": h["keep"]}}
            assert r["label"] is None or r["N"] == int(p["feats"]["idx"].numel()), "routed-token count mismatch between oracle and classifier rows"
            keep_rows.append(p)
            lab.append(None if r["label"] is None else r["label"].float())
            cef.append(r["cef"])
            heur.append(r["heuristic"])
        out[split] = (keep_rows, lab, cef, heur)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt-format", default="hf")
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--val-root", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--eval-data-root", required=True)
    ap.add_argument("--n-train", type=int, default=32)
    ap.add_argument("--n-val", type=int, default=32)
    ap.add_argument("--val-extra-from-train", type=int, default=32)
    ap.add_argument("--w-keep", default="2,4")
    ap.add_argument("--lam-emb", default="1e-3,1e-2,1e-1")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "out"))
    ap.add_argument("--name", default="oracle_cls")
    ap.add_argument("--test-rows", type=int, default=16)
    a = ap.parse_args()
    t0 = time.time()
    model, tok, ids, vocab, prepped, dev = O.setup(a, ["train", "val"])
    L = load_labels(a.task, prepped, a.out_dir, model, ids)
    tr, ytr, cef_tr, h_tr = L["train"]
    va, yva, cef_va, h_va = L["val"]
    log(f"classifier {a.task}: {len(tr)} train / {len(va)} val rows with oracle labels; setup {time.time() - t0:.0f}s")
    c_bar = float(np.mean([h["comp"] for h in h_va]))
    d_bar = float(np.mean([h["dce"] for h in h_va]))
    d_emb = int(model.embeddings.weight.shape[1])
    ytr_all = torch.cat(ytr)
    va_lab = [i for i, y in enumerate(yva) if y is not None]
    yva_all = torch.cat([yva[i] for i in va_lab]) if va_lab else None
    res = {"task": a.task, "train_size": len(tr), "val_size": len(va), "heuristic_val": {"comp": c_bar, "dce": d_bar},
           "oracle_label_keep_train": float(ytr_all.mean()), "configs": []}
    best = None
    for wk in [float(x) for x in a.w_keep.split(",")]:
        for le in [float(x) for x in a.lam_emb.split(",")]:
            tf = time.time()
            r = fit(tr, ytr, d_emb, wk, le, dev=dev)
            with torch.no_grad():
                ztr = torch.cat([r.logits(p["feats"], p["e"].float()) for p in tr])
                zva = torch.cat([r.logits(va[i]["feats"], va[i]["e"].float()) for i in va_lab]) if va_lab else None
            c = E.match_comp_offset(model, r, va, c_bar)
            v = E.val_threshold(model, r, va, cef_va, c)
            pr = [x - h["dce"] for x, h in zip(v["per_row_dce"], h_va)]
            cfg = {"w_keep": wk, "lam_emb": le, "auc_train": auc(ztr, ytr_all > 0.5), "auc_val": auc(zva, yva_all > 0.5) if va_lab else float("nan"), "val_labelled_rows": len(va_lab),
                   "val_dce": v["dce"], "val_comp": v["comp"], "val_paired_vs_heur": float(np.mean(pr)),
                   "val_paired_se": float(np.std(pr, ddof=1) / math.sqrt(len(pr))), "offset": c, "fit_sec": time.time() - tf,
                   "w_gold": float(r.w_gold), "w_marker": float(r.w_marker), "b": float(r.b), "w_emb_norm": float(r.w_emb.norm())}
            res["configs"].append(cfg)
            log(f"[cfg w_keep {wk:g} lam_emb {le:g}] AUC train {cfg['auc_train']:.3f} val {cfg['auc_val']:.3f} | val @ heuristic T2/T "
                f"{c_bar:.3f}: dCE {v['dce']:+.4f} (T2/T {v['comp']:.3f}) vs heuristic {d_bar:+.4f}, paired {cfg['val_paired_vs_heur']:+.4f} "
                f"+- {cfg['val_paired_se']:.4f} | w_gold {cfg['w_gold']:+.2f} w_marker {cfg['w_marker']:+.2f} b {cfg['b']:+.2f} ({cfg['fit_sec']:.0f}s)")
            if best is None or v["dce"] < best[0]["val_dce"]:
                best = (cfg, r)
    cfg, r = best
    res["selected"] = cfg
    log(f"selected w_keep {cfg['w_keep']:g} lam_emb {cfg['lam_emb']:g} (val dCE {cfg['val_dce']:+.4f})")
    wdir = os.path.join(os.path.dirname(HERE), "weights", a.task)
    os.makedirs(wdir, exist_ok=True)
    st = r.state()
    st.update({"keep_rule": "p>0.5", "oracle_cls": {k: cfg[k] for k in ("w_keep", "lam_emb")}})
    torch.save(st, os.path.join(wdir, f"{a.name}.pt"))
    schemes_c = []
    for tc in (c_bar, 0.7 * c_bar):
        c = E.match_comp_offset(model, r, va, tc)
        st2 = dict(st)
        st2["b"] = st["b"] + c
        st2["match_comp"] = tc
        nm = f"{a.name}_c{tc:.3f}"
        torch.save(st2, os.path.join(wdir, f"{nm}.pt"))
        vm = E.val_threshold(model, r, va, cef_va, c)
        res[f"val_at_{tc:.3f}"] = {k: v for k, v in vm.items() if k != "per_row_dce"}
        log(f"[val] T2/T target {tc:.3f}: dCE {vm['dce']:+.4f} realised {vm['comp']:.3f} -> {nm}.pt")
        schemes_c.append(nm)
    # test rows: the grid driver's cell logic in-process (same compaction path as every grid scheme)
    schemes = {"full": None, "gold_fl20p8_noslot": G.SCHEMES["gold_fl20p8_noslot"]}
    for nm in schemes_c:
        schemes[f"router_{nm}"] = dict(G.SCHEMES["router_l0.2"], router=nm)
    pieces = tok.convert_ids_to_tokens(list(range(vocab)))
    res_dir = os.path.join(os.path.dirname(os.path.dirname(HERE)), "devloss_grid", "results_router_oracle")
    os.makedirs(res_dir, exist_ok=True)
    ea = types.SimpleNamespace(task=f"ctc_{a.task}", rung="2k", rows=a.test_rows, ckpt=a.ckpt, ckpt_format=a.ckpt_format,
                               data_root=a.eval_data_root, cpt_source=None, seed=42, cpt_block=512, attn_backend="flash_2",
                               tokenizer=a.tokenizer)
    out_json = os.path.join(res_dir, f"{a.task}_2k.json")
    G.run_cell(ea, model, tok, pieces, lambda t: tok.decode([int(t)]), ids, vocab, schemes, time.time(), out_json)
    d = json.load(open(out_json))
    pr = d["per_row"]
    full, b = np.asarray(pr["full"]["ce"]), np.asarray(pr["gold_fl20p8_noslot"]["ce"])
    res["test"] = {"eval_size": len(full), "bar": {"comp": float(np.mean(pr["gold_fl20p8_noslot"]["compaction"])), "dce": float((b - full).mean())}}
    for nm in schemes_c:
        rr = np.asarray(pr[f"router_{nm}"]["ce"])
        diff = rr - b
        res["test"][nm] = {"comp": float(np.mean(pr[f"router_{nm}"]["compaction"])), "dce": float((rr - full).mean()),
                           "paired": float(diff.mean()), "paired_se": float(diff.std(ddof=1) / math.sqrt(len(diff)))}
        t_ = res["test"][nm]
        log(f"[test ⚠ eval_size {len(full)}] {nm}: T2/T {t_['comp']:.3f} dCE {t_['dce']:+.4f} | bar T2/T {res['test']['bar']['comp']:.3f} "
            f"dCE {res['test']['bar']['dce']:+.4f} | paired {t_['paired']:+.4f} +- {t_['paired_se']:.4f}")
    res["git_commit"] = G.git_commit()
    res["sec"] = time.time() - t0
    json.dump(res, open(os.path.join(a.out_dir, a.task, f"classify_{a.name}.json"), "w"), indent=1)
    log(f"done in {res['sec']:.0f}s")


if __name__ == "__main__":
    main()
