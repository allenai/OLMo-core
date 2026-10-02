"""Compute-vs-loss frontier: frozen v6a TOKEN router -> learned per-(token, layer) SKIP router.

    python debug/learned_router/layerskip/frontier.py --task nq --ckpt ... \\
        --tok "bar1=v6avg_s0_rhobar:1.0,bar0.7=v6avg_s0_rhobar:0.7" --name fr_bar

For each token configuration ``label=weights_name:f`` the token router keeps, on every row, the top
``round(f * k_bar)`` routed tokens (body + markers), where ``k_bar`` is the number of routed tokens
``gold_fl20p8_noslot`` keeps on that row (the token agent's ``@pair`` budget, scaled by f). Drop
semantics are the grid's (``check_hand_fl.router_real``: routed markers, original positions).
The layer-skip router (``--variant``, default ``shared``) then runs on the ELIGIBLE tokens (kept
document body tokens, not markers): keep 0.75, then 0.5 warm-started from the 0.75 router. Each
point is scored on train (32), val (64), test64 and the grid's 16 test rows (2k): per-row CE, and
FLOPs vs the full row (``layerskip_lib.flops_ratio``: per-layer active columns incl. the attention
quadratic and GDN terms). Controls: token router only (keep 1.0; flash and soft paths), uniform
random layer skip at the same keep (``--rand-seeds``), and the bar itself (heuristic_real mask).

Output: ``runs/<task>/<name>.json``; ``summarize_frontier.py`` builds the frontier.
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
import train_layerskip as TLS  # noqa: E402

sys.path.insert(0, os.path.dirname(HERE))
import router_lib as RL  # noqa: E402
import train_router as TR  # noqa: E402
from check_hand_fl import heuristic_real, router_real  # noqa: E402

G = TR.G
log = TLS.log


def base_prep(model, r, ids, dev):
    """Per-row things shared by every token configuration: features (routed markers), chunk ids, the
    bar's real mask and its routed budget, full-model answer log-probs."""
    from olmo_core.nn.attention.chunked_mask import build_chunk_ids_from_tokens

    p = TR.prep_row(model, r, ids, dev, route_markers=True)
    cid = build_chunk_ids_from_tokens(p["x"].cpu(), doc_start_id=ids.doc_start, doc_end_id=ids.doc_end,
                                      eos_id=ids.eos, mode="chunked")[0]
    gold = sorted(r["gold"]) if r["gold"] else None
    real_bar = heuristic_real(p["x"][0].cpu(), cid, gold, ids)
    p["cid"], p["real_bar"] = cid, real_bar
    p["k_bar"] = int(real_bar[p["feats"]["idx"].cpu()].sum())
    with torch.no_grad():
        lg = model(p["x"], logits_to_keep=p["pred"][None])[0].float()
    lpf = F.log_softmax(lg, -1)
    p["lpf"], p["ce_full"] = lpf.to(torch.bfloat16), float(F.nll_loss(lpf, p["tgt"]))
    return p


def compact(p, real: torch.Tensor, elig_full: torch.Tensor):
    """Item for train_layerskip.evaluate / train_one from a (S,) bool real-token mask."""
    dev = p["x"].device
    real = real.to(dev)
    kept = real.nonzero().flatten()
    cols = torch.searchsorted(kept, p["pred"])
    assert bool((kept[cols] == p["pred"]).all()), "answer prediction position dropped"
    elig_full = elig_full.to(dev) & real
    f = p["feats"]
    sel = elig_full[f["idx"]]
    return {"x": p["x"][:, kept], "pos": kept[None], "elig": elig_full[kept][None], "pred": cols, "tgt": p["tgt"],
            "tokfeat": {"pos": f["pos"][sel], "gold": f["gold"][sel], "e": p["e"][sel]},
            "lpf": p["lpf"], "ce_full": p["ce_full"], "T": p["T"], "T2": int(kept.numel()), "n_elig": int(sel.sum())}


def token_item(p, router, f, ids):
    """The token router at f x the bar's per-row routed budget (top-k), grid drop semantics."""
    fe = p["feats"]
    with torch.no_grad():
        z = router.logits(fe, None)
    k = int(round(f * p["k_bar"]))
    keep = RL.topk_keep(z, fe, k, getattr(router, "span", 0))
    mask = RL.keep_mask_from(fe, keep, p["T"])
    real = router_real(p["x"][0].cpu(), p["cid"], mask.cpu(), ids)
    elig = torch.zeros(p["T"], dtype=torch.bool, device=p["x"].device)
    body = keep.to(fe["idx"].device) & (fe["is_marker"] < 0.5)
    elig[fe["idx"][body]] = True
    return compact(p, real, elig)


@torch.no_grad()
def flash_ce(model, it):
    lg = model(it["x"], logits_to_keep=it["pred"][None], position_ids=it["pos"])[0].float()
    return float(F.cross_entropy(lg, it["tgt"]))


def split_eval(model, items, a, gater_fn, L, ref_full):
    r_ = TLS.evaluate(model, items, a.pad_id, gater_fn, L)
    ne = sum(it["n_elig"] for it in items)
    return {"ce": r_["ce"], "flops": r_["flops"], "keep_pairs": r_["keep_pairs"], "kl": float(np.mean(r_["kl"])),
            "skip_rate_per_layer": [v / max(1, ne) for v in r_["skip_per_layer"]],
            "dce": float(np.mean(np.array(r_["ce"]) - np.array(ref_full)))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt-format", default="hf", choices=["hf", "distcp"])
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--val-root", required=True)
    ap.add_argument("--test64-root", required=True)
    ap.add_argument("--test16-root", required=True)
    ap.add_argument("--tok", required=True, help="comma list label=weights_name:f (weights under learned_router/weights/<task>/)")
    ap.add_argument("--n-train", type=int, default=32)
    ap.add_argument("--n-val", type=int, default=32)
    ap.add_argument("--val-extra-from-train", type=int, default=32)
    ap.add_argument("--keeps", default="0.75,0.5", help="layer keep targets; each after the first is warm-started from the previous")
    ap.add_argument("--variant", default="shared")
    ap.add_argument("--epochs", type=int, default=20)
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
    ap.add_argument("--rand-seeds", type=int, default=2)
    ap.add_argument("--name", default="fr")
    ap.add_argument("--wandb-group", default=None)
    ap.add_argument("--wandb-project", default="memory-networks")
    ap.add_argument("--wandb-entity", default="prasann-uc-berkeley-electrical-engineering-computer-sciences")
    ap.add_argument("--tokenizer", default=os.environ.get("DEVLOSS_TOKENIZER", G.TOKENIZER_BY_FAMILY[G.FAMILY]))
    ap.add_argument("--out-dir", default=HERE)
    a = ap.parse_args()
    t_start = time.time()
    log(f"frontier task={a.task} tok={a.tok} keeps={a.keeps} variant={a.variant} epochs={a.epochs}")
    from transformers import AutoTokenizer

    from olmo_core.nn.attention.fla_autotune import freeze_fla_length_autotune

    ids = G.RESERVED_IDS[G.FAMILY]
    vocab = G.VOCAB_BY_FAMILY[G.FAMILY]
    a.pad_id = ids.eos
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rows = {"train": TR.load_split(a.train_root, a.task, ["2k"], tok, ids, a.n_train)}
    va_rows = TR.load_split(a.val_root, a.task, ["2k"], tok, ids, a.n_val)
    if a.val_extra_from_train:
        extra = TR.load_split(a.train_root, a.task, ["2k"], tok, ids, a.n_train + a.val_extra_from_train)
        seen = {tuple(r["ids"]) for r in rows["train"]}
        va_rows += [r for r in extra if tuple(r["ids"]) not in seen]
    rows["val"] = va_rows
    rows["test64"] = TR.load_split(a.test64_root, a.task, ["2k"], tok, ids, 64)
    rows["test16"] = TR.load_split(a.test16_root, a.task, ["2k"], tok, ids, 16)
    all_ids = np.concatenate([np.asarray(r["ids"], dtype=np.int64) for r in rows["train"]])
    pieces = tok.convert_ids_to_tokens(list(range(vocab)))
    stop_ids, _, tables = G.build_tables(all_ids, vocab, pieces, ids, lambda t: tok.decode([int(t)]))
    model = G.load_model(a.ckpt, a.ckpt_format, vocab, ids, 42, stop_ids, "flash_2")
    for p_ in model.parameters():
        p_.requires_grad_(False)
    freeze_fla_length_autotune()
    dev = next(model.parameters()).device
    TR.configure(model, stop_ids, tables.to(dev))
    max_pos = 2 * max(len(r["ids"]) for rr in rows.values() for r in rr) + 8
    for mod in model.modules():
        rope = getattr(mod, "rope", None)
        if rope is not None and hasattr(rope, "warmup_cache"):
            rope.warmup_cache(max_pos, dev)
    model.eval()
    L, d = len(model.blocks), int(model.embeddings.weight.shape[1])

    t0 = time.time()
    base = {}
    for nm, rr in rows.items():
        base[nm] = []
        for j, r in enumerate(rr):
            base[nm].append(base_prep(model, r, ids, dev))
            if j + 1 in (1, 2, 5, 10) or (j + 1) % 25 == 0 or j + 1 == len(rr):
                log(f"base prep {nm} {j + 1}/{len(rr)} ({time.time() - t0:.0f}s, ETA {(time.time() - t0) / (j + 1) * (len(rr) - j - 1):.0f}s for this split)")
    out = {"task": a.task, "argv": sys.argv, "git_commit": G.git_commit(), "eval_size": {k: len(v) for k, v in base.items()},
           "hparams": {k: v for k, v in vars(a).items() if k != "pad_id"}, "full": {}, "bar": {}, "configs": {}}
    os.makedirs(os.path.join(a.out_dir, "runs", a.task), exist_ok=True)
    os.makedirs(os.path.join(a.out_dir, "weights", a.task), exist_ok=True)
    opath = os.path.join(a.out_dir, "runs", a.task, f"{a.name}.json")
    if os.path.exists(opath):  # preempted + requeued: keep finished configs
        prev = json.load(open(opath))
        out["configs"] = prev.get("configs", {})
        log(f"resuming: finished configs {sorted(out['configs'])}")

    def dump():
        json.dump(out, open(opath + ".part", "w"))
        os.replace(opath + ".part", opath)

    # the bar (gold_fl20p8_noslot via heuristic_real) and the full model, per split
    for nm, ps in base.items():
        out["full"][nm] = [p["ce_full"] for p in ps]
        bar_items = [compact(p, p["real_bar"], torch.zeros(p["T"], dtype=torch.bool)) for p in ps]
        out["bar"][nm] = {"ce": [flash_ce(model, it) for it in bar_items],
                          "flops": [LS.flops_ratio(it["T"], it["T2"], [0] * L) for it in bar_items],
                          "comp": [it["T2"] / it["T"] for it in bar_items]}
        b = out["bar"][nm]
        log(f"{nm}: {len(ps)} rows, CE_full {np.mean(out['full'][nm]):.4f}; bar dCE {np.mean(b['ce']) - np.mean(out['full'][nm]):+.4f} "
            f"T2/T {np.mean(b['comp']):.3f} FLOPs x{np.mean(b['flops']):.4f}")
    dump()

    for spec in a.tok.split(","):
        label, rest = spec.split("=")
        wname, f = rest.rsplit(":", 1)
        f = float(f)
        if label in out["configs"] and out["configs"][label].get("done"):
            log(f"[{label}] done (resumed); skipping")
            continue
        wpath = os.path.join(os.path.dirname(HERE), "weights", a.task, f"{wname}.pt")
        st = torch.load(wpath, map_location="cpu")
        tokr = RL.LinearRouter.from_state(st).to(dev)
        assert tokr.route_markers and not tokr.use_emb, (tokr.route_markers, tokr.use_emb)
        assert not getattr(tokr, "marker_follow", False)
        items = {nm: [token_item(p, tokr, f, ids) for p in ps] for nm, ps in base.items()}
        cfg = {"weights": wname, "f": f, "splits": {}, "points": {}, "random": {}}
        for nm, it in items.items():
            soft = split_eval(model, it, a, lambda el, tf: LS.Gater(el, "ones"), L, out["full"][nm])
            cfg["splits"][nm] = {"tok_flash": [flash_ce(model, x_) for x_ in it], "tok": soft["ce"], "tok_flops": soft["flops"],
                                 "comp": [x_["T2"] / x_["T"] for x_ in it], "elig_frac": [x_["n_elig"] / x_["T2"] for x_ in it]}
            s_ = cfg["splits"][nm]
            log(f"[{label}] {nm}: token-only dCE (flash) {np.mean(s_['tok_flash']) - np.mean(out['full'][nm]):+.4f} (soft) "
                f"{soft['dce']:+.4f}; T2/T {np.mean(s_['comp']):.3f} FLOPs x{np.mean(soft['flops']):.4f}; eligible {np.mean(s_['elig_frac']):.2f} of T2")
        init = None
        for kt in [float(x) for x in a.keeps.split(",")]:
            for nm in ("val", "test64"):
                ces, fl = [], []
                for sd in range(a.rand_seeds):
                    def rfn(el, tf, sd=sd, kt=kt):
                        g_ = torch.Generator().manual_seed(100 + sd)
                        return LS.Gater(el, "fixed", values=(torch.rand(L, *el.shape, generator=g_) < kt).float().to(el.device))
                    r_ = TLS.evaluate(model, items[nm], a.pad_id, rfn, L)
                    ces.append(r_["ce"])
                    fl.append(r_["flops"])
                cfg["random"].setdefault(f"{kt:g}", {})[nm] = {"ce": np.mean(ces, 0).tolist(), "flops": np.mean(fl, 0).tolist()}
            rt = cfg["random"][f"{kt:g}"]["test64"]
            log(f"[{label}] random keep {kt:g}: test64 dCE {np.mean(rt['ce']) - np.mean(out['full']['test64']):+.4f} FLOPs x{np.mean(rt['flops']):.4f}")
            a.rho_start = 1.0 if init is None else float(init["target"])
            wb = None
            if a.wandb_group:
                try:
                    import wandb

                    wb = wandb.init(entity=a.wandb_entity, project=a.wandb_project, group=a.wandb_group, reinit=True,
                                    name=f"{a.task}-{a.name}-{label}-k{kt:g}-{os.environ.get('SLURM_JOB_ID', 'local')}",
                                    config={**{k: v for k, v in vars(a).items()}, "tok_label": label, "f": f, "target": kt},
                                    settings=wandb.Settings(init_timeout=90))
                except Exception as ex:  # noqa: BLE001
                    log(f"wandb init failed ({ex!r}); continuing without wandb")
                    wb = None
            router, hist = TLS.train_one(model, a.variant, kt, items["train"], items["val"], a, L, d, wb, init_state=init)
            cs = hist["epochs"][-1]["c_star"]
            state = router.state()
            state.update({"c_star": cs, "target": kt, "tok_router": wname, "tok_f": f})
            torch.save(state, os.path.join(a.out_dir, "weights", a.task, f"{a.name}_{label}_k{kt:g}.pt"))
            pt = {"c_star": cs, "train_sec": hist["train_sec"]}
            for nm, it in items.items():
                pt[nm] = split_eval(model, it, a, lambda el, tf, r_=router, c_=cs: LS.Gater(el, "det", router=r_, c=c_, tokfeat=tf), L,
                                    out["full"][nm])
            cfg["points"][f"{kt:g}"] = pt
            te = pt["test64"]
            d_tok = np.array(te["ce"]) - np.array(cfg["splits"]["test64"]["tok"])
            log(f"[{label} k{kt:g}] test64 dCE vs full {te['dce']:+.4f}, vs token-only {d_tok.mean():+.4f} +- {d_tok.std(ddof=1) / math.sqrt(len(d_tok)):.4f}; "
                f"FLOPs x{np.mean(te['flops']):.4f} (tok-only x{np.mean(cfg['splits']['test64']['tok_flops']):.4f}) | val dCE {pt['val']['dce']:+.4f} train {pt['train']['dce']:+.4f}")
            if wb is not None:
                wb.summary.update({"test64_dce": te["dce"], "test64_flops": float(np.mean(te["flops"])), "val_dce": pt["val"]["dce"]})
                wb.finish()
            init = state
            out["configs"][label] = cfg
            dump()
        cfg["done"] = True
        out["configs"][label] = cfg
        dump()
    log(f"all done in {time.time() - t_start:.0f}s -> {opath}")


if __name__ == "__main__":
    main()
