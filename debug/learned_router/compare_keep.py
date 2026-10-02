"""What does a router keep vs gold_fl20p8_noslot, at the bar's T2/T, on a rung's test rows? CPU only
(position-only routers need no embedding). Diagnostic.

    python debug/learned_router/compare_keep.py --task textgroups --rung 32k --router v5_s0_rhobar --comp 0.245

Per row, the router keeps the top-k routed tokens with k set so the row's T2/T equals ``--comp`` (the grid's
``@c`` rule); the bar's REAL set comes from the model's own marking calls (check_hand_fl.heuristic_real).
Reports, for each, the fraction kept of: gold-document body, non-gold id region (body j < 8), markers,
non-gold body j >= 8. ⚠ eval_size = rows.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from check_hand_fl import G, RL, heuristic_real  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--rung", required=True)
    ap.add_argument("--router", required=True)
    ap.add_argument("--comp", type=float, required=True)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--data-root", default="/net/sneetches/data/prasann/devloss_grid/data")
    ap.add_argument("--tokenizer", default=sorted(glob.glob("/net/sneetches/data/prasann/hf_cache/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/*"))[-1])
    a = ap.parse_args()
    from transformers import AutoTokenizer

    from olmo_core.nn.attention.chunked_mask import build_chunk_ids_from_tokens

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    ids = G.RESERVED_IDS[G.FAMILY]
    row = G.ROSTER[f"ctc_{a.task}"]
    st = torch.load(os.path.join(HERE, "weights", a.task, f"{a.router}.pt"), map_location="cpu")
    router = RL.LinearRouter.from_state(st)
    E = None
    if router.use_emb:  # embedding routers need the input-embedding table (audit read of one tensor over /net)
        import json as _json

        from analyze_weights import load_embedding

        ck = _json.load(open(os.path.join(os.path.dirname(HERE), "devloss_grid", "manifest.json")))["tasks"][a.task]["ckpt"]
        E = load_embedding(ck.replace("/data/", "/net/sneetches/data/"))
    acc = {"router": {}, "bar": {}}
    for ex in G.load_examples(a.data_root, row, a.rung, a.rows):
        r_ids, _, n_spans = G.render_ctc_row(tok, ex, row["seg_task"], ids)
        if G.SEG_CFG[row["seg_task"]]["chunk_by"] == "document" and n_spans != len(ex.get("documents") or []):
            continue
        x = torch.tensor(r_ids)
        cid = build_chunk_ids_from_tokens(x[None], doc_start_id=ids.doc_start, doc_end_id=ids.doc_end, eos_id=ids.eos, mode="chunked")[0]
        gold = G.gold_docs(row["spec"], ex)
        f = RL.routed_features(x, cid, ids.doc_start, ids.doc_end, int(cid.max()) + 1, sorted(gold) if gold else None, route_markers=True)
        T, N = len(r_ids), int(f["idx"].numel())
        if getattr(router, "marker_follow", False):
            # markers follow their document: route body tokens only, markers kept iff their doc is touched
            fb = RL.routed_features(x, cid, ids.doc_start, ids.doc_end, int(cid.max()) + 1, sorted(gold) if gold else None)
            with torch.no_grad():
                zb = router.logits(fb, RL.rms_embed(E, fb["tok"]) if E is not None else None)
            mode = "whole" if router.marker_follow == "whole" else "any"
            kb = RL.follow_keep_budget(zb, fb, int(round(a.comp * T - (T - int(fb["idx"].numel()) - int(fb["mk_idx"].numel())))), mode)
            touched = torch.zeros(int(cid.max()) + 1, dtype=torch.bool)
            touched[fb["doc"][kb]] = True
            pos_keep = torch.zeros(T, dtype=torch.bool)
            pos_keep[fb["idx"][kb]] = True
            if mode == "whole":
                touched = torch.bincount(fb["doc"][kb], minlength=touched.numel()) == torch.bincount(fb["doc"], minlength=touched.numel())
            pos_keep[fb["mk_idx"][touched[fb["mk_doc"]]]] = True
            keep_r = pos_keep[f["idx"]]
        else:
            with torch.no_grad():
                z = router.logits(f, RL.rms_embed(E, f["tok"]) if E is not None else None)
            k = min(N, max(0, int(round(a.comp * T - (T - N)))))
            keep_r = RL.topk_keep(z, f, k, router.span)
        keep_b = heuristic_real(x, cid, gold, ids)[f["idx"]]
        mk = f["is_marker"] > 0
        g = (f["gold"] > 0) & ~mk
        idr = ~mk & (f["gold"] == 0) & (f["j"] < 8)
        rest = ~mk & (f["gold"] == 0) & (f["j"] >= 8)
        for name, kk in (("router", keep_r), ("bar", keep_b)):
            for cat, m in (("gold body", g), ("non-gold id region j<8", idr), ("markers", mk), ("non-gold body j>=8", rest), ("all routed", torch.ones_like(mk))):
                if bool(m.any()):
                    acc[name].setdefault(cat, []).append(float(kk[m].float().mean()))
            acc[name].setdefault("share of kept: gold", []).append(float((kk & g).sum()) / max(1, int(kk.sum())))
            acc[name].setdefault("share of kept: id region", []).append(float((kk & idr).sum()) / max(1, int(kk.sum())))
    n = len(next(iter(acc["bar"].values())))
    print(f"{a.task} @ {a.rung}, T2/T {a.comp}, router {a.router}  (⚠ {n} rows)")
    for cat in acc["bar"]:
        print(f"  {cat:26s} router {np.mean(acc['router'].get(cat, [np.nan])):.3f}   bar {np.mean(acc['bar'][cat]):.3f}")


if __name__ == "__main__":
    main()
