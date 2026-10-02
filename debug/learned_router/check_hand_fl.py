"""Representability check (diagnostic only): how well does a hand-set ``relpos`` + routed-marker weight
vector reproduce ``gold_fl20p8_noslot``'s keep set on the grid's test rows? CPU only.

    python debug/learned_router/check_hand_fl.py [--tasks nq,outlier] [--rung 2k] [--rows 16] [--fit]

Per row, the heuristic's REAL set is computed with the model's own marking calls (header-free 8 body
tokens, first_last top ceil(0.2 R), gold documents whole incl. markers); the router's with
``mark_positions_free(free_markers="mask")`` (exactly the grid's router path). Reports the fraction of
routed tokens (body + markers) whose keep decision differs, and both T2/T. ``--fit``: also fit the
BEST linear vector in the class (logistic regression on the heuristic mask, features only, no
embedding) and report its mismatch -- a representability bound, never used as an init.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "devloss_grid"))
import ctc_devloss_grid as G  # noqa: E402
import router_lib as RL  # noqa: E402

TASKS = "nq,outlier,contradiction,strmatch,scifact,rerank,fiqa,msmarco,niah,obliq,oolong,outlier_amzn,qdmatch_hpqa,reorder,textgroups,grouping,absence"


def heuristic_real(x, cid, gold, ids):
    from olmo_core.nn.attention.chunked_mask import FREE_CHUNK_ID, mark_doc_headers_free, mark_doc_topk_tokens_free
    from olmo_core.nn.pooled_soft_token import first_last_scores

    n_docs = int(cid.max()) + 1
    c = mark_doc_headers_free(cid[None], x[None], doc_start_id=ids.doc_start, doc_end_id=ids.doc_end, stop_id=None,
                              stop_count=1, extra_tokens=8, cap=32)
    sc = first_last_scores(x[None], c, doc_start_id=ids.doc_start, doc_end_id=ids.doc_end, n_docs=n_docs)
    c = mark_doc_topk_tokens_free(c, x[None], sc, doc_start_id=ids.doc_start, doc_end_id=ids.doc_end, k=0.2, n_docs=n_docs)[0]
    real = (c == FREE_CHUNK_ID) | (cid < 0)
    for g in gold or []:
        real |= cid == int(g)
    return real


def router_real(x, cid, keep_mask, ids):
    from olmo_core.nn.attention.chunked_mask import FREE_CHUNK_ID, mark_positions_free

    c = mark_positions_free(cid[None], x[None], keep_mask[None], doc_start_id=ids.doc_start, doc_end_id=ids.doc_end,
                            n_docs=int(cid.max()) + 1, free_markers="mask")[0]
    return (c == FREE_CHUNK_ID) | (cid < 0)


def feats_matrix(f):
    return torch.cat([f["pos"][:, : 2 * RL.N_OFF], f["rel"], f["gold"][:, None], f["is_marker"][:, None]], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=TASKS)
    ap.add_argument("--rung", default="2k")
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--data-root", default="/net/sneetches/data/prasann/devloss_grid/data")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--save", action="store_true", help="write weights/<task>/hand_fl.pt (+ hand_fit.pt with --fit) for grid re-scoring")
    ap.add_argument("--tokenizer", default=sorted(glob.glob("/net/sneetches/data/prasann/hf_cache/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/*"))[-1])
    a = ap.parse_args()
    from transformers import AutoTokenizer

    from olmo_core.nn.attention.chunked_mask import build_chunk_ids_from_tokens

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    ids = G.RESERVED_IDS[G.FAMILY]
    hand = RL.LinearRouter.from_state(RL.fl20p8_hand_state(2560))
    out = {}
    for t in a.tasks.split(","):
        row = G.ROSTER[f"ctc_{t}"]
        exs = G.load_examples(a.data_root, row, a.rung, a.rows)
        recs, X, Y = [], [], []
        for ex in exs:
            r_ids, _, n_spans = G.render_ctc_row(tok, ex, row["seg_task"], ids)
            if G.SEG_CFG[row["seg_task"]]["chunk_by"] == "document" and n_spans != len(ex.get("documents") or []):
                continue  # the grid drops these rows too
            x = torch.tensor(r_ids)
            cid = build_chunk_ids_from_tokens(x[None], doc_start_id=ids.doc_start, doc_end_id=ids.doc_end, eos_id=ids.eos, mode="chunked")[0]
            gold = G.gold_docs(row["spec"], ex)
            f = RL.routed_features(x, cid, ids.doc_start, ids.doc_end, int(cid.max()) + 1, sorted(gold) if gold else None, route_markers=True)
            with torch.no_grad():
                k = hand.logits(f, None) > 0
            hr = heuristic_real(x, cid, gold, ids)
            rr = router_real(x, cid, RL.keep_mask_from(f, k, len(r_ids)), ids)
            routed = torch.zeros(len(r_ids), dtype=torch.bool)
            routed[f["idx"]] = True
            mism = float((hr != rr)[routed].float().mean())
            recs.append({"T": len(r_ids), "mismatch": mism, "comp_heur": float(hr.float().mean()), "comp_hand": float(rr.float().mean()),
                         "hand_extra": float((rr & ~hr)[routed].float().mean()), "hand_missing": float((hr & ~rr)[routed].float().mean())})
            X.append(feats_matrix(f))
            Y.append(hr[f["idx"]].float())
        res = {k: float(np.mean([r[k] for r in recs])) for k in recs[0]}
        res["rows"] = len(recs)
        if a.fit:
            Xc, Yc = torch.cat(X), torch.cat(Y)
            w = torch.zeros(Xc.shape[1] + 1, requires_grad=True)
            opt = torch.optim.LBFGS([w], max_iter=500, line_search_fn="strong_wolfe")

            def clo():
                opt.zero_grad()
                z = Xc @ w[1:] + w[0]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(z, Yc) + 1e-4 * w.pow(2).sum()
                loss.backward()
                return loss

            opt.step(clo)
            with torch.no_grad():
                res["fit_mismatch"] = float(((Xc @ w[1:] + w[0] > 0).float() != Yc).float().mean())
                fr = RL.LinearRouter(2560, "relpos")
                fr.route_markers = True
                nO = 2 * RL.N_OFF
                fr.b.fill_(float(w[0]))
                fr.w_pos[:nO] = w[1 : 1 + nO]
                fr.w_rel.copy_(w[1 + nO : 1 + nO + RL.N_REL])
                fr.w_gold.fill_(float(w[1 + nO + RL.N_REL]))
                fr.w_marker.fill_(float(w[2 + nO + RL.N_REL]))
                st = fr.state()
                st.update({"keep_rule": "p>0.5", "hand": "best linear fit to gold_fl20p8_noslot's mask (diagnostic)"})
                if a.save:
                    torch.save(st, os.path.join(HERE, "weights", t, "hand_fit.pt"))
        if a.save:
            torch.save(RL.fl20p8_hand_state(2560), os.path.join(HERE, "weights", t, "hand_fl.pt"))
        out[t] = res
        print(f"{t:14s} rows {res['rows']:2d}  token mismatch {res['mismatch']:.3%} (hand extra {res['hand_extra']:.3%}, missing {res['hand_missing']:.3%})"
              f"  T2/T heur {res['comp_heur']:.3f} hand {res['comp_hand']:.3f}" + (f"  best-linear-fit mismatch {res['fit_mismatch']:.3%}" if a.fit else ""), flush=True)
    fo = os.path.join(HERE, f"hand_fl_check_{a.rung}.json")
    prev = json.load(open(fo)) if os.path.exists(fo) else {}
    prev.update(out)
    json.dump(prev, open(fo, "w"), indent=1)


if __name__ == "__main__":
    main()
