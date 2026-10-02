"""Per test row: what did a trained router keep? Fraction of GOLD-document tokens kept, of non-gold
tokens kept, and the row's dCE from the grid-format results file. CPU only (router + embedding table).

    python debug/learned_router/inspect_e2e_rows.py --task nq --rung 8k --routers e2e_rho0.1,e2e_rho0.2 \\
        --results debug/devloss_grid/results_router_e2e --data-root /net/sneetches/data/prasann/devloss_grid/data \\
        --ckpt /net/sneetches/data/prasann/devloss_grid/ckpts/ctc-4b-nq-full

(Reading one tensor of a checkpoint over /net: an audit read, not job I/O.)
"""
import argparse
import glob
import json
import os
import sys
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "devloss_grid"))
import ctc_devloss_grid as G  # noqa: E402
import router_lib as RL  # noqa: E402
from analyze_weights import load_embedding  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--rung", required=True)
    ap.add_argument("--routers", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--tag", default="")
    ap.add_argument("--tokenizer", default=sorted(glob.glob("/net/sneetches/data/prasann/hf_cache/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/*"))[-1])
    a = ap.parse_args()
    from transformers import AutoTokenizer

    from olmo_core.nn.attention.chunked_mask import build_chunk_ids_from_tokens

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    ids = G.RESERVED_IDS[G.FAMILY]
    row = G.ROSTER[f"ctc_{a.task}"]
    exs = G.load_examples(a.data_root, row, a.rung, a.rows)
    E = load_embedding(a.ckpt)
    res = json.load(open(os.path.join(a.results, f"{a.task}_{a.rung}.json")))
    full = np.array(res["per_row"]["full"]["ce"])
    out = {}
    for name in a.routers.split(","):
        st = torch.load(os.path.join(HERE, "weights", a.task, f"{name}.pt"), map_location="cpu")
        router = RL.LinearRouter.from_state(st)
        dce = np.array(res["per_row"][f"router_{name}"]["ce"]) - full
        recs = []
        for i, ex in enumerate(exs):
            r_ids, _, n_spans = G.render_ctc_row(tok, ex, row["seg_task"], ids)
            if n_spans != len(ex.get("documents") or []):
                continue
            x = torch.tensor(r_ids)
            cid = build_chunk_ids_from_tokens(x[None], doc_start_id=ids.doc_start, doc_end_id=ids.doc_end, eos_id=ids.eos, mode="chunked")[0]
            n_docs = int(cid.max()) + 1
            gold = G.gold_docs(row["spec"], ex)
            f = RL.routed_features(x, cid, ids.doc_start, ids.doc_end, n_docs, sorted(gold) if gold else None)
            with torch.no_grad():
                k = router.logits(f, RL.rms_embed(E, f["tok"])) > 0
            g = f["gold"] > 0
            idr = f["j"] < 4  # doc-id region: first 4 body tokens of each document (`Document [N]:`)
            recs.append({"row": i, "T": len(r_ids), "dce": float(dce[len(recs)]) if len(recs) < len(dce) else None,
                         "gold_kept": float(k[g].float().mean()) if g.any() else None,
                         "idregion_kept": float(k[idr].float().mean()), "comp_router_keep": float(k.float().mean()),
                         "nongold_kept": float(k[~g].float().mean()), "n_gold_tok": int(g.sum())})
        out[name] = {"w_gold": float(st["w_gold"]), "b": float(st["b"]), "rows": recs}
        print(f"== {a.task}@{a.rung} {name}: w_gold {float(st['w_gold']):+.2f} b {float(st['b']):+.2f}")
        for r in recs:
            flag = "  <<<" if r["dce"] is not None and r["dce"] > 0.1 else ""
            print(f"   row {r['row']:2d} T={r['T']:5d} dCE {r['dce']:+.3f}  gold kept {r['gold_kept'] if r['gold_kept'] is None else round(r['gold_kept'], 3)} "
                  f"({r['n_gold_tok']} tok)  non-gold kept {r['nongold_kept']:.3f}  id-region kept {r['idregion_kept']:.3f}{flag}")
        gk = [r["gold_kept"] for r in recs if r["gold_kept"] is not None]
        print(f"   MEAN gold kept {np.mean(gk) if gk else float('nan'):.3f}  id-region kept {np.mean([r['idregion_kept'] for r in recs]):.3f}  "
              f"routed kept {np.mean([r['comp_router_keep'] for r in recs]):.3f}  dCE {np.mean([r['dce'] for r in recs]):+.3f}")
    json.dump(out, open(os.path.join(HERE, "runs", a.task, f"inspect_{a.rung}{a.tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
