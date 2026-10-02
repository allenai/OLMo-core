"""Compaction floor per task/rung: the share of each row that no token router can drop.

    python debug/learned_router/floors.py --tasks nq,outlier --rung 2k --rows 16

``floor_markers_kept`` = (T - routed body tokens) / T: prompt, query, answer, EOS AND every doc's two
markers (the default router, markers always kept). ``floor_markers_routed`` = the same without the
markers (``--route-markers`` routers). Means over the grid's test rows.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "devloss_grid"))
import ctc_devloss_grid as G  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--rung", default="2k")
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--data-root", default="/net/sneetches/data/prasann/devloss_grid/data")
    ap.add_argument("--tokenizer", default=sorted(glob.glob("/net/sneetches/data/prasann/hf_cache/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/*"))[-1])
    a = ap.parse_args()
    from transformers import AutoTokenizer

    from olmo_core.nn.attention.chunked_mask import build_chunk_ids_from_tokens

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    ids = G.RESERVED_IDS[G.FAMILY]
    out = {}
    for t in a.tasks.split(","):
        row = G.ROSTER[f"ctc_{t}"]
        fk, fr = [], []
        for ex in G.load_examples(a.data_root, row, a.rung, a.rows):
            r, _, _ = G.render_ctc_row(tok, ex, row["seg_task"], ids)
            x = torch.tensor(r)
            cid = build_chunk_ids_from_tokens(x[None], doc_start_id=ids.doc_start, doc_end_id=ids.doc_end, eos_id=ids.eos, mode="chunked")[0]
            mk = (x == ids.doc_start) | (x == ids.doc_end)
            body = int(((cid >= 0) & ~mk).sum())
            nmk = int((mk & (cid >= 0)).sum())
            fk.append((len(r) - body) / len(r))
            fr.append((len(r) - body - nmk) / len(r))
        out[t] = {"floor_markers_kept": float(np.mean(fk)), "floor_markers_routed": float(np.mean(fr))}
        print(f"{t:14} {a.rung}: floor (markers kept) x{np.mean(fk):.3f}   floor (markers routed) x{np.mean(fr):.3f}")
    json.dump(out, open(os.path.join(HERE, f"floors_{a.rung}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
