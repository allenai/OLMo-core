"""CPU check of the idf{K} selector: render a few real rows, build the tables exactly as the driver
does, run the `rule` scorer with weights {"idf": 1.0}, and decode which body tokens survive per doc.

    HF_HUB_OFFLINE=1 python debug/devloss_grid/check_idf_cpu.py --task ctc_nq --rung 2k --rows 4 \
        --data-root /scratch/users/prasann/devloss_grid_data [--df]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctc_devloss_grid as D  # noqa: E402

from olmo_core.nn.attention.chunked_mask import build_chunk_ids_from_tokens, mark_doc_topk_tokens_free  # noqa: E402
from olmo_core.nn.pooled_soft_token import keep_token_scores  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ctc_nq")
    ap.add_argument("--rung", default="2k")
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--data-root", default="/scratch/users/prasann/devloss_grid_data")
    ap.add_argument("--tokenizer", default=os.environ.get("DEVLOSS_TOKENIZER", D.TOKENIZER_BY_FAMILY[D.FAMILY]))
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    ids = D.RESERVED_IDS[D.FAMILY]
    row = D.ROSTER[a.task]
    examples = D.load_examples(a.data_root, row, a.rung, a.rows)
    rows = [D.render_ctc_row(tok, ex, row["seg_task"], ids)[0] for ex in examples]
    vocab = D.VOCAB_BY_FAMILY[D.FAMILY]
    pieces = tok.convert_ids_to_tokens(list(range(min(vocab, len(tok)))))
    pieces = list(pieces) + [None] * (vocab - len(pieces))
    all_ids = np.concatenate([np.asarray(r, dtype=np.int64) for r in rows])
    stop_ids, shown, tables = D.build_tables(all_ids, vocab, pieces, ids, lambda i: tok.decode([int(i)]))
    idf = tables.idf.numpy()
    print(f"idf table: nonzero {int((idf > 0).sum())}, min {idf.min():.2f} max {idf.max():.2f}; "
          f"idf(doc_start)={idf[ids.doc_start]:.2f} idf(' the')={idf[tok.encode(' the')[0]]:.2f}")
    x = torch.tensor(rows[0])[None]
    cid = build_chunk_ids_from_tokens(x, doc_start_id=ids.doc_start, doc_end_id=ids.doc_end, eos_id=ids.eos, mode="chunked")
    n_docs = int(cid.max()) + 1
    scores = keep_token_scores(x, cid, tables=tables, weights={"idf": 1.0}, doc_start_id=ids.doc_start,
                               doc_end_id=ids.doc_end, n_docs=n_docs)
    nz = int((scores != 0).sum())
    print(f"row0: T={x.shape[1]} n_docs={n_docs} scored(nonzero) positions={nz}")
    cid2 = mark_doc_topk_tokens_free(cid, x, scores, doc_start_id=ids.doc_start, doc_end_id=ids.doc_end, k=a.k, n_docs=n_docs)
    free = (cid2 != cid)[0]
    print(f"freed positions: {int(free.sum())} (expect ~{a.k}*docs={a.k * n_docs} minus short docs)")
    # per-doc decode of kept tokens vs the first 12 tokens of the doc
    starts = (x[0] == ids.doc_start).nonzero().flatten().tolist()
    ends = (x[0] == ids.doc_end).nonzero().flatten().tolist()
    for d, (s, e) in enumerate(zip(starts, ends)):
        kept = [int(x[0, p]) for p in range(s + 1, e) if free[p]]
        body = x[0, s + 1 : e].tolist()
        print(f"--- doc {d}: len {len(body)}, kept {len(kept)}")
        print("   head : " + repr(tok.decode(body[:14])))
        print("   kept : " + repr(tok.decode(kept[:40])))
        if d >= 3:
            break


if __name__ == "__main__":
    main()
