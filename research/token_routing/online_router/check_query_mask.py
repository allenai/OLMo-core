"""Per-task check of the router's view of real shard rows (CPU): body tokens, documents, query tokens found by
``query_mask`` (FREE tokens after the last document, before the answer) and gold coverage, so a shard built with
a different query_position (mt11: ``both``) cannot silently leave the router without a query.

    python research/token_routing/online_router/check_query_mask.py <shard dir> [rows per task]
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"))
from olmo_core.nn.attention.chunked_mask import build_chunk_ids_from_tokens  # noqa: E402
from olmo_core.nn.attention.gold_grad_mask import content_fingerprint_from_row  # noqa: E402
from olmo_core.nn.online_token_router import query_mask, structural_features  # noqa: E402

shard = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
meta = json.load(open(os.path.join(shard, "metadata.json")))
EOS, DS, DE = meta["eos_token_id"], meta["doc_start_id"], meta["doc_end_id"]
gold = json.load(open(os.path.join(shard, "gold_fingerprints.json"))) if os.path.exists(os.path.join(shard, "gold_fingerprints.json")) else {}
print(f"query_position={meta.get('query_position')} max_example_len={meta.get('max_example_len')}")
for ti, src in enumerate(meta["merged_from"]):
    ids = np.fromfile(os.path.join(shard, f"token_ids_part_t{ti}_000.npy"), dtype=np.uint32)
    lm = np.fromfile(os.path.join(shard, f"labels_mask_t{ti}_000.npy"), dtype=np.bool_)
    ends = np.nonzero(ids == EOS)[0][:n]
    starts = np.concatenate([[0], ends[:-1] + 1])
    stats = []
    for s, e in zip(starts, ends):
        x = torch.from_numpy(ids[s:e + 1].astype(np.int64))[None]
        m = torch.from_numpy(lm[s:e + 1])
        lab = torch.full_like(x, -100)
        lab[0, :-1] = torch.where(m[1:], x[0, 1:], torch.full_like(x[0, 1:], -100))
        cid = build_chunk_ids_from_tokens(x, DS, DE, EOS, mode="chunked")
        _, body = structural_features(x, cid, doc_start_id=DS, doc_end_id=DE)
        q = query_mask(cid, lab)
        g = gold.get(content_fingerprint_from_row(x[0].tolist(), EOS))
        first_doc = int((cid[0] >= 0).nonzero()[0]) if (cid[0] >= 0).any() else -1
        pre_free = int(((cid[0] == -1) & (torch.arange(x.shape[1]) < first_doc)).sum())
        stats.append((x.shape[1], int(body.sum()), int(cid.max()) + 1, int(q.sum()), pre_free, int(m.sum()), g is not None))
    a = np.array([s[:6] for s in stats])
    print(f"t{ti} {src['task']:13s} T~{a[:,0].mean():6.0f} body~{a[:,1].mean():6.0f} docs~{a[:,2].mean():5.1f} "
          f"query(after docs)~{a[:,3].mean():5.1f} min {a[:,3].min()} | free before 1st doc~{a[:,4].mean():5.0f} "
          f"answer toks~{a[:,5].mean():5.1f} | gold fp hit {sum(s[6] for s in stats)}/{len(stats)}")
