"""What do the oracle labels keep? Mean soft label by token group (gold / non-gold body, markers, doc-id
region j<4, start/end deciles). CPU; reads the rows over /net (audit read).

    python debug/learned_router/oracle/inspect_labels.py --task rerank
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import router_lib as RL  # noqa: E402
import train_router as TR  # noqa: E402

G = TR.G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--root", default="/net/sneetches/data/prasann/devloss_grid/data/router_train_e2e")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "out"))
    ap.add_argument("--tokenizer", default=sorted(glob.glob("/net/sneetches/data/prasann/hf_cache/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/*"))[-1])
    a = ap.parse_args()
    from transformers import AutoTokenizer

    from olmo_core.nn.attention.chunked_mask import build_chunk_ids_from_tokens

    import hashlib

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    ids = G.RESERVED_IDS[G.FAMILY]
    rows = TR.load_split(a.root, a.task, ["2k"], tok, ids, a.n)
    recs = []
    for f in sorted(glob.glob(os.path.join(a.out_dir, a.task, "labels_train*.pt"))):
        recs += torch.load(f, map_location="cpu")
    by = {r["key"]: r for r in recs}
    acc = {}

    def add(k, v):
        acc.setdefault(k, []).append(v)

    for r in rows:
        x = torch.tensor(np.asarray(r["ids"]))
        key = hashlib.sha1(x[None].numpy().tobytes()).hexdigest()[:16]
        if key not in by:
            continue
        cid = build_chunk_ids_from_tokens(x[None], doc_start_id=ids.doc_start, doc_end_id=ids.doc_end, eos_id=ids.eos, mode="chunked")[0]
        f = RL.routed_features(x, cid, ids.doc_start, ids.doc_end, int(cid.max()) + 1, sorted(r["gold"]) if r["gold"] else None, route_markers=True)
        y = by[key]["label"]
        assert y.numel() == f["idx"].numel()
        mk = f["is_marker"] > 0
        g = f["gold"] > 0
        body = ~mk
        for name, m in (("all", torch.ones_like(mk)), ("marker", mk), ("gold body", g & body), ("non-gold body", ~g & body),
                        ("id region j<4", body & (f["j"] < 4)), ("j 4..15", body & (f["j"] >= 4) & (f["j"] < 16))):
            if bool(m.any()):
                add(name, float(y[m].mean()))
        for d in range(10):
            m = body & ((10 * f["j"]) // f["n"].clamp(min=1) == d)
            if bool(m.any()):
                add(f"start decile {d}", float(y[m].mean()))
        add("gold frac of routed", float(g.float().mean()))
    for k, v in acc.items():
        print(f"{k:22s} {np.mean(v):.3f}  (rows {len(v)})")


if __name__ == "__main__":
    main()
