"""
Arm C phase-1 shard: the LAST ``--rows`` rows of train32k, each 32,768-token row cut into four 8,192-token
rows (labels mask cut the same way). Phase 2 reads train32k from row 0, so the two phases never share
text as long as phase 2 stays below row 43,000 - rows (it needs <= 3,840 rows at the 250M budget).

    python build_c8k.py --src /data/prasann/olmo3_cptdrop/data/train32k --out /data/prasann/olmo3_cptdrop/data/c8k
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=3840, help="32k source rows (4 x 8k rows each)")
    ap.add_argument("--piece", type=int, default=8192)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.src, "metadata.json")))
    L = meta["max_example_len"]
    assert L % a.piece == 0
    tok = np.fromfile(os.path.join(a.src, "token_ids_part_0000.npy"), dtype=np.uint32).reshape(-1, L)
    msk = np.fromfile(os.path.join(a.src, "labels_mask_0000.npy"), dtype=bool).reshape(-1, L)
    n = tok.shape[0]
    lo = n - a.rows
    t = np.ascontiguousarray(tok[lo:]).reshape(-1, a.piece)
    m = np.ascontiguousarray(msk[lo:]).reshape(-1, a.piece)
    os.makedirs(a.out, exist_ok=True)
    t.reshape(-1).tofile(os.path.join(a.out, "token_ids_part_0000.npy"))
    m.reshape(-1).tofile(os.path.join(a.out, "labels_mask_0000.npy"))
    out = dict(meta)
    out.update(
        num_instances=int(t.shape[0]), num_tokens=int(t.size), num_loss_tokens=int(m.sum()),
        max_example_len=a.piece, min_example_len=a.piece,
        source_rows=f"train32k rows {lo}..{n - 1} cut into {L // a.piece} x {a.piece}",
    )
    json.dump(out, open(os.path.join(a.out, "metadata.json"), "w"), indent=1)
    print(f"wrote {a.out}: {t.shape[0]} rows x {a.piece} = {t.size:,} tokens from train32k rows {lo}..{n - 1}")


if __name__ == "__main__":
    main()
