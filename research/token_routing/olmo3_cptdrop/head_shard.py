"""
First ``--rows`` rows of a fixed-length shard as a new shard (train_ctc_suite layout). Arm C's 32k phase trains
on ``train32k_head`` (rows 0..39,159): the data loader shuffles the whole shard, so pointing phase 2 at
train32k could re-read the tail rows that ``c8k`` (phase 1) was cut from.

    python head_shard.py --src .../data/train32k --out .../data/train32k_head --rows 39160
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, required=True)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.src, "metadata.json")))
    L = meta["max_example_len"]
    assert a.rows <= meta["num_instances"]
    os.makedirs(a.out, exist_ok=True)
    for name, dtype in (("token_ids_part_0000.npy", np.uint32), ("labels_mask_0000.npy", bool)):
        src = np.memmap(os.path.join(a.src, name), dtype=dtype, mode="r")
        src[: a.rows * L].tofile(os.path.join(a.out, name))
    with open(os.path.join(a.src, "rows.tsv")) as f, open(os.path.join(a.out, "rows.tsv"), "w") as g:
        for i, line in enumerate(f):
            if i > a.rows:
                break
            g.write(line)
    out = dict(meta)
    msk = np.memmap(os.path.join(a.out, "labels_mask_0000.npy"), dtype=bool, mode="r")
    out.update(num_instances=a.rows, num_tokens=a.rows * L, num_loss_tokens=int(msk.sum()),
               source_rows=f"{os.path.basename(a.src)} rows 0..{a.rows - 1}")
    json.dump(out, open(os.path.join(a.out, "metadata.json"), "w"), indent=1)
    if os.path.exists(os.path.join(a.src, "doc_ids.txt")):
        shutil.copy(os.path.join(a.src, "doc_ids.txt"), os.path.join(a.out, "doc_ids.txt"))
    print(f"wrote {a.out}: {a.rows} rows x {L}")


if __name__ == "__main__":
    main()
