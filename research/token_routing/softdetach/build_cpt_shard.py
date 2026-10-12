"""
Build marker-wrapped CPT shards for the soft-detach CPT comparison (records/softdetach-cpt-plan.md).

Reads amandab's tokenized dolma3_longmino sample (raw uint32 ``part-*.npy``, EOS-separated docs,
Qwen3.5 ids), concatenates documents in file order, and cuts the stream into fixed rows of
``blocks`` pseudo-documents: each pseudo-document is ``<|doc_start|>`` + ``body`` body tokens +
``<|doc_end|>``. A row ends with EOS. Every arm (dense AND soft) trains on the SAME shard, so the
markers are in-distribution for both -- the ds64 convention.

Output = the ``train_ctc_suite.py`` shard layout: raw ``token_ids_part_0000.npy`` (uint32, no npy
header -- np.load on it kills the run, records/ds64-handoff.md trap 14), raw ``labels_mask_0000.npy``
(bool; True on body tokens, False on markers/EOS) and ``metadata.json``.

The dev shard is cut from a DIFFERENT source part file than the train shard (``--dev-parts``), so
no dev token was ever trained on.

    python build_cpt_shard.py --src /weka/.../dolma3_longmino_mix_sample15B_qwen3_5 \
        --out /weka/.../softdetach_cpt/shards/cpt_u128M --tokens 128000000 --parts 0-7
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np

from olmo_core.data.document_chunk_landmark import RESERVED_IDS


def iter_docs(paths, eos, dtype):
    for p in paths:
        arr = np.memmap(p, dtype=dtype, mode="r")
        cuts = np.flatnonzero(arr == eos)
        start = 0
        for c in cuts:
            if c > start:
                yield np.asarray(arr[start:c])
            start = int(c) + 1
        if start < len(arr):
            yield np.asarray(arr[start:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, required=True, help="row tokens to emit (train) / cap")
    ap.add_argument("--parts", default="0-7", help="source part index range a-b (inclusive)")
    ap.add_argument("--body", type=int, default=510)
    ap.add_argument("--blocks", type=int, default=127, help="127 x 512 = 65024 + EOS <= 65536")
    ap.add_argument("--family", default="qwen3_5")
    ap.add_argument("--min-doc", type=int, default=64, help="skip docs shorter than this")
    a = ap.parse_args()
    ids = RESERVED_IDS[a.family]
    eos = ids.eos
    paths = sorted(glob.glob(os.path.join(a.src, "part-*.npy")))
    lo, hi = [int(x) for x in a.parts.split("-")]
    paths = paths[lo : hi + 1]
    if not paths:
        raise SystemExit(f"no part files in {a.src} for range {a.parts}")
    print(f"[build] {len(paths)} source parts: {paths[0]} .. {paths[-1]}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    row_len = a.blocks * (a.body + 2) + 1
    ids_out, mask_out = [], []
    buf = []  # token stream (docs concatenated, no EOS -- the block structure replaces it)
    n_rows, n_tok, t0 = 0, 0, time.time()
    stream = iter_docs(paths, eos, np.uint32)
    exhausted = False
    while n_tok < a.tokens and not exhausted:
        need = a.blocks * a.body
        while sum(len(b) for b in buf) < need:
            try:
                d = next(stream)
            except StopIteration:
                exhausted = True
                break
            if len(d) >= a.min_doc:
                buf.append(d)
        if exhausted:
            break
        flat = np.concatenate(buf)
        take, rest = flat[:need], flat[need:]
        buf = [rest] if len(rest) else []
        row = np.empty(row_len, dtype=np.uint32)
        msk = np.zeros(row_len, dtype=bool)
        pos = 0
        for b in range(a.blocks):
            row[pos] = ids.doc_start
            row[pos + 1 : pos + 1 + a.body] = take[b * a.body : (b + 1) * a.body]
            row[pos + 1 + a.body] = ids.doc_end
            msk[pos + 1 : pos + 1 + a.body] = True
            pos += a.body + 2
        row[pos] = eos
        ids_out.append(row)
        mask_out.append(msk)
        n_rows += 1
        n_tok += row_len
        if n_rows in (1, 2, 5, 10) or n_rows % 200 == 0:
            print(f"[build] rows {n_rows} tokens {n_tok:,} ({time.time() - t0:.0f}s)", flush=True)
    tok = np.concatenate(ids_out)
    msk = np.concatenate(mask_out)
    tok.tofile(os.path.join(a.out, "token_ids_part_0000.npy"))
    msk.tofile(os.path.join(a.out, "labels_mask_0000.npy"))
    meta = {
        "task": "cpt", "emit": "dense", "cot_mode": "none", "chunk_by": "block512",
        "wrap_docs": True, "marker_set": a.family, "eos_token_id": int(eos),
        "doc_start_id": int(ids.doc_start), "doc_end_id": int(ids.doc_end),
        "landmark_token_id": None, "pad_token_id": None, "mem_freq": None,
        "dtype": "uint32", "mask_dtype": "bool", "num_instances": n_rows, "num_dropped": 0,
        "num_tokens": int(n_tok), "num_loss_tokens": int(msk.sum()),
        "max_example_len": row_len, "min_example_len": row_len,
        "source": a.src, "source_parts": a.parts, "body_tokens": a.body, "blocks_per_row": a.blocks,
        "query_position": "n/a",
    }
    json.dump(meta, open(os.path.join(a.out, "metadata.json"), "w"), indent=1)
    print(f"[build] DONE {a.out}: {n_rows} rows, {n_tok:,} tokens, {int(msk.sum()):,} loss tokens", flush=True)


if __name__ == "__main__":
    main()
