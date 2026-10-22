"""
Rewrite a fixed-length training shard from EOS-FIRST rows ([EOS, body...]) to EOS-LAST rows ([body..., EOS]).

Why: the trainer's document source splits the token stream at EOS (a document ENDS at its EOS). With EOS-first
rows the stream starts with a lone EOS, i.e. a degenerate 1-token document padded to a full instance (43,001
instances for 43,000 rows). The soft-token variant's length-sorted micro-batching then gives the rank holding
that instance a different micro-batch structure, its FSDP collectives fall out of step, and the job hangs
(olmo3_cptdrop 2026-10-22: every drop arm hung at step 118, where seed 34521 puts that instance on rank 7).
With EOS-last rows every document is exactly one row and identical to the instances the trainer already built
([body, EOS]); only the degenerate document disappears. The label mask keeps EOS unscored, as before.

    python eos_to_end.py --src .../data/train32k --out .../data/train32k_ee
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
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.src, "metadata.json")))
    L, eos = meta["max_example_len"], meta["eos_token_id"]
    tok = np.fromfile(os.path.join(a.src, "token_ids_part_0000.npy"), dtype=np.uint32).reshape(-1, L)
    msk = np.fromfile(os.path.join(a.src, "labels_mask_0000.npy"), dtype=bool).reshape(-1, L)
    assert (tok[:, 0] == eos).all(), "source rows must all start with EOS"
    assert int((tok[:, 1:] == eos).sum()) == 0, "source bodies must contain no EOS"
    new_t = np.empty_like(tok)
    new_t[:, :-1] = tok[:, 1:]
    new_t[:, -1] = eos
    new_m = np.empty_like(msk)
    new_m[:, :-1] = msk[:, 1:]
    new_m[:, -1] = False
    os.makedirs(a.out, exist_ok=True)
    new_t.reshape(-1).tofile(os.path.join(a.out, "token_ids_part_0000.npy"))
    new_m.reshape(-1).tofile(os.path.join(a.out, "labels_mask_0000.npy"))
    for f in ("rows.tsv", "doc_ids.txt"):
        if os.path.exists(os.path.join(a.src, f)):
            shutil.copy(os.path.join(a.src, f), os.path.join(a.out, f))
    meta.update(eos_position="last", num_loss_tokens=int(new_m.sum()),
                derived_from=f"{os.path.basename(a.src)} (EOS moved from row start to row end)")
    json.dump(meta, open(os.path.join(a.out, "metadata.json"), "w"), indent=1)
    print(f"wrote {a.out}: {tok.shape[0]} rows x {L}, every row ends with EOS, loss tokens {int(new_m.sum()):,}")


if __name__ == "__main__":
    main()
