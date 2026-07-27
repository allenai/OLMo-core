"""Explain the token delta between the chunked and standard shard sets.

If the two builds really differ ONLY by the document boundary tokens, then across the FULL shard
set (not just part 0):

    tokens(chunked) - tokens(standard)  ==  2 * n_documents  ==  box_start + box_end

Any residual has to come from the instances each build dropped for exceeding --seq-len (the marker
build is longer, so it drops slightly more), which is why the instance counts differ. This
reconciles the delta against both terms instead of eyeballing it.
"""

import glob
import json

import numpy as np

DS, DE, EOS = 248049, 248050, 248044
CHUNKED = "/data/prasann/xlong5/shards"
STANDARD = "/data/prasann/xlong5/shards_nomarker"


def totals(d: str) -> dict:
    ds = de = ntok = 0
    for f in sorted(glob.glob(d + "/token_ids_part_*.npy")):
        a = np.fromfile(f, dtype=np.uint32)
        ntok += len(a)
        ds += int((a == DS).sum())
        de += int((a == DE).sum())
    meta = json.load(open(d + "/metadata.json"))
    return {"tokens": ntok, "ds": ds, "de": de,
            "instances": meta["num_instances"], "dropped": meta["num_dropped"]}


print(f"{'task':15s}{'tok(chk)':>14}{'tok(std)':>14}{'delta':>12}{'box_s+box_e':>13}"
      f"{'resid':>10}{'inst chk/std':>14}")
for d in sorted(glob.glob(CHUNKED + "/*_train")):
    task = d.split("/")[-1].replace("_train", "")
    c = totals(d)
    s = totals(STANDARD + "/" + task + "_train")
    delta = c["tokens"] - s["tokens"]
    boxes = c["ds"] + c["de"]
    print(f"{task:15s}{c['tokens']:>14,}{s['tokens']:>14,}{delta:>12,}{boxes:>13,}"
          f"{delta - boxes:>10,}{c['instances']:>7}/{s['instances']:<6}")
    print(f"{'':15s}  docs={c['ds']:,}  avg docs/inst={c['ds'] / c['instances']:.1f}  "
          f"dropped chk={c['dropped']} std={s['dropped']}")
