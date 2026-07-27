"""Confirm the chunked / marker-free shard PAIR differs only by the document boundary tokens.

The chunked arm needs `<|box_start|>`/`<|box_end|>` to derive chunk ids; the standard (full) arm
must not see them at all. Both are built from the same pools with the same rendering, so:
  * the marker build has box_start == box_end == (number of documents),
  * the marker-free build has ZERO of either,
  * instance counts match, and the token delta is ~2 per document.
"""

import glob
import json

import numpy as np

DS, DE = 248049, 248050
ROOTS = [("chunked", "/data/prasann/xlong5/shards"),
         ("nomarker", "/data/prasann/xlong5/shards_nomarker")]

rows = {}
for var, root in ROOTS:
    for d in sorted(glob.glob(root + "/*_train")):
        task = d.split("/")[-1].replace("_train", "")
        meta = json.load(open(d + "/metadata.json"))
        part0 = sorted(glob.glob(d + "/token_ids_part_*.npy"))[0]
        a = np.fromfile(part0, dtype=np.uint32)
        rows.setdefault(task, {})[var] = {
            "instances": meta["num_instances"],
            "tokens": meta["num_tokens"],
            "markers_meta": meta.get("doc_markers"),
            "ds": int((a == DS).sum()),
            "de": int((a == DE).sum()),
        }

hdr = f"{'task':16s}{'inst(chk)':>10}{'inst(nom)':>10}{'tok(chk)':>15}{'tok(nom)':>15}{'box chk':>10}{'box nom':>9}"
print(hdr)
ok = True
for task, v in sorted(rows.items()):
    c, n = v.get("chunked"), v.get("nomarker")
    if not c or not n:
        print(f"{task:16s}  MISSING one variant")
        ok = False
        continue
    # `doc_markers` was added to metadata.json with the --no-doc-markers flag, so shards built
    # before it predate the key: absent == the old always-wrapped behaviour == True.
    chk_meta = True if c["markers_meta"] is None else c["markers_meta"]
    good = (n["ds"] == 0 and n["de"] == 0 and c["ds"] > 0 and c["ds"] == c["de"]
            and chk_meta is True and n["markers_meta"] is False)
    ok = ok and good
    print(f"{task:16s}{c['instances']:>10}{n['instances']:>10}{c['tokens']:>15,}"
          f"{n['tokens']:>15,}{c['ds']:>10,}{n['ds']:>9,}  {'OK' if good else 'BAD'}")
print("\nPAIR OK - chunked has markers, standard has none" if ok else "\nPROBLEM - inspect")
