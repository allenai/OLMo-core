"""Merge several fast8k task shards into ONE multi-task shard for the research trainer.

    python3 research/token_routing/ds64_fast8k/merge_tasks.py --out <shards>/mt4_g64M \
        --src <shards>/contradiction_g16M <shards>/nq_g16M <shards>/outlier_g16M <shards>/oolong_g16M

``train_ctc_suite.py`` reads one ``--data`` dir (``token_ids_part_*.npy`` + ``labels_mask_*.npy`` by
glob, ``metadata.json``, ``gold_fingerprints.json``). This copies every source's npy parts under a
per-source prefix (``token_ids_part_t<i>_<orig>.npy`` / ``labels_mask_t<i>_<orig>.npy``, so the two
globs still sort into aligned pairs), unions the gold sidecars (keyed by content fingerprint; a key
present in two sources with different gold is an error), and writes a merged ``metadata.json``.
Rows from different tasks interleave at train time because the data loader shuffles instances.
Stdlib only.
"""

import argparse
import glob
import json
import os
import shutil

MUST_AGREE = ("marker_set", "doc_start_id", "doc_end_id", "query_position")
SUMMED = ("num_instances", "num_tokens", "num_loss_tokens", "num_dropped")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--name", default="mt4", help="task name written to metadata.json")
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out, "metadata.json")):
        raise SystemExit(f"{a.out} already has metadata.json; refusing to overwrite")
    os.makedirs(a.out, exist_ok=True)
    metas, gold = [], {}
    for i, src in enumerate(a.src):
        meta = json.load(open(os.path.join(src, "metadata.json")))
        metas.append(meta)
        toks = sorted(glob.glob(os.path.join(src, "token_ids_part_*.npy")))
        masks = sorted(glob.glob(os.path.join(src, "labels_mask_*.npy")))
        if len(toks) != len(masks) or not toks:
            raise SystemExit(f"{src}: {len(toks)} token parts vs {len(masks)} mask parts")
        for t, m in zip(toks, masks):
            if t.rsplit("_", 1)[-1] != m.rsplit("_", 1)[-1]:
                raise SystemExit(f"unaligned parts {t} / {m}")
            shutil.copy2(t, os.path.join(a.out, f"token_ids_part_t{i}_{os.path.basename(t)[len('token_ids_part_'):]}"))
            shutil.copy2(m, os.path.join(a.out, f"labels_mask_t{i}_{os.path.basename(m)[len('labels_mask_'):]}"))
        side = json.load(open(os.path.join(src, "gold_fingerprints.json")))
        clash = [k for k in side if k in gold and gold[k] != side[k]]
        if clash:
            raise SystemExit(f"{src}: {len(clash)} fingerprints with conflicting gold, e.g. {clash[0]}")
        gold.update(side)
        print(f"[merge] t{i} {src}: {meta.get('num_instances')} rows, {len(toks)} parts, "
              f"{len(side)} gold rows, max_example_len {meta.get('max_example_len')}")
    for k in MUST_AGREE:
        vals = {json.dumps(m.get(k)) for m in metas}
        if len(vals) > 1:
            raise SystemExit(f"sources disagree on {k}: {vals}")
    merged = dict(metas[0])
    for k in SUMMED:
        if all(k in m for m in metas):
            merged[k] = sum(int(m[k]) for m in metas)
    merged["max_example_len"] = max(int(m.get("max_example_len", 0)) for m in metas)
    merged["min_example_len"] = min(int(m.get("min_example_len", 0)) for m in metas)
    merged["task"] = a.name
    merged["merged_from"] = [{"src": s, "task": m.get("task"), "num_instances": m.get("num_instances")}
                             for s, m in zip(a.src, metas)]
    json.dump(gold, open(os.path.join(a.out, "gold_fingerprints.json"), "w"))
    json.dump(merged, open(os.path.join(a.out, "metadata.json"), "w"), indent=1)
    print(f"[merge] wrote {a.out}: {merged.get('num_instances')} rows, {len(gold)} gold rows, "
          f"max_example_len {merged['max_example_len']}")


if __name__ == "__main__":
    main()
