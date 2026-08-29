"""Drop over-length instances from an already-tokenised SFT shard, in TOKEN space.

Why this exists. At 128k the choice of ``--seq-len`` is boxed in from three sides:

* ``--pack`` (mandatory for the mix arms -- without it PadToLength pads every 2k example up to
  the full window, a ~64x compute waste) asserts a **power-of-2** ``--seq-len``
  (``train_ctc_suite.py``), so the only usable dense value near 128k is exactly 131072.
* ``read_shard_metadata`` refuses a ``--seq-len`` below the shard's ``max_example_len``.
* the n=880 pool has a thin tail above 131072 (median is 128,335, but the max runs ~0.5% over).

Re-tokenising an arm to ``--max-seq-len 131072`` would cost hours; the shards are headerless raw
uint32 + raw bool written by ``ndarray.tofile()``, so the same filtering is exact and cheap in
token space: split on EOS, drop the long instances, re-concatenate.

This DROPS data, so it is opt-in per shard and always reports what it removed. The pure-128k arms
do not need it (they can run un-packed at ``--seq-len 132096``, where padding waste is ~3%).
"""

import argparse
import json
import pathlib
import shutil

import numpy as np

EOS = 248044


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True, help="shard dir (token_ids_part_000000.npy + labels_mask_000000.npy)")
    ap.add_argument("--cap", type=int, default=131072, help="drop instances longer than this")
    ap.add_argument("--backup", action="store_true", help="keep the pre-cap shard as <dir>.uncapped")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    d = pathlib.Path(args.shard)

    meta = json.load(open(d / "metadata.json"))
    ids = np.fromfile(d / "token_ids_part_000000.npy", dtype=np.uint32)
    mask = np.fromfile(d / "labels_mask_000000.npy", dtype=bool)
    assert ids.size == mask.size, (ids.size, mask.size)

    ends = np.flatnonzero(ids == EOS)
    starts = np.concatenate(([0], ends[:-1] + 1))
    lens = ends - starts + 1
    assert lens.size == meta["num_instances"], (lens.size, meta["num_instances"])

    keep = lens <= args.cap
    n_drop = int((~keep).sum())
    print(f"[{d.name}] {lens.size} instances, max {int(lens.max()):,}, cap {args.cap:,} "
          f"-> drop {n_drop} ({100*n_drop/lens.size:.2f}%)")
    if n_drop:
        print(f"    dropped lengths: {sorted(int(x) for x in lens[~keep])}")
    if args.dry_run or n_drop == 0:
        print("    (no write)" if n_drop == 0 else "    (dry run)")
        return

    if args.backup:
        shutil.copytree(d, d.with_suffix(d.suffix + ".uncapped"), dirs_exist_ok=True)
    sel = np.concatenate([np.arange(s, e + 1) for s, e, k in zip(starts, ends, keep) if k])
    ids[sel].tofile(d / "token_ids_part_000000.npy")
    mask[sel].tofile(d / "labels_mask_000000.npy")

    new_lens = lens[keep]
    meta.update(num_instances=int(keep.sum()), num_tokens=int(sel.size),
                num_loss_tokens=int(mask[sel].sum()), median_len=int(np.median(new_lens)),
                max_len=int(new_lens.max()), max_example_len=int(new_lens.max()),
                capped_at=args.cap, num_dropped_over_cap=n_drop)
    json.dump(meta, open(d / "metadata.json", "w"), indent=2)
    print(f"    -> {meta['num_instances']} instances, {meta['num_tokens']:,} tokens, "
          f"max_example_len {meta['max_example_len']:,}")


if __name__ == "__main__":
    main()
