"""Post-build step for the 128k arms: write the seq-len guard key into each shard's metadata.json.

`convert_unified_to_sft.py` historically wrote only `max_len`, but the ctc_suite trainers guard
`--seq-len` with ``meta.get("max_example_len", 0) > seq_len`` -- so the guard silently defaulted to
0 and a too-small `--seq-len` made PadToLength skip the long examples instead of failing the launch.
The converter now emits `max_example_len`; this script backfills it (plus the exact `median_len`
that the chunked-tokenisation merge could not compute) for shards built before that fix, and
records the two capacity counts that matter at 128k:

  over_131072  -- examples that do not fit a seq_len=131072 dense window
  over_129024  -- examples that do not fit sparselandmark's content capacity at seq_len=131072
                  (2048 blocks x 63 content tokens); the landmark packer DROPS these with only a
                  log warning, so they must be counted before a launch, not after.
"""

import argparse
import json
import math
import pathlib

import numpy as np

EOS = 248044
LANDMARK_CONTENT_AT_131072 = 131072 * 63 // 64  # 129024


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/data/prasann/outlier_lengthmix")
    ap.add_argument("--shards", nargs="+", required=True,
                    help="shard dirs relative to --work, e.g. arms_tokenized/p128k_500")
    args = ap.parse_args()
    work = pathlib.Path(args.work)

    summary = {}
    for rel in args.shards:
        d = work / rel
        meta = json.load(open(d / "metadata.json"))
        ids = np.fromfile(d / "token_ids_part_000000.npy", dtype=np.uint32)
        lens = np.diff(np.concatenate(([-1], np.flatnonzero(ids == EOS))))
        assert lens.size == meta["num_instances"], (rel, lens.size, meta["num_instances"])
        meta["median_len"] = int(np.median(lens))
        meta["max_len"] = int(lens.max())
        meta["max_example_len"] = int(lens.max())
        meta["over_131072"] = int((lens > 131072).sum())
        meta["over_129024_landmark_content"] = int((lens > LANDMARK_CONTENT_AT_131072).sum())
        json.dump(meta, open(d / "metadata.json", "w"), indent=2)
        summary[d.name] = {k: meta[k] for k in
                           ("num_instances", "num_tokens", "median_len", "max_example_len",
                            "over_131072", "over_129024_landmark_content")}
        print(f"[{d.name}] " + json.dumps(summary[d.name]))

    # Smallest multiple of 1024 that is also a multiple of 64 (landmark) and clears every example,
    # for dense (needs seq_len >= max) and landmark (needs seq_len*63/64 >= max).
    mx = max(v["max_example_len"] for v in summary.values())
    dense = math.ceil(mx / 1024) * 1024
    # landmark: lm_len = ceil(content/63)*64 must be <= seq_len (LandmarkPackingInstanceSource
    # ._landmark_len), and seq_len must be a multiple of the block size 64.
    lm = math.ceil(math.ceil(mx / 63) * 64 / 1024) * 1024
    assert math.ceil(mx / 63) * 64 <= lm and lm % 64 == 0
    print(f"\nmax_example_len across shards = {mx:,}")
    print(f"  dense           : --seq-len {dense} (>= {mx:,})")
    print(f"  sparselandmark  : --seq-len {lm} (content capacity {lm // 64 * 63:,} >= {mx:,})")
    (work / "SEQLEN_128k.json").write_text(json.dumps(
        {"shards": summary, "max_example_len": mx,
         "recommended_seq_len_dense": dense, "recommended_seq_len_sparselandmark": lm}, indent=2))


if __name__ == "__main__":
    main()
