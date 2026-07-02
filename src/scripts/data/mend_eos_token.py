"""
In-place "mend" of the document-separator token in already-written raw uint32 ``.npy`` token
files produced by ``tokenize_dolma3_longmino_sample.py``.

Background: an earlier run of the tokenizer separated documents with the Qwen3.5 tokenizer's *chat*
``eos_token`` ``<|im_end|>`` (id 248046) instead of the intended pretraining document separator
``<|endoftext|>`` (id 248044, which is what ``TokenizerConfig.qwen3_5()`` expects). This script walks
every ``part-*.npy`` in the output folder and rewrites all occurrences of the old id to the new id,
editing the files in place (memory-mapped, processed in bounded chunks so a ~2 GB part file never has
to be fully loaded).

The swap is idempotent and resumable: re-running only ever turns remaining OLD ids into NEW, so a
job interrupted mid-folder can simply be run again. It also (optionally) fixes the ``eos_token_id``
field recorded in ``progress.json``.

NOTE: only mend files that are *finished* being written. If the tokenization job is still running and
appending new (still-wrong) part files, stop it first -- otherwise newly written parts will also need
mending. Once the tokenization script itself is fixed (uses ``<|endoftext|>``), any parts it writes
afterwards are already correct and don't need this.

Run locally (weka mounted) or via gantry on a CPU node::

    python src/scripts/data/mend_eos_token.py \\
        --out-dir /weka/oe-training-default/ai2-llm/checkpoints/amandab/dolma3_longmino_mix_sample15B_qwen3_5

Use ``--dry-run`` first to count occurrences without modifying anything.
"""

import argparse
import glob
import json
import logging
import os
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("mend_eos")

DTYPE = np.uint32
# Old (incorrect) separator written by the buggy run: Qwen3.5 chat eos <|im_end|>.
DEFAULT_OLD_EOS = 248046
# New (correct) pretraining document separator: Qwen3.5 <|endoftext|> (matches TokenizerConfig.qwen3_5()).
DEFAULT_NEW_EOS = 248044

DEFAULT_OUT = (
    "/weka/oe-training-default/ai2-llm/checkpoints/amandab/"
    "dolma3_longmino_mix_sample15B_qwen3_5"
)


def mend_file(path: str, old_eos: int, new_eos: int, chunk: int, dry_run: bool) -> int:
    """Swap ``old_eos`` -> ``new_eos`` in a raw uint32 ``.npy`` file in place. Returns swap count."""
    nbytes = os.path.getsize(path)
    if nbytes % DTYPE().itemsize != 0:
        raise ValueError(f"{path}: size {nbytes} not a multiple of {DTYPE().itemsize}; not raw uint32?")

    mode = "r" if dry_run else "r+"
    mm = np.memmap(path, dtype=DTYPE, mode=mode)
    n = mm.shape[0]
    swapped = 0
    try:
        for i in range(0, n, chunk):
            block = mm[i : i + chunk]
            mask = block == old_eos
            c = int(mask.sum())
            if c and not dry_run:
                block[mask] = new_eos
            swapped += c
        if swapped and not dry_run:
            mm.flush()
    finally:
        del mm  # close the memmap / release the handle
    return swapped


def update_progress(progress_path: str, old_eos: int, new_eos: int, dry_run: bool) -> None:
    if not os.path.exists(progress_path):
        return
    with open(progress_path) as f:
        prog = json.load(f)
    if prog.get("eos_token_id") != old_eos:
        log.info(
            f"progress.json eos_token_id={prog.get('eos_token_id')} (not {old_eos}); leaving as-is"
        )
        return
    if dry_run:
        log.info(f"[dry-run] would set progress.json eos_token_id {old_eos} -> {new_eos}")
        return
    prog["eos_token_id"] = new_eos
    prog["eos_token_id_mended_from"] = old_eos
    tmp = progress_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(prog, f)
    os.replace(tmp, progress_path)
    log.info(f"updated progress.json eos_token_id {old_eos} -> {new_eos}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--old-eos", type=int, default=DEFAULT_OLD_EOS)
    parser.add_argument("--new-eos", type=int, default=DEFAULT_NEW_EOS)
    parser.add_argument(
        "--chunk", type=int, default=1 << 26, help="elements per chunk (default 64M = 256 MB)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="count occurrences without modifying files"
    )
    parser.add_argument(
        "--glob", default="part-*.npy", help="filename glob within --out-dir to mend"
    )
    args = parser.parse_args()

    assert args.old_eos != args.new_eos, "old and new EOS ids are identical; nothing to do"
    for v in (args.old_eos, args.new_eos):
        assert 0 <= v <= np.iinfo(DTYPE).max, f"{v} out of uint32 range"

    paths = sorted(glob.glob(os.path.join(args.out_dir, args.glob)))
    if not paths:
        log.warning(f"no files matching {args.glob!r} in {args.out_dir}")
        return

    log.info(
        f"{'[dry-run] ' if args.dry_run else ''}mending {len(paths)} file(s) in {args.out_dir}: "
        f"swap {args.old_eos} -> {args.new_eos}"
    )

    total = 0
    for p in paths:
        c = mend_file(p, args.old_eos, args.new_eos, args.chunk, args.dry_run)
        total += c
        log.info(f"{'would swap' if args.dry_run else 'swapped'} {c:,} in {os.path.basename(p)}")

    update_progress(os.path.join(args.out_dir, "progress.json"), args.old_eos, args.new_eos, args.dry_run)

    log.info(
        f"DONE: {'would swap' if args.dry_run else 'swapped'} {total:,} occurrences of "
        f"{args.old_eos} -> {args.new_eos} across {len(paths)} file(s)"
    )


if __name__ == "__main__":
    main()
