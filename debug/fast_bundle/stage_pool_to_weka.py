"""Copy the wiki100w article pool from a Beaker dataset onto weka. Runs ON a Beaker node.

The planted outlier build needs the article pool, the pool is a 431 MB pickle that lives on the
Berkeley cluster, and weka is not mountable from there -- so it goes up as a Beaker dataset once and
this lands it beside the other corpora. Every rung build then reads it from weka like any other
source.

The image has no ``beaker`` CLI but does ship ``beaker-py`` and a ``BEAKER_TOKEN``, which is what
this uses. Run it once; it is a no-op when the destination already exists.

    python debug/fast_bundle/stage_pool_to_weka.py <dataset-id> [--dest PATH]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

DEST_DEFAULT = (
    "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/cr_suite_data/"
    "wiki100w_article_pool.pkl"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset")
    ap.add_argument("--dest", default=DEST_DEFAULT)
    ap.add_argument("--workspace", default="ai2/flex2")
    args = ap.parse_args()

    dest = Path(args.dest)
    if dest.exists():
        print(f"already staged: {dest} ({dest.stat().st_size:,} bytes)")
        return 0

    from beaker import Beaker

    beaker = Beaker.from_env(default_workspace=args.workspace)
    dataset = beaker.dataset.get(args.dataset)
    print(f"dataset {dataset.id}")

    # beaker-py 2.x has no `fetch`; files are listed and streamed one at a time.
    files = list(beaker.dataset.list_files(dataset))
    print(f"{len(files)} file(s): {[getattr(f, 'path', f) for f in files]}")
    if len(files) != 1:
        print("expected exactly one file in the dataset", file=sys.stderr)
        return 1

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    written = 0
    with open(partial, "wb") as handle:
        # stream_file takes the file's PATH, not the DatasetFile object it was listed as.
        for chunk in beaker.dataset.stream_file(dataset, files[0].path):
            handle.write(chunk)
            written += len(chunk)
    print(f"downloaded {written:,} bytes")

    # Rename into place only once the bytes are down: a killed job must never leave a truncated
    # pickle behind, which every later build would load and fail on in a confusing way.
    shutil.move(str(partial), str(dest))
    print(f"staged {dest} ({dest.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
