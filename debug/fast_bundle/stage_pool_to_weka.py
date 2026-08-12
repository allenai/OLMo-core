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

    staging = dest.parent / "_pool_staging"
    staging.mkdir(parents=True, exist_ok=True)
    beaker.dataset.fetch(dataset, target=staging, force=True)

    files = sorted(p for p in staging.rglob("*.pkl") if p.is_file())
    if not files:
        files = sorted(p for p in staging.rglob("*") if p.is_file())
    if len(files) != 1:
        print(f"expected one file in the dataset, found {len(files)}: {files}", file=sys.stderr)
        return 1

    # Move into place only once the bytes are down, so a killed job never leaves a truncated pickle
    # that every later build would load and fail on in a confusing way.
    shutil.move(str(files[0]), str(dest))
    shutil.rmtree(staging, ignore_errors=True)
    print(f"staged {dest} ({dest.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
