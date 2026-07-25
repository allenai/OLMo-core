"""
Stage A of the custom 50B "longmino-512k" mix: materialise the raw text corpus on weka.

Two parts:

* **A1** -- snapshot the whole ``allenai/dolma3_longmino_mix-50B-1025`` repo (~83 GB). We keep
  essentially all of it (the midtraining data and every long-context stratum), so there is nothing
  to sample and no reason to rewrite any files; stratum routing is a path mapping applied later.
* **A2** -- pull a seeded random sample of ``allenai/dolma3_longmino_pool`` shards from the three
  new length buckets (2e16 / 2e17 / 2e18), ~12 GB. Shard selection is done from the *listed file
  sizes* converted to a token budget via the pool's published per-bucket token totals, so no
  decompression or tokenisation is needed to decide what to download.

Downloading the text once and tokenising from it (see ``tokenize_longmino_512k.py``) means the
Qwen3 and Qwen3.5 runs read literally the same bytes, making their document sets identical by
construction rather than by after-the-fact reconciliation.

Run (via gantry, from the repo root)::

    gantry run \\
        --workspace ai2/flex2 --budget ai2/oe-other \\
        --cluster ai2/jupiter-cirrascale-2 \\
        --weka oe-training-default:/weka/oe-training-default \\
        --cpus 32 --gpus 0 --priority urgent --shared-memory 8GiB \\
        --env-secret HF_TOKEN=amandab_HF_TOKEN \\
        --env HF_HUB_ENABLE_HF_TRANSFER=1 \\
        --python-manager conda --system-python \\
        --install "pip install zstandard hf_transfer 'huggingface_hub>=0.24'" \\
        --yes \\
        -- python src/scripts/data/download_longmino_512k_raw.py
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from longmino_512k_common import (  # noqa: E402
    MIX_DATASET,
    POOL_BUCKET_TOKENS,
    POOL_DATASET,
    POOL_HEADROOM,
    POOL_TARGET_TOKENS,
    WEKA_ROOT,
    pool_bucket_for_subset,
    stratum_for_mix_subset,
    stratum_for_pool_bucket,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("download_raw")


def list_pool_shards(buckets: Tuple[str, ...], workers: int) -> Dict[str, List[Tuple[str, int]]]:
    """
    List every shard in the requested pool length buckets, with its size.

    Uses one ``list_repo_tree`` call per subset directory (the HF tree endpoint silently truncates
    at 1000 entries per directory, which ``list_repo_tree`` handles by paginating).

    :returns: Mapping of bucket -> list of ``(repo_path, size_bytes)``.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    subsets = sorted(
        {
            f.split("/")[1]
            for f in api.list_repo_files(POOL_DATASET, repo_type="dataset")
            if f.startswith("data/") and f.endswith(".jsonl.zst")
        }
    )
    wanted = [(s, pool_bucket_for_subset(s)) for s in subsets]
    wanted = [(s, b) for s, b in wanted if b in buckets]
    log.info(f"listing {len(wanted)} pool subset dirs across buckets {buckets}")

    def one(item):
        subset, bucket = item
        out = []
        for entry in api.list_repo_tree(
            POOL_DATASET, path_in_repo=f"data/{subset}", repo_type="dataset"
        ):
            size = getattr(entry, "size", None)
            if size is None or not entry.path.endswith(".jsonl.zst"):
                continue
            out.append((entry.path, size))
        return bucket, out

    by_bucket: Dict[str, List[Tuple[str, int]]] = {b: [] for b in buckets}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for bucket, entries in pool.map(one, wanted):
            by_bucket[bucket].extend(entries)
    for b, entries in by_bucket.items():
        total = sum(s for _, s in entries)
        log.info(f"  {b}: {len(entries):,} shards, {total / 1e9:.1f} GB")
    return by_bucket


def select_pool_shards(
    by_bucket: Dict[str, List[Tuple[str, int]]], target_tokens: int, headroom: float, seed: int
) -> Dict[str, dict]:
    """
    Choose which shards to download per bucket, by seeded shuffle then cumulative byte budget.

    The byte budget is derived from the bucket's published token total, so we never need to
    decompress anything to know roughly how many tokens we are taking.
    """
    selection: Dict[str, dict] = {}
    for bucket, entries in by_bucket.items():
        total_bytes = sum(s for _, s in entries)
        bytes_per_token = total_bytes / POOL_BUCKET_TOKENS[bucket]
        budget_tokens = int(target_tokens * headroom)
        budget_bytes = budget_tokens * bytes_per_token

        shuffled = sorted(entries)  # deterministic base order before shuffling
        random.Random(f"{seed}:{bucket}").shuffle(shuffled)

        chosen: List[str] = []
        acc = 0
        for path, size in shuffled:
            if acc >= budget_bytes:
                break
            chosen.append(path)
            acc += size
        selection[bucket] = {
            "stratum": stratum_for_pool_bucket(bucket),
            "n_shards": len(chosen),
            "bytes": acc,
            "bytes_per_token": bytes_per_token,
            "est_tokens": int(acc / bytes_per_token),
            "budget_tokens": budget_tokens,
            "shards": chosen,
        }
        log.info(
            f"  {bucket}: selected {len(chosen):,}/{len(shuffled):,} shards, "
            f"{acc / 1e9:.1f} GB, ~{acc / bytes_per_token / 1e9:.2f}B tokens "
            f"({100 * acc / total_bytes:.1f}% of bucket)"
        )
    return selection


def snapshot_with_retry(repo: str, local_dir: str, workers: int, attempts: int = 12):
    """
    ``snapshot_download`` the whole repo, retrying on transient hub errors.

    A single failed file (HF's Xet CAS returns 429 under sustained parallel load) aborts the entire
    ``snapshot_download`` call, which would otherwise throw away hours of progress. Each retry
    resumes: already-downloaded files are skipped, so every round only fetches what is left.
    """
    from huggingface_hub import snapshot_download

    for attempt in range(attempts):
        try:
            snapshot_download(
                repo,
                repo_type="dataset",
                local_dir=local_dir,
                allow_patterns=["data/**", "README.md"],
                max_workers=workers,
            )
            return
        except Exception as e:  # noqa: BLE001
            wait = min(300, 10 * 2**attempt)
            log.warning(
                f"snapshot_download attempt {attempt + 1}/{attempts} failed: {e}; "
                f"resuming in {wait}s"
            )
            time.sleep(wait)
    raise RuntimeError(f"snapshot_download of {repo} failed after {attempts} attempts")


def download_files(repo: str, paths: List[str], local_dir: str, workers: int, retries: int = 5):
    """Download ``paths`` from ``repo`` into ``local_dir``, preserving repo-relative layout."""
    from huggingface_hub import hf_hub_download

    os.makedirs(local_dir, exist_ok=True)
    done = {"n": 0}

    def one(path: str):
        for attempt in range(retries):
            try:
                hf_hub_download(
                    repo, path, repo_type="dataset", local_dir=local_dir, etag_timeout=60
                )
                done["n"] += 1
                if done["n"] % 200 == 0:
                    log.info(f"  {done['n']:,}/{len(paths):,} shards")
                return
            except Exception as e:  # noqa: BLE001
                wait = 2**attempt
                log.warning(
                    f"download failed ({path}) attempt {attempt + 1}: {e}; retry in {wait}s"
                )
                time.sleep(wait)
        raise RuntimeError(f"failed to download {path} after {retries} attempts")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, paths))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=WEKA_ROOT)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--target-tokens", type=int, default=POOL_TARGET_TOKENS)
    parser.add_argument("--headroom", type=float, default=POOL_HEADROOM)
    parser.add_argument("--skip-mix", action="store_true", help="Skip stage A1.")
    parser.add_argument("--skip-pool", action="store_true", help="Skip stage A2.")
    parser.add_argument(
        "--plan-only", action="store_true", help="Write selection.json but download nothing."
    )
    args = parser.parse_args()

    raw = os.path.join(args.root, "raw")
    os.makedirs(raw, exist_ok=True)

    # ---------------------------------------------------------------- A2 planning (cheap, first)
    selection_path = os.path.join(raw, "selection.json")
    selection = None
    if not args.skip_pool:
        if os.path.exists(selection_path):
            with open(selection_path) as f:
                selection = json.load(f)["buckets"]
            log.info(f"reusing existing selection from {selection_path}")
        else:
            by_bucket = list_pool_shards(tuple(POOL_BUCKET_TOKENS), args.workers)
            selection = select_pool_shards(by_bucket, args.target_tokens, args.headroom, args.seed)
            tmp = selection_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(
                    {
                        "dataset": POOL_DATASET,
                        "seed": args.seed,
                        "target_tokens": args.target_tokens,
                        "headroom": args.headroom,
                        "buckets": selection,
                    },
                    f,
                    indent=2,
                )
            os.replace(tmp, selection_path)
            log.info(f"wrote {selection_path}")

    # ------------------------------------------------------------------------- A1: snapshot mix
    if not args.skip_mix and not args.plan_only:
        mix_dir = os.path.join(raw, "mix")
        log.info(f"A1: snapshotting {MIX_DATASET} -> {mix_dir} (~83 GB, resumable)")
        snapshot_with_retry(MIX_DATASET, mix_dir, args.workers)
        log.info("A1: done")

    # --------------------------------------------------------------------- A2: sampled pool pull
    if not args.skip_pool and not args.plan_only:
        assert selection is not None
        pool_dir = os.path.join(raw, "pool")
        paths = [p for b in selection.values() for p in b["shards"]]
        log.info(f"A2: downloading {len(paths):,} pool shards -> {pool_dir}")
        download_files(POOL_DATASET, paths, pool_dir, args.workers)
        log.info("A2: done")

    # ---------------------------------------------------------------------------- strata mapping
    strata: Dict[str, str] = {}
    mix_data = os.path.join(raw, "mix", "data")
    if os.path.isdir(mix_data):
        for subset in sorted(os.listdir(mix_data)):
            if os.path.isdir(os.path.join(mix_data, subset)):
                strata[f"mix/data/{subset}"] = stratum_for_mix_subset(subset)
    pool_data = os.path.join(raw, "pool", "data")
    if os.path.isdir(pool_data):
        for subset in sorted(os.listdir(pool_data)):
            bucket = pool_bucket_for_subset(subset)
            if bucket is not None:
                strata[f"pool/data/{subset}"] = stratum_for_pool_bucket(bucket)
    strata_path = os.path.join(raw, "strata.json")
    with open(strata_path + ".tmp", "w") as f:
        json.dump(strata, f, indent=2, sort_keys=True)
    os.replace(strata_path + ".tmp", strata_path)
    log.info(f"wrote {strata_path} ({len(strata)} source dirs)")
    log.info("STAGE A COMPLETE")


if __name__ == "__main__":
    main()
