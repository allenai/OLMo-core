"""
Stage B/C of the custom 50B "longmino-512k" mix: tokenise the staged raw text into per-stratum
``part-*.npy`` trees.

Reads the corpus written by ``download_longmino_512k_raw.py`` from weka and writes raw ``uint32``
token arrays (EOS-separated documents, no header) in the on-disk format
:class:`~olmo_core.data.composable.NumpyDocumentSource` expects. Run once per tokenizer; the runs
are independent and can go in parallel. Because both read the same staged files and share the
routing/filter rules in ``longmino_512k_common.py``, their document sets are identical by
construction.

Output layout (mirrors the strata, per-stratum and per-length)::

    <out>/midtrain/<family>/part-*.npy
    <out>/lc/real_s2pdf/{2e13,2e14,2e15,2e16,2e17,2e18}/part-*.npy
    <out>/lc/synth_rex/2e15/part-*.npy
    <out>/lc/synth_cwe/2e15/part-*.npy
    <out>/progress.json
    <out>/token_counts.json

Documents are **never truncated**: no ``max_length``/``truncation`` is passed to the tokenizer.
Batching is by total characters rather than document count, because a 256k-512k document batched
1000-at-a-time would put ~500M tokens in a single encode call.

Run (via gantry, from the repo root)::

    gantry run \\
        --workspace ai2/flex2 --budget ai2/oe-other \\
        --cluster ai2/jupiter-cirrascale-2 \\
        --weka oe-training-default:/weka/oe-training-default \\
        --cpus 64 --gpus 0 --priority urgent --shared-memory 32GiB \\
        --env TOKENIZERS_PARALLELISM=true \\
        --python-manager conda --system-python \\
        --install "pip install zstandard 'transformers>=4.40' numpy" \\
        --yes \\
        -- python src/scripts/data/tokenize_longmino_512k.py \\
            --tokenizer Qwen/Qwen3-0.6B --out-name qwen3
"""

import argparse
import io
import json
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from longmino_512k_common import (  # noqa: E402
    WEKA_ROOT,
    is_usable_text,
    pool_bucket_for_subset,
    stratum_for_mix_subset,
    stratum_for_pool_bucket,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("tokenize")

DTYPE = np.uint32
SEP_TOKEN = "<|endoftext|>"


def collect_sources(raw_root: str) -> Dict[str, List[str]]:
    """
    Walk the staged corpus and group shard paths by output stratum.

    File order within a stratum is sorted, so both tokenizer runs consume documents in the same
    order and their part files line up.
    """
    by_stratum: Dict[str, List[str]] = defaultdict(list)

    mix_data = os.path.join(raw_root, "mix", "data")
    if os.path.isdir(mix_data):
        for subset in sorted(os.listdir(mix_data)):
            subset_dir = os.path.join(mix_data, subset)
            if not os.path.isdir(subset_dir):
                continue
            stratum = stratum_for_mix_subset(subset)
            for fn in sorted(os.listdir(subset_dir)):
                if fn.endswith(".jsonl.zst"):
                    by_stratum[stratum].append(os.path.join(subset_dir, fn))

    pool_data = os.path.join(raw_root, "pool", "data")
    if os.path.isdir(pool_data):
        for subset in sorted(os.listdir(pool_data)):
            subset_dir = os.path.join(pool_data, subset)
            if not os.path.isdir(subset_dir):
                continue
            bucket = pool_bucket_for_subset(subset)
            if bucket is None:
                continue
            stratum = stratum_for_pool_bucket(bucket)
            for fn in sorted(os.listdir(subset_dir)):
                if fn.endswith(".jsonl.zst"):
                    by_stratum[stratum].append(os.path.join(subset_dir, fn))

    return {k: sorted(v) for k, v in sorted(by_stratum.items())}


def iter_texts(shard_path: str):
    """
    Yield usable ``text`` fields from a zstd-compressed JSONL shard.

    A truncated or corrupt shard yields whatever it decoded before the error rather than killing a
    job that is part-way through tens of thousands of shards.
    """
    import zstandard as zstd

    dctx = zstd.ZstdDecompressor()
    try:
        with open(shard_path, "rb") as fh, dctx.stream_reader(fh) as reader:
            for line in io.TextIOWrapper(reader, encoding="utf-8", errors="replace"):
                line = line.strip()
                if not line:
                    continue
                try:
                    text = json.loads(line).get("text")
                except json.JSONDecodeError:
                    log.warning(f"skipping unparseable line in {shard_path}")
                    continue
                if is_usable_text(text):
                    yield text
    except (zstd.ZstdError, OSError) as e:
        log.error(f"truncated/corrupt shard {shard_path}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=WEKA_ROOT)
    parser.add_argument("--raw-dir", default=None, help="Defaults to <root>/raw")
    parser.add_argument("--out-name", required=True, help="Output subdir, e.g. 'qwen3' or 'qwen35'")
    parser.add_argument("--tokenizer", required=True, help="e.g. Qwen/Qwen3-0.6B")
    parser.add_argument("--flush-tokens", type=int, default=500_000_000, help="~2 GB part files")
    parser.add_argument("--batch-chars", type=int, default=4_000_000)
    parser.add_argument("--only-stratum", default=None, help="Substring filter, for smoke tests.")
    parser.add_argument("--max-files-per-stratum", type=int, default=0, help="0 = no limit.")
    args = parser.parse_args()

    raw_root = args.raw_dir or os.path.join(args.root, "raw")
    out_dir = os.path.join(args.root, args.out_name)
    os.makedirs(out_dir, exist_ok=True)
    progress_path = os.path.join(out_dir, "progress.json")

    processed = set()
    stats: Dict[str, dict] = {}
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            prog = json.load(f)
        processed = set(prog.get("processed", []))
        stats = prog.get("strata", {})
        log.info(f"resuming: {len(processed):,} shards already done")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    # Documents are never truncated; raise the limit so the fast tokenizer stops warning about it.
    tok.model_max_length = int(1e12)
    eos_token_id = tok.convert_tokens_to_ids(SEP_TOKEN)
    if eos_token_id is None or eos_token_id == tok.unk_token_id:
        raise SystemExit(f"tokenizer {args.tokenizer!r} has no {SEP_TOKEN!r} token")
    if eos_token_id > np.iinfo(DTYPE).max:
        raise SystemExit(f"separator id {eos_token_id} does not fit in {DTYPE}")
    log.info(f"tokenizer {args.tokenizer!r}: separator {SEP_TOKEN!r}={eos_token_id}")

    sources = collect_sources(raw_root)
    if args.only_stratum:
        sources = {k: v for k, v in sources.items() if args.only_stratum in k}
    if args.max_files_per_stratum:
        sources = {k: v[: args.max_files_per_stratum] for k, v in sources.items()}
    total_files = sum(len(v) for v in sources.values())
    log.info(f"{len(sources)} strata, {total_files:,} shards under {raw_root}")
    for stratum, files in sources.items():
        log.info(f"  {stratum:44s} {len(files):6,d} shards")

    def save_progress():
        tmp = progress_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(
                {
                    "tokenizer": args.tokenizer,
                    "eos_token_id": eos_token_id,
                    "dtype": "uint32",
                    "raw_dir": raw_root,
                    "processed": sorted(processed),
                    "strata": stats,
                },
                f,
            )
        os.replace(tmp, progress_path)

    t_start = time.time()
    for stratum, files in sources.items():
        todo = [f for f in files if f not in processed]
        st = stats.setdefault(stratum, {"tokens": 0, "docs": 0, "next_part": 0})
        if not todo:
            log.info(f"{stratum}: already complete ({st['tokens']:,} tokens)")
            continue

        stratum_dir = os.path.join(out_dir, stratum)
        os.makedirs(stratum_dir, exist_ok=True)
        log.info(f"{stratum}: {len(todo):,} shards to go")

        buffer: List[np.ndarray] = []
        buffered = 0
        pending: List[str] = []

        def flush():
            nonlocal buffer, buffered, pending
            if buffered == 0:
                # Still record consumed-but-empty shards so we do not re-read them on resume.
                processed.update(pending)
                pending = []
                return
            arr = np.concatenate(buffer)
            part = os.path.join(stratum_dir, f"part-{st['next_part']:05d}.npy")
            tmp = part + ".tmp"
            arr.tofile(tmp)  # raw uint32, no header
            os.replace(tmp, part)
            st["tokens"] += int(arr.size)
            st["next_part"] += 1
            # Only mark shards done once their tokens are durably on disk.
            processed.update(pending)
            pending = []
            buffer = []
            buffered = 0
            save_progress()
            log.info(
                f"  {stratum}: wrote {os.path.basename(part)} ({arr.size:,} tok); "
                f"stratum total {st['tokens']:,}"
            )

        batch: List[str] = []
        batch_chars = 0

        def encode_batch():
            nonlocal batch, batch_chars, buffered
            if not batch:
                return
            enc = tok(batch, add_special_tokens=False)["input_ids"]
            for ids in enc:
                arr = np.empty(len(ids) + 1, dtype=DTYPE)
                arr[:-1] = ids
                arr[-1] = eos_token_id
                buffer.append(arr)
                buffered += arr.size
                st["docs"] += 1
            batch = []
            batch_chars = 0

        for i, shard in enumerate(todo):
            for text in iter_texts(shard):
                batch.append(text)
                batch_chars += len(text)
                if batch_chars >= args.batch_chars:
                    encode_batch()
            pending.append(shard)
            if buffered >= args.flush_tokens:
                encode_batch()
                flush()
            if (i + 1) % 500 == 0:
                log.info(
                    f"  {stratum}: {i + 1:,}/{len(todo):,} shards, "
                    f"{st['tokens'] + buffered:,} tok, {time.time() - t_start:.0f}s elapsed"
                )
        encode_batch()
        flush()
        log.info(f"{stratum}: DONE {st['tokens']:,} tokens, {st['docs']:,} docs")

    save_progress()
    counts_path = os.path.join(out_dir, "token_counts.json")
    with open(counts_path + ".tmp", "w") as f:
        json.dump(
            {
                "tokenizer": args.tokenizer,
                "eos_token_id": eos_token_id,
                "total_tokens": sum(v["tokens"] for v in stats.values()),
                "total_docs": sum(v["docs"] for v in stats.values()),
                "strata": stats,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    os.replace(counts_path + ".tmp", counts_path)
    total = sum(v["tokens"] for v in stats.values())
    log.info(f"COMPLETE: {total:,} tokens across {len(stats)} strata -> {out_dir}")


if __name__ == "__main__":
    main()
