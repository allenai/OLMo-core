"""
Stream a ~15B-token proportional sample of the ``allenai/dolma3_longmino_mix-100B-1125``
HuggingFace dataset, tokenize it with the Qwen3.5 tokenizer, and write raw uint32 token arrays
(EOS-separated documents) to weka -- in the same on-disk format the OLMo-core
``NumpyDocumentSource`` expects (raw memmap, no header).

Designed to run on a Beaker CPU node with the weka bucket mounted. The dataset is ~470 GB of
zstd-compressed JSONL spread over ~59k shards, but we only ever keep a tiny working set on local
disk: a bounded queue of downloaded shards (each ~5-8 MB) plus an in-memory token buffer that is
flushed to weka as ~2 GB part files. Total local disk stays well under 10 GB.

Sampling: shards are shuffled with a fixed seed and streamed in that order until the token target is
reached. Because each source contributes shards roughly in proportion to its size, this yields an
approximately proportional sample of the full mix.

The job is resumable: a ``progress.json`` on weka records which shards have been consumed, the
running token count, and the next part index, so a preempted job picks up where it left off.

Run (via gantry, from the repo root)::

    gantry run \\
        --workspace ai2/flex2 --budget ai2/oe-other \\
        --cluster ai2/jupiter-cirrascale-2 \\
        --weka oe-training-default:/weka/oe-training-default \\
        --cpus 64 --gpus 0 --priority urgent --shared-memory 32GiB \\
        --env-secret HF_TOKEN=amandab_HF_TOKEN \\
        --env HF_HUB_ENABLE_HF_TRANSFER=1 --env TOKENIZERS_PARALLELISM=true \\
        --install "pip install zstandard hf_transfer 'huggingface_hub>=0.24' tokenizers numpy" \\
        --yes \\
        -- python src/scripts/data/tokenize_dolma3_longmino_sample.py
"""

import argparse
import io
import json
import logging
import os
import queue
import random
import tempfile
import threading
import time
from typing import List, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("tokenize_sample")

DATASET = "allenai/dolma3_longmino_mix-100B-1125"
# Qwen3.5 tokenizer. Documents are separated by the <|endoftext|> token (NOT the tokenizer's chat
# eos_token <|im_end|>); its id is resolved from the tokenizer at runtime in main(). This matches
# olmo_core.data.TokenizerConfig.qwen3_5() (eos/bos/pad = 248044 for Qwen3.5-0.8B).
TOKENIZER = "Qwen/Qwen3.5-0.8B"
SEP_TOKEN = "<|endoftext|>"
DTYPE = np.uint32

DEFAULT_OUT = (
    "/weka/oe-training-default/ai2-llm/checkpoints/amandab/"
    "dolma3_longmino_mix_sample15B_qwen3_5"
)


def list_shards(seed: int) -> List[str]:
    from huggingface_hub import HfApi

    files = [
        f
        for f in HfApi().list_repo_files(DATASET, repo_type="dataset")
        if f.endswith(".jsonl.zst")
    ]
    files.sort()  # deterministic base order before shuffling
    random.Random(seed).shuffle(files)
    return files


def download_shard(repo_file: str, dest_dir: str, token: Optional[str], retries: int = 5) -> str:
    """Download a single shard via direct HTTP (no HF cache) to ``dest_dir``; return local path."""
    import requests
    from huggingface_hub import hf_hub_url

    url = hf_hub_url(DATASET, repo_file, repo_type="dataset")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    dest = os.path.join(dest_dir, repo_file.replace("/", "__"))
    for attempt in range(retries):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(dest, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
            return dest
        except Exception as e:  # noqa: BLE001
            wait = 2**attempt
            log.warning(f"download failed ({repo_file}) attempt {attempt + 1}: {e}; retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"failed to download {repo_file} after {retries} attempts")


def iter_texts(shard_path: str):
    import zstandard as zstd

    dctx = zstd.ZstdDecompressor()
    with open(shard_path, "rb") as fh, dctx.stream_reader(fh) as reader:
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            text = json.loads(line).get("text")
            if text:
                yield text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument(
        "--tokenizer",
        default=TOKENIZER,
        help="HF tokenizer to use. Default is the Qwen3.5 tokenizer; pass e.g. 'Qwen/Qwen3-0.6B' "
        "to produce shards compatible with the Qwen3-4B (eos 151643) model line. The doc "
        "separator '<|endoftext|>' is resolved per-tokenizer.",
    )
    parser.add_argument("--target-tokens", type=int, default=15_000_000_000)
    parser.add_argument("--flush-tokens", type=int, default=500_000_000)  # ~2 GB part files
    parser.add_argument("--batch-docs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--prefetch", type=int, default=16)
    parser.add_argument(
        "--dry-run-shards", type=int, default=0, help="If >0, process at most N shards to a local dir."
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, "progress.json")

    processed = set()
    total_tokens = 0
    next_part = 0
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            prog = json.load(f)
        processed = set(prog.get("processed_shards", []))
        total_tokens = int(prog.get("total_tokens", 0))
        next_part = int(prog.get("next_part", 0))
        log.info(
            f"Resuming: {len(processed):,} shards done, {total_tokens:,} tokens, next_part={next_part}"
        )

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    assert tok.vocab_size <= np.iinfo(DTYPE).max
    eos_token_id = tok.convert_tokens_to_ids(SEP_TOKEN)
    assert eos_token_id is not None and eos_token_id != tok.unk_token_id, (
        f"tokenizer {args.tokenizer!r} has no {SEP_TOKEN!r} token"
    )
    assert eos_token_id <= np.iinfo(DTYPE).max
    log.info(
        f"Using tokenizer {args.tokenizer!r} (vocab {tok.vocab_size:,}, "
        f"doc separator {SEP_TOKEN!r}={eos_token_id})"
    )

    shards = list_shards(args.seed)
    todo = [s for s in shards if s not in processed]
    log.info(f"{len(shards):,} total shards; {len(todo):,} remaining; target {args.target_tokens:,} tokens")

    scratch = tempfile.mkdtemp(prefix="dolma3_shards_")
    dl_queue: "queue.Queue" = queue.Queue(maxsize=args.prefetch)
    stop = threading.Event()

    # Parallel downloaders feed a bounded queue. The bound (``--prefetch``) caps how many shards sit
    # on local disk at once (each ~5-8 MB), keeping the working set tiny; backpressure pauses
    # downloads when the consumer falls behind.
    work_idx = {"i": 0}
    work_lock = threading.Lock()

    def downloader():
        while not stop.is_set():
            with work_lock:
                i = work_idx["i"]
                work_idx["i"] += 1
            if i >= len(todo):
                return
            repo_file = todo[i]
            try:
                path = download_shard(repo_file, scratch, token)
                dl_queue.put((repo_file, path))
            except Exception as e:  # noqa: BLE001
                log.error(f"giving up on shard {repo_file}: {e}")

    n_dl = max(1, args.prefetch)
    dl_threads = [threading.Thread(target=downloader, daemon=True) for _ in range(n_dl)]
    for t in dl_threads:
        t.start()

    def joiner():
        for t in dl_threads:
            t.join()
        dl_queue.put(None)  # single sentinel once all downloaders are done

    threading.Thread(target=joiner, daemon=True).start()

    buffer: List[np.ndarray] = []
    buffered_tokens = 0

    def flush():
        nonlocal buffer, buffered_tokens, next_part, total_tokens
        if buffered_tokens == 0:
            return
        arr = np.concatenate(buffer)
        part = os.path.join(args.out_dir, f"part-{next_part:05d}.npy")
        tmp = part + ".tmp"
        arr.tofile(tmp)  # raw uint32, no header (matches NumpyDocumentSource memmap format)
        os.replace(tmp, part)
        total_tokens += int(arr.size)
        log.info(f"wrote {part} ({arr.size:,} tokens); total {total_tokens:,}")
        next_part += 1
        buffer = []
        buffered_tokens = 0
        with open(progress_path + ".tmp", "w") as f:
            json.dump(
                {
                    "processed_shards": sorted(processed),
                    "total_tokens": total_tokens,
                    "next_part": next_part,
                    "target_tokens": args.target_tokens,
                    "dataset": DATASET,
                    "tokenizer": args.tokenizer,
                    "eos_token_id": eos_token_id,
                    "dtype": "uint32",
                },
                f,
            )
        os.replace(progress_path + ".tmp", progress_path)

    def encode_batch(texts: List[str]):
        nonlocal buffered_tokens
        enc = tok(texts, add_special_tokens=False)["input_ids"]
        flat: List[int] = []
        for ids in enc:
            flat.extend(ids)
            flat.append(eos_token_id)
        if flat:
            arr = np.asarray(flat, dtype=DTYPE)
            buffer.append(arr)
            buffered_tokens += arr.size

    n_shards_done = 0
    done = total_tokens >= args.target_tokens
    while not done:
        item = dl_queue.get()
        if item is None:
            break
        repo_file, path = item
        try:
            batch: List[str] = []
            for text in iter_texts(path):
                batch.append(text)
                if len(batch) >= args.batch_docs:
                    encode_batch(batch)
                    batch = []
            if batch:
                encode_batch(batch)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        processed.add(repo_file)
        n_shards_done += 1

        if buffered_tokens >= args.flush_tokens:
            flush()
        if total_tokens + buffered_tokens >= args.target_tokens:
            done = True
        if args.dry_run_shards and n_shards_done >= args.dry_run_shards:
            done = True

    stop.set()
    flush()
    # drain producer / queue so its thread can exit cleanly
    try:
        while True:
            leftover = dl_queue.get_nowait()
            if leftover is not None:
                os.remove(leftover[1])
    except (queue.Empty, OSError):
        pass

    with open(progress_path + ".tmp", "w") as f:
        json.dump(
            {
                "processed_shards": sorted(processed),
                "total_tokens": total_tokens,
                "next_part": next_part,
                "target_tokens": args.target_tokens,
                "dataset": DATASET,
                "tokenizer": args.tokenizer,
                "eos_token_id": eos_token_id,
                "dtype": "uint32",
                "complete": total_tokens >= args.target_tokens,
            },
            f,
        )
    os.replace(progress_path + ".tmp", progress_path)
    log.info(
        f"DONE: {total_tokens:,} tokens across {next_part} part files in {args.out_dir} "
        f"({len(processed):,} shards consumed)"
    )
    try:
        os.rmdir(scratch)
    except OSError:
        pass


if __name__ == "__main__":
    main()
