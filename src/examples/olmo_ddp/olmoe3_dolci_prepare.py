"""Reuse audited Dolci tokenization, restore shard-spanning records, and pack identically."""

import gzip
import hashlib
import json
import os
import shutil
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from olmoe3_dolci_hero_plan import AUTO, DATA, INPUT
from olmoe3_lr_sweep_watch import atomic_json


def ordered_chunks(paths, workers=8, chunk_bytes=128 * 1024 * 1024):
    """Prefetch bounded ranges, preserving exact source-file and byte order."""
    ranges = iter(
        (path, offset, min(chunk_bytes, path.stat().st_size - offset))
        for path in paths
        for offset in range(0, path.stat().st_size, chunk_bytes)
    )

    def read_range(item):
        path, offset, size = item
        with path.open("rb") as handle:
            handle.seek(offset)
            data = handle.read(size)
        assert len(data) == size, (path, offset, size, len(data))
        return data

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        for _ in range(workers):
            item = next(ranges, None)
            if item is not None:
                pending.append(pool.submit(read_range, item))
        while pending:
            data = pending.popleft().result()
            item = next(ranges, None)
            if item is not None:
                pending.append(pool.submit(read_range, item))
            yield data


def prepare():
    """Build an owned data copy; never edit the coworker's shared tokenization cache."""
    if (AUTO / "dolci-data-plan.json").exists():
        assert json.loads((AUTO / "dolci-data-plan.json").read_text())["passed"]
        return
    DATA.mkdir(parents=True, exist_ok=True)
    stats = json.loads((INPUT / "dataset_statistics.json").read_text())["overall_statistics"]
    sources = sorted(INPUT.glob("token_ids_part_*.npy"))
    assert len(sources) == 93
    whole = []
    offset = 0
    prior_ended = True
    for source in sources:
        tokens = np.memmap(source, mode="r", dtype=np.uint32)
        with gzip.open(source.with_suffix(".csv.gz"), "rt") as f:
            group = [tuple(int(x) + offset for x in line.split(",")) for line in f]
        assert group[0][0] == offset
        if whole and not prior_ended:
            whole[-1] = (whole[-1][0], group[0][1])
            whole.extend(group[1:])
        else:
            whole.extend(group)
        prior_ended = int(tokens[-1]) == 100257
        offset += len(tokens)
    assert len(whole) == stats["total_instances"] and offset == stats["total_tokens"]
    assert all(b - a <= 65536 for a, b in whole)
    assert all(a[1] == b[0] for a, b in zip(whole, whole[1:]))
    # Tiny deterministic holdout, identical in both lineages, excluded from training.
    split = whole[256][0]
    for name, itemsize in [("token_ids", 4), ("labels_mask", 1)]:
        train = DATA / "train-single-source"
        valid = DATA / "validation"
        train.mkdir(exist_ok=True)
        valid.mkdir(exist_ok=True)
        a = train / (name + "_part_0000.npy")
        b = valid / (name + "_part_0000.npy")
        if a.exists() and b.exists():
            assert (
                a.stat().st_size == (offset - split) * itemsize
                and b.stat().st_size == split * itemsize
            )
            continue
        assert not a.exists() and not b.exists()
        at = 0
        digest = hashlib.sha256()
        with (
            a.with_suffix(".partial").open("wb") as out,
            b.with_suffix(".partial").open("wb") as val,
        ):
            for chunk in ordered_chunks(sorted(INPUT.glob(name + "_part_*.npy"))):
                n = min(len(chunk), max(0, split * itemsize - at))
                val.write(chunk[:n])
                out.write(chunk[n:])
                digest.update(chunk)
                at += len(chunk)
                if at // (4 * 1024**3) != (at - len(chunk)) // (4 * 1024**3):
                    print("DOLCI_COPY_PROGRESS", name, at, offset * itemsize, flush=True)
            out.flush()
            val.flush()
            os.fsync(out.fileno())
            os.fsync(val.fileno())
        assert at == offset * itemsize
        a.with_suffix(".partial").rename(a)
        b.with_suffix(".partial").rename(b)
        print("DOLCI_CONSOLIDATED", name, at, digest.hexdigest(), flush=True)
    for sub, rows, bias in [
        ("validation", whole[:256], 0),
        ("train-single-source", whole[256:], split),
    ]:
        with gzip.open(DATA / sub / "token_ids_part_0000.csv.gz", "wt") as f:
            for a, b in rows:
                f.write(f"{a-bias},{b-bias}\n")
    tokenizer = DATA / "train/tokenizer"
    tokenizer.parent.mkdir(exist_ok=True)
    if not tokenizer.exists():
        shutil.copytree(INPUT / "tokenizer", tokenizer)
    from olmoe3_hero_sft_data import SFTPackedDatasetConfig
    from olmo_core.data import TokenizerConfig

    lengths = {}
    for split_name, directory in [("train", "train-single-source"), ("validation", "validation")]:
        ds = SFTPackedDatasetConfig.glob(
            str(DATA / directory / "token_ids_part_*.npy"),
            label_mask_paths=[str(DATA / directory / "labels_mask_part_*.npy")],
            tokenizer=TokenizerConfig.dolma2(),
            sequence_length=65536,
            generate_doc_lengths=True,
            source_group_size=1,
            work_dir=str(DATA / "packing-cache" / split_name),
            instance_filter_config=None,
        ).build()
        ds.prepare()
        lengths[split_name] = len(ds)
        for i in [0, len(ds) // 2, len(ds) - 1]:
            row = ds[i]
            assert sum(row["doc_lens"]) == 65536 and row["label_mask"].any()
        print("DOLCI_PACKED", split_name, len(ds), flush=True)
    steps = lengths["train"] // 128
    atomic_json(
        AUTO / "dolci-data-plan.json",
        dict(
            passed=True,
            sequence_length=65536,
            packed_instances=lengths,
            steps_per_epoch=steps,
            total_steps=steps * 2,
            batch_tokens=8388608,
            total_steps_by_epochs={"2": steps * 2},
            dropped_packed_instances_per_epoch=lengths["train"] % 128,
            source=str(INPUT),
            dataset="allenai/Dolci-Think-SFT",
            holdout_records=256,
            raw_train_tokens=offset - split,
            epochs=2,
        ),
    )
    print("DOLCI_DATA_READY", json.loads((AUTO / "dolci-data-plan.json").read_text()), flush=True)
