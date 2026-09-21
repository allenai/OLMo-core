"""Gate on corrected arrays, preserve record boundaries, and build fresh packing indexes."""

import fcntl
import gzip
import hashlib
import json
import os
import shutil

import numpy as np
import olmoe3_corrected_sft_plan as p
from olmoe3_dolci_prepare import ordered_chunks
from olmoe3_lr_sweep_watch import atomic_json


def sha(path):
    """Fingerprint a provenance file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokenizer_check(path):
    """Check actual AutoTokenizer encoding, not just the vocabulary or tokenizer name."""
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    reference = Tokenizer.from_file(str(p.INPUT.parent / "payload/tokenizer-source/tokenizer.json"))
    # The payload is a sibling of datasets, not a child of it.
    actual = AutoTokenizer.from_pretrained(path, local_files_only=True)
    backend = json.loads(actual.backend_tokenizer.to_str())
    raw = json.loads(reference.to_str())
    assert backend["pre_tokenizer"] == raw["pre_tokenizer"]
    assert actual.bos_token_id is None and actual.eos_token_id == 100257
    assert actual.pad_token_id == 100277 and actual.vocab_size == 100278
    texts = [
        "9078563412",
        "12345678901234567890",
        "def f(x):\n    return x + 1234\n",
        "中文 123456",
        "  \n\t\n\n",
        "I'm testing contractions, aren't we?",
    ]
    for text in texts:
        assert (
            actual.encode(text, add_special_tokens=False)
            == reference.encode(text, add_special_tokens=False).ids
        )
    assert actual.encode("9078563412", add_special_tokens=False) == [23505, 25505, 16546, 17]
    return actual


def tokenizer_config(path):
    """The loader consumes integer IDs; record the exact corrected tokenizer provenance."""
    from olmo_core.data import TokenizerConfig

    return TokenizerConfig(
        vocab_size=100278,
        eos_token_id=100257,
        pad_token_id=100277,
        bos_token_id=None,
        identifier=str(path),
    )


def prepare(kind):
    """Consolidate only owned outputs; never modify old arrays or coworker directories."""
    from olmoe3_hero_sft_data import SFTPackedDatasetConfig

    source, target = p.INPUT / kind, p.DATA / kind
    target.mkdir(parents=True, exist_ok=True)
    lock = (target / "prepare.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((source / "manifest.json").read_text())
    assert manifest["status"] == "complete" and not manifest["smoke"]
    assert manifest["tokenizer_revision"] == p.TOKENIZER_REV
    assert not manifest["source_arrays_reused"] and manifest["max_seq_length"] == 65536
    assert len(manifest["parts"]) == len(manifest["part_source_files"])
    proof = json.loads((source / "tokenizer-proof.json").read_text())
    assert proof["canonical_tokenizer_sha256"] == (
        "18e309ad7f9c60037eaf26aca401a96128187cad2bdfa01a508dca7526bf6300"
    )
    tokenizer_check(source / "tokenizer")
    token_dest = target / "train/tokenizer"
    if not token_dest.exists():
        token_dest.parent.mkdir(exist_ok=True)
        shutil.copytree(source / "tokenizer", token_dest)
    tokenizer_check(token_dest)
    plan_path = target / "data-plan.json"
    if plan_path.exists():
        saved = json.loads(plan_path.read_text())
        assert saved["passed"] and saved["source_manifest_sha256"] == sha(source / "manifest.json")
        print("CORRECTED_DATA_ALREADY_READY", kind, flush=True)
        return
    paths, records, offset = [], [], 0
    validation_paths = []
    for part in manifest["parts"]:
        files = sorted((source / part["path"]).glob("token_ids_part_*.npy"))
        assert len(files) == 1, "Writer must preserve whole source records"
        if part["path"] == "validation":
            validation_paths = files
            continue
        f = files[0]
        with gzip.open(f.with_suffix(".csv.gz"), "rt") as handle:
            rows = [tuple(map(int, line.split(","))) for line in handle]
        count = f.stat().st_size // 4
        assert count == part["statistics"]["total_tokens"]
        assert rows[0][0] == 0 and rows[-1][1] == count
        assert len(rows) == part["statistics"]["total_instances"]
        assert all(a[1] == b[0] for a, b in zip(rows, rows[1:]))
        assert all(0 < b - a <= 65536 for a, b in rows)
        masks = f.with_name(f.name.replace("token_ids", "labels_mask"))
        assert masks.stat().st_size == count
        # Read actual saved records, comparing decoded text against the canonical encoder.
        ids = np.memmap(f, mode="r", dtype=np.uint32)
        mask = np.memmap(masks, mode="r", dtype=np.bool_)
        tok = tokenizer_check(token_dest)
        for n in (0, len(rows) // 2, len(rows) - 1):
            a, b = rows[n]
            observed = ids[a:b].tolist()
            assert (
                tok.encode(
                    tok.decode(observed, skip_special_tokens=False), add_special_tokens=False
                )
                == observed
            )
            assert not mask[a] and mask[a:b].any()
        records.extend((a + offset, b + offset) for a, b in rows)
        paths.append(f)
        offset += count
    holdout = 256 if kind == "dolci-think" else 0
    split = records[holdout][0] if holdout else 0
    train_dir, val_dir = target / "train-single-source", target / "validation"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    for name, itemsize in (("token_ids", 4), ("labels_mask", 1)):
        train = train_dir / f"{name}_part_0000.npy"
        valid = val_dir / f"{name}_part_0000.npy"
        sources = [f.with_name(f.name.replace("token_ids", name)) for f in paths]
        if not train.exists():
            with train.with_suffix(".partial").open("wb") as out:
                val = valid.with_suffix(".partial").open("wb") if holdout else None
                at = 0
                for chunk in ordered_chunks(sources):
                    n = min(len(chunk), max(0, split * itemsize - at))
                    if val:
                        val.write(chunk[:n])
                    out.write(chunk[n:])
                    at += len(chunk)
                out.flush()
                os.fsync(out.fileno())
                if val:
                    val.flush()
                    os.fsync(val.fileno())
                    val.close()
            assert at == offset * itemsize
            train.with_suffix(".partial").rename(train)
            if holdout:
                valid.with_suffix(".partial").rename(valid)
        assert train.stat().st_size == (offset - split) * itemsize
        if not holdout and not valid.exists():
            assert len(validation_paths) == 1
            f = validation_paths[0]
            shutil.copyfile(f.with_name(f.name.replace("token_ids", name)), valid)
    with gzip.open(train_dir / "token_ids_part_0000.csv.gz", "wt") as handle:
        for a, b in records[holdout:]:
            handle.write(f"{a-split},{b-split}\n")
    if holdout:
        with gzip.open(val_dir / "token_ids_part_0000.csv.gz", "wt") as handle:
            for a, b in records[:holdout]:
                handle.write(f"{a},{b}\n")
    else:
        shutil.copyfile(
            validation_paths[0].with_suffix(".csv.gz"), val_dir / "token_ids_part_0000.csv.gz"
        )
    lengths = {}
    probes = {}
    for label, directory in (("train", train_dir), ("validation", val_dir)):
        ds = SFTPackedDatasetConfig.glob(
            str(directory / "token_ids_part_*.npy"),
            label_mask_paths=[str(directory / "labels_mask_part_*.npy")],
            tokenizer=tokenizer_config(token_dest),
            sequence_length=65536,
            generate_doc_lengths=True,
            source_group_size=1,
            instance_filter_config=None,
            work_dir=str(target / "packing-cache" / label),
        ).build()
        ds.prepare()
        lengths[label] = len(ds)
        probes[label] = []
        for i in (0, len(ds) // 2, len(ds) - 1):
            row = ds[i]
            assert sum(row["doc_lens"]) == 65536 and row["label_mask"].any()
            assert not row["label_mask"][row["input_ids"] == 100277].any()
            probes[label].append(
                dict(
                    index=i,
                    input_sha256=hashlib.sha256(row["input_ids"].numpy().tobytes()).hexdigest(),
                    supervised_tokens=int(row["label_mask"].sum()),
                )
            )
        print("CORRECTED_SFT_PACKED", kind, label, lengths[label], flush=True)
    steps = lengths["train"] // 128
    assert steps > 0 and lengths["validation"] > 0
    atomic_json(
        plan_path,
        dict(
            passed=True,
            dataset=manifest["dataset"],
            sequence_length=65536,
            packed_instances=lengths,
            steps_per_epoch=steps,
            total_steps=2 * steps,
            total_steps_by_epochs={"2": 2 * steps},
            batch_tokens=8388608,
            epochs=2,
            dropped_packed_instances_per_epoch=lengths["train"] % 128,
            source_manifest_sha256=sha(source / "manifest.json"),
            tokenizer_sha256=sha(token_dest / "tokenizer.json"),
            tokenizer_revision=p.TOKENIZER_REV,
            new_holdout_records=holdout,
            raw_train_tokens=offset - split,
            loader_probes=probes,
        ),
    )
    print("CORRECTED_SFT_DATA_READY", kind, steps, flush=True)
