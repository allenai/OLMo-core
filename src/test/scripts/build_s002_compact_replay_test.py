from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict

import pytest
import torch


def _load_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "data" / "build_s002_compact_replay.py"
    spec = importlib.util.spec_from_file_location("_build_s002_compact_replay_test_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    return _load_module()


class FakeObjectStore:
    """Exact-object fake with no list operation and adversarial response controls."""

    def __init__(self, builder, sources, data_by_index: Dict[int, bytes]):
        self.builder = builder
        self.sources = {source.parent_path_index: source for source in sources}
        self.data_by_index = data_by_index
        self.generations = {index: "100" for index in data_by_index}
        self.etags = {index: f"etag-{index}" for index in data_by_index}
        self.list_calls = 0
        self.head_calls: list[int] = []
        self.range_calls: list[tuple[int, int, int, str]] = []
        self.wrong_range = False
        self.mutate_before_get = False
        self.mutate_during_closing = False
        self.initial_heads = len(sources)

    def list_objects(self):
        """Fail if the implementation attempts a forbidden object listing."""
        self.list_calls += 1
        raise AssertionError("compact replay must never list remote objects")

    def head(self, parent_path: str, parent_path_index: int):
        self.head_calls.append(parent_path_index)
        if self.mutate_during_closing and len(self.head_calls) > self.initial_heads:
            self.generations[parent_path_index] = "101"
        data = self.data_by_index[parent_path_index]
        source = self.sources[parent_path_index]
        assert parent_path == source.parent_path
        return self.builder.RemoteObject(
            parent_path_index=parent_path_index,
            parent_path=parent_path,
            mirror_uri=parent_path.replace("s3://", "gs://", 1),
            size_bytes=len(data),
            num_tokens=len(data) // 4,
            generation=self.generations[parent_path_index],
            etag=self.etags[parent_path_index],
            md5_hash=None,
            crc32c=base64.b64encode(parent_path_index.to_bytes(4, "big")).decode(),
            source_etag=f'"multipart-{parent_path_index}-2"',
        )

    def get_range(self, snapshot, start: int, stop: int):
        self.range_calls.append((snapshot.parent_path_index, start, stop, snapshot.generation))
        if self.mutate_before_get:
            self.generations[snapshot.parent_path_index] = "101"
        if snapshot.generation != self.generations[snapshot.parent_path_index]:
            raise ValueError("generation precondition failed")
        data = self.data_by_index[snapshot.parent_path_index][start:stop]
        content_range = f"bytes {start}-{stop - 1}/{snapshot.size_bytes}"
        if self.wrong_range:
            content_range = "bytes 0-0/1"
        return self.builder.RangeResult(
            data=data,
            content_range=content_range,
            content_length=len(data),
            generation=snapshot.generation,
            etag=snapshot.etag,
        )


def _synthetic_fixture(builder, *, objects: int = 4, windows_per_object: int = 8):
    sources = []
    data_by_index = {}
    for index in range(objects):
        parent_path = f"s3://ai2-llm/preprocessed/test/source-{index}.npy"
        path_digest = hashlib.sha256(parent_path.encode()).hexdigest()
        sources.append(
            builder.ParentSource(
                source_id=f"s002-{index:06d}-{path_digest[:16]}",
                source_name=("web" if index < objects - 1 else "books"),
                parent_path_index=index,
                parent_path=parent_path,
            )
        )
        size = windows_per_object * builder.WINDOW_SIZE_BYTES
        pattern = hashlib.sha256(f"object-{index}".encode()).digest()
        data_by_index[index] = (pattern * (size // len(pattern) + 1))[:size]
    fingerprint = builder.reconstruct_parent_dataset_fingerprint(
        [source.parent_path for source in sources],
        [len(data_by_index[index]) for index in range(objects)],
    )
    identity = builder.ParentIdentity(
        sources=tuple(sources),
        config_sha256="1" * 64,
        paths_sha256="2" * 64,
        mix_sha256="3" * 64,
        trainer_state_sha256="4" * 64,
        dataset_fingerprint=fingerprint,
    )
    return identity, FakeObjectStore(builder, sources, data_by_index)


def _build_small(builder, monkeypatch, output: Path, store, identity, *, workers: int = 3):
    monkeypatch.setattr(
        builder,
        "_load_parent_identity",
        lambda *, expected_objects: identity,
    )
    return builder._build_compact_replay(
        output,
        store=store,
        workers=workers,
        train_windows=6,
        holdout_windows=2,
        seed=17,
        expected_objects=len(identity.sources),
        require_production_counts=False,
    )


def _tree_bytes(root: Path):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _semantic_tree(root: Path):
    values = _tree_bytes(root)
    for name in (
        "native-text-replay-verification.json",
        "native-text-replay-train.json",
        "native-text-replay-holdout.json",
    ):
        data = json.loads(values[name])
        if name == "native-text-replay-verification.json":
            for split in ("train", "holdout"):
                for source in data["splits"][split]:
                    for field in ("mtime_ns", "ctime_ns", "inode", "device"):
                        source[field] = "<physical-stat>"
            data["compact_materialization_sha256"] = "<physical-materialization>"
            data["manifest_contract_sha256"] = {
                "train": "<physical-contract>",
                "holdout": "<physical-contract>",
            }
        else:
            data["provenance"]["compact_materialization_sha256"] = "<physical-materialization>"
            data["provenance"]["verification_receipt_sha256"] = "<physical-receipt>"
        values[name] = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return values


def test_affine_grid_golden_parity(builder):
    assert builder._apportion_exact(11, {"books": 3, "code": 7, "web": 19}) == {
        "books": 1,
        "code": 3,
        "web": 7,
    }
    assert builder._permutation_prefix(17, 8, seed=6198, source_id="s002-golden") == (
        0,
        10,
        3,
        13,
        6,
        16,
        9,
        2,
    )


def test_parallel_build_is_byte_deterministic_and_loader_valid(
    tmp_path: Path, builder, monkeypatch
):
    identity, first_store = _synthetic_fixture(builder)
    output = tmp_path / "artifact"
    first = _build_small(builder, monkeypatch, output, first_store, identity, workers=4)
    assert first_store.list_calls == 0
    assert len(first_store.head_calls) == len(identity.sources) * 2
    assert all(
        stop - start == builder.WINDOW_SIZE_BYTES for _, start, stop, _ in first_store.range_calls
    )
    first_semantic_tree = _semantic_tree(output)
    backup = tmp_path / "first-artifact"
    output.rename(backup)

    _, second_store = _synthetic_fixture(builder)
    second = _build_small(builder, monkeypatch, output, second_store, identity, workers=1)
    assert _semantic_tree(output) == first_semantic_tree
    assert first.remote_snapshot_sha256 == second.remote_snapshot_sha256

    from olmo_core.data.multimodal.native_text_replay import (
        NativeTextReplayManifest,
        NativeTextReplayVerificationReceipt,
    )

    train = NativeTextReplayManifest.load(second.train_manifest)
    holdout = NativeTextReplayManifest.load(second.holdout_manifest)
    receipt = NativeTextReplayVerificationReceipt.load(
        second.verification_receipt,
        expected_sha256=second.verification_receipt_sha256,
    )
    receipt.validate_manifest(train)
    receipt.validate_manifest(holdout)
    assert train.num_windows == 6
    assert holdout.num_windows == 2
    train_pairs = {
        (source.parent_path_index, start)
        for source in train.sources
        for start in source.parent_window_starts
    }
    holdout_pairs = {
        (source.parent_path_index, start)
        for source in holdout.sources
        for start in source.parent_window_starts
    }
    assert train_pairs.isdisjoint(holdout_pairs)
    remote_sources = json.loads(second.verification_receipt.read_text())["remote_sources"]
    assert all(source["md5_hash"] is None for source in remote_sources)
    assert all("multipart" in source["source_etag"] for source in remote_sources)
    receipt_data = json.loads(second.verification_receipt.read_text())
    for split in ("train", "holdout"):
        for source in receipt_data["splits"][split]:
            token_stat = (output / source["path"]).stat()
            assert (
                source["mtime_ns"],
                source["ctime_ns"],
                source["inode"],
                source["device"],
            ) == (
                token_stat.st_mtime_ns,
                token_stat.st_ctime_ns,
                token_stat.st_ino,
                token_stat.st_dev,
            )

    head_calls = len(second_store.head_calls)
    range_calls = len(second_store.range_calls)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _build_small(builder, monkeypatch, output, second_store, identity)
    assert len(second_store.head_calls) == head_calls
    assert len(second_store.range_calls) == range_calls


def test_saved_fingerprint_mismatch_fails_before_download(tmp_path: Path, builder, monkeypatch):
    identity, store = _synthetic_fixture(builder)
    identity = builder.ParentIdentity(
        sources=identity.sources,
        config_sha256=identity.config_sha256,
        paths_sha256=identity.paths_sha256,
        mix_sha256=identity.mix_sha256,
        trainer_state_sha256=identity.trainer_state_sha256,
        dataset_fingerprint="0" * 64,
    )
    with pytest.raises(ValueError, match="reconstruct dataset fingerprint"):
        _build_small(builder, monkeypatch, tmp_path / "artifact", store, identity)
    assert store.range_calls == []


@pytest.mark.parametrize("failure", ["mutation", "wrong-range"])
def test_generation_and_range_response_fail_closed(
    tmp_path: Path, builder, monkeypatch, failure: str
):
    identity, store = _synthetic_fixture(builder)
    if failure == "mutation":
        store.mutate_before_get = True
        match = "generation precondition"
    else:
        store.wrong_range = True
        match = "Range response identity differs"
    with pytest.raises(ValueError, match=match):
        _build_small(builder, monkeypatch, tmp_path / failure, store, identity)


def test_resume_rejects_compact_byte_drift(tmp_path: Path, builder, monkeypatch):
    identity, store = _synthetic_fixture(builder)
    store.mutate_during_closing = True
    output = tmp_path / "artifact"
    with pytest.raises(ValueError, match="Remote object snapshot changed"):
        _build_small(builder, monkeypatch, output, store, identity)
    staging = next(tmp_path.glob(".artifact.*.building"))
    token_path = next((staging / "tokens").rglob("*.npy"))
    with token_path.open("r+b") as file_handle:
        file_handle.seek(0)
        file_handle.write(b"drift")

    _, stable_store = _synthetic_fixture(builder)
    with pytest.raises(ValueError, match="Resumable compact bytes drifted"):
        _build_small(builder, monkeypatch, output, stable_store, identity)


@pytest.mark.parametrize("orphan", ["token", "sidecar", "partial"])
def test_resume_recovers_plan_scoped_orphans(tmp_path: Path, builder, monkeypatch, orphan: str):
    identity, interrupted_store = _synthetic_fixture(builder)
    interrupted_store.mutate_during_closing = True
    output = tmp_path / "artifact"
    with pytest.raises(ValueError, match="Remote object snapshot changed"):
        _build_small(builder, monkeypatch, output, interrupted_store, identity)

    staging = next(tmp_path.glob(".artifact.*.building"))
    token_path = next((staging / "tokens" / "train").glob("*.npy"))
    resume_path = staging / "resume" / "train" / f"{token_path.stem}.json"
    if orphan == "token":
        resume_path.unlink()
    elif orphan == "sidecar":
        token_path.unlink()
    else:
        token_path.unlink()
        resume_path.unlink()
        plan = json.loads((staging / builder.PLAN_FILENAME).read_text())
        partial_path = builder._selection_partial_path(token_path, plan["content_sha256"])
        partial_path.write_bytes(b"stale interrupted bytes")

    _, stable_store = _synthetic_fixture(builder)
    result = _build_small(builder, monkeypatch, output, stable_store, identity)
    assert result.output_dir == output
    assert stable_store.range_calls
    assert not any(output.rglob("*.partial"))


def test_resume_rejects_symlinked_completion(tmp_path: Path, builder, monkeypatch):
    identity, interrupted_store = _synthetic_fixture(builder)
    interrupted_store.mutate_during_closing = True
    output = tmp_path / "artifact"
    with pytest.raises(ValueError, match="Remote object snapshot changed"):
        _build_small(builder, monkeypatch, output, interrupted_store, identity)

    staging = next(tmp_path.glob(".artifact.*.building"))
    token_path = next((staging / "tokens" / "train").glob("*.npy"))
    resume_path = staging / "resume" / "train" / f"{token_path.stem}.json"
    resume_path.unlink()
    resume_path.symlink_to(staging / builder.PLAN_FILENAME)
    _, stable_store = _synthetic_fixture(builder)
    with pytest.raises(ValueError, match="regular non-symlink"):
        _build_small(builder, monkeypatch, output, stable_store, identity)


def test_staging_descendant_symlink_fails_before_range_get(tmp_path: Path, builder, monkeypatch):
    identity, store = _synthetic_fixture(builder)
    original_open_staging = builder._open_staging
    escape = tmp_path / "escape"
    escape.mkdir()

    def open_attacked_staging(output_dir, plan):
        staging, lock_fd = original_open_staging(output_dir, plan)
        (staging / "tokens").symlink_to(escape, target_is_directory=True)
        return staging, lock_fd

    monkeypatch.setattr(builder, "_open_staging", open_attacked_staging)
    with pytest.raises(ValueError, match="non-symlink directory"):
        _build_small(builder, monkeypatch, tmp_path / "artifact", store, identity)
    assert store.range_calls == []


@pytest.mark.parametrize("attack", ["same-size", "extra", "symlink"])
def test_descriptor_closing_rejects_tree_attacks(tmp_path: Path, builder, monkeypatch, attack: str):
    identity, store = _synthetic_fixture(builder)
    original_validate = builder._validate_and_fsync_staging

    def validate_attacked_staging(staging, expected_files):
        token_relative = next(path for path in expected_files if path.startswith("tokens/"))
        token_path = staging / token_relative
        if attack == "same-size":
            raw = bytearray(token_path.read_bytes())
            raw[0] ^= 1
            token_path.write_bytes(raw)
        elif attack == "extra":
            (staging / "unexpected-file").write_text("not allowed")
        else:
            token_path.unlink()
            token_path.symlink_to(staging / builder.PLAN_FILENAME)
        return original_validate(staging, expected_files)

    monkeypatch.setattr(builder, "_validate_and_fsync_staging", validate_attacked_staging)
    with pytest.raises(ValueError, match="Closing artifact"):
        _build_small(builder, monkeypatch, tmp_path / attack, store, identity)


@pytest.mark.parametrize(
    ("field", "value"),
    [("parent_path_index", True), ("num_tokens", True)],
)
def test_remote_metadata_rejects_bool_integer_alias(builder, field: str, value):
    source = builder.ParentSource(
        source_id="s002-000001-test",
        source_name="web",
        parent_path_index=1,
        parent_path="s3://ai2-llm/preprocessed/test/source.npy",
    )
    remote = builder.RemoteObject(
        parent_path_index=1,
        parent_path=source.parent_path,
        mirror_uri=source.parent_path.replace("s3://", "gs://", 1),
        size_bytes=4,
        num_tokens=1,
        generation="1",
        etag="etag",
        md5_hash=None,
        crc32c="AAAAAA==",
        source_etag=None,
    )
    with pytest.raises(ValueError, match="Remote object metadata differs"):
        builder._validate_remote_object(replace(remote, **{field: value}), source)


def test_pinned_parent_bytes_and_saved_fingerprint_are_validated(
    tmp_path: Path, builder, monkeypatch
):
    tokenizer = dict(builder.S002_TOKENIZER)
    tokenizer["_CLASS_"] = "olmo_core.data.tokenizer.TokenizerConfig"
    config = {
        "_CLASS_": "olmo_core.internal.experiment.ExperimentConfig",
        "dataset": {
            "_CLASS_": "olmo_core.data.numpy_dataset.NumpyFSLDatasetConfig",
            "tokenizer": tokenizer,
            "mix": builder.S002_PARENT_MIX,
            "mix_base_dir": "s3://ai2-llm",
            "expand_glob": False,
            "ignore_fingerprint_mismatch": False,
            "sequence_length": builder.SEQUENCE_LENGTH,
            "max_target_sequence_length": builder.SEQUENCE_LENGTH,
            "instance_filter_config": {
                **builder.S002_INSTANCE_FILTER,
                "_CLASS_": "olmo_core.data.numpy_dataset.InstanceFilterConfig",
            },
        },
        "data_loader": {
            "_CLASS_": "olmo_core.data.data_loader.NumpyDataLoaderConfig",
            "type": "numpy",
            "ignore_fingerprint_mismatch": False,
        },
    }
    paths = [
        "s3://ai2-llm/preprocessed/test/web/a.npy",
        "s3://ai2-llm/preprocessed/test/books/b.npy",
    ]
    mix = ["web,preprocessed/test/web/a.npy", "books,preprocessed/test/books/b.npy"]
    config_path = tmp_path / "config.json"
    paths_path = tmp_path / "data_paths.txt"
    mix_path = tmp_path / "mix.txt"
    trainer_path = tmp_path / "rank0.pt"
    config_path.write_text(json.dumps(config))
    paths_path.write_text("\n".join(paths) + "\n")
    mix_path.write_text("\n".join(mix) + "\n")
    torch.save(
        {
            "data_loader": {
                "dataset_fingerprint_version": builder.S002_PARENT_DATASET_FINGERPRINT_VERSION,
                "dataset_fingerprint": "a" * 64,
                "dataset_type": "fsl",
                "sequence_length": builder.SEQUENCE_LENGTH,
                "max_target_sequence_length": builder.SEQUENCE_LENGTH,
            }
        },
        trainer_path,
    )
    for name, path in (
        ("S002_PARENT_CONFIG_FILE", config_path),
        ("S002_PARENT_PATHS_FILE", paths_path),
        ("S002_PARENT_MIX_FILE", mix_path),
        ("S002_PARENT_TRAINER_STATE_FILE", trainer_path),
    ):
        monkeypatch.setattr(builder, name, str(path))
    for name, path in (
        ("S002_PARENT_CONFIG_SHA256", config_path),
        ("S002_PARENT_PATHS_SHA256", paths_path),
        ("S002_PARENT_MIX_SHA256", mix_path),
        ("S002_PARENT_TRAINER_STATE_SHA256", trainer_path),
    ):
        monkeypatch.setattr(builder, name, hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(builder, "S002_PARENT_DATASET_FINGERPRINT", "a" * 64)
    identity = builder._load_parent_identity(expected_objects=2)
    assert [source.parent_path for source in identity.sources] == paths
    assert identity.dataset_fingerprint == "a" * 64

    config_path.write_text(config_path.read_text() + " ")
    with pytest.raises(ValueError, match="Pinned s002 config has SHA-256"):
        builder._load_parent_identity(expected_objects=2)
