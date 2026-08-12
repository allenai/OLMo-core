#!/usr/bin/env python
"""Materialize the two reviewed FineVision sources as compatible immutable Arrow shards.

The canonical FineVision parquet metadata was authored by a newer ``datasets`` release and
cannot be parsed by the pinned training environment. This command does not reinterpret rows:
it streams each parquet row group into an Arrow IPC shard with the same physical schema while
removing only the incompatible schema metadata. The resulting directories are consumed through
``load_from_disk_compat`` and are pinned by a byte-level source/output manifest.

Production uses the code-pinned source root and exact shard/row counts. The public
``materialize_sources`` function accepts explicit specs solely so focused tests can build tiny
artifacts without reading Weka.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from olmo_core.data.multimodal.dataset_compat import load_from_disk_compat

MATERIALIZATION_FORMAT = "vision_alignment_finevision_materialization"
MATERIALIZATION_VERSION = 1
CANONICAL_SOURCE_ROOT = Path(
    "/weka/oe-training-default/mm-olmo/hf_datasets/HuggingFaceM4___FineVision"
)


@dataclass(frozen=True)
class FineVisionSourceSpec:
    """One exact FineVision source expected by the materializer.

    :param name: Runtime FineVision configuration name.
    :param output_name: Filesystem-safe artifact subdirectory.
    :param expected_shards: Exact number of source parquet shards.
    :param expected_rows: Exact total number of source rows.
    """

    name: str
    output_name: str
    expected_shards: int
    expected_rows: int


CANONICAL_SOURCES = (
    FineVisionSourceSpec(
        name="visualwebinstruct(filtered)",
        output_name="visualwebinstruct-filtered",
        expected_shards=73,
        expected_rows=263_581,
    ),
    FineVisionSourceSpec(
        name="geo170k(align)",
        output_name="geo170k-align",
        expected_shards=1,
        expected_rows=35_297,
    ),
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    raw = _canonical_bytes(value) + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    """Flush every published file and directory without rereading file contents."""
    directories = [root]
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            directories.append(path)
        elif path.is_file():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    for directory in reversed(directories):
        _fsync_directory(directory)


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing any existing target."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        # The production Linux runtime provides renameat2. Keep a guarded fallback for
        # development platforms, where the sibling writer lock is still authoritative.
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite immutable artifact {destination}")
        os.rename(source, destination)
        return
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(f"Refusing to overwrite immutable artifact {destination}")
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _builder_sha256() -> str:
    return _sha256_file(Path(__file__).resolve())


def output_dataset_fingerprint(
    *,
    source_name: str,
    rows: int,
    physical_schema_sha256: str,
    shards: Sequence[Mapping[str, Any]],
    dataset_info_sha256: str,
) -> str:
    """Return a path-independent content identity for one materialized dataset."""
    return hashlib.sha256(
        _canonical_bytes(
            {
                "version": "vision-alignment-finevision-arrow-content-v1",
                "source_name": source_name,
                "rows": rows,
                "physical_schema_sha256": physical_schema_sha256,
                "shards": [
                    {
                        "rows": shard["rows"],
                        "sha256": shard["sha256"],
                    }
                    for shard in shards
                ],
                "dataset_info_sha256": dataset_info_sha256,
            }
        )
    ).hexdigest()


def _source_shards(root: Path, spec: FineVisionSourceSpec) -> tuple[Path, ...]:
    source_dir = root / spec.name
    shards = tuple(sorted(source_dir.glob("train-*.parquet")))
    if len(shards) != spec.expected_shards:
        raise ValueError(
            f"FineVision source {spec.name!r} has {len(shards)} shards; "
            f"expected {spec.expected_shards}"
        )
    return shards


def _source_inventory(root: Path, spec: FineVisionSourceSpec) -> dict[str, Any]:
    shards = _source_shards(root, spec)
    entries = []
    total_rows = 0
    schema_sha256: Optional[str] = None
    metadata_sha256: Optional[str] = None
    for path in shards:
        parquet = pq.ParquetFile(path)
        rows = parquet.metadata.num_rows
        total_rows += rows
        current_schema_sha = hashlib.sha256(
            parquet.schema_arrow.remove_metadata().serialize().to_pybytes()
        ).hexdigest()
        current_metadata_sha = hashlib.sha256(
            _canonical_bytes(
                {
                    key.decode("utf-8"): value.decode("utf-8")
                    for key, value in sorted((parquet.schema_arrow.metadata or {}).items())
                }
            )
        ).hexdigest()
        if schema_sha256 is None:
            schema_sha256 = current_schema_sha
            metadata_sha256 = current_metadata_sha
        elif current_schema_sha != schema_sha256 or current_metadata_sha != metadata_sha256:
            raise ValueError(f"FineVision source {spec.name!r} has inconsistent shard schemas")
        entries.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "rows": rows,
                "sha256": _sha256_file(path),
            }
        )
    if total_rows != spec.expected_rows:
        raise ValueError(
            f"FineVision source {spec.name!r} has {total_rows} rows; "
            f"expected {spec.expected_rows}"
        )
    assert schema_sha256 is not None and metadata_sha256 is not None
    return {
        "name": spec.name,
        "output_name": spec.output_name,
        "shards": entries,
        "shard_count": len(entries),
        "rows": total_rows,
        "physical_schema_sha256": schema_sha256,
        "source_metadata_sha256": metadata_sha256,
    }


def _write_arrow_shard(source: Path, output: Path) -> int:
    parquet = pq.ParquetFile(source)
    schema = parquet.schema_arrow.remove_metadata()
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    rows = 0
    try:
        with pa.OSFile(str(temporary), "wb") as sink:
            with ipc.new_stream(sink, schema) as writer:
                for row_group in range(parquet.num_row_groups):
                    table = parquet.read_row_group(row_group).replace_schema_metadata(None)
                    if table.schema != schema:
                        raise ValueError(f"Physical schema drift in {source} row group {row_group}")
                    writer.write_table(table)
                    rows += len(table)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return rows


def _dataset_info(source: Path, *, rows: int) -> dict[str, Any]:
    metadata = pq.ParquetFile(source).schema_arrow.metadata or {}
    raw = metadata.get(b"huggingface")
    if raw is None:
        raise ValueError(f"FineVision shard {source} lacks Hugging Face feature metadata")
    try:
        value = json.loads(raw)
        info = value["info"]
        features = info["features"]
    except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"FineVision shard {source} has invalid feature metadata") from error
    return {**info, "features": features, "num_examples": rows}


def _validate_specs(specs: Sequence[FineVisionSourceSpec]) -> None:
    if not specs:
        raise ValueError("At least one FineVision source specification is required")
    names = [spec.name for spec in specs]
    output_names = [spec.output_name for spec in specs]
    if len(set(names)) != len(names) or len(set(output_names)) != len(output_names):
        raise ValueError("FineVision source names and output names must be unique")
    for spec in specs:
        output_path = Path(spec.output_name)
        if (
            not spec.name
            or not spec.output_name
            or output_path.is_absolute()
            or len(output_path.parts) != 1
            or spec.output_name in (".", "..")
            or spec.expected_shards < 1
            or spec.expected_rows < 1
        ):
            raise ValueError(f"Invalid FineVision source specification: {spec!r}")


def _verify_materialized_outputs(
    staging: Path,
    specs: Sequence[FineVisionSourceSpec],
    inventories: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
) -> None:
    """Revalidate every output byte, receipt, and live dataset before publication."""
    if len(outputs) != len(specs):
        raise ValueError("FineVision output count differs from the build plan")
    for spec, inventory, output_entry in zip(specs, inventories, outputs):
        source_output = staging / spec.output_name
        dataset_info_path = source_output / "dataset_info.json"
        if (
            output_entry.get("name") != spec.name
            or output_entry.get("path") != spec.output_name
            or output_entry.get("rows") != spec.expected_rows
            or not dataset_info_path.is_file()
            or _sha256_file(dataset_info_path) != output_entry.get("dataset_info_sha256")
        ):
            raise ValueError(f"FineVision output identity differs for {spec.name!r}")
        shard_entries = output_entry.get("shards")
        if not isinstance(shard_entries, list) or len(shard_entries) != spec.expected_shards:
            raise ValueError(f"FineVision output shard count differs for {spec.name!r}")
        for shard_index, (source_entry, shard_entry) in enumerate(
            zip(inventory["shards"], shard_entries)
        ):
            output_path = source_output / (
                f"data-{shard_index:05d}-of-{spec.expected_shards:05d}.arrow"
            )
            expected_path = str(output_path.relative_to(staging))
            if (
                shard_entry.get("path") != expected_path
                or not output_path.is_file()
                or output_path.stat().st_size != shard_entry.get("bytes")
                or _sha256_file(output_path) != shard_entry.get("sha256")
                or shard_entry.get("rows") != source_entry["rows"]
            ):
                raise ValueError(f"FineVision output shard differs: {output_path}")
            receipt_path = output_path.with_suffix(".receipt.json")
            expected_receipt = {
                "source_sha256": source_entry["sha256"],
                "output_sha256": shard_entry["sha256"],
                "rows": source_entry["rows"],
            }
            if (
                not receipt_path.is_file()
                or json.loads(receipt_path.read_text()) != expected_receipt
            ):
                raise ValueError(f"FineVision output receipt differs: {receipt_path}")
        expected_fingerprint = output_dataset_fingerprint(
            source_name=spec.name,
            rows=spec.expected_rows,
            physical_schema_sha256=inventory["physical_schema_sha256"],
            shards=shard_entries,
            dataset_info_sha256=output_entry["dataset_info_sha256"],
        )
        live = load_from_disk_compat(source_output)
        if (
            len(live) != spec.expected_rows
            or output_entry.get("dataset_fingerprint") != expected_fingerprint
        ):
            raise ValueError(f"FineVision live output identity differs for {spec.name!r}")


def materialize_sources(
    *,
    source_root: Path,
    output_dir: Path,
    specs: Sequence[FineVisionSourceSpec] = CANONICAL_SOURCES,
    resume: bool = False,
) -> Path:
    """Build and atomically publish compatible Arrow shards.

    :param source_root: Directory containing the source config subdirectories.
    :param output_dir: Immutable final artifact directory.
    :param specs: Exact source expectations. Production uses :data:`CANONICAL_SOURCES`.
    :param resume: Reuse a matching sibling staging directory after interruption.
    :returns: The final materialization manifest path.
    :raises FileExistsError: If the final artifact or another writer already exists.
    :raises ValueError: If source identity, a resume plan, or output validation differs.
    """
    source_root = source_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    _validate_specs(specs)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite immutable artifact {output_dir}")
    staging = output_dir.with_name(f".{output_dir.name}.building")
    lock = output_dir.with_name(f".{output_dir.name}.lock")
    lock_fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(lock_fd)
        raise RuntimeError(f"Another FineVision materializer holds {lock}") from error
    try:
        if staging.exists() and not resume:
            raise FileExistsError(f"Staging artifact exists; rerun with --resume: {staging}")
        staging.mkdir(parents=False, exist_ok=resume)
        inventories = [_source_inventory(source_root, spec) for spec in specs]
        plan_identity = {
            "format": MATERIALIZATION_FORMAT,
            "version": MATERIALIZATION_VERSION,
            "builder_sha256": _builder_sha256(),
            "source_root": str(source_root),
            "sources": inventories,
        }
        plan_path = staging / "build-plan.json"
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
            if not isinstance(plan, dict) or set(plan) != {*plan_identity, "created_at"}:
                raise ValueError("FineVision resume plan fields differ")
            created_at = plan.get("created_at")
            if not isinstance(created_at, str) or not created_at:
                raise ValueError("FineVision resume plan lacks a stable creation time")
            if {key: plan[key] for key in plan_identity} != plan_identity:
                raise ValueError("FineVision resume plan differs from the current build")
        else:
            plan = {
                **plan_identity,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            _write_json_atomic(plan_path, plan)
        complete_path = staging / "COMPLETE"
        if complete_path.exists():
            prior_manifest = staging / "vision-alignment-finevision-materialization.json"
            if not prior_manifest.is_file() or complete_path.read_text().strip() != _sha256_file(
                prior_manifest
            ):
                raise ValueError("Completed FineVision staging marker or manifest differs")
            # Re-run every shard/live-dataset check below before republishing. The marker is
            # recreated only after the refreshed manifest has been committed.
            complete_path.unlink()

        outputs = []
        for spec, inventory in zip(specs, inventories):
            source_shards = _source_shards(source_root, spec)
            source_output = staging / spec.output_name
            source_output.mkdir(exist_ok=True)
            shard_entries = []
            for shard_index, (source, source_entry) in enumerate(
                zip(source_shards, inventory["shards"])
            ):
                output = source_output / (
                    f"data-{shard_index:05d}-of-{len(source_shards):05d}.arrow"
                )
                receipt_path = output.with_suffix(".receipt.json")
                if output.exists() != receipt_path.exists():
                    if not resume:
                        raise ValueError(f"Arrow shard and receipt presence disagree: {output}")
                    # The receipt is the per-shard commit marker. A crash between the Arrow
                    # rename and receipt write leaves a recoverable pair, not a poisoned build.
                    output.unlink(missing_ok=True)
                    receipt_path.unlink(missing_ok=True)
                    _fsync_directory(source_output)
                if output.exists():
                    receipt = json.loads(receipt_path.read_text())
                    expected_receipt = {
                        "source_sha256": source_entry["sha256"],
                        "output_sha256": _sha256_file(output),
                        "rows": source_entry["rows"],
                    }
                    if receipt != expected_receipt:
                        raise ValueError(f"Resumed Arrow shard receipt differs: {output}")
                    rows = receipt["rows"]
                else:
                    rows = _write_arrow_shard(source, output)
                if rows != source_entry["rows"]:
                    raise ValueError(f"Arrow row count differs for {output}")
                output_sha = _sha256_file(output)
                if not receipt_path.exists():
                    _write_json_atomic(
                        receipt_path,
                        {
                            "source_sha256": source_entry["sha256"],
                            "output_sha256": output_sha,
                            "rows": rows,
                        },
                    )
                shard_entries.append(
                    {
                        "path": str(output.relative_to(staging)),
                        "bytes": output.stat().st_size,
                        "rows": rows,
                        "sha256": output_sha,
                    }
                )
            dataset_info_path = source_output / "dataset_info.json"
            _write_json_atomic(
                dataset_info_path,
                _dataset_info(source_shards[0], rows=spec.expected_rows),
            )
            dataset_info_sha256 = _sha256_file(dataset_info_path)
            live = load_from_disk_compat(source_output)
            if len(live) != spec.expected_rows:
                raise ValueError(f"Materialized FineVision row count differs for {spec.name!r}")
            fingerprint = output_dataset_fingerprint(
                source_name=spec.name,
                rows=len(live),
                physical_schema_sha256=inventory["physical_schema_sha256"],
                shards=shard_entries,
                dataset_info_sha256=dataset_info_sha256,
            )
            outputs.append(
                {
                    "name": spec.name,
                    "path": spec.output_name,
                    "rows": len(live),
                    "dataset_fingerprint": fingerprint,
                    "dataset_info_sha256": dataset_info_sha256,
                    "physical_schema_sha256": inventory["physical_schema_sha256"],
                    "shards": shard_entries,
                }
            )

        if [_source_inventory(source_root, spec) for spec in specs] != inventories:
            raise ValueError("FineVision source inventory changed during materialization")
        if _builder_sha256() != plan["builder_sha256"]:
            raise ValueError("FineVision materializer changed during construction")
        _verify_materialized_outputs(staging, specs, inventories, outputs)

        manifest = {
            **plan,
            "status": "verified",
            "outputs": outputs,
        }
        manifest["content_sha256"] = hashlib.sha256(_canonical_bytes(manifest)).hexdigest()
        manifest_path = staging / "vision-alignment-finevision-materialization.json"
        _write_json_atomic(manifest_path, manifest)
        manifest_sha = _sha256_file(manifest_path)
        complete = staging / "COMPLETE"
        with complete.open("x") as handle:
            handle.write(manifest_sha + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_tree(staging)
        _rename_directory_no_replace(staging, output_dir)
        _fsync_directory(output_dir.parent)
        return output_dir / manifest_path.name
    except BaseException:
        # Keep staging for an explicitly requested, byte-plan-validated resume.
        raise
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Materialize the canonical FineVision sources."""
    args = _parser().parse_args(argv)
    manifest = materialize_sources(
        source_root=CANONICAL_SOURCE_ROOT,
        output_dir=Path(args.output_dir),
        resume=args.resume,
    )
    print(
        json.dumps(
            {"manifest": str(manifest), "sha256": _sha256_file(manifest)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
