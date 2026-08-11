import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict

import numpy as np
import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[2] / "scripts" / "data" / "build_s002_replay_manifest.py"
    )
    spec = importlib.util.spec_from_file_location("_build_s002_replay_manifest_test_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    return _load_module()


def _write_tokens(path: Path, num_tokens: int) -> str:
    np.arange(num_tokens, dtype=np.uint32).tofile(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_catalog(tmp_path: Path, builder):
    sources = []
    parent_paths = []
    parent_mix_rows = []
    provenance_sources = []
    for parent_path_index, (source_id, source_name, num_tokens) in enumerate(
        (
            ("web-a", "web", 48),
            ("web-b", "web", 24),
            ("code-a", "code", 24),
            ("books-a", "books", 8),
        )
    ):
        token_path = tmp_path / f"{source_id}.npy"
        relative_path = f"preprocessed/test/{source_name}/{source_id}.npy"
        parent_path = f"s3://ai2-llm/{relative_path}"
        source_sha256 = _write_tokens(token_path, num_tokens)
        parent_paths.append(parent_path)
        parent_mix_rows.append(f"{source_name},{relative_path}")
        sources.append(
            {
                "parent_path_index": parent_path_index,
                "path": token_path.name,
            }
        )
        provenance_sources.append(
            {
                "parent_path_index": parent_path_index,
                "parent_path": parent_path,
                "num_tokens": num_tokens,
                "sha256": source_sha256,
            }
        )
    parent_paths_file = tmp_path / "data_paths.txt"
    parent_paths_file.write_text("\n".join(parent_paths) + "\n")
    builder.S002_PARENT_PATHS_FILE = str(parent_paths_file)
    builder.S002_PARENT_PATHS_SHA256 = hashlib.sha256(parent_paths_file.read_bytes()).hexdigest()
    parent_mix_file = tmp_path / "OLMo-mix-0925.txt"
    parent_mix_file.write_text("\n".join(parent_mix_rows) + "\n")
    builder.S002_PARENT_MIX_FILE = str(parent_mix_file)
    builder.S002_PARENT_MIX_SHA256 = hashlib.sha256(parent_mix_file.read_bytes()).hexdigest()
    provenance = {
        "format": builder.UPSTREAM_PROVENANCE_FORMAT,
        "version": builder.UPSTREAM_PROVENANCE_VERSION,
        "hash_algorithm": "sha256",
        "parent_paths_sha256": builder.S002_PARENT_PATHS_SHA256,
        "parent_mix_sha256": builder.S002_PARENT_MIX_SHA256,
        "sources": provenance_sources,
    }
    provenance_path = tmp_path / "upstream-provenance.json"
    _pin_provenance(provenance_path, builder, provenance)
    catalog = {
        "format": builder.SOURCE_CATALOG_FORMAT,
        "version": builder.SOURCE_CATALOG_VERSION,
        "sources": sources,
    }
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog, indent=2))
    return catalog_path, catalog, provenance_path, provenance


def _pin_provenance(path: Path, builder, provenance) -> None:
    path.write_text(json.dumps(provenance, indent=2))
    builder.S002_UPSTREAM_PROVENANCE_FILE = str(path)
    builder.S002_UPSTREAM_PROVENANCE_SHA256 = hashlib.sha256(path.read_bytes()).hexdigest()


def _counts_by_label(manifest):
    counts: Dict[str, int] = defaultdict(int)
    for source in manifest["sources"]:
        counts[source["source"]] += len(source["window_starts"])
    return dict(counts)


def _starts_by_id(manifest):
    return {source["id"]: set(source["window_starts"]) for source in manifest["sources"]}


def test_builds_exact_deterministic_disjoint_native_manifests(tmp_path: Path, builder):
    catalog_path, _, _, _ = _write_catalog(tmp_path, builder)
    catalog = builder.load_source_catalog(catalog_path)
    assert [source.source_name for source in catalog.sources] == ["web", "web", "code", "books"]
    for index, source in enumerate(catalog.sources):
        path_digest = hashlib.sha256(source.parent_path.encode("utf-8")).hexdigest()
        assert source.source_id == f"s002-{index:06d}-{path_digest[:16]}"
    output_dir = tmp_path / "manifests"
    kwargs = {
        "manifest_dir": output_dir,
        "sequence_length": 3,
        "train_usable_tokens": 30,
        "holdout_usable_tokens": 10,
        "seed": 17,
        "train_minimum_source_tokens": {"code": 4, "books": 2},
        "holdout_minimum_source_tokens": {"code": 2, "books": 2},
    }

    manifests = builder.build_replay_manifests(catalog, **kwargs)
    repeated = builder.build_replay_manifests(catalog, **kwargs)
    assert manifests == repeated
    assert manifests.verification_receipt is not None
    assert manifests.manifest_dir == output_dir.resolve()
    assert manifests.train["num_windows"] == 15
    assert manifests.holdout["num_windows"] == 5
    assert manifests.train["provenance"]["usable_tokens"] == 30
    assert manifests.holdout["provenance"]["usable_tokens"] == 10
    assert manifests.train["provenance"]["minimum_source_usable_tokens"] == {
        "books": 2,
        "code": 4,
    }
    assert sum(manifests.train["provenance"]["source_usable_tokens"].values()) == 30

    train_counts = _counts_by_label(manifests.train)
    holdout_counts = _counts_by_label(manifests.holdout)
    assert train_counts["code"] >= 2
    assert train_counts["books"] >= 1
    assert holdout_counts["code"] >= 1
    assert holdout_counts["books"] >= 1

    train_starts = _starts_by_id(manifests.train)
    holdout_starts = _starts_by_id(manifests.holdout)
    for source_id in set(train_starts) | set(holdout_starts):
        assert train_starts.get(source_id, set()).isdisjoint(holdout_starts.get(source_id, set()))
        assert all(start % 3 == 0 for start in train_starts.get(source_id, set()))
        assert all(start % 3 == 0 for start in holdout_starts.get(source_id, set()))

    train_path = output_dir / "train.json"
    holdout_path = output_dir / "holdout.json"
    train, holdout = builder.write_replay_manifests(
        manifests,
        train_path=train_path,
        holdout_path=holdout_path,
    )
    assert train.num_windows == 15
    assert holdout.num_windows == 5
    assert train.tokenizer == builder.S002_TOKENIZER
    assert train.provenance["parent_mix"] == "OLMo-mix-0925"
    assert train.provenance["parent_checkpoint"] == builder.S002_PARENT_CHECKPOINT
    assert train.provenance["parent_paths_sha256"] == catalog.parent_paths_sha256
    assert train.provenance["parent_mix_sha256"] == catalog.parent_mix_sha256
    assert train.provenance["upstream_provenance_sha256"] == catalog.upstream_provenance_sha256
    assert train.provenance["instance_filter"] == builder.S002_INSTANCE_FILTER
    assert train.provenance["materialized_sources_sha256"] == catalog.materialized_sources_sha256
    receipt_path = output_dir / builder.VERIFICATION_RECEIPT_FILENAME
    assert receipt_path.is_file()
    assert (
        train.provenance["verification_receipt_sha256"]
        == hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    )
    assert (
        holdout.provenance["verification_receipt_sha256"]
        == train.provenance["verification_receipt_sha256"]
    )
    assert train.provenance["split"] == "train"
    assert holdout.provenance["split"] == "holdout"

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        builder.write_replay_manifests(
            manifests,
            train_path=train_path,
            holdout_path=holdout_path,
        )


def test_catalog_validation_rejects_schema_size_hash_and_remote_paths(tmp_path: Path, builder):
    catalog_path, catalog_data, provenance_path, provenance_data = _write_catalog(tmp_path, builder)

    catalog_path.write_text(
        '{"format":"olmo_native_text_replay_source_catalog",'
        '"version":1,"version":1,"sources":[]}'
    )
    with pytest.raises(ValueError, match="repeats object key 'version'"):
        builder.load_source_catalog(catalog_path)

    # Identity and byte metadata must not be caller-supplied through the local-path catalog.
    for field_name, value in (
        ("id", "caller-id"),
        ("source", "caller-source"),
        ("parent_path", "s3://caller/path.npy"),
        ("num_tokens", 48),
        ("sha256", "0" * 64),
    ):
        unknown = deepcopy(catalog_data)
        unknown["sources"][0][field_name] = value
        catalog_path.write_text(json.dumps(unknown))
        with pytest.raises(ValueError, match="unknown fields"):
            builder.load_source_catalog(catalog_path)

    catalog_path.write_text(json.dumps(catalog_data))
    wrong_size = deepcopy(provenance_data)
    wrong_size["sources"][0]["num_tokens"] += 1
    _pin_provenance(provenance_path, builder, wrong_size)
    with pytest.raises(ValueError, match="bytes, expected"):
        builder.load_source_catalog(catalog_path)

    wrong_parent_path = deepcopy(provenance_data)
    wrong_parent_path["sources"][0]["parent_path"] = "s3://ai2-llm/not-the-parent.npy"
    _pin_provenance(provenance_path, builder, wrong_parent_path)
    with pytest.raises(ValueError, match="does not match the pinned checkpoint"):
        builder.load_source_catalog(catalog_path)

    _pin_provenance(provenance_path, builder, provenance_data)
    missing_parent_path = deepcopy(catalog_data)
    missing_parent_path["sources"].pop()
    catalog_path.write_text(json.dumps(missing_parent_path))
    with pytest.raises(ValueError, match="every pinned parent path"):
        builder.load_source_catalog(catalog_path)

    catalog_path.write_text(json.dumps(catalog_data))
    wrong_hash = deepcopy(provenance_data)
    wrong_hash["sources"][0]["sha256"] = "0" * 64
    _pin_provenance(provenance_path, builder, wrong_hash)
    with pytest.raises(ValueError, match="expected authoritative"):
        builder.load_source_catalog(catalog_path)
    # The explicit development escape hatch still uses byte metadata from pinned provenance.
    builder.load_source_catalog(catalog_path, verify_source_hashes=False)

    uppercase_hash = deepcopy(provenance_data)
    uppercase_hash["sources"][0]["sha256"] = uppercase_hash["sources"][0]["sha256"].upper()
    _pin_provenance(provenance_path, builder, uppercase_hash)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        builder.load_source_catalog(catalog_path, verify_source_hashes=False)

    _pin_provenance(provenance_path, builder, provenance_data)
    remote = deepcopy(catalog_data)
    remote["sources"][0]["path"] = "s3://bucket/tokens.npy"
    catalog_path.write_text(json.dumps(remote))
    with pytest.raises(ValueError, match="materialized locally"):
        builder.load_source_catalog(catalog_path, verify_source_hashes=False)


def test_production_fails_closed_without_code_pinned_upstream_provenance(tmp_path: Path, builder):
    catalog_path, _, _, _ = _write_catalog(tmp_path, builder)
    builder.S002_UPSTREAM_PROVENANCE_FILE = None
    builder.S002_UPSTREAM_PROVENANCE_SHA256 = None

    with pytest.raises(ValueError, match="Production native replay is closed"):
        builder.load_source_catalog(catalog_path)


def test_pinned_parent_mix_and_upstream_provenance_bytes_cannot_drift(tmp_path: Path, builder):
    catalog_path, _, provenance_path, _ = _write_catalog(tmp_path, builder)

    provenance_path.write_text("{}")
    with pytest.raises(ValueError, match="Pinned s002 upstream byte provenance has SHA-256"):
        builder.load_source_catalog(catalog_path)

    _, _, _, _ = _write_catalog(tmp_path, builder)
    parent_mix_path = Path(builder.S002_PARENT_MIX_FILE)
    parent_mix_path.write_text(parent_mix_path.read_text() + "tampered,row\n")
    with pytest.raises(ValueError, match="Pinned OLMo-mix-0925 manifest has SHA-256"):
        builder.load_source_catalog(catalog_path)


def test_hash_verification_bypass_cannot_emit_a_production_receipt(tmp_path: Path, builder):
    catalog_path, _, _, _ = _write_catalog(tmp_path, builder)
    catalog = builder.load_source_catalog(catalog_path, verify_source_hashes=False)

    manifests = builder.build_replay_manifests(
        catalog,
        manifest_dir=tmp_path / "output",
        sequence_length=3,
        train_usable_tokens=20,
        holdout_usable_tokens=4,
    )

    assert manifests.verification_receipt is None
    assert "verification_receipt_sha256" not in manifests.train["provenance"]
    assert "verification_receipt_sha256" not in manifests.holdout["provenance"]


def test_budget_and_rare_source_quota_validation_is_exact(tmp_path: Path, builder):
    catalog_path, _, _, _ = _write_catalog(tmp_path, builder)
    catalog = builder.load_source_catalog(catalog_path)
    base = {
        "catalog": catalog,
        "manifest_dir": tmp_path / "output",
        "sequence_length": 3,
        "train_usable_tokens": 20,
        "holdout_usable_tokens": 4,
    }

    with pytest.raises(ValueError, match="never rounded"):
        builder.build_replay_manifests(**dict(base, train_usable_tokens=21))
    with pytest.raises(ValueError, match="unknown source label"):
        builder.build_replay_manifests(
            **base,
            train_minimum_source_tokens={"not-in-parent-mix": 2},
        )
    with pytest.raises(ValueError, match="above capacity"):
        builder.build_replay_manifests(
            **base,
            train_minimum_source_tokens={"books": 6},
        )
    with pytest.raises(ValueError, match="materialized sources hold only"):
        builder.build_replay_manifests(
            **dict(base, train_usable_tokens=66, holdout_usable_tokens=4)
        )


def test_cli_dry_run_is_read_only_and_reports_exact_allocation(tmp_path: Path, builder, capsys):
    catalog_path, _, _, _ = _write_catalog(tmp_path, builder)
    output_dir = tmp_path / "dry-run-output"

    result = builder.main(
        [
            "--catalog",
            str(catalog_path),
            "--output-dir",
            str(output_dir),
            "--sequence-length",
            "3",
            "--train-tokens",
            "20",
            "--holdout-tokens",
            "4",
            "--train-min-source-tokens",
            "books=2",
            "--holdout-min-source-tokens",
            "books=2",
            "--dry-run",
        ]
    )

    assert result == 0
    assert not output_dir.exists()
    summary = json.loads(capsys.readouterr().out)
    assert summary["dry_run"] is True
    assert summary["train"]["usable_tokens"] == 20
    assert summary["holdout"]["usable_tokens"] == 4
    assert summary["raw_tokens_per_window"] == 3
    assert summary["loss_tokens_per_window"] == 2
    help_text = builder._build_parser().format_help()
    assert "already-materialized" in help_text
    assert "never modified" in help_text
    assert "SOURCE=TOKENS" in help_text
    assert "--parent-checkpoint" not in help_text
