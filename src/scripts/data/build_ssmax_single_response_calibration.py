"""Build immutable SSMax single-response loss-mass calibration evidence.

The producer reuses the exact train population and epoch/index panel from an already reviewed
perception or joint source audit.  Every visual row is rebuilt through the canonical selected
dataset and :class:`SSMaxSingleResponseDataset`; branch receipts and complete serialized-row
hashes are retained as compact cumulative digests.  Joint native-text replay is not projected,
so its audited mean is carried unchanged and explicitly identified.

Example::

    python src/scripts/data/build_ssmax_single_response_calibration.py \
      --phase perception \
      --selection-manifest /path/to/perception-provenance.json \
      --expected-selection-manifest-sha256 <sha256> \
      --source-audit /path/to/perception-source-audit.json \
      --expected-source-audit-sha256 <sha256> \
      --expected-source-audit-fingerprint <sha256> \
      --hf-cache-dir /weka/.../hf-cache/hub \
      --output /path/to/ssmax-perception-single-response-calibration.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from olmo_core.data.multimodal.ssmax_single_response import (
    SSMAX_SINGLE_RESPONSE_CALIBRATION_FORMAT,
    SSMAX_SINGLE_RESPONSE_CALIBRATION_VERSION,
    SSMAX_SINGLE_RESPONSE_PROJECTION_SEED,
    SSMaxSingleResponseDataset,
    ssmax_single_response_calibration_summary,
    ssmax_single_response_projection_contract,
)
from olmo_core.data.multimodal.vision_alignment_joint_provenance import (
    build_selected_joint_dataset,
    load_joint_visual_projection_manifest,
)
from olmo_core.data.multimodal.vision_alignment_joint_sources import (
    JOINT_VISUAL_SOURCE_NAMES,
    VisionAlignmentJointSourceSpec,
)
from olmo_core.data.multimodal.vision_alignment_perception_provenance import (
    PERCEPTION_SOURCE_NAMES,
    build_selected_perception_dataset,
    load_perception_provenance_manifest,
)
from olmo_core.data.multimodal.vision_alignment_perception_sources import (
    VisionAlignmentPerceptionSourceSpec,
)
from olmo_core.data.multimodal.vision_alignment_sources import (
    load_pinned_vision_alignment_tokenizer,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("perception", "joint"), required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--expected-selection-manifest-sha256", required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--expected-source-audit-sha256", required=True)
    parser.add_argument("--expected-source-audit-fingerprint", required=True)
    parser.add_argument("--hf-cache-dir", required=True)
    parser.add_argument("--seed", type=int, default=SSMAX_SINGLE_RESPONSE_PROJECTION_SEED)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--created-at")
    return parser.parse_args(argv)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"Duplicate JSON key {key!r}")
        output[key] = value
    return output


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_bytes(), object_pairs_hook=_strict_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Invalid JSON artifact {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact {path} must contain an object")
    return payload


def _audit_panel(source: Mapping[str, Any], *, phase: str) -> tuple[tuple[int, int], ...]:
    indices = source.get("probe_indices")
    raw_epochs = source.get("probe_epochs")
    if not isinstance(indices, list) or any(type(index) is not int for index in indices):
        raise ValueError("Source-audit probe indices are invalid")
    if phase == "perception":
        if type(raw_epochs) is not int or raw_epochs <= 0:
            raise ValueError("Perception source-audit probe epochs are invalid")
        epochs = tuple(range(raw_epochs))
    else:
        if not isinstance(raw_epochs, list) or any(type(epoch) is not int for epoch in raw_epochs):
            raise ValueError("Joint source-audit probe epochs are invalid")
        epochs = tuple(raw_epochs)
    panel = tuple((int(index), int(epoch)) for epoch in epochs for index in indices)
    row_hashes = source.get("serialized_row_hashes")
    if not isinstance(row_hashes, list) or len(row_hashes) != len(panel):
        raise ValueError("Source-audit row hashes and epoch/index panel differ")
    return panel


def _artifact_reference(path: Path, *, semantic_sha256: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "raw_sha256": _sha256_file(path)}
    if semantic_sha256 is not None:
        result["content_sha256"] = semantic_sha256
    return result


def _visual_loss_token_weighting(source_spec: Any, *, phase: str) -> str:
    """Resolve the exact parent visual weighting from either canonical phase spec."""

    if phase == "perception":
        if type(source_spec) is not VisionAlignmentPerceptionSourceSpec:
            raise ValueError("Perception calibration requires its exact visual source spec type")
        source_spec.validate_production_contract()
        parent_spec = source_spec
    elif phase == "joint":
        if type(source_spec) is not VisionAlignmentJointSourceSpec:
            raise ValueError("Joint calibration requires its exact joint source spec type")
        source_spec.validate_production_contract()
        parent_spec = source_spec.perception_spec
        if type(parent_spec) is not VisionAlignmentPerceptionSourceSpec:
            raise ValueError("Joint calibration lacks its exact parent perception source spec")
    else:
        raise ValueError(f"Unknown SSMax calibration phase {phase!r}")
    weighting = parent_spec.loss_token_weighting
    canonical = source_spec.as_canonical_dict()
    if (
        not isinstance(weighting, str)
        or canonical.get("loss_token_weighting") != weighting
        or weighting != "root_subsegments_root_tokens"
    ):
        raise ValueError("SSMax visual loss-token weighting differs from the production parent")
    return weighting


def _fsync_directory(path: Path) -> None:
    """Best-effort directory fsync for filesystems that expose the POSIX primitive."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        # Some network filesystems provide atomic links but reject directory fsync.
        pass
    finally:
        os.close(descriptor)


def _write_json_no_replace(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably publish canonical JSON without ever replacing an existing target."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file_handle:
            file_handle.write(_canonical_bytes(payload) + b"\n")
            file_handle.flush()
            os.fsync(file_handle.fileno())
        # A same-directory hard link is atomic and fails with FileExistsError if another
        # producer won the race after our initial preflight.
        os.link(temporary, path)
        temporary.unlink()
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.seed != SSMAX_SINGLE_RESPONSE_PROJECTION_SEED:
        raise ValueError(
            "--seed must equal the canonical SSMax data/projection seed "
            f"{SSMAX_SINGLE_RESPONSE_PROJECTION_SEED}"
        )
    selection_path = args.selection_manifest.expanduser().resolve()
    audit_path = args.source_audit.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable calibration {output_path}")
    if _sha256_file(selection_path) != args.expected_selection_manifest_sha256:
        raise ValueError("Selection manifest differs from its CLI raw SHA-256 pin")
    audit_raw_sha = _sha256_file(audit_path)
    if audit_raw_sha != args.expected_source_audit_sha256:
        raise ValueError("Source audit differs from its CLI raw SHA-256 pin")
    audit = _load_json(audit_path)
    unsigned_audit = dict(audit)
    recorded_fingerprint = unsigned_audit.pop("fingerprint", None)
    if (
        recorded_fingerprint != args.expected_source_audit_fingerprint
        or _canonical_sha256(unsigned_audit) != recorded_fingerprint
        or audit.get("status") != "ok"
        or audit.get("phase") != args.phase
        or audit.get("failures") != []
    ):
        raise ValueError("Source audit semantic identity or success status differs")
    raw_inputs = audit.get("inputs")
    raw_summaries = audit.get("sources")
    if not isinstance(raw_inputs, Mapping) or not isinstance(raw_summaries, Mapping):
        raise ValueError("Source audit lacks inputs/summaries")

    selection_payload = _load_json(selection_path)
    selection_content_sha = selection_payload.get("content_sha256")
    if (
        not isinstance(selection_content_sha, str)
        or _canonical_sha256(
            {key: value for key, value in selection_payload.items() if key != "content_sha256"}
        )
        != selection_content_sha
    ):
        raise ValueError("Selection manifest semantic SHA-256 differs")

    if args.phase == "perception":
        manifest = load_perception_provenance_manifest(
            selection_path,
            expected_sha256=args.expected_selection_manifest_sha256,
            verify_finevision_materialization=False,
            load_image_path_signatures=False,
        )
        source_spec = manifest.source_spec
        visual_sources = tuple(PERCEPTION_SOURCE_NAMES)
        unprojected_sources: tuple[str, ...] = ()

        def build_dataset(source: str, logical_split: str) -> Any:
            return build_selected_perception_dataset(
                manifest,
                tokenizer,
                token_ids,
                source,
                logical_split=logical_split,
                validate_required_annotations=True,
                verify_finevision_materialization=False,
            )

    else:
        # The joint loader additionally verifies that the projection was built for these exact
        # tokenizer image-token IDs, so it is loaded after tokenizer preparation below.
        manifest = None
        source_spec = None
        visual_sources = tuple(JOINT_VISUAL_SOURCE_NAMES)
        unprojected_sources = ("native_text_replay",)

    spec_payload = (
        source_spec.as_canonical_dict()
        if source_spec is not None
        else selection_payload.get("source_spec")
    )
    if not isinstance(spec_payload, Mapping):
        raise ValueError("Selection manifest lacks a canonical source specification")
    tokenizer, token_ids = load_pinned_vision_alignment_tokenizer(
        identifier=str(spec_payload["tokenizer_id"]),
        revision=str(spec_payload["tokenizer_revision"]),
        expected_fingerprint=str(spec_payload["tokenizer_fingerprint"]),
        cache_dir=args.hf_cache_dir,
    )
    if args.phase == "joint":
        manifest = load_joint_visual_projection_manifest(
            selection_path,
            expected_token_ids=token_ids,
            expected_sha256=args.expected_selection_manifest_sha256,
            verify_finevision_materialization=False,
            load_image_path_signatures=False,
        )
        source_spec = manifest.source_spec

        def build_dataset(source: str, logical_split: str) -> Any:
            return build_selected_joint_dataset(
                manifest,
                tokenizer,
                token_ids,
                source,
                logical_split=logical_split,
                validate_required_annotations=True,
            )

    assert source_spec is not None
    loss_token_weighting = _visual_loss_token_weighting(source_spec, phase=args.phase)
    contract = ssmax_single_response_projection_contract(
        seed=args.seed,
        loss_token_weighting=loss_token_weighting,
    )
    if not set(visual_sources).issubset(raw_inputs) or set(raw_inputs) != set(raw_summaries):
        raise ValueError("Source-audit population differs from the selected visual sources")

    source_results: dict[str, Any] = {}
    validation_results: dict[str, Any] = {}
    projected_means: dict[str, float] = {}
    for source in visual_sources:
        selected = build_dataset(source, "train")
        projected = SSMaxSingleResponseDataset(
            selected,
            source_name=source,
            logical_split="train",
            seed=args.seed,
            loss_token_weighting=loss_token_weighting,
        )
        summary = ssmax_single_response_calibration_summary(
            projected,
            _audit_panel(raw_inputs[source], phase=args.phase),
        )
        source_results[source] = summary
        projected_means[source] = float(summary["mean_sum_loss_masks"])
        print(
            json.dumps(
                {"phase": args.phase, "source": source, "split": "train", "summary": summary},
                sort_keys=True,
            ),
            flush=True,
        )

        selected_validation = build_dataset(source, "validation")
        if len(selected_validation) < 512:
            raise ValueError(f"SSMax {source!r} validation projection requires at least 512 rows")
        projected_validation = SSMaxSingleResponseDataset(
            selected_validation,
            source_name=source,
            logical_split="validation",
            seed=args.seed,
            loss_token_weighting=loss_token_weighting,
        )
        validation_summary = ssmax_single_response_calibration_summary(
            projected_validation,
            tuple((index, 0) for index in range(len(projected_validation))),
        )
        validation_results[source] = validation_summary
        print(
            json.dumps(
                {
                    "phase": args.phase,
                    "source": source,
                    "split": "validation",
                    "summary": validation_summary,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    for source in unprojected_sources:
        summary = raw_summaries.get(source)
        if not isinstance(summary, Mapping):
            raise ValueError(f"Source audit lacks {source!r} summary")
        mean = summary.get("mean_sum_loss_masks")
        if isinstance(mean, bool) or not isinstance(mean, (int, float)) or float(mean) <= 0:
            raise ValueError(f"Source audit has invalid {source!r} mean loss weight")
        projected_means[source] = float(mean)

    script_path = Path(__file__).resolve()
    implementation_path = Path(SSMaxSingleResponseDataset.__module__.replace(".", "/") + ".py")
    implementation_path = script_path.parents[2] / implementation_path
    payload: dict[str, Any] = {
        "format": SSMAX_SINGLE_RESPONSE_CALIBRATION_FORMAT,
        "version": SSMAX_SINGLE_RESPONSE_CALIBRATION_VERSION,
        "status": "ok",
        "created_at": args.created_at or datetime.now(timezone.utc).isoformat(),
        "phase": args.phase,
        "producer": {
            "path": str(script_path.relative_to(script_path.parents[3])),
            "sha256": _sha256_file(script_path),
        },
        "projection_implementation": {
            "path": str(implementation_path.relative_to(script_path.parents[3])),
            "sha256": _sha256_file(implementation_path),
        },
        "projection_contract": contract,
        "source_audit": _artifact_reference(audit_path, semantic_sha256=str(recorded_fingerprint)),
        "selection_manifest": _artifact_reference(
            selection_path, semantic_sha256=selection_content_sha
        ),
        "sources": source_results,
        "validation_preflight": validation_results,
        "unprojected_sources": list(unprojected_sources),
        "projected_mean_loss_weight": projected_means,
        "errors": [],
    }
    payload["content_sha256"] = _canonical_sha256(payload)
    _write_json_no_replace(output_path, payload)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "raw_sha256": _sha256_file(output_path),
                "content_sha256": payload["content_sha256"],
                "projected_mean_loss_weight": projected_means,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
