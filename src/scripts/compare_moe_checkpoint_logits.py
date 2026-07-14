#!/usr/bin/env python3
"""Compare every logit from legacy and converted OLMoDDP checkpoint exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--require-exact",
        action="store_true",
        help="Fail unless all stored FP32 logit values are bitwise equal.",
    )
    return parser.parse_args()


def load_artifact(path: Path) -> Dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a dictionary in {path}")
    for key in ("input_ids", "logits", "metadata"):
        if key not in value:
            raise KeyError(f"Missing {key!r} in {path}")
    return value


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def tensor_bitwise_equal(reference: torch.Tensor, candidate: torch.Tensor) -> bool:
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        return False
    reference_bytes = reference.detach().cpu().contiguous().view(torch.uint8)
    candidate_bytes = candidate.detach().cpu().contiguous().view(torch.uint8)
    return bool(torch.equal(reference_bytes, candidate_bytes))


def unravel_index(flat_index: int, shape: tuple[int, ...]) -> list[int]:
    result = []
    for size in reversed(shape):
        result.append(flat_index % size)
        flat_index //= size
    return list(reversed(result))


def compare_intermediates(
    reference_artifact: Dict[str, Any], candidate_artifact: Dict[str, Any]
) -> list[Dict[str, Any]]:
    reference_values = reference_artifact.get("intermediates", {})
    candidate_values = candidate_artifact.get("intermediates", {})
    if not isinstance(reference_values, dict) or not isinstance(candidate_values, dict):
        raise TypeError("intermediates must be dictionaries")
    if reference_values.keys() != candidate_values.keys():
        raise RuntimeError(
            "Intermediate key mismatch: "
            f"legacy_only={sorted(reference_values.keys() - candidate_values.keys())}, "
            f"olmo_ddp_only={sorted(candidate_values.keys() - reference_values.keys())}"
        )

    comparisons = []
    for name in sorted(reference_values):
        reference = reference_values[name]
        candidate = candidate_values[name]
        if not isinstance(reference, torch.Tensor) or not isinstance(candidate, torch.Tensor):
            raise TypeError(f"Intermediate {name!r} is not a tensor")
        if reference.shape != candidate.shape:
            raise RuntimeError(
                f"Intermediate shape mismatch for {name}: "
                f"{tuple(reference.shape)} != {tuple(candidate.shape)}"
            )
        equal = reference == candidate
        bitwise_equal = tensor_bitwise_equal(reference, candidate)
        comparison: Dict[str, Any] = {
            "name": name,
            "shape": list(reference.shape),
            "dtype": str(reference.dtype),
            "candidate_dtype": str(candidate.dtype),
            "bitwise_equal": bitwise_equal,
            "equal_fraction": float(equal.float().mean()),
            "mismatch_count": reference.numel() - int(equal.sum()),
        }
        if reference.is_floating_point() and candidate.is_floating_point():
            difference = (candidate.float() - reference.float()).abs()
            comparison.update(
                max_abs_diff=float(difference.max()),
                mean_abs_diff=float(difference.mean()),
            )
        comparisons.append(comparison)
    return comparisons


def main() -> None:
    args = parse_args()
    reference_artifact = load_artifact(args.reference)
    candidate_artifact = load_artifact(args.candidate)

    reference_ids = reference_artifact["input_ids"]
    candidate_ids = candidate_artifact["input_ids"]
    reference = reference_artifact["logits"]
    candidate = candidate_artifact["logits"]
    if not all(
        isinstance(value, torch.Tensor)
        for value in (reference_ids, candidate_ids, reference, candidate)
    ):
        raise TypeError("input_ids and logits must all be tensors")
    if not torch.equal(reference_ids, candidate_ids):
        raise RuntimeError("Legacy and OLMoDDP artifacts used different input IDs")
    if reference.shape != candidate.shape:
        raise RuntimeError(
            f"Logit shape mismatch: legacy={tuple(reference.shape)}, "
            f"OLMoDDP={tuple(candidate.shape)}"
        )
    if reference.dtype != torch.float32 or candidate.dtype != torch.float32:
        raise TypeError(f"Expected stored FP32 logits, got {reference.dtype} and {candidate.dtype}")
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise RuntimeError("Non-finite logits found")
    if args.top_k < 1 or args.top_k > reference.shape[-1]:
        raise ValueError(f"Invalid --top-k={args.top_k}")

    difference = (candidate - reference).abs()
    flat_difference = difference.reshape(-1)
    flat_worst = int(flat_difference.argmax())
    worst_index = unravel_index(flat_worst, tuple(reference.shape))
    worst_index_tuple = tuple(worst_index)

    exact_mask = candidate == reference
    tolerance = args.atol + args.rtol * reference.abs()
    tolerance_mask = difference <= tolerance
    reference_topk = torch.topk(reference, args.top_k, dim=-1).indices
    candidate_topk = torch.topk(candidate, args.top_k, dim=-1).indices
    topk_overlap = (candidate_topk.unsqueeze(-1) == reference_topk.unsqueeze(-2)).any(dim=-1)

    exact_count = int(exact_mask.sum())
    within_tolerance_count = int(tolerance_mask.sum())
    numel = reference.numel()
    argmax_match = reference.argmax(dim=-1) == candidate.argmax(dim=-1)
    topk_order_match = reference_topk == candidate_topk
    intermediate_comparisons = compare_intermediates(reference_artifact, candidate_artifact)
    intermediates_bitwise_equal = all(
        comparison["bitwise_equal"] for comparison in intermediate_comparisons
    )
    logits_bitwise_equal = tensor_bitwise_equal(reference, candidate)
    exact_match = logits_bitwise_equal and intermediates_bitwise_equal
    report = {
        "status": (
            "LOGITS_MATCH"
            if bool(tolerance_mask.all()) and intermediates_bitwise_equal
            else "LOGITS_DIFFER"
        ),
        "reference": str(args.reference.resolve()),
        "candidate": str(args.candidate.resolve()),
        "reference_metadata": reference_artifact["metadata"],
        "candidate_metadata": candidate_artifact["metadata"],
        "shape": list(reference.shape),
        "num_logits": numel,
        "bitwise_equal": logits_bitwise_equal,
        "intermediates_bitwise_equal": intermediates_bitwise_equal,
        "exact_match": exact_match,
        "exact_count": exact_count,
        "exact_fraction": exact_count / numel,
        "atol": args.atol,
        "rtol": args.rtol,
        "within_tolerance_count": within_tolerance_count,
        "within_tolerance_fraction": within_tolerance_count / numel,
        "max_abs_diff": float(difference.max()),
        "mean_abs_diff": float(difference.mean()),
        "rmse": math.sqrt(float(torch.mean((candidate - reference).square()))),
        "worst_index": worst_index,
        "worst_reference_logit": float(reference[worst_index_tuple]),
        "worst_candidate_logit": float(candidate[worst_index_tuple]),
        "argmax_match_count": int(argmax_match.sum()),
        "argmax_match_fraction": float(argmax_match.float().mean()),
        "top_k": args.top_k,
        "topk_order_match_fraction": float(topk_order_match.float().mean()),
        "topk_set_overlap_fraction": float(topk_overlap.float().mean()),
        "input_ids_sha256": tensor_sha256(reference_ids),
        "reference_logits_sha256": tensor_sha256(reference),
        "candidate_logits_sha256": tensor_sha256(candidate),
        "intermediate_comparisons": intermediate_comparisons,
    }
    if args.require_exact and not exact_match:
        report["status"] = "LOGITS_DIFFER"

    rendered = json.dumps(report, indent=2)
    print(rendered, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    if args.require_exact and not exact_match:
        raise SystemExit("Full logits and captured intermediates are not bitwise identical")
    if not bool(tolerance_mask.all()):
        raise SystemExit("Full logits exceed the requested numerical tolerance")


if __name__ == "__main__":
    main()
