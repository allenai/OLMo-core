#!/usr/bin/env python3
"""Require bitwise equality for two deterministic checkpoint-forward artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def require_equal(name: str, expected: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    if expected.shape != actual.shape:
        raise RuntimeError(
            f"{name}: shape mismatch {tuple(expected.shape)} != {tuple(actual.shape)}"
        )
    if expected.dtype != actual.dtype:
        raise RuntimeError(f"{name}: dtype mismatch {expected.dtype} != {actual.dtype}")
    if not torch.equal(expected, actual):
        unequal = expected.contiguous().view(torch.uint8) != actual.contiguous().view(torch.uint8)
        first = int(torch.nonzero(unequal.reshape(-1), as_tuple=False)[0].item())
        raise RuntimeError(
            f"{name}: values are not bitwise equal; first differing byte={first}, "
            f"differing_bytes={int(unequal.sum().item()):,}"
        )
    return {"shape": list(expected.shape), "dtype": str(expected.dtype), "numel": expected.numel()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=True)
    checks: dict[str, Any] = {}
    checks["input_ids"] = require_equal("input_ids", reference["input_ids"], candidate["input_ids"])
    checks["logits"] = require_equal("logits", reference["logits"], candidate["logits"])
    reference_intermediates = reference["intermediates"]
    candidate_intermediates = candidate["intermediates"]
    if set(reference_intermediates) != set(candidate_intermediates):
        raise RuntimeError(
            "Intermediate-key mismatch: "
            "reference_only="
            f"{sorted(set(reference_intermediates) - set(candidate_intermediates))}, "
            f"candidate_only={sorted(set(candidate_intermediates) - set(reference_intermediates))}"
        )
    for name in sorted(reference_intermediates):
        checks[name] = require_equal(
            name, reference_intermediates[name], candidate_intermediates[name]
        )

    report = {
        "status": "STRICT_PORT_PARITY_PASS",
        "bitwise_equal": True,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "reference_metadata": reference["metadata"],
        "candidate_metadata": candidate["metadata"],
        "tensor_checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
