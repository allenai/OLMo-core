"""Batch-shape filters shared by ladder experiment plotters."""

from __future__ import annotations

import re


CANONICAL_BATCH_K_BY_CX = {
    1: 256,
    2: 384,
    4: 512,
    8: 768,
}

BATCH_RE = re.compile(r"b([0-9]+)k")


def has_canonical_batch_for_cx(name: str, cx: int) -> bool:
    """Return whether a run name matches the canonical batch shape for a Cx.

    New experiment names carry explicit `b*K` tags. For those, require the tag
    to match the canonical global batch for the Chinchilla multiple. Some older
    baseline names predate explicit batch tags; keep those unless they are a
    known non-canonical legacy shape, such as 275M Cx2 `gpu2-ep1mb16`/b256k.
    """

    canonical_k = CANONICAL_BATCH_K_BY_CX.get(cx)
    if canonical_k is None:
        return True

    match = BATCH_RE.search(name)
    if match is not None:
        return int(match.group(1)) == canonical_k

    lowered = name.lower()
    if cx == 2 and "olmoe3-tiny-275m-cx2" in lowered and "gpu2-ep1mb16" in lowered:
        return False

    return True
