"""Pure naming/geometry helpers for bounded, retained-allocation medium probes."""

import re

VARIANTS = (
    "baseline",
    "optimized",
    "no-wgrad",
    "no-rs",
    "no-kda",
    "no-routing",
    "core-only",
    "optimized-ackda",
    "optimized-ackda-half",
    "optimized-metrics5",
    "optimized-simple",
    "optimized-lb-batched",
)


def sample_offsets(size, limit=2048):
    """Return exact bounded indices; floating endpoints overflow for large shards."""
    if size < 0 or limit < 1:
        raise ValueError((size, limit))
    count = min(size, limit)
    return [i * (size - 1) // max(count - 1, 1) for i in range(count)]


def parse_test(label):
    """Optional explicit geometry suffix; never silently change a model setting."""
    match = re.fullmatch(r"(.+)-mb([124])-b(8|16|32)mi", label)
    variant, mb, batch = (
        (match[1], int(match[2]), int(match[3]) * 1024**2) if match else (label, None, None)
    )
    if variant not in VARIANTS:
        raise ValueError(f"Unknown medium follow-up test: {label}")
    return variant, mb, batch
