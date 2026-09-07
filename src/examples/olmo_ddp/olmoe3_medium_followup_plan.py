"""Pure naming/geometry helpers for bounded, retained-allocation medium probes."""

import re

VARIANTS = ("baseline", "optimized", "no-wgrad", "no-rs", "no-kda", "no-routing", "core-only")


def parse_test(label):
    """Optional explicit geometry suffix; never silently change a model setting."""
    match = re.fullmatch(r"(.+)-mb([124])-b(8|16|32)mi", label)
    variant, mb, batch = (
        (match[1], int(match[2]), int(match[3]) * 1024**2) if match else (label, None, None)
    )
    if variant not in VARIANTS:
        raise ValueError(f"Unknown medium follow-up test: {label}")
    return variant, mb, batch
