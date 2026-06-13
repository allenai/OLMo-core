from __future__ import annotations

import re
from dataclasses import dataclass


CURRENT_FAMILY_MARKERS = (
    "b384k-gpu2-ep1mb8",
    "gpu2-ep1mb16",
    "gpu2-ep1mb8",
    "gpu4-ep1mb4",
    "gpu4-ep1mb8",
    "gpu4-ep1mb16",
    "gpu8-ep1mb2",
    "gpu8-ep1mb4",
    "gpu8-ep1mb16",
)

LR_TAG_RE = re.compile(r"lr([0-9]+(?:\.[0-9]+)?e-[0-9]+)")


@dataclass(frozen=True)
class RunSpec:
    cx: int
    batch_label: str
    batch_tokens: int
    lr_tag: str
    lr: float
    current_family: bool


BATCH_TAGS: tuple[tuple[str, str, int], ...] = (
    ("b128k", "128k", 131_072),
    ("b256k", "256k", 262_144),
    ("b384k", "384k", 393_216),
    ("b512k", "512k", 524_288),
    ("b768k", "768k", 786_432),
    ("b1m", "1M", 1_048_576),
)


def is_ladder_run_name(name: str) -> bool:
    return any(
        marker in name
        for marker in (
            "olmoe3-tiny-275m-cx",
            "olmoe3-moe-a0-810m-cx",
            "olmoe3-moe-a0-1p2b-cx",
            "olmoe3-810m-cx",
            "m480-cx",
        )
    )


def parse_run_spec(name: str) -> RunSpec | None:
    if not is_ladder_run_name(name):
        return None

    cx_match = re.search(r"cx([0-9]+)", name)
    lr_match = LR_TAG_RE.search(name)
    if cx_match is None or lr_match is None:
        return None

    batch_label: str | None = None
    batch_tokens: int | None = None
    for marker, label, tokens in BATCH_TAGS:
        if marker in name:
            batch_label = label
            batch_tokens = tokens
            break

    if batch_label is None:
        if "cx1-lr" in name:
            batch_label, batch_tokens = "2M", 2_097_152
        else:
            return None

    lr_tag = lr_match.group(1)
    return RunSpec(
        cx=int(cx_match.group(1)),
        batch_label=batch_label,
        batch_tokens=batch_tokens,
        lr_tag=lr_tag,
        lr=float(lr_tag),
        current_family=any(marker in name for marker in CURRENT_FAMILY_MARKERS),
    )


def is_analysis_run(name: str) -> bool:
    lowered = name.lower()
    ignored_markers = ("smoke", "smoketest", "sanity", "pilot")
    return not any(marker in lowered for marker in ignored_markers)


def model_label_from_name(name: str) -> str:
    if "tiny-275m" in name or "eg-275m" in name:
        return "275m"
    if "mid-480m" in name or "mid_480m" in name or "m480-cx" in name or "480m" in name:
        return "480m"
    if "810m" in name:
        return "810m"
    if "1p2b" in name:
        return "1p2b"
    return "unknown"


def family_label_from_name(name: str) -> str:
    if "b384k" in name and "gpu2-ep1mb8" in name:
        return "b384k-gpu2-ep1mb8"
    for marker in (
        "gpu8-ep1mb16",
        "gpu8-ep1mb4",
        "gpu8-ep1mb2",
        "gpu4-ep1mb16",
        "gpu4-ep1mb8",
        "gpu4-ep1mb4",
        "gpu2-ep1mb16",
        "gpu2-ep1mb8",
    ):
        if marker in name:
            return marker
    if "-n2-" in name or "-n2_" in name:
        return "n2"
    # New stable names intentionally omit systems settings. Keep optimizer-batch
    # policy visible as the family when no GPU/EP/microbatch marker is present.
    if "b384k" in name:
        return "b384k"
    if "b512k" in name:
        return "b512k"
    if "b768k" in name:
        return "b768k"
    if "b256k" in name:
        return "b256k"
    return "original"
