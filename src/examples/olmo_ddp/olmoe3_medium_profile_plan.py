"""Bounded medium-model comparison; no architecture or precision search."""

import math
import re
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class MediumProfileTopology:
    """Keep PP1/EP8/MB2 and16Mi tokens fixed at either supported GPU count."""

    gpus: int
    ep: int = 8
    microbatch: int = 2
    batch_tokens: int = 16_777_216
    sequence_length: int = 8192

    def __post_init__(self):
        if (
            self.gpus not in (64, 128)
            or self.ep != 8
            or self.microbatch != 2
            or self.batch_tokens != 16_777_216
            or self.sequence_length != 8192
        ):
            raise ValueError("Medium profiling requires64/128GPUs, PP1EP8MB2,16Mi/8192")

    @property
    def nodes(self):
        return self.gpus // 8

    @property
    def accumulation(self):
        return self.batch_tokens // (self.gpus * self.microbatch * self.sequence_length)

    def validate_rank_groups(self, groups, hosts):
        """Require node-local EP groups partitioning the actual world."""
        if sorted(rank for group in groups for rank in group) != list(range(self.gpus)):
            raise ValueError("EP groups must partition all ranks")
        if any(len(group) != 8 or len({hosts[rank] for rank in group}) != 1 for group in groups):
            raise ValueError("Every EP8 group must be node-local")


def named_medium_passes(prefix, capture=False):
    """Use the same naming source for the launcher and its CPU collectors."""
    if not re.fullmatch(r"[a-z0-9-]+", prefix):
        raise ValueError("Expected a plain run name")
    passes = [
        (f"{prefix}-repeat{repeat}-{arm}", arm, "timing")
        for repeat, arms in ((1, ("baseline", "optimized")), (2, ("optimized", "baseline")))
        for arm in arms
    ]
    if capture:
        passes.extend(
            (f"{prefix}-{arm}", arm, mode)
            for mode in ("nsys", "torch")
            for arm in ("baseline", "optimized")
        )
    return passes


def inspect_route_metrics(metrics):
    """Validate per-block route fractions without counting aggregate sums twice.

    Nonzero fractions are supported by the configured capacity-limited EP path.
    They invalidate a dropless throughput claim, not the training execution.
    """
    values = {
        name: float(value)
        for name, value in metrics.items()
        if re.fullmatch(r"train/block \d+/token drop rate", name)
    }
    for name, value in values.items():
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid per-block route fraction: {name}={value}")
    return {
        "blocks": len(values),
        "max_block_drop_fraction": max(values.values()) if values else None,
        "mean_block_drop_fraction": statistics.mean(values.values()) if values else None,
        "blocks_with_drops": sum(value > 0 for value in values.values()),
    }


def routing_window_summary(rows, expected_steps, expected_blocks=23):
    """Keep diagnostic timing separate from a verified dropless workload window."""
    expected_steps = list(expected_steps)
    snapshots = [inspect_route_metrics(row) for row in rows]
    actual_steps = [row["step"] for row in rows]
    complete = (
        len(actual_steps) == len(set(actual_steps))
        and sorted(actual_steps) == sorted(expected_steps)
        and bool(expected_steps)
        and all(item["blocks"] == expected_blocks for item in snapshots)
    )
    measured = [item for item in snapshots if item["blocks"]]
    dropless = complete and all(item["max_block_drop_fraction"] == 0 for item in measured)
    return {
        "telemetry_complete": complete,
        "expected_blocks": expected_blocks,
        "expected_steps": len(expected_steps),
        "observed_steps": len(actual_steps),
        "steps_with_drops": sum(item["blocks_with_drops"] > 0 for item in snapshots),
        "max_block_step_drop_fraction": (
            max(item["max_block_drop_fraction"] for item in measured) if measured else None
        ),
        "mean_block_step_drop_fraction": (
            statistics.mean(item["mean_block_drop_fraction"] for item in measured)
            if measured
            else None
        ),
        "verified_dropless_window": dropless,
        "matched_workload_review_required": not dropless,
        "caveat": (
            "Fractions measure dropped expert assignments, not whole input tokens. "
            "Nonzero drops or missing telemetry require matched-workload review before "
            "claiming an optimization speedup; do not label these as dropless throughput."
        ),
    }
