"""Bounded medium-model comparison; no architecture or precision search."""

import re
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
