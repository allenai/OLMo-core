#!/usr/bin/env python3
"""
Standalone helper to plot a composable staged LR schedule.

Edit the constants in the "Config" section and run:
    python src/scripts/train/plot_lr.py
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sqrt
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Stage:
    duration_tokens: int
    shape: str  # "linear" or "cosine"
    start_lr: Optional[float] = None
    start_lr_fraction: Optional[float] = None
    end_lr: Optional[float] = None
    end_lr_fraction: Optional[float] = None


# -----------------------------------------------------------------------------
# Config (edit in place)
# -----------------------------------------------------------------------------
SEQUENCE_LENGTH = 4096
GLOBAL_BATCH_SIZE_SEQ = (8 * 8) * 4
GLOBAL_BATCH_SIZE = GLOBAL_BATCH_SIZE_SEQ * SEQUENCE_LENGTH

MAX_DURATION_TOKENS = int(7000e9)

BASE_LR = 3e-3
BASE_LR = BASE_LR * sqrt(GLOBAL_BATCH_SIZE / (1 * 1024 * 1024))
NUM_EXPERTS = 64 + 16
TOP_K = 3
EXPERT_LR = BASE_LR * sqrt(TOP_K / NUM_EXPERTS)

def as_aligned_tokens(num_tokens: float) -> int:
    return int((num_tokens // GLOBAL_BATCH_SIZE) * GLOBAL_BATCH_SIZE)


STAGES = [
    Stage(
        duration_tokens=as_aligned_tokens(5e9),
        shape="linear",
        start_lr_fraction=0.0,
        end_lr_fraction=1.0,
    ),
    Stage(
        duration_tokens=as_aligned_tokens(45e9),
        shape="cosine",
        end_lr_fraction=0.4,
    ),
    Stage(
        duration_tokens=as_aligned_tokens(6000e9),
        shape="linear",
        end_lr_fraction=0.1,
    ),
]

NUM_SAMPLES = 5000
PLOT_IN_STEPS = False
INCLUDE_EXPERT_CURVE = True

OUTPUT_PATH = Path(__file__).with_name("plot_lr.png")
SHOW_PLOT = True


def _resolve_from_initial(initial_lr: float, value: Optional[float], fraction: Optional[float]) -> float:
    if value is not None:
        return value
    if fraction is None:
        raise ValueError("Expected either absolute value or fraction.")
    return initial_lr * fraction


def _interpolate(shape: str, start_lr: float, end_lr: float, current: int, duration: int) -> float:
    if shape == "linear":
        return start_lr + (end_lr - start_lr) * current / duration
    if shape == "cosine":
        return end_lr + (start_lr - end_lr) * (1 + cos(pi * current / duration)) / 2
    raise ValueError(f"Unsupported shape: {shape}")


def staged_lr(initial_lr: float, current: int, stages: list[Stage]) -> float:
    current = max(current, 0)
    stage_start = 0
    previous_end_lr = initial_lr
    for stage in stages:
        if stage.duration_tokens <= 0:
            raise ValueError("Stage duration_tokens must be > 0.")
        if stage.start_lr is not None and stage.start_lr_fraction is not None:
            raise ValueError("Specify at most one of start_lr or start_lr_fraction.")
        if (stage.end_lr is None) == (stage.end_lr_fraction is None):
            raise ValueError("Specify exactly one of end_lr or end_lr_fraction per stage.")

        if stage.start_lr is None and stage.start_lr_fraction is None:
            start_lr = previous_end_lr
        else:
            start_lr = _resolve_from_initial(initial_lr, stage.start_lr, stage.start_lr_fraction)
        end_lr = _resolve_from_initial(initial_lr, stage.end_lr, stage.end_lr_fraction)

        stage_end = stage_start + stage.duration_tokens
        if current < stage_end:
            return _interpolate(stage.shape, start_lr, end_lr, current - stage_start, stage.duration_tokens)

        previous_end_lr = end_lr
        stage_start = stage_end

    return previous_end_lr


def main() -> None:
    if MAX_DURATION_TOKENS <= 0 or GLOBAL_BATCH_SIZE <= 0:
        raise ValueError("MAX_DURATION_TOKENS and GLOBAL_BATCH_SIZE must be > 0.")
    if len(STAGES) == 0:
        raise ValueError("STAGES must be non-empty.")

    xs_tokens = np.linspace(0, MAX_DURATION_TOKENS, NUM_SAMPLES)
    base_curve = [staged_lr(BASE_LR, int(x), STAGES) for x in xs_tokens]

    if PLOT_IN_STEPS:
        xs = xs_tokens / GLOBAL_BATCH_SIZE
        xlabel = "Step"
        stage_marks = np.cumsum([s.duration_tokens for s in STAGES]) / GLOBAL_BATCH_SIZE
    else:
        xs = xs_tokens / 1e9
        xlabel = "Tokens (billions)"
        stage_marks = np.cumsum([s.duration_tokens for s in STAGES]) / 1e9

    plt.figure(figsize=(10, 5))
    plt.plot(xs, base_curve, label=f"base LR (init={BASE_LR:.3e})", linewidth=2)

    if INCLUDE_EXPERT_CURVE:
        expert_curve = [staged_lr(EXPERT_LR, int(x), STAGES) for x in xs_tokens]
        plt.plot(xs, expert_curve, label=f"expert LR (init={EXPERT_LR:.3e})", linewidth=2, alpha=0.9)

    for i, mark in enumerate(stage_marks, start=1):
        plt.axvline(mark, linestyle="--", alpha=0.5, label=f"stage {i} end")

    plt.title("Composable Staged LR")
    plt.xlabel(xlabel)
    plt.ylabel("Learning rate")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=160)
    print(f"Saved plot to: {OUTPUT_PATH}")

    if SHOW_PLOT:
        plt.show()


if __name__ == "__main__":
    main()
