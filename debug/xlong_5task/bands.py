"""Shared band spec + Qwen3.5 calibration for the 5-task 2k->256k short-skewed build.

Counts fall roughly as 1/T^2 (short-heavy), with a hard floor of 300 examples in the top
128-256k band per the user's requirement. See ``PLAN.md``.
"""

#: (lo_tokens, hi_tokens, n_examples) per band. Sums to 20,000 examples / ~344M tokens per task.
BANDS = [
    (2048, 4096, 6000),
    (4096, 8192, 5000),
    (8192, 16384, 4000),
    (16384, 32768, 2700),
    (32768, 65536, 1400),
    (65536, 131072, 600),
    (131072, 262144, 300),  # <- floor: never drop below 300 here
]

#: Measured Qwen3.5-4B-Base fits: tokens = intercept + tok_per_doc * n_docs.
#: From debug/xlong_5task/calibrate_and_audit.py over 300 real examples/task (MAPE 1.2-3.3%).
CALIB = {
    "contradiction": (188.1, 42.413),
    "nq": (26.1, 156.538),
    "outlier": (-5.4, 144.328),
    "rerank": (13.8, 85.227),
}


def n_for_tokens(task: str, tokens: float) -> int:
    """Document count that renders to roughly ``tokens`` Qwen3.5 tokens for ``task``."""
    a, b = CALIB[task]
    return max(1, int(round((tokens - a) / b)))


def band_label(tokens: float) -> str:
    """Human label for the band a rendered length falls in."""
    for lo, hi, _ in BANDS:
        if tokens < hi:
            return f"{lo // 1024}k-{hi // 1024}k"
    return "256k+"


def draw_plan(seed: int = 42):
    """Deterministic list of per-example target token counts, shuffled across bands.

    Shuffling matters: a sharded run must not put every long example in one shard, or that
    shard alone dominates wall-clock and memory.
    """
    import random

    rng = random.Random(seed)
    plan = []
    for lo, hi, count in BANDS:
        for _ in range(count):
            plan.append(rng.uniform(lo, hi))
    rng.shuffle(plan)
    return plan
