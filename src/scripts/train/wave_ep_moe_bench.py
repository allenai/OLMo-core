"""Compatibility wrapper for the MoE EP benchmark.

The implementation lives under ``scripts.train.benchmark`` so the rowwise and
DeepEP benchmark pieces can evolve independently.
"""

from __future__ import annotations

from scripts.train.benchmark.wave_ep_moe_bench import main


if __name__ == "__main__":
    main()
