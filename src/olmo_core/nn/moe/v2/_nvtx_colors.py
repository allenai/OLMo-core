"""
nvtx range colors for the fused MoE modules.

The shared :func:`olmo_core._nvtx.maybe_nvtx_annotate` helper is subsystem-agnostic — it just takes a
color. This module holds the MoE domain's color convention so every MoE range is colored
consistently: call sites pick the constant for their subsystem rather than a raw color string.
"""

ROUTING_COLOR = "blue"  # token routing + per-block forward orchestration
EXPERTS_COLOR = "purple"  # expert compute and expert-weight preparation
COMM_COLOR = "green"  # communication / token movement (permute, all-to-all, drop/restore)
TBO_COLOR = "orange"  # two-batch-overlap orchestration
