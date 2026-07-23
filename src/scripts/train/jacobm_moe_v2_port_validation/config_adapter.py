"""Validation-facing aliases for the canonical migration adapter."""

from __future__ import annotations

from pathlib import Path

from olmo_core.nn.transformer import OLMoDDPModelConfig
from scripts.train.jacobm_olmoe_ladder.v2.moe_v2_core_adapter import (
    adapt_train_module_payload,
    build_model_config_from_recorded,
    load_recorded_config,
)


def build_model_config(path: Path) -> OLMoDDPModelConfig:
    return build_model_config_from_recorded(path)


__all__ = ["adapt_train_module_payload", "build_model_config", "load_recorded_config"]
