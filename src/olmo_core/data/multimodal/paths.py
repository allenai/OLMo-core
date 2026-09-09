"""Shared on-disk data roots for Molmo2 multimodal training."""

from __future__ import annotations

import os

_DEFAULT_MOLMO_DATA = "/weka/oe-training-default/mm-olmo"
_DEFAULT_TORCH_DATASETS = f"{_DEFAULT_MOLMO_DATA}/torch_datasets"
_DEFAULT_EXPERIMENT_DATA = "/weka/oe-training-default/donovanc/molmo-experimental-data"

MOLMO_DATA_DIR = os.environ.get("MOLMO_DATA_DIR", _DEFAULT_MOLMO_DATA)
MOLMO_EXPERIMENT_DATA_DIR = os.environ.get(
    "MOLMO_EXPERIMENT_DATA_DIR", _DEFAULT_EXPERIMENT_DATA
)
TORCH_DATASETS = os.path.join(MOLMO_DATA_DIR, "torch_datasets")
PIXMO_DATASETS = os.path.join(TORCH_DATASETS, "pixmo_datasets")
TULU4_DATA = os.path.join(TORCH_DATASETS, "olmo-3-instruct-sft-no-tools-classified-v3")
ACADEMIC_DATASETS = os.path.join(TORCH_DATASETS, "academic_datasets")

__all__ = [
    "MOLMO_DATA_DIR",
    "MOLMO_EXPERIMENT_DATA_DIR",
    "TORCH_DATASETS",
    "PIXMO_DATASETS",
    "TULU4_DATA",
    "ACADEMIC_DATASETS",
]
