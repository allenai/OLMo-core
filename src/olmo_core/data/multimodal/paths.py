"""Shared on-disk data roots for Molmo2 multimodal training."""

from __future__ import annotations

import os
from typing import Optional

from olmo_core.exceptions import OLMoConfigurationError

_DEFAULT_MOLMO_DATA = "/weka/oe-training-default/mm-olmo"
_DEFAULT_TORCH_DATASETS = f"{_DEFAULT_MOLMO_DATA}/torch_datasets"

MOLMO_DATA_DIR = os.environ.get("MOLMO_DATA_DIR", _DEFAULT_MOLMO_DATA)

# Staging root for datasets that aren't in the shared corpus yet (v10's FineVision
# symlinks and the generated DynaMath variants). Deliberately has *no* default: it used
# to fall back to an individual's home directory on weka, which made a library default
# depend on one person's scratch space. Callers that need it must set the env var or
# pass an explicit path.
MOLMO_EXPERIMENT_DATA_DIR: Optional[str] = os.environ.get("MOLMO_EXPERIMENT_DATA_DIR")

TORCH_DATASETS = os.path.join(MOLMO_DATA_DIR, "torch_datasets")
PIXMO_DATASETS = os.path.join(TORCH_DATASETS, "pixmo_datasets")
TULU4_DATA = os.path.join(TORCH_DATASETS, "olmo-3-instruct-sft-no-tools-classified-v3")
ACADEMIC_DATASETS = os.path.join(TORCH_DATASETS, "academic_datasets")


def require_experiment_data_dir(what: str) -> str:
    """Return ``$MOLMO_EXPERIMENT_DATA_DIR``, or raise explaining what needed it.

    Read from the environment on each call rather than at import time so tests and
    launch scripts can set it after this module is imported.
    """
    root = os.environ.get("MOLMO_EXPERIMENT_DATA_DIR")
    if not root:
        raise OLMoConfigurationError(
            f"{what} needs the experimental-data staging root, but "
            "MOLMO_EXPERIMENT_DATA_DIR is not set. Export it to the directory holding "
            "the prepared datasets, or pass an explicit path on the dataset config."
        )
    return root


__all__ = [
    "MOLMO_DATA_DIR",
    "MOLMO_EXPERIMENT_DATA_DIR",
    "TORCH_DATASETS",
    "PIXMO_DATASETS",
    "TULU4_DATA",
    "ACADEMIC_DATASETS",
    "require_experiment_data_dir",
]
