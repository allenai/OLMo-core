"""Shared on-disk data roots for Molmo2 multimodal training."""

from __future__ import annotations

import os

_DEFAULT_MOLMO_DATA = "/weka/oe-training-default/mm-olmo"
_DEFAULT_TORCH_DATASETS = f"{_DEFAULT_MOLMO_DATA}/torch_datasets"

MOLMO_DATA_DIR = os.environ.get("MOLMO_DATA_DIR", _DEFAULT_MOLMO_DATA)
TORCH_DATASETS = os.path.join(MOLMO_DATA_DIR, "torch_datasets")
PIXMO_DATASETS = os.path.join(TORCH_DATASETS, "pixmo_datasets")
TULU4_DATA = os.path.join(TORCH_DATASETS, "olmo-3-instruct-sft-no-tools-classified-v3")
ACADEMIC_DATASETS = os.path.join(TORCH_DATASETS, "academic_datasets")
# allenai/olmOCR-mix-1025 as materialised by mm_olmo's ``OlmOcrMixConfig.download``: the
# per-subset/split parquets plus the PDF tarballs expanded into ``pdfs/<chunk>/<arcname>``.
OLMOCR_MIX = os.path.join(TORCH_DATASETS, "olmocr_mix_1025")
# The oe-encoder team's webdataset-style caption tars (``<key>.jpg|png`` + ``<key>.json`` per
# sample), used for the OCR sources in :mod:`.mixtures.ocr`. Another project's directory, so
# it is overridable with the OE_ENCODER_DATA_DIR env var.
OE_ENCODER_DATA = os.environ.get("OE_ENCODER_DATA_DIR", "/weka/oe-training-default/oe-encoder")

# HARDCODED personal dataset (chrisc's audited, image-grouped PixMo-Points build on weka).
# mm_olmo's ``PixMoPointV2.PATH`` reads this same directory, so it is the canonical location
# for now. Override with the PIXMO_POINTS_V2_DIR env var, or per run through the dataset
# config (``--pointing_v2.dataset_path=...`` in Molmo2-Stage1.py).
PIXMO_POINTS_V2 = os.environ.get(
    "PIXMO_POINTS_V2_DIR", "/weka/oe-training-default/chrisc/pixmo-points-with-masks-v17"
)

__all__ = [
    "MOLMO_DATA_DIR",
    "TORCH_DATASETS",
    "PIXMO_DATASETS",
    "PIXMO_POINTS_V2",
    "TULU4_DATA",
    "ACADEMIC_DATASETS",
    "OLMOCR_MIX",
    "OE_ENCODER_DATA",
]
