"""
Metrics and evaluator classes.
"""

from .evaluator import Evaluator
from .lm_evaluator import LMEvaluator
from .matched_wrong_image import (
    MultimodalFixedValidationDataset,
    MultimodalMatchedWrongImageDataset,
    build_matched_wrong_image_pairing,
    matched_wrong_image_pairing_sha256,
    serialize_matched_wrong_image_pairing,
    validate_matched_wrong_image_pairing,
)
from .metrics import MeanMetric, Metric
from .multimodal_lm_evaluator import (
    MultimodalBlankImageEvaluator,
    MultimodalLMEvaluator,
)
from .ssmax_attention_diagnostics import (
    ProbeSequence,
    SSMaxAttentionDiagnosticsCollector,
    SSMaxProbeBatch,
    SSMaxProbeManifest,
    build_probe_manifest,
    capture_ssmax_probe_batches,
    compare_ssmax_attention_reports,
    iter_ssmax_probe_batches,
    probe_manifest_sha256,
    serialize_probe_manifest,
    validate_ssmax_attention_report,
)

__all__ = [
    "Evaluator",
    "LMEvaluator",
    "MultimodalLMEvaluator",
    "MultimodalBlankImageEvaluator",
    "MultimodalFixedValidationDataset",
    "MultimodalMatchedWrongImageDataset",
    "build_matched_wrong_image_pairing",
    "matched_wrong_image_pairing_sha256",
    "serialize_matched_wrong_image_pairing",
    "validate_matched_wrong_image_pairing",
    "Metric",
    "MeanMetric",
    "ProbeSequence",
    "SSMaxAttentionDiagnosticsCollector",
    "SSMaxProbeBatch",
    "SSMaxProbeManifest",
    "build_probe_manifest",
    "capture_ssmax_probe_batches",
    "compare_ssmax_attention_reports",
    "iter_ssmax_probe_batches",
    "probe_manifest_sha256",
    "serialize_probe_manifest",
    "validate_ssmax_attention_report",
]
