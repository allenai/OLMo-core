"""
The shared contract: what data generation and evaluation must agree on.

Prompt templates, document serialization, the task registry, answer parsers, metrics, and the
train/eval format fingerprint. Pure Python -- no torch, no transformers, no olmo_core -- so every
other module can depend on it freely. One definition each, read by both halves of the pipeline.

Two layers of protection against drift, guarding different things:

* :mod:`ctc.format.fingerprint` guards **a checkpoint against the data it will be graded on**, at
  runtime. It catches a config that no longer matches how the model was trained.
* ``ctc/tests/format/test_golden_parity.py`` guards **this code against itself**, at test time,
  using a byte-level snapshot of the pre-migration implementation. It catches an edit here that
  would change what any of it emits.

Neither subsumes the other: correct code can be pointed at the wrong checkpoint, and the right
checkpoint can be fed by quietly-changed code.
"""

from . import documents, fingerprint, metrics, parsing, prompts, registry, rungs
from .fingerprint import FormatFingerprint, FormatMismatchError
from .registry import TaskSpec

__all__ = [
    "documents",
    "fingerprint",
    "metrics",
    "parsing",
    "prompts",
    "registry",
    "rungs",
    "FormatFingerprint",
    "FormatMismatchError",
    "TaskSpec",
]
