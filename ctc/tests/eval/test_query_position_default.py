"""
The eval-side ``query_position`` default matches what the checkpoints were trained with.

``query_position`` is a real knob and stays configurable, but its DEFAULT is not free: the CTC-suite
checkpoints train with ``both`` (``{questions}\\n\\n{documents}\\n\\n{questions}``) and the
pre-migration evaluator defaulted to ``both`` to match. A default of ``after`` does not raise -- it
produces a prompt layout the model never saw, measured at nq 0.860 -> 0.074. That is a collapse
reported as a result.

The format fingerprint catches this for any checkpoint that has one. Most existing checkpoints do
not, and there it degrades to a warning, so the default has to be right on its own.
"""

from __future__ import annotations

import inspect

import pytest

from ctc.eval import prefill, runner
from ctc.eval.backends import hf, native, vllm

EXPECTED = "both"


def _default(fn, name="query_position"):
    return inspect.signature(fn).parameters[name].default


#: Where each backend actually exposes the knob. The native backend takes it on the constructor;
#: hf and vllm take it on their module-level ``build()``, which is what ``backends.load`` calls.
#: Checking whichever entry point a caller really reaches is the point -- a default that is right on
#: an unused constructor and wrong on the used factory is no default at all.
ENTRY_POINTS = [
    ("native.NativeBackend", native.NativeBackend.__init__),
    ("hf.build", hf.build),
    ("vllm.build", vllm.build),
    ("prefill.StructuralPrefill", prefill.StructuralPrefill.__init__),
]


@pytest.mark.parametrize("name,fn", ENTRY_POINTS, ids=[n for n, _ in ENTRY_POINTS])
def test_every_entry_point_defaults_to_the_trained_layout(name, fn):
    assert _default(fn) == EXPECTED


def test_the_runner_config_agrees():
    from dataclasses import fields

    field = next(f for f in fields(runner.EvalConfig) if f.name == "query_position")
    assert field.default == EXPECTED
