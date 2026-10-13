"""
Alias of the canonical module named in ``_TARGET`` -- edit that one, not this.

Importing this module returns the canonical module object itself (same globals), and running it as a
script or with ``python -m`` runs the canonical module's ``__main__``.
"""

import importlib
import sys

import corpus_reasoning  # noqa: F401  (puts recipes/eval/legacy_native on sys.path for ctc_eval)

_TARGET = "ctc_eval.lib.adapter_save"

if __name__ == "__main__":
    import runpy

    runpy.run_module(_TARGET, run_name="__main__", alter_sys=True)
else:
    sys.modules[__name__] = importlib.import_module(_TARGET)
