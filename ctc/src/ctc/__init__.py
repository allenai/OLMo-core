"""
Corpus-reasoning task generation and evaluation.

Independent of olmo-core by construction: nothing in this package imports it except
:mod:`ctc.eval.backends.native` and :mod:`ctc.eval.masking.native`, both behind the ``native``
extra. That is what lets ``pip install ctc`` work on a machine with no GPU and no compiler.
"""

__version__ = "0.1.0"
