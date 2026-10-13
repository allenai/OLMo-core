"""Corpus-reasoning: long-context task data generation, evaluation harness, and viz.

Vendored from the standalone PrasannS/corpus-reasoning repo; only the files that research/ code imports.
Import as ``corpus_reasoning.*`` (``PYTHONPATH=$PWD/research/common``). The large tokenized corpora and
checkpoints stay on /scratch -- only code lives here.

Most ``corpus_reasoning.lib.*`` / ``corpus_reasoning.eval.*`` modules are now aliases of their
canonical copies: prompt rendering / data format / io live in
:mod:`olmo_core.data.corpus_reasoning_prompts`, and the eval harness (metrics, eval_tasks, evaluate,
...) lives in the ``ctc_eval`` package under ``recipes/eval/legacy_native``. Importing this package
puts that directory on ``sys.path`` so the aliases (and the research-only modules here that import
them) resolve without a separate ``PYTHONPATH`` entry.
"""

import os as _os
import sys as _sys

_LEGACY_NATIVE = _os.path.normpath(
    _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "recipes", "eval", "legacy_native")
)
if _os.path.isdir(_LEGACY_NATIVE) and _LEGACY_NATIVE not in _sys.path:
    _sys.path.append(_LEGACY_NATIVE)
