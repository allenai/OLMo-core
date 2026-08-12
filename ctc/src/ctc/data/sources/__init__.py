"""
Corpus loaders: one module per corpus, and the pool types the generators build from.

Two rules hold across every module here, and both are load-bearing.

**The pool type is a plain dataclass; only the loader touches the network.** So a test builds a
five-document pool by hand and exercises the *construction* -- gold placement, index base, filler
provenance, ladder nesting -- with no HF download, no Lucene index and no GPU. The pre-migration
tree had no such seam: every generator's ``main()`` loaded its corpus first, so nothing about
example construction could be tested at all.

**Heavy imports live inside the loader that needs them.** ``datasets``, ``pyserini``,
``transformers`` and ``torch`` are never imported at module scope, so ``import
ctc.data.sources.beir`` works on a bare ``pip install ./ctc``. The pre-migration tree worked around
the same problem by duplicating a cross-encoder class three times and reimplementing an LLM client
from stdlib; the seam removes the reason for both.

Pools that are consumed one-question-per-example also implement ``for_split(split)``, returning a
disjoint slice for ``"train"`` and ``"eval"``. Train/eval separation is a property of the *pool*,
not of the loop over it -- five pre-migration generators each re-derived it, with different
fractions and different rounding.
"""

from __future__ import annotations

__all__: list = []
