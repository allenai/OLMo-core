"""
Corpus-reasoning prompt rendering: the ONE copy of the CTC task prompts, shared by training and eval.

``data_format`` / ``io`` / ``prompts`` were vendored from the corpus-reasoning repo (origin/main @
2b20f8d, ``scripts/lib/*``) and are now the canonical home of that code: the training converters,
the library (:mod:`olmo_core.data.document_chunk_landmark`), and the legacy native eval harness
(``recipes/eval/legacy_native/ctc_eval``) all import it from here, so train and eval prompts cannot
drift apart. ``research/common/corpus_reasoning/lib/{data_format,io,prompts}`` are aliases of these
modules.

Call ``build_prompt(ex, task=..., cot_mode=..., use_alpaca=False)`` to get the bare
``(user_content, answer)`` for the Qwen3 chat template. Eval callers that must still render legacy
binary (no ``ce_scores``) rerank files pass ``allow_binary_rerank=True``.
"""

from .data_format import build_prompt, build_prompt_parts  # noqa: F401

__all__ = ["build_prompt", "build_prompt_parts"]
