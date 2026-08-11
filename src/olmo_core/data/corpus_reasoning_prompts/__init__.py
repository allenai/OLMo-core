"""
Vendored corpus-reasoning prompt builder for the unified task-suite SFT converter.

``_data_format`` / ``_io`` / ``_prompts`` are copied VERBATIM from the corpus-reasoning repo
(origin/main @ 2b20f8d, ``scripts/lib/*``); only the internal ``scripts.lib.*`` imports were
rewritten to relative imports. This keeps the **training** prompts byte-identical to the
corpus-reasoning baselines AND to the oe-eval ``cr_*`` suite eval (which vendors the same
``build_prompt``), so train/eval inputs match exactly. Do not edit casually; re-vendor instead.

Call ``build_prompt(ex, task=..., cot_mode=..., use_alpaca=False)`` to get the bare
``(user_content, answer)`` for the Qwen3 chat template.
"""

from ._data_format import build_prompt, build_prompt_parts  # noqa: F401

__all__ = ["build_prompt", "build_prompt_parts"]
