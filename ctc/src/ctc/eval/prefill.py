"""
Turning an example into the token ids a model is actually fed.

Split out of the native backend because **every** backend needs it and they must all agree. If one
backend plain-tokenizes the prompt string while another emits the document-marker scaffold, the two
grade different inputs, and the gap is small enough to read as numerical noise -- roughly -0.01 f1
on contradiction @2k -- which is exactly what makes it dangerous. A cross-backend parity test that
let each backend build its own prefill would compare two different experiments and call the
difference a backend bug.

Two builders:

* :func:`plain_prefill` -- ``tokenizer(prompt)``. Correct only for a model trained on flat text.
* :func:`structural_prefill` -- rebuilds the prompt from the example's document structure and wraps
  each document in ``<|box_start|>``/``<|box_end|>``. **The markers are present in the ``full`` arm
  too.** ``full`` describes the attention *mask*; the token stream is the same one training used.
  Getting this backwards is the single easiest way to under-report a full-attention baseline.

``structural_prefill`` needs olmo-core, since the segmenter and emitters live there. It is imported
lazily and only by this function, so ``ctc`` stays installable without olmo-core -- but a vLLM or
HuggingFace run against a document-chunked checkpoint does need it, which the error message says.
"""

from __future__ import annotations

from typing import Any, Callable, List, Mapping, Optional, Protocol, Sequence

__all__ = ["Prefill", "plain_prefill", "structural_prefill", "StructuralPrefill"]

#: Landmark stride. 63 content tokens per landmark, matching the training default.
DEFAULT_MEM_FREQ = 63


class Prefill(Protocol):
    """``(prompt, example) -> token ids``. What every backend calls to build its input."""

    def __call__(self, prompt: str, example: Optional[Mapping[str, Any]] = None) -> List[int]: ...


def plain_prefill(tokenizer: Any) -> Prefill:
    """
    Tokenize the assembled prompt string, nothing more.

    :param tokenizer: A HuggingFace tokenizer.

    :returns: A :class:`Prefill` that ignores ``example``.
    """

    def build(prompt: str, example: Optional[Mapping[str, Any]] = None) -> List[int]:
        del example
        return list(tokenizer(prompt, add_special_tokens=False)["input_ids"])

    return build


class StructuralPrefill:
    """
    Rebuild the prompt from the example's structure, marker scaffold included.

    :param tokenizer: A HuggingFace tokenizer.
    :param task: Task name -- the segmenter dispatches on it.
    :param attn: ``full``, ``chunked`` or ``landmark``. ``full`` and ``chunked`` share one token
        stream and differ only in the mask; ``landmark`` interleaves landmark tokens, so it is the
        one arm whose *tokens* differ.
    :param query_position: Must match what :mod:`ctc.format` used to assemble the prompt. The
        segmenter rebuilds the prompt itself rather than parsing the assembled string, so a
        disagreement here is silent.
    :param mem_freq: Landmark stride, ``landmark`` only.
    :param doc_start_id: Override the opening marker id.
    :param doc_end_id: Override the closing marker id.

    :raises ModuleNotFoundError: If olmo-core is absent, naming the extra to install.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        task: str,
        attn: str = "full",
        query_position: str = "after",
        mem_freq: int = DEFAULT_MEM_FREQ,
        doc_start_id: Optional[int] = None,
        doc_end_id: Optional[int] = None,
    ):
        try:
            from olmo_core.data.document_chunk_landmark import (  # noqa: F401
                DOC_END_ID,
                DOC_START_ID,
                LANDMARK_TOKEN_ID,
                PAD_TOKEN_ID,
                emit_document_chunk_dense,
                emit_document_chunk_landmark,
                segment_prompt_to_chunks,
            )
        except ModuleNotFoundError as e:  # pragma: no cover - depends on the install
            raise ModuleNotFoundError(
                "the structural prefill needs olmo-core (the segmenter and chunk emitters live "
                "there).\n  pip install 'ctc[native]'\n"
                "Grading a document-chunked checkpoint without it would plain-tokenize the "
                "prompt, dropping the document markers the model was trained with."
            ) from e

        self._segment = segment_prompt_to_chunks
        self._emit_dense = emit_document_chunk_dense
        self._emit_landmark = emit_document_chunk_landmark
        self._landmark_id = LANDMARK_TOKEN_ID
        self._pad_id = PAD_TOKEN_ID

        self.tok = tokenizer
        self.task = task
        self.attn = attn
        self.query_position = query_position
        self.mem_freq = mem_freq
        self.doc_start_id = DOC_START_ID if doc_start_id is None else doc_start_id
        self.doc_end_id = DOC_END_ID if doc_end_id is None else doc_end_id

    def __call__(self, prompt: str, example: Optional[Mapping[str, Any]] = None) -> List[int]:
        """
        :param prompt: Unused -- the structure comes from ``example``, not from the rendered text.
            Recovering document boundaries from a flat string means re-deriving what the generator
            already knew, and getting it subtly wrong.
        :param example: A unified-format example.

        :returns: Prompt token ids, markers included.

        :raises ValueError: If ``example`` is absent. Falling back to plain tokenization here would
            silently produce the very token stream this class exists to avoid.
        """
        del prompt
        if example is None:
            raise ValueError(
                "structural prefill requires the example, not just the prompt string. Passing the "
                "prompt alone would drop the document markers -- worth about -0.01 f1, small "
                "enough to look like noise."
            )
        segs, _, _ = self._segment(
            self.tok,
            example,
            self.task,
            query_position=self.query_position,
            cot_mode="none",
            chunk_by="document",
            item_regex=r"\|\|",
            include_answer=False,
            doc_start_id=self.doc_start_id,
            doc_end_id=self.doc_end_id,
        )
        if self.attn == "landmark":
            ids, _ = self._emit_landmark(
                segs, mem_freq=self.mem_freq, mem_id=self._landmark_id, pad_id=self._pad_id
            )
        else:
            # full and chunked share one token stream; the mask is what differs, and it is set on
            # the model at load time, not here.
            ids, _ = self._emit_dense(segs)
        return list(ids)


def structural_prefill(tokenizer: Any, **kwargs: Any) -> Prefill:
    """
    :param tokenizer: A HuggingFace tokenizer.
    :param kwargs: See :class:`StructuralPrefill`.

    :returns: The structural builder.
    """
    return StructuralPrefill(tokenizer, **kwargs)


def build_prefills(
    prefill: Prefill,
    prompts: Sequence[str],
    examples: Optional[Sequence[Mapping[str, Any]]],
) -> List[List[int]]:
    """
    Apply a prefill builder across a rung.

    Every backend uses this rather than looping itself, so "which prompt paired with which example"
    cannot drift between them.

    :param prefill: The builder.
    :param prompts: Rendered prompts.
    :param examples: The examples they came from, or ``None``.

    :returns: Token ids per prompt.

    :raises ValueError: If ``examples`` is given but does not line up with ``prompts``.
    """
    if examples is not None and len(examples) != len(prompts):
        raise ValueError(
            f"{len(prompts)} prompts but {len(examples)} examples; they are zipped positionally, "
            "so a mismatch would grade each prompt against another example's structure"
        )
    return [prefill(p, None if examples is None else examples[i]) for i, p in enumerate(prompts)]


#: Convenience alias, so callers can annotate a parameter without importing Protocol.
PrefillFn = Callable[..., List[int]]
