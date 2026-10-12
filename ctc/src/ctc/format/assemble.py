"""
Prompt assembly: a unified-format example becomes the exact text a model sees.

**This module's output is the training data.** Every shard we have tokenized was produced by
running examples through the pre-migration equivalent of this code, so a one-character difference
here makes newly built data incompatible with every checkpoint trained before it — and the failure
mode is a quiet accuracy drop, not an error. The 16 cases pinned in
``ctc/tests/fixtures/golden_format.json`` were snapshotted from that implementation *before* this
one was written, so they are an independent target rather than a description of this code.

The assembled text has one of two shapes, decided by the task, not by a caller:

``unified``
    A generic alpaca header, with the task instruction itself occupying the positioned slot next to
    the documents. Used by the 11 tasks that have **no per-example query** — for contradiction,
    "find every contradicting pair" *is* the whole ask, so there is nothing else to position.

``classic``
    The task instruction in the alpaca header, and the per-example question positioned. Used by the
    9 tasks that carry a real question: retrieval, qa, oolong, grouping, grouping_labeled, outlier,
    rerank, summarization, cot_retrieval.

The pre-migration code also had a ``unified_prompt`` argument that could force the unified shape
onto any task. It was never enabled in any config, so the hardcoded task set was the entire
behaviour; that set is now declared per task as :attr:`~ctc.format.registry.TaskSpec.unified`.

What lives elsewhere, deliberately:

* The **per-task query text** — which instruction, and in what order relative to the example's own
  question — is a property of one task and lives in that task's ``spec.py`` as ``build_query``.
  The old chain of 17 ``if task ==`` branches hid real per-task quirks: ``cycle`` and ``ruler`` put
  the example's question *before* the instruction, everything else puts it after.
* The **per-task target** likewise lives in ``spec.py`` as ``build_target``.
* **Document serialization** is :mod:`ctc.format.documents`.

Ported from ``corpus_reasoning/lib/data_format.py`` (``build_prompt`` / ``build_prompt_parts``) and
``corpus_reasoning/lib/io.py`` (the alpaca template), minus the CoT paths.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

from .documents import format_documents

__all__ = [
    "ALPACA_TEMPLATE",
    "QUERY_POSITIONS",
    "format_alpaca_prompt",
    "insert_dummy_tokens",
    "build_questions_block",
    "assemble",
    "assemble_parts",
]

#: The wrapper every trained model expects. Byte-identical to the pre-migration template — the
#: models were trained on these exact tokens, down to the newlines.
ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

#: Where the positioned text sits relative to the documents. All three are in real use; ``"both"``
#: repeats the whole block on the far side of the documents, which costs context budget at every
#: rung and matters most on the long ladders.
QUERY_POSITIONS = ("before", "after", "both")

Example = Mapping[str, Any]


def format_alpaca_prompt(instruction: str, input_text: str) -> str:
    """
    Wrap an instruction and input in the alpaca template.

    :param instruction: The ``### Instruction:`` body.
    :param input_text: The ``### Input:`` body.

    :returns: The full prompt, ending with ``### Response:\\n``.
    """
    return ALPACA_TEMPLATE.format(instruction=instruction, input=input_text)


def insert_dummy_tokens(
    input_text: str,
    before_dummy: int = 0,
    after_dummy: int = 0,
    dummy_token: str = "* ",
) -> str:
    """
    Pad around the document block, for the query-distance ablation.

    Pushes documents further from the query in the sequence to test whether the model depends on
    positional proximity rather than content.

    :param input_text: The assembled input (documents plus positioned text).
    :param before_dummy: Repetitions to insert before the document block.
    :param after_dummy: Repetitions to insert after it.
    :param dummy_token: The string repeated.

    :returns: The padded input, unchanged when both counts are zero.
    """
    if before_dummy == 0 and after_dummy == 0:
        return input_text
    parts = input_text.split("\n\n")
    if before_dummy > 0:
        parts.insert(0, (dummy_token * before_dummy).rstrip())
    if after_dummy > 0:
        parts.append((dummy_token * after_dummy).rstrip())
    return "\n\n".join(parts)


def build_questions_block(queries: Sequence[str]) -> str:
    """
    Render an example's own question(s).

    :param queries: One or more questions. Multi-query tasks number them.

    :returns: ``"Question: ..."`` for one, or numbered ``"Question N: ..."`` lines for several.
    """
    if len(queries) == 1:
        return f"Question: {queries[0]}"
    return "\n".join(f"Question {i + 1}: {q}" for i, q in enumerate(queries))


def _position(context: str, positioned: str, query_position: str) -> str:
    """
    Place the positioned text relative to the document block.

    :param context: The serialized documents.
    :param positioned: The task query or question block.
    :param query_position: One of :data:`QUERY_POSITIONS`.

    :returns: The assembled input text.

    :raises ValueError: On an unknown position.
    """
    if query_position == "before":
        return f"{positioned}\n\n{context}"
    if query_position == "both":
        return f"{positioned}\n\n{context}\n\n{positioned}"
    if query_position == "after":
        return f"{context}\n\n{positioned}"
    raise ValueError(f"query_position must be one of {QUERY_POSITIONS}, got {query_position!r}")


def assemble_parts(
    example: Example,
    *,
    task: str,
    unified: bool,
    header: str,
    positioned: str,
    query_position: str = "after",
    use_titles: bool = True,
    before_dummy: int = 0,
    after_dummy: int = 0,
) -> Tuple[str, str]:
    """
    Assemble the header and input text, without wrapping.

    Kept separate from :func:`assemble` because some consumers need the two halves apart — the
    axolotl conversion path wants an alpaca ``instruction``/``input`` pair rather than a single
    string.

    :param example: A unified-format example.
    :param task: Task name, used to pick the document serializer.
    :param unified: Whether this task uses the unified shape. A property of the task; pass
        ``spec.unified``.
    :param header: The ``### Instruction:`` body. For a unified task this is the generic string;
        for a classic task it is the task instruction.
    :param positioned: The text placed relative to the documents. For a unified task this is the
        task query; for a classic task the question block.
    :param query_position: One of :data:`QUERY_POSITIONS`.
    :param use_titles: Include document titles where the serializer honours them.
    :param before_dummy: Dummy repetitions before the documents.
    :param after_dummy: Dummy repetitions after them.

    :returns: ``(header, input_text)``.
    """
    del unified  # recorded by the caller's fingerprint; the shape is already baked into `header`
    context = format_documents(example["documents"], task, use_titles=use_titles)
    input_text = _position(context, positioned, query_position)
    if before_dummy > 0 or after_dummy > 0:
        input_text = insert_dummy_tokens(input_text, before_dummy, after_dummy)
    return header, input_text


def assemble(
    example: Example,
    *,
    task: str,
    unified: bool,
    header: str,
    positioned: str,
    query_position: str = "after",
    use_titles: bool = True,
    before_dummy: int = 0,
    after_dummy: int = 0,
    use_alpaca: bool = True,
) -> str:
    """
    Assemble the full prompt text for one example.

    :param example: A unified-format example.
    :param task: Task name.
    :param unified: Whether this task uses the unified shape.
    :param header: The instruction header.
    :param positioned: The text placed relative to the documents.
    :param query_position: One of :data:`QUERY_POSITIONS`.
    :param use_titles: Include document titles where the serializer honours them.
    :param before_dummy: Dummy repetitions before the documents.
    :param after_dummy: Dummy repetitions after them.
    :param use_alpaca: Wrap in the alpaca template. ``True`` for every trained model; ``False``
        emits ``"{header}\\n\\n{input}\\n"``, which is the non-alpaca SFT path. Note the trailing
        newline — it is part of the format.

    :returns: The prompt, ready to tokenize.
    """
    header, input_text = assemble_parts(
        example,
        task=task,
        unified=unified,
        header=header,
        positioned=positioned,
        query_position=query_position,
        use_titles=use_titles,
        before_dummy=before_dummy,
        after_dummy=after_dummy,
    )
    if use_alpaca:
        return format_alpaca_prompt(header, input_text)
    return f"{header}\n\n{input_text}\n"
