"""Unified data format and prompt building.

All training data is stored in a structured JSONL format with documents kept
separate from formatting. Prompt construction (query position, dummy tokens,
document IDs, alpaca wrapping) happens at train/eval time via build_prompt().

Unified JSONL format:
    {
        "documents": [{"title": "...", "text": "..."}, ...],
        "queries": ["question text"],           # list, even for single-query
        "answers": ["answer text"],             # list, even for single-answer
        "gold_doc_indices": [3],                # 0-indexed positions in documents
        "hard_neg_indices": [4, 7, 12],         # optional: BM25 hard negatives
        "source": "nq|hotpotqa",               # dataset origin
    }

Documents are stored already shuffled so the model sees gold at a random
position. `gold_doc_indices` and (optionally) `hard_neg_indices` track where
each provenance class landed after the shuffle, so inspection tools and
post-processing can still identify gold / hard-negative / random-pool docs
without assuming any particular storage order.

For multi-query tasks (multiple independent queries over shared documents):
    {
        "documents": [...],
        "queries": ["q1", "q2", ...],
        "answers": ["a1", "a2", ...],
        "gold_doc_indices": [[0, 2], [1, 4]],  # per-query gold doc indices
        "hard_neg_indices": [5, 6, 7],         # single list, shared across queries
        "source": "hotpotqa",
    }

At train/eval time, build_prompt() converts this into a formatted prompt string
with the appropriate instruction, document formatting, query position, etc.
"""

import hashlib
import json as _json
import random as _random
import re

from corpus_reasoning.lib.attn_smoothing import Q_END, Q_START
from corpus_reasoning.lib.io import format_alpaca_prompt, insert_dummy_tokens
from corpus_reasoning.lib.prompts import (
    PASSAGE_TEMPLATE, PASSAGE_TEMPLATE_NO_TITLE,
    PASSAGE_TEMPLATE_ID, PASSAGE_TEMPLATE_NO_TITLE_ID,
    QA_INSTRUCTION, MULTI_QA_INSTRUCTION,
    RETRIEVAL_INSTRUCTION_SINGLE, RETRIEVAL_INSTRUCTION_MULTI_DOC,
    RETRIEVAL_INSTRUCTION_MULTI_QUERY,
    COT_RETRIEVAL_INSTRUCTION_SINGLE, COT_RETRIEVAL_INSTRUCTION_MULTI_DOC,
    CONTRADICTION_INSTRUCTION, CLAIM_TEMPLATE,
    REORDER_INSTRUCTION, PASSAGE_TEMPLATE_REORDER,
    GROUPING_INSTRUCTION, GROUPING_LABELED_INSTRUCTION,
    PASSAGE_TEMPLATE_GROUPING,
    OUTLIER_INSTRUCTION,
    GENERIC_INSTRUCTION,
)


def is_multi_query(example):
    """Check if example has multiple independent queries."""
    return len(example["queries"]) > 1


def _has_multi_gold(example):
    """Check if example has multiple gold documents for any query."""
    gold = example["gold_doc_indices"]
    if not gold:
        return False
    if isinstance(gold[0], list):
        return any(len(g) > 1 for g in gold)
    return len(gold) > 1


def _get_instruction(example, task):
    """Select the appropriate instruction based on task type and query count."""
    if task == "contradiction":
        return CONTRADICTION_INSTRUCTION
    if task == "reorder":
        return REORDER_INSTRUCTION
    if task == "grouping":
        return GROUPING_INSTRUCTION
    if task == "grouping_labeled":
        return GROUPING_LABELED_INSTRUCTION
    if task == "outlier":
        return OUTLIER_INSTRUCTION
    multi = is_multi_query(example)
    if task == "cot_retrieval":
        # CoT retrieval doesn't support multi-query
        return (COT_RETRIEVAL_INSTRUCTION_MULTI_DOC if _has_multi_gold(example)
                else COT_RETRIEVAL_INSTRUCTION_SINGLE)
    elif task == "retrieval":
        if multi:
            return RETRIEVAL_INSTRUCTION_MULTI_QUERY
        return (RETRIEVAL_INSTRUCTION_MULTI_DOC if _has_multi_gold(example)
                else RETRIEVAL_INSTRUCTION_SINGLE)
    else:
        return MULTI_QA_INSTRUCTION if multi else QA_INSTRUCTION


def _format_doc(doc, use_titles=True, doc_id=None):
    """Format a single document dict using the passage template."""
    title = doc.get("title")
    text = doc["text"]
    if doc_id is not None:
        if use_titles and title:
            return PASSAGE_TEMPLATE_ID.format(id=doc_id, title=title, text=text)
        return PASSAGE_TEMPLATE_NO_TITLE_ID.format(id=doc_id, text=text)
    if use_titles and title:
        return PASSAGE_TEMPLATE.format(title=title, text=text)
    return PASSAGE_TEMPLATE_NO_TITLE.format(text=text)


def _format_documents(documents, task, use_titles=True):
    """Format all documents, adding [N] IDs for retrieval tasks."""
    if task == "contradiction":
        # Separate claims with \n\n (not \n) so each claim becomes its own
        # paragraph when wrap_documents splits on \n\n, and therefore its own
        # isolated chunk under chunked attention.
        return "\n\n".join(
            CLAIM_TEMPLATE.format(id=i + 1, text=doc["text"])
            for i, doc in enumerate(documents)
        )
    if task == "reorder":
        # Gutenberg passages often contain internal \n\n paragraph breaks.
        # Collapse those to \n so each passage remains a single paragraph when
        # wrap_documents splits on \n\n — otherwise only the first chunk of a
        # passage gets wrapped and the rest leaks into the "free" region.
        return "\n\n".join(
            PASSAGE_TEMPLATE_REORDER.format(
                id=i + 1, text=doc["text"].replace("\n\n", "\n"))
            for i, doc in enumerate(documents)
        )
    if task in ("grouping", "grouping_labeled"):
        return "\n\n".join(
            PASSAGE_TEMPLATE_GROUPING.format(
                id=i + 1, title=doc.get("title", ""), text=doc["text"])
            for i, doc in enumerate(documents)
        )
    use_ids = task in ("retrieval", "cot_retrieval", "outlier")
    formatted = []
    for i, doc in enumerate(documents):
        doc_id = i + 1 if use_ids else None  # 1-indexed for retrieval
        formatted.append(_format_doc(doc, use_titles=use_titles, doc_id=doc_id))
    return "\n\n".join(formatted)


def remap_cot_doc_ids(cot_text, id_mapping):
    """Remap document IDs in CoT text when document positions change.

    Used when scaling to more documents or reshuffling: the CoT was generated
    with documents at certain positions, but at training time positions differ.

    Args:
        cot_text: The chain-of-thought string containing references like
            "Document [3]", "[7]", etc.
        id_mapping: Dict mapping old 1-indexed IDs to new 1-indexed IDs.
            E.g., {3: 45, 7: 72} means old Document [3] is now Document [45].

    Returns:
        CoT text with all [N] references remapped.
    """
    if not id_mapping or not cot_text:
        return cot_text

    def _replace_id(match):
        old_id = int(match.group(1))
        new_id = id_mapping.get(old_id, old_id)
        return f"[{new_id}]"

    return re.sub(r'\[(\d+)\]', _replace_id, cot_text)


def _build_retrieval_ids(gold):
    """Format gold doc indices as 1-indexed ID string: '[3]' or '[3], [7]'."""
    if isinstance(gold[0], list):
        gids = gold[0]
    else:
        gids = gold
    return ", ".join(f"[{g + 1}]" for g in sorted(gids))


_OUTLIER_RATINGS = [1, 2, 3, 4, 5]
_OUTLIER_CATEGORIES = [
    "Books", "Beauty_and_Personal_Care", "Home_and_Kitchen", "Electronics",
]


def _outlier_rng(example):
    key = _json.dumps({
        "gold": sorted(example.get("gold_doc_indices", []) or []),
        "src": example.get("source", ""),
        "mj": (example.get("meta") or {}).get("majority_label"),
        "mn": (example.get("meta") or {}).get("minority_label"),
    }, sort_keys=True, default=str)
    seed = int(hashlib.sha1(key.encode()).hexdigest()[:16], 16)
    return _random.Random(seed)


def _outlier_random_content_words(example, n, rng):
    words = []
    for doc in example.get("documents", []):
        words.extend(re.findall(r"[A-Za-z]{4,}", doc.get("text", "")))
    if not words:
        return ""
    if len(words) >= n:
        sample = rng.sample(words, n)
    else:
        sample = [rng.choice(words) for _ in range(n)]
    return " ".join(sample)


_CONTRADICTION_COT_MAX_QUOTE_CHARS = 300


def _quote_claim(text: str) -> str:
    text = text.strip()
    if len(text) > _CONTRADICTION_COT_MAX_QUOTE_CHARS:
        text = text[: _CONTRADICTION_COT_MAX_QUOTE_CHARS - 3].rstrip() + "..."
    return text


def _build_contradiction_template_cot(example) -> str:
    """Deterministic CoT for contradiction: quote both claims per pair, assert
    they disagree. Forces the model to retrieve each claim's text before
    emitting the final JSON, giving chunked-family patterns much more
    supervision per example than the ~20-token JSON answer alone.
    """
    docs = example["documents"]
    lines = ["Reasoning:"]
    for pair in example["gold_doc_indices"]:
        a, b = pair
        lines.append(f"- Claim [{a}]: \"{_quote_claim(docs[a - 1]['text'])}\"")
        lines.append(f"  Claim [{b}]: \"{_quote_claim(docs[b - 1]['text'])}\"")
        lines.append(f"  These two claims disagree.")
    return "\n".join(lines)


def _build_output(example, task, cot_mode="label"):
    """Build the expected output string from the structured example.

    cot_mode applies to the outlier task only:
      "label" (default): real majority/minority labels in the CoT sentence.
      "dummy": fixed phrase that names the attribute axis (rating or category)
        but strips the specific label — ablation for "does any scaffolding
        help, or only the one that reveals the label?"
      "mislabel": label-template sentence with the majority/minority values
        replaced by random wrong labels (deterministic per-example).
      "random_words": CoT is 50 random content words drawn from the documents
        (deterministic per-example).
      "none": no CoT prefix, just the "Outliers: ..." line.
    """
    if task == "contradiction":
        import json
        # gold_doc_indices stores the contradiction pairs as [[a, b], [c, d]]
        # These are already 1-indexed claim IDs
        answer = json.dumps(example["gold_doc_indices"])
        if cot_mode == "template":
            cot = _build_contradiction_template_cot(example)
            return f"{cot}\nContradicting pairs: {answer}"
        return answer
    if task == "reorder":
        import json
        # gold_order is already a list of 1-indexed display IDs in source order.
        return json.dumps(example["gold_order"])
    if task == "grouping":
        return example["answers"][0]
    if task == "grouping_labeled":
        import json
        labels = example.get("cluster_labels") or []
        clusters = example["gold_doc_indices"]
        groups = []
        for i, c in enumerate(clusters):
            lbl = labels[i] if i < len(labels) else ""
            groups.append({"label": lbl, "doc_ids": [int(d) + 1 for d in c]})
        return json.dumps({"groups": groups})
    if task == "outlier":
        gold = example["gold_doc_indices"]
        ids_str = ", ".join(f"[{g + 1}]" for g in sorted(gold))
        meta = example.get("meta") or {}
        src = example.get("source", "")
        maj = meta.get("majority_label")
        minn = meta.get("minority_label")
        if cot_mode == "none":
            cot = ""
        elif cot_mode == "dummy":
            if src == "review_outlier_rating":
                cot = ("I'll first look for the majority rating before "
                       "outputting the final IDs.")
            elif src == "review_outlier_category":
                cot = ("I'll first look for the majority category before "
                       "outputting the final IDs.")
            else:
                cot = ""
        elif cot_mode == "mislabel":
            rng = _outlier_rng(example)
            if src == "review_outlier_rating" and maj is not None and minn is not None:
                alts_maj = [r for r in _OUTLIER_RATINGS if r != maj and r != minn]
                new_maj = rng.choice(alts_maj)
                alts_min = [r for r in _OUTLIER_RATINGS
                            if r != new_maj and r != maj and r != minn]
                new_min = rng.choice(alts_min)
                cot = (f"Most reviews are {new_maj}-star ratings and the outliers "
                       f"are {new_min}-star reviews.")
            elif src == "review_outlier_category" and maj and minn:
                alts_maj = [c for c in _OUTLIER_CATEGORIES if c != maj and c != minn]
                new_maj = rng.choice(alts_maj)
                alts_min = [c for c in _OUTLIER_CATEGORIES
                            if c != new_maj and c != maj and c != minn]
                new_min = rng.choice(alts_min)
                cot = (f"Most reviews are about {new_maj.replace('_', ' ')} "
                       f"and the outliers are about {new_min.replace('_', ' ')}.")
            else:
                cot = ""
        elif cot_mode == "random_words":
            rng = _outlier_rng(example)
            cot = _outlier_random_content_words(example, n=50, rng=rng)
        else:  # "label"
            if src == "review_outlier_rating" and maj is not None and minn is not None:
                cot = (f"Most reviews are {maj}-star ratings and the outliers "
                       f"are {minn}-star reviews.")
            elif src == "review_outlier_category" and maj and minn:
                maj_s = str(maj).replace("_", " ")
                min_s = str(minn).replace("_", " ")
                cot = (f"Most reviews are about {maj_s} and the outliers are "
                       f"about {min_s}.")
            elif src == "review_outlier_category" and maj is None and minn:
                # v2 mixed-majority: the prefix spans multiple categories
                min_s = str(minn).replace("_", " ")
                cot = (f"The reviews cover several product categories, and the "
                       f"outliers are about {min_s}.")
            else:
                cot = ""
        if cot:
            return f"{cot}\nOutliers: {ids_str}"
        return f"Outliers: {ids_str}"
    if task == "cot_retrieval":
        gold = example["gold_doc_indices"]
        cot = example.get("chain_of_thought", "")
        ids_str = _build_retrieval_ids(gold)
        has_multi = _has_multi_gold(example)
        prefix = "Relevant Documents" if has_multi else "Relevant Document"
        # Remap doc IDs in CoT if positions have changed (e.g., scaled to more docs)
        id_mapping = example.get("cot_id_mapping")
        if cot and id_mapping:
            cot = remap_cot_doc_ids(cot, id_mapping)
        if cot:
            return f"{cot}\n{prefix}: {ids_str}"
        else:
            # Fallback: no CoT available, just output IDs
            return f"{prefix}: {ids_str}"
    elif task == "retrieval":
        gold = example["gold_doc_indices"]
        if is_multi_query(example):
            # Multi-query: "Q1: [3], [7]; Q2: [1], [5]; ..."
            parts = []
            for qi, gids in enumerate(gold):
                ids_str = ", ".join(f"[{g + 1}]" for g in sorted(gids))  # 0→1 indexed
                parts.append(f"Q{qi + 1}: {ids_str}")
            return "; ".join(parts)
        else:
            return _build_retrieval_ids(gold)
    else:
        # QA task
        if is_multi_query(example):
            return ", ".join(example["answers"])
        else:
            return example["answers"][0]


def _wrap_q(text: str, wrap: bool) -> str:
    """Wrap text with smoothing markers when wrap=True."""
    if not wrap:
        return text
    return f"{Q_START}{text}{Q_END}"


def _build_questions_block(queries, wrap_question=False):
    """Format the question(s) section of the prompt."""
    if len(queries) == 1:
        block = f"Question: {queries[0]}"
    else:
        block = "\n".join(f"Question {i+1}: {q}" for i, q in enumerate(queries))
    return _wrap_q(block, wrap_question)


def _build_task_query(example, task, queries, wrap_question=False):
    """Build the positioned "query" text for a task under unified-prompt mode.

    This is the task-specific ask that gets placed before/after/both relative
    to the documents. The alpaca header uses GENERIC_INSTRUCTION separately,
    so two tasks in a mixed dataset produce identically-structured prefills
    and differ only in this query text.

    The per-task strings below are intentionally self-contained — each carries
    the format spec the model needs, so in qafter mode the tokens right before
    "### Response:\\n" fully describe the task at hand.
    """
    if task == "contradiction":
        return CONTRADICTION_INSTRUCTION
    if task == "reorder":
        return REORDER_INSTRUCTION
    if task == "grouping":
        return f"{GROUPING_INSTRUCTION}\n\n{_wrap_q(queries[0], wrap_question)}"
    if task == "grouping_labeled":
        return f"{GROUPING_LABELED_INSTRUCTION}\n\n{_wrap_q(queries[0], wrap_question)}"
    if task == "outlier":
        return f"{OUTLIER_INSTRUCTION}\n\n{_wrap_q(queries[0], wrap_question)}"
    # retrieval / cot_retrieval / qa / multi-qa: use the task instruction so
    # the task type is identifiable at the positioned slot, followed by the
    # per-example question(s).
    task_instruction = _get_instruction(example, task)
    return f"{task_instruction}\n\n{_build_questions_block(queries, wrap_question=wrap_question)}"


def build_prompt(example, task="retrieval", query_position="after",
                 use_titles=True, before_dummy=0, after_dummy=0,
                 use_alpaca=True, unified_prompt=False, cot_mode="label",
                 wrap_question=False):
    """Build a formatted prompt + output from a unified example.

    This is the single entry point for converting structured data into
    the text format consumed by training and evaluation.

    Args:
        example: Dict with unified format (documents, queries, answers, gold_doc_indices).
        task: "retrieval" (output doc IDs) or "qa" (output answer text).
        query_position: "after" (default), "before", or "both".
        use_titles: Whether to include document titles.
        before_dummy: Number of dummy token repetitions before documents.
        after_dummy: Number of dummy token repetitions after documents.
        use_alpaca: Whether to wrap in alpaca template (True for trained models).
        unified_prompt: If True, use GENERIC_INSTRUCTION as the alpaca header
            for every task and move the task-specific ask into the positioned
            query slot. This is for mixed-task datasets where the pre-query
            prefill should be textually identical across tasks. Breaks
            backward compatibility with models trained under the old per-task
            instruction headers, so defaults to False.

    Returns:
        (prompt, output) tuple of strings.
    """
    docs = example["documents"]
    queries = example["queries"]

    # Handle no-document (closed-book) case
    if not docs:
        instruction = GENERIC_INSTRUCTION if unified_prompt else _get_instruction(example, task)
        questions = _build_questions_block(queries, wrap_question=wrap_question)
        output = _build_output(example, task, cot_mode=cot_mode)
        if use_alpaca:
            prompt = format_alpaca_prompt(instruction, questions)
        else:
            prompt = f"{instruction}\n\n{questions}\n"
        return prompt, output

    context = _format_documents(docs, task, use_titles=use_titles)
    output = _build_output(example, task, cot_mode=cot_mode)

    # Contradiction and reorder always use the unified-style prompt: there is
    # no per-example query, so the task instruction itself plays the role of
    # the positioned ask and gets placed before/after/both relative to the
    # documents. The alpaca header is GENERIC_INSTRUCTION.
    force_unified = task in ("contradiction", "reorder")

    # Unified path: every task shares the same structural prefill. The
    # task-specific ask lives in `query`, positioned relative to the docs.
    if unified_prompt or force_unified:
        query = _build_task_query(example, task, queries, wrap_question=wrap_question)
        if query_position == "before":
            input_text = f"{query}\n\n{context}"
        elif query_position == "both":
            input_text = f"{query}\n\n{context}\n\n{query}"
        else:  # "after"
            input_text = f"{context}\n\n{query}"
        if before_dummy > 0 or after_dummy > 0:
            input_text = insert_dummy_tokens(input_text, before_dummy, after_dummy)
        if use_alpaca:
            prompt = format_alpaca_prompt(GENERIC_INSTRUCTION, input_text)
        else:
            prompt = f"{GENERIC_INSTRUCTION}\n\n{input_text}\n"
        return prompt, output

    # ── Legacy per-task paths (preserved for backward compatibility) ──

    # Grouping / outlier: docs followed by the raw query string (no "Question:" prefix)
    if task in ("grouping", "grouping_labeled", "outlier"):
        input_text = f"{context}\n\n{_wrap_q(queries[0], wrap_question)}"
        instruction = _get_instruction(example, task)
        if use_alpaca:
            prompt = format_alpaca_prompt(instruction, input_text)
        else:
            prompt = f"{instruction}\n\n{input_text}\n"
        return prompt, output

    questions = _build_questions_block(queries, wrap_question=wrap_question)

    if query_position == "before":
        input_text = f"{questions}\n\n{context}"
    elif query_position == "both":
        input_text = f"{questions}\n\n{context}\n\n{questions}"
    else:  # "after" (default)
        input_text = f"{context}\n\n{questions}"

    if before_dummy > 0 or after_dummy > 0:
        input_text = insert_dummy_tokens(input_text, before_dummy, after_dummy)

    instruction = _get_instruction(example, task)
    if use_alpaca:
        prompt = format_alpaca_prompt(instruction, input_text)
    else:
        prompt = f"{instruction}\n\n{input_text}\n"

    return prompt, output


def build_prompt_parts(example, task="retrieval", query_position="after",
                       use_titles=True, before_dummy=0, after_dummy=0,
                       cot_mode="label", wrap_question=False):
    """Like build_prompt, but returns (instruction, input_text, output) separately.

    Used by scripts/train/train.py to convert unified JSONL to Axolotl-compatible
    alpaca format (cached under data/.cache/) when dispatching to axolotl training.
    """
    docs = example["documents"]
    queries = example["queries"]
    output = _build_output(example, task, cot_mode=cot_mode)

    # Contradiction/reorder use unified-style: generic header + task
    # instruction placed in the positioned query slot.
    if task in ("contradiction", "reorder"):
        instruction = GENERIC_INSTRUCTION
    else:
        instruction = _get_instruction(example, task)

    if not docs:
        return instruction, _build_questions_block(queries, wrap_question=wrap_question), output

    context = _format_documents(docs, task, use_titles=use_titles)

    if task in ("contradiction", "reorder"):
        query = _build_task_query(example, task, queries, wrap_question=wrap_question)
        if query_position == "before":
            input_text = f"{query}\n\n{context}"
        elif query_position == "both":
            input_text = f"{query}\n\n{context}\n\n{query}"
        else:  # "after"
            input_text = f"{context}\n\n{query}"
    elif task in ("grouping", "grouping_labeled", "outlier"):
        input_text = f"{context}\n\n{_wrap_q(queries[0], wrap_question)}"
    else:
        questions = _build_questions_block(queries, wrap_question=wrap_question)
        if query_position == "before":
            input_text = f"{questions}\n\n{context}"
        elif query_position == "both":
            input_text = f"{questions}\n\n{context}\n\n{questions}"
        else:
            input_text = f"{context}\n\n{questions}"

    if before_dummy > 0 or after_dummy > 0:
        input_text = insert_dummy_tokens(input_text, before_dummy, after_dummy)

    return instruction, input_text, output
