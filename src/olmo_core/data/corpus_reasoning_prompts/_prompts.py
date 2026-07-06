"""Shared prompt templates and task instructions.

All data generation scripts, evaluation scripts, and example generators import
from here to ensure training and eval prompts stay in sync.

There are two axes of variation:
  1. Task type: QA (answer the question) vs Retrieval (identify relevant doc IDs)
  2. Task scope: single-doc (NQ), multi-doc (HotpotQA), multi-query (multi-HotpotQA)

Passage templates control how individual documents are formatted within the prompt.
The _ID variants add numeric document identifiers for the retrieval task.
"""

# ── Passage templates ──
# Used by all data generation and eval scripts to format individual documents.
# The base template matches HELMET eval format exactly.
# ── Unified-prompt instruction ──
# When `unified_prompt=True` is passed to build_prompt, every task uses this
# string as its alpaca "### Instruction:" header, and the task-specific ask
# moves into the positioned "query" slot. This makes the pre-query prefix
# textually identical across tasks (e.g. in qafter mode:
# "<alpaca>{GENERIC_INSTRUCTION}\n\n### Input:\n<docs>\n\n") so a mixed-task
# dataset shows the model the same structural prefill regardless of task type.
GENERIC_INSTRUCTION = (
    "You will be asked to do a long-context processing task for a context."
)


PASSAGE_TEMPLATE = "Document (Title: {title}): {text}"
PASSAGE_TEMPLATE_NO_TITLE = "Document: {text}"
PASSAGE_TEMPLATE_ID = "Document [{id}] (Title: {title}): {text}"
PASSAGE_TEMPLATE_NO_TITLE_ID = "Document [{id}]: {text}"

# ── QA task instructions ──
# Single-query QA (used by NQ and HotpotQA): model outputs a short answer.
QA_INSTRUCTION = (
    "Use the given documents to write a concise and short answer to the question. "
    "Write your answer in the following format:\nAnswer: [answer]"
)

# Multi-query QA (used by multi-HotpotQA): model outputs comma-separated answers.
MULTI_QA_INSTRUCTION = (
    "Use the given documents to answer each of the following questions. "
    "Write a concise and short answer for each question, in order, as a comma-separated list.\n"
    "Write your answer in the following format:\nAnswers: [answer1], [answer2], ..."
)

# ── Retrieval task instructions ──
# Single-doc retrieval (NQ): model identifies one relevant document.
RETRIEVAL_INSTRUCTION_SINGLE = (
    "Use the given documents to identify which document is most relevant to "
    "answering the question.\n"
    "Write your answer in the following format:\nRelevant Document: [id]"
)

# Multi-doc retrieval (HotpotQA): model identifies multiple relevant documents.
RETRIEVAL_INSTRUCTION_MULTI_DOC = (
    "Use the given documents to identify which documents are relevant to "
    "answering the question. List all relevant document IDs.\n"
    "Write your answer in the following format:\nRelevant Documents: [id1], [id2]"
)

# CoT retrieval: single-doc (NQ) — reason about relevance, then output ID.
COT_RETRIEVAL_INSTRUCTION_SINGLE = (
    "Use the given documents to identify which document is most relevant to "
    "answering the question. Think step by step about why the document is "
    "relevant, then give your answer.\n"
    "Write your answer in the following format:\n"
    "[chain of thought reasoning]\n"
    "Relevant Document: [id]"
)

# CoT retrieval: multi-doc (HotpotQA) — reason about relevance, then output IDs.
COT_RETRIEVAL_INSTRUCTION_MULTI_DOC = (
    "Use the given documents to identify which documents are relevant to "
    "answering the question. Think step by step about why the documents are "
    "relevant, then list all relevant document IDs.\n"
    "Write your answer in the following format:\n"
    "[chain of thought reasoning]\n"
    "Relevant Documents: [id1], [id2]"
)

# Multi-query retrieval (multi-HotpotQA): per-query relevant document IDs.
RETRIEVAL_INSTRUCTION_MULTI_QUERY = (
    "Use the given documents to identify which documents are relevant to "
    "answering each of the following questions. For each question, list the "
    "relevant document IDs.\n"
    "Write your answer in the following format:\n"
    "Relevant Documents: Q1: [id1], [id2]; Q2: [id3], [id4]; ..."
)

# ── Re-ranking task (MS MARCO) ──
# The model is given a query and a candidate pool, and must output the document
# IDs ordered from most to least relevant. Scored by MRR@10. Reuses the numbered
# retrieval document format.
RERANK_INSTRUCTION = (
    "Rank the documents by how relevant each is to the question, from most to "
    "least relevant. Output the document IDs in ranked order.\n"
    "Write your answer in the following format:\nRanking: [id1], [id2], [id3], ..."
)


def rerank_instruction(top_k=-1):
    """Re-ranking instruction, optionally truncated to the top-K most relevant.

    top_k <= 0 ranks the entire pool (the default, `RERANK_INSTRUCTION`).
    top_k > 0 asks for only the K most relevant IDs in ranked order. (HELMET
    itself requests the FULL ranking and scores NDCG@10 — see the exact-match
    `rerank_helmet` task; this top-K knob is our own compact-output option.)"""
    if top_k and top_k > 0:
        return (
            "Rank the documents by how relevant each is to the question. Output "
            f"the IDs of the {top_k} most relevant documents, in ranked order "
            "from most to least relevant.\n"
            "Write your answer in the following format:\nRanking: [id1], [id2], ..."
        )
    return RERANK_INSTRUCTION


# ── HELMET MS MARCO passage re-ranking (exact-match variant) ──
# Verbatim from HELMET's data.py `load_msmarco_rerank` (INCLUDING the upstream
# "relelvant" typo) so a model trained/evaluated on this format matches HELMET's
# eval prompt token-for-token. The pool uses explicit per-doc IDs and the output
# is a full permutation of those IDs joined by " > " (not our compact bracketed
# form). Relevance is graded (0-3), gold order = label descending.
HELMET_RERANK_USER_TEMPLATE = (
    "You are provided with a list of documents, each indicated by their ID. "
    "Rank each document based on their relevance to the question in descending "
    "order from most relelvant to least relevant texts. Include all documents in "
    "the rankings. Write your answer using the unique IDs, with the following "
    "format:\nRanking: ID3 > ID1 > ID2\n\n{demos}{context}\n\nQuery: {question}"
)
HELMET_RERANK_SYSTEM_TEMPLATE = "Ranking:"


def helmet_rerank_passage(doc_id, text, title=None):
    """One candidate line in HELMET's rerank context block."""
    if title is not None:
        return f"[ID: {doc_id}] Document (Title: {title}): {text}"
    return f"[ID: {doc_id}] Document: {text}"


def helmet_rerank_prompt(context, question, demos=""):
    """Full HELMET rerank prompt: filled user template + "\\n" + "Ranking:".

    Matches HELMET's `prompt_template = user_template + "\\n" + system_template`,
    so the prompt ends with "Ranking:" and the model/target continues with
    " ID3 > ID1 > ...". `context` = passages joined by "\\n\\n"; `demos` = the
    few-shot block (each demo already ends with "\\n\\n") or "" for zero-shot."""
    user = HELMET_RERANK_USER_TEMPLATE.format(demos=demos, context=context,
                                              question=question)
    return user + "\n" + HELMET_RERANK_SYSTEM_TEMPLATE


# ── Contradiction task instructions ──
# Claims are formatted as numbered items; model identifies contradicting pairs.
CONTRADICTION_INSTRUCTION = (
    "Given the following corpus of numbered claims, identify all pairs of claims "
    "that contradict each other. A pair of claims is contradictory if they cannot "
    "both be true at the same time.\n\n"
    "Output your answer as a JSON list of pairs, where each pair is a list of two "
    "claim IDs. For example: [[1, 4], [3, 7]]\n"
    "If there are no contradicting pairs, output: []"
)

CLAIM_TEMPLATE = "Claim {id}: {text}"

# ── Query-document matching task (qdmatch — sparse N²-ified retrieval) ──
# A numbered list mixing M queries and N documents (single shared index, each
# line tagged Query:/Document:). Exactly k query-document pairs are relevant
# (the document answers the query). Output the relevant [query_id, doc_id]
# pairs — query number first, then the matching document number (ordered).
QDMATCH_INSTRUCTION = (
    "Below is a numbered list of items. Each item is labeled either 'Query:' "
    "(a question) or 'Document:' (a passage). A few query-document pairs are "
    "relevant: the document answers the query. Identify every relevant pair.\n\n"
    "Output your answer as a JSON list of pairs, where each pair is "
    "[query_id, document_id] — the query's number first, then the matching "
    "document's number. For example: [[1, 8], [3, 4]]\n"
    "If no pairs are relevant, output: []"
)

# ── Redundancy-detection task ──
# Mirror of contradiction: find all pairs of claims that are REDUNDANT (state
# the same fact / are paraphrases of each other). Reuses CLAIM_TEMPLATE and the
# same JSON-pairs output schema, so the same parser and pair metrics apply.
REDUNDANCY_INSTRUCTION = (
    "Given the following corpus of numbered claims, identify all pairs of claims "
    "that are redundant with each other. A pair of claims is redundant if they "
    "state the same fact or finding (one is a paraphrase or restatement of the "
    "other), even if worded differently.\n\n"
    "Output your answer as a JSON list of pairs, where each pair is a list of two "
    "claim IDs. For example: [[1, 4], [3, 7]]\n"
    "If there are no redundant pairs, output: []"
)

# ── AbsenceBench task ──
# The inverse of needle-in-a-haystack: the model is given the full numbered
# original corpus, then a second version with some claims removed, and must
# identify which numbered claims are MISSING. Reuses CLAIM_TEMPLATE; the second
# (modified) version lives in the positioned query. Output is a set of IDs.
ABSENCE_INSTRUCTION = (
    "The numbered claims above are the original corpus. The text below is a "
    "second version of that corpus with some claims removed. Identify which of "
    "the numbered claims are MISSING from the second version (present above but "
    "absent below).\n"
    "Write your answer in the following format:\nMissing: [id1], [id2], ..."
)

# ── Gutenberg text-diff absence ──
# Two versions of a prose passage (Version A = the full segment, Version B =
# the same segment with some whole sentences deleted). The model reports the
# first four words of every sentence that is in A but missing from B, as a JSON
# list of strings in order of occurrence. No document IDs.
ABSENCE_GUTENBERG_INSTRUCTION = (
    "Above are two versions of the same passage. Version B is identical to "
    "Version A except that some whole sentences have been removed. Identify "
    "every sentence that appears in Version A but is MISSING from Version B. "
    "For each missing sentence, write its first four words.\n"
    "Write your answer as a JSON list of strings, in order of occurrence, for "
    'example:\n["Bob was not happy", "Jane felt sad that"]'
)

# ── Cross-corpus absence (xabsence) ──
# Two numbered corpora A and B. Almost every claim has a paraphrase in the OTHER
# corpus; a few are unmatched (no paraphrase anywhere in the other corpus). Find
# the unmatched ones. Output a set of IDs -> scored by the absence set-F1.
XABSENCE_INSTRUCTION = (
    "Below are two corpora of numbered claims, A and B. Almost every claim has a "
    "paraphrase — a restatement of the same fact — somewhere in the OTHER corpus "
    "(an A claim paraphrased in B, or a B claim paraphrased in A). A few claims "
    "are UNMATCHED: they have no paraphrase anywhere in the other corpus. "
    "Identify the unmatched claims.\n"
    "Write your answer in the following format:\nUnmatched: [id1], [id2], ..."
)

# ── Cycle-comparison task ──
# Claims assert a strict comparison (A ranks strictly above B). The model finds
# every set of claims whose edges form a directed cycle (an impossible loop).
# Reuses CLAIM_TEMPLATE. The per-example query states the task; this instruction
# fixes the output format. Output is a JSON list of cycles (variable length).
CYCLE_INSTRUCTION = (
    "Output your answer as a JSON list of cycles, where each cycle is a list "
    "of the claim IDs whose comparisons form the loop. For example: "
    "[[3, 8, 12]]\n"
    "If there are no cycles, output: []"
)

# ── Matching n-gram task ──
# A list of N short n-grams; some pairs are character-identical. The model
# identifies the matching pairs. Same output schema as contradiction.
MATCHING_NGRAM_INSTRUCTION = (
    "Given the following numbered list of short text snippets, identify all "
    "pairs of snippets whose text is exactly identical to each other.\n\n"
    "Output your answer as a JSON list of pairs, where each pair is a list of "
    "two snippet IDs. For example: [[1, 4], [3, 7]]\n"
    "If there are no matching pairs, output: []"
)

NGRAM_TEMPLATE = "Snippet {id}: {text}"

# ── Mathmatch task ──
# A list of N arithmetic expressions; some pairs of answers satisfy a closeness
# criterion (e.g. |val(a)-val(b)| <= x). Model identifies those pairs. Same
# output schema as contradiction; the per-example query specifies the criterion.
MATHMATCH_INSTRUCTION = (
    "Given the following numbered list of arithmetic expressions, identify all "
    "pairs of expressions matching the criterion below.\n\n"
    "Output your answer as a JSON list of pairs, where each pair is a list of "
    "two expression IDs. For example: [[1, 4], [3, 7]]\n"
    "If there are no matching pairs, output: []"
)

EXPRESSION_TEMPLATE = "Expression {id}: {text}"

# ── Groups-of-4 task ──
# A list of N arithmetic expressions; some groups of G satisfy a closeness
# criterion (all answers within X of each other). Model identifies those groups.
# Reuses EXPRESSION_TEMPLATE; per-example query states the criterion. Output is
# a JSON list of groups (same shape as the cycle task, scored by the same
# set-of-groups metric).
GROUPS4_INSTRUCTION = (
    "Given the following numbered list of arithmetic expressions, identify all "
    "groups of expressions matching the criterion below.\n\n"
    "Output your answer as a JSON list of groups, where each group is a list of "
    "the matching expression IDs. For example: [[3, 8, 12, 19]]\n"
    "If there are no matching groups, output: []"
)

# ── Strmatch task ──
# A list of N short strings; some pairs satisfy a string-similarity criterion
# (a shared contiguous run of words, or a shared word count). The model
# identifies those pairs. Same output schema as contradiction/mathmatch; the
# per-example query states the criterion.
STRMATCH_INSTRUCTION = (
    "Given the following numbered list of strings, identify all pairs of "
    "strings matching the criterion below.\n\n"
    "Output your answer as a JSON list of pairs, where each pair is a list of "
    "two string IDs. For example: [[1, 4], [3, 7]]\n"
    "If there are no matching pairs, output: []"
)

STRING_TEMPLATE = "String {id}: {text}"

# ── Textgroups task ──
# A list of N short natural-language passages. Each passage carries a *textual*
# feature value (e.g. its noun count, or how many times a common word appears).
# Some groups of G passages satisfy an aggregate criterion (their feature values
# add up to a target T). The model identifies those groups. Same output schema
# as the cycle/groups4 tasks (a JSON list of ID-groups, scored set-of-groups);
# the per-example query states which feature and what target.
TEXTGROUPS_INSTRUCTION = (
    "Given the following numbered list of passages, identify all groups of "
    "passages matching the criterion below.\n\n"
    "Output your answer as a JSON list of groups, where each group is a list of "
    "the matching passage IDs. For example: [[3, 8, 12]]\n"
    "If there are no matching groups, output: []"
)

TEXTGROUPS_TEMPLATE = "Passage {id}: {text}"

# ── Reorder task ──
# Passages are shown with shuffled display IDs; model outputs a permutation
# that restores the original document order as a JSON array of those IDs.
PASSAGE_TEMPLATE_REORDER = "Passage [{id}]: {text}"

REORDER_INSTRUCTION = (
    "You are given a list of text passages presented in a random order. They "
    "were originally consecutive segments of a single document. Output the "
    "permutation that restores them to their original order, as a JSON array "
    "of the passage IDs.\n"
    "Write your answer in the following format:\n[id1, id2, id3, ...]"
)

# ── Grouping task ──
PASSAGE_TEMPLATE_GROUPING = "Document [{id}](Title: {title}) {text}"

GROUPING_INSTRUCTION = (
    "You are given a list of scientific paper abstracts. Group them into the "
    "requested number of categories based on what they are about. Output a "
    'JSON object of the form {"groups": [{"doc_ids": [...]}, ...]} where each '
    "group is a list of 1-indexed document IDs. Every document must appear in "
    "exactly one group."
)

# Labeled variant: model also names each group (shared topic) before listing IDs.
GROUPING_LABELED_INSTRUCTION = (
    "You are given a list of scientific paper abstracts. Group them into the "
    "requested number of categories based on what they are about. For each "
    "group, give a short label describing the shared topic, then list the "
    '1-indexed document IDs. Output a JSON object of the form '
    '{"groups": [{"label": "<topic>", "doc_ids": [...]}, ...]}. Every '
    "document must appear in exactly one group."
)

# ── Outlier task ──
# Documents are product reviews; the majority share a common attribute (rating
# or category) and a few are outliers. Model outputs the outlier IDs.
OUTLIER_INSTRUCTION = (
    "You are given a list of product reviews. Most share a common attribute "
    "(star rating or product category); a few are outliers with a different "
    "value. First state what the majority attribute is and what the outlier "
    "attribute is, then list the 1-indexed document IDs of the outliers.\n"
    "Write your answer in the following format:\n"
    "[one sentence describing majority and outlier attributes]\n"
    "Outliers: [id1], [id2], ..."
)

# ── Summarization task (HELMET: ∞Bench Sum, Multi-LexSum) ──
SUMMARIZATION_INSTRUCTION = (
    "Write a concise, faithful summary of the document(s) below. Cover the key "
    "facts and findings; do not add information that is not in the document(s)."
)

# ── OOLONG task ──
# Aggregate/distributional reasoning over many labeled items (Oolong,
# arXiv:2511.02817). The per-example `question` already carries the full
# instruction and answer-format spec, so the header is minimal and the raw
# context block is shown verbatim (no per-document wrapper).
OOLONG_INSTRUCTION = (
    "Read the data below and answer the question. Compute the exact answer by "
    "analyzing every item; do not guess or approximate."
)

# ── RULER tasks (native long-context synthetic benchmark) ──
# Each subtask carries a self-contained answer-format spec so the tokens right
# before "### Response:\n" fully describe the task. The per-example question
# (which key/value/variable is asked about) lives in queries[0] and is appended
# to the instruction by _build_task_query. The context is the haystack rendered
# verbatim (needles read as natural sentences; no per-document [N] wrapper).
RULER_INSTRUCTION_DEFAULT = (
    "Some special magic numbers are hidden within the text above. Read the text "
    "and answer the question below using only the information in the text.\n"
    "Write your answer in the following format:\nAnswer: [answer]"
)

RULER_INSTRUCTIONS = {
    "niah_single": (
        "A special magic number is hidden within the text above. Find it and "
        "answer the question below.\n"
        "Write your answer in the following format:\nAnswer: [the magic number]"
    ),
    "niah_multikey": (
        "Several special magic numbers, each tied to a different key, are hidden "
        "within the text above. Find the one for the key in the question below.\n"
        "Write your answer in the following format:\nAnswer: [the magic number]"
    ),
    "niah_multivalue": (
        "Several special magic numbers for the same key are hidden within the "
        "text above. Find ALL of them and answer the question below.\n"
        "Write your answer in the following format:\n"
        "Answer: [number1, number2, ...]"
    ),
    "niah_multiquery": (
        "Special magic numbers for several keys are hidden within the text "
        "above. Find the number for EACH key asked about in the question below.\n"
        "Write your answer in the following format:\n"
        "Answer: [number1, number2, ...]"
    ),
    "vt": (
        "The text above contains chains of variable assignments. Variables can "
        "be assigned a number directly or set equal to another variable. Trace "
        "the assignments to answer the question below.\n"
        "Write your answer as the list of all matching variable names in the "
        "following format:\nAnswer: [VAR1, VAR2, ...]"
    ),
    "cwe": (
        "Write your answer as a comma-separated list in the following format:\n"
        "Answer: [word1, word2, ...]"
    ),
    "fwe": (
        "Write your answer as a comma-separated list in the following format:\n"
        "Answer: [word1, word2, ...]"
    ),
}


# ── HELMET base-model eval templates (non-alpaca) ──
# These are used when evaluating base models (no fine-tuning) with few-shot demos.
# Trained models use the alpaca template from lib/io.py instead.
DEMO_TEMPLATE = "{documents}\n\nQuestion: {question}\nAnswer: {answer}"

HELMET_TEMPLATE = (
    "Use the given documents to write a concise and short answer to the question. "
    "Write your answer in the following format:\nAnswer: [answer]\n\n"
    "{demos}{context}\n\nQuestion: {question}"
)
HELMET_TEMPLATE_QUERY_BEFORE = (
    "Use the given documents to write a concise and short answer to the question. "
    "Write your answer in the following format:\nAnswer: [answer]\n\n"
    "{demos}Question: {question}\n\n{context}"
)
HELMET_TEMPLATE_QUERY_BOTH = (
    "Use the given documents to write a concise and short answer to the question. "
    "Write your answer in the following format:\nAnswer: [answer]\n\n"
    "{demos}Question: {question}\n\n{context}\n\nQuestion: {question}"
)


def format_doc(text, title=None, use_titles=True, doc_id=None):
    """Format a document string using the appropriate passage template.

    Args:
        text: Document text content.
        title: Document title (optional).
        use_titles: Whether to include the title in the formatted output.
        doc_id: Numeric document ID for retrieval tasks. When provided,
                uses the _ID template variants that show [id] before the doc.

    Returns:
        Formatted document string matching the HELMET passage template format.
    """
    if doc_id is not None:
        if use_titles and title:
            return PASSAGE_TEMPLATE_ID.format(id=doc_id, title=title, text=text)
        return PASSAGE_TEMPLATE_NO_TITLE_ID.format(id=doc_id, text=text)
    if use_titles and title:
        return PASSAGE_TEMPLATE.format(title=title, text=text)
    return PASSAGE_TEMPLATE_NO_TITLE.format(text=text)


def format_doc_dict(doc, use_titles=True, doc_id=None):
    """Format a document dict (with 'title' and 'text' keys) using the passage template.

    Convenience wrapper around format_doc() for HotpotQA-style document dicts.
    """
    title = doc.get("title") if use_titles else None
    return format_doc(doc["text"], title=title, use_titles=use_titles, doc_id=doc_id)
