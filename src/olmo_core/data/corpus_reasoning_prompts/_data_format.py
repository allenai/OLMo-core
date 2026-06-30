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

from ._io import format_alpaca_prompt, insert_dummy_tokens
from ._prompts import (
    PASSAGE_TEMPLATE, PASSAGE_TEMPLATE_NO_TITLE,
    PASSAGE_TEMPLATE_ID, PASSAGE_TEMPLATE_NO_TITLE_ID,
    QA_INSTRUCTION, MULTI_QA_INSTRUCTION,
    RETRIEVAL_INSTRUCTION_SINGLE, RETRIEVAL_INSTRUCTION_MULTI_DOC,
    RETRIEVAL_INSTRUCTION_MULTI_QUERY, RERANK_INSTRUCTION, rerank_instruction,
    helmet_rerank_passage, helmet_rerank_prompt,
    COT_RETRIEVAL_INSTRUCTION_SINGLE, COT_RETRIEVAL_INSTRUCTION_MULTI_DOC,
    CONTRADICTION_INSTRUCTION, CLAIM_TEMPLATE,
    QDMATCH_INSTRUCTION,
    REDUNDANCY_INSTRUCTION,
    ABSENCE_INSTRUCTION,
    ABSENCE_GUTENBERG_INSTRUCTION,
    XABSENCE_INSTRUCTION,
    CYCLE_INSTRUCTION,
    MATCHING_NGRAM_INSTRUCTION, NGRAM_TEMPLATE,
    MATHMATCH_INSTRUCTION, EXPRESSION_TEMPLATE,
    GROUPS4_INSTRUCTION,
    STRMATCH_INSTRUCTION, STRING_TEMPLATE,
    TEXTGROUPS_INSTRUCTION, TEXTGROUPS_TEMPLATE,
    REORDER_INSTRUCTION, PASSAGE_TEMPLATE_REORDER,
    GROUPING_INSTRUCTION, GROUPING_LABELED_INSTRUCTION,
    PASSAGE_TEMPLATE_GROUPING,
    OUTLIER_INSTRUCTION,
    OOLONG_INSTRUCTION,
    SUMMARIZATION_INSTRUCTION,
    RULER_INSTRUCTIONS, RULER_INSTRUCTION_DEFAULT,
    GENERIC_INSTRUCTION,
)


def _ruler_subtask(example):
    """RULER subtask id from _meta, used to pick the right instruction."""
    return (example.get("_meta") or {}).get("ruler_subtask", "")


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


def _get_instruction(example, task, output_top_k=-1):
    """Select the appropriate instruction based on task type and query count."""
    if task == "contradiction":
        return CONTRADICTION_INSTRUCTION
    if task == "qdmatch":
        return QDMATCH_INSTRUCTION
    if task == "xabsence":
        return XABSENCE_INSTRUCTION
    if task == "redundancy":
        return REDUNDANCY_INSTRUCTION
    if task == "absence":
        return ABSENCE_INSTRUCTION
    if task == "matching_ngram":
        return MATCHING_NGRAM_INSTRUCTION
    if task == "mathmatch":
        return MATHMATCH_INSTRUCTION
    if task == "groups4":
        return GROUPS4_INSTRUCTION
    if task == "textgroups":
        return TEXTGROUPS_INSTRUCTION
    if task == "strmatch":
        return STRMATCH_INSTRUCTION
    if task == "cycle":
        return CYCLE_INSTRUCTION
    if task == "reorder":
        return REORDER_INSTRUCTION
    if task == "grouping":
        return GROUPING_INSTRUCTION
    if task == "grouping_labeled":
        return GROUPING_LABELED_INSTRUCTION
    if task == "outlier":
        return OUTLIER_INSTRUCTION
    if task == "oolong":
        return OOLONG_INSTRUCTION
    if task == "summarization":
        return SUMMARIZATION_INSTRUCTION
    if task == "ruler":
        return RULER_INSTRUCTIONS.get(_ruler_subtask(example),
                                      RULER_INSTRUCTION_DEFAULT)
    if task == "rerank":
        return rerank_instruction(output_top_k)
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
    if task == "qdmatch":
        # Items are a pre-ordered mix of queries and documents (single shared
        # 1-based index; order encodes the separate-vs-shuffled layout). Each
        # carries a `type` tag. \n\n-separated so every item is its own isolated
        # chunk under chunked attention.
        lines = []
        for i, item in enumerate(documents):
            tag = "Query" if item.get("type") == "query" else "Document"
            lines.append(f"[{i + 1}] {tag}: {item['text']}")
        return "\n\n".join(lines)
    if task == "xabsence":
        # Two corpora A and B, pre-ordered (A block then B block), single shared
        # 1-based index; each item tagged by its `corpus`. \n\n-separated so each
        # claim is its own isolated chunk under chunked attention.
        lines = []
        for i, item in enumerate(documents):
            lines.append(f"[{i + 1}] {item.get('corpus', 'A')}: {item['text']}")
        return "\n\n".join(lines)
    if task in ("contradiction", "redundancy"):
        # Separate claims with \n\n (not \n) so each claim becomes its own
        # paragraph when wrap_documents splits on \n\n, and therefore its own
        # isolated chunk under chunked attention.
        return "\n\n".join(
            CLAIM_TEMPLATE.format(id=i + 1, text=doc["text"])
            for i, doc in enumerate(documents)
        )
    if task == "absence":
        # Neutral numbered items — elements may be claims, poem lines, numbers,
        # or diff lines depending on the source domain.
        return "\n\n".join(
            f"[{i + 1}] {doc['text']}" for i, doc in enumerate(documents)
        )
    if task == "matching_ngram":
        return "\n\n".join(
            NGRAM_TEMPLATE.format(id=i + 1, text=doc["text"])
            for i, doc in enumerate(documents)
        )
    if task in ("mathmatch", "groups4"):
        return "\n\n".join(
            EXPRESSION_TEMPLATE.format(id=i + 1, text=doc["text"])
            for i, doc in enumerate(documents)
        )
    if task == "textgroups":
        return "\n\n".join(
            TEXTGROUPS_TEMPLATE.format(id=i + 1, text=doc["text"])
            for i, doc in enumerate(documents)
        )
    if task == "strmatch":
        return "\n\n".join(
            STRING_TEMPLATE.format(id=i + 1, text=doc["text"])
            for i, doc in enumerate(documents)
        )
    if task == "cycle":
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
    if task in ("oolong", "summarization"):
        # oolong: benchmark pre-formats the block. summarization: concatenate
        # the source document(s) verbatim, no per-document wrapper.
        return "\n\n".join(doc["text"] for doc in documents)
    if task == "ruler":
        # Haystack chunks and needles are rendered verbatim — needles must read
        # as natural sentences (no [N] prefix) so the model can't shortcut by
        # position. \n\n-separated so each chunk/needle is isolated under
        # chunked attention.
        return "\n\n".join(doc["text"] for doc in documents)
    use_ids = task in ("retrieval", "cot_retrieval", "outlier", "rerank")
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


def _build_contradiction_enumerate_cot(example) -> str:
    """Per-document enumeration CoT: for each claim 1..N, say either
    'No conflict' or 'conflict with X'. Scales with N (unlike template which
    only lists gold pairs) so chunked models are forced to emit something for
    every claim — including non-conflicting ones.
    """
    n = len(example["documents"])
    partners: dict[int, list[int]] = {}
    for pair in example["gold_doc_indices"]:
        a, b = pair
        partners.setdefault(a, []).append(b)
        partners.setdefault(b, []).append(a)
    parts = []
    for i in range(1, n + 1):
        if i in partners:
            parts.append(f"{i}: conflict with {', '.join(str(p) for p in sorted(partners[i]))}")
        else:
            parts.append(f"{i}: No conflict")
    return "Reasoning: " + ", ".join(parts)


def _build_contradiction_conflicts_cot(example) -> str:
    """Sparse version of the enumerate CoT: list only conflicting claims in
    numeric order. Same schema as `enumerate` minus the 'No conflict' lines,
    so supervision scales with the number of contradictions, not N.
    """
    partners: dict[int, list[int]] = {}
    for pair in example["gold_doc_indices"]:
        a, b = pair
        partners.setdefault(a, []).append(b)
        partners.setdefault(b, []).append(a)
    parts = []
    for i in sorted(partners):
        parts.append(f"{i}: conflict with {', '.join(str(p) for p in sorted(partners[i]))}")
    return "Reasoning: " + ", ".join(parts)


def _build_reorder_successor_cot(example) -> str:
    """For each non-final doc in the gold order, state its immediate successor.
    Listed in ascending doc-id order (not gold order), so the chunked model must
    recover adjacency info per claim from its own chunk + the usual causal
    prefix — same structural idea as the contradiction 'conflicts' CoT.
    """
    order = example["gold_order"]
    succ: dict[int, int] = {order[i]: order[i + 1] for i in range(len(order) - 1)}
    parts = [f"{d}: right before {succ[d]}" for d in sorted(succ)]
    return "Reasoning: " + ", ".join(parts)


def _build_reorder_anchor_cot(example, n_anchor: int = 3) -> str:
    """Natural-language successor: quote the last n_anchor words of each chunk
    and the first n_anchor words of its successor, so the CoT exhibits the
    lexical cohesion that makes adjacency inferrable.

    Same ascending-display-id ordering as `successor` for chunked compatibility.
    """
    order = example["gold_order"]
    docs = example["documents"]
    succ: dict[int, int] = {order[i]: order[i + 1] for i in range(len(order) - 1)}
    lines = []
    for i in sorted(succ):
        j = succ[i]
        end_w = docs[i - 1]["text"].split()[-n_anchor:]
        start_w = docs[j - 1]["text"].split()[:n_anchor]
        lines.append(
            f'Chunk {i} → Chunk {j}: "...{" ".join(end_w)}" → "{" ".join(start_w)}..."'
        )
    return "Reasoning:\n" + "\n".join(lines)


def _build_mathmatch_template_cot(example) -> str:
    """Templated CoT for mathmatch: for each gold pair, name the two document
    ids, quote each one's numeric value, and state that their absolute
    difference is within the tolerance. This forces the model to surface each
    matched value before emitting the final JSON (more supervision per example
    than the bare ~20-token answer).

    Parse-safety: the eval pair-parser (_parse_pairs in evaluate.py) falls back
    to scanning for `[a, b]` / `(a, b)` two-number patterns across the WHOLE
    response. So document references here use SINGLE-number brackets (`[id]`,
    like the contradiction CoT's `Claim [a]`) and the difference is written in
    prose — only the final "Matching pairs: [...]" line carries a two-number
    pair the parser should pick up.
    """
    docs = example["documents"]
    meta = example.get("_meta", {}) or {}
    tol = meta.get("tolerance")
    vals = meta.get("answer_values")

    def value_of(doc_id):  # 1-indexed doc id -> numeric value
        if vals is not None:
            return vals[doc_id - 1]
        return int(str(docs[doc_id - 1]["text"]).strip())

    tol_clause = f"at most {tol}" if tol is not None else "within the allowed amount"
    lines = ["Reasoning:"]
    for pair in example["gold_doc_indices"]:
        a, b = pair
        va, vb = value_of(a), value_of(b)
        lines.append(
            f"- Number [{a}] = {va} and Number [{b}] = {vb} differ by "
            f"{abs(va - vb)}, which is {tol_clause}."
        )
    return "\n".join(lines)


def _longest_word_run(wa, wb):
    """Longest contiguous shared word run between two token lists."""
    best = []
    for i in range(len(wa)):
        for j in range(len(wb)):
            k = 0
            while i + k < len(wa) and j + k < len(wb) and wa[i + k] == wb[j + k]:
                k += 1
            if k > len(best):
                best = wa[i:i + k]
    return best


def _safe_num(expr):
    """Evaluate a bare number or +/- arithmetic expression; None if not numeric."""
    s = str(expr).strip()
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    if re.fullmatch(r"[-\d+ ]+", s):
        try:
            return eval(s)  # only digits, spaces, +/-
        except (SyntaxError, ValueError):
            return None
    return None


def _build_strmatch_template_cot(example) -> str:
    """For each gold pair, quote the shared contiguous run (or shared words).
    Single-[id] brackets keep the pair-parser locked onto the final JSON line."""
    docs = example["documents"]
    rel = (example.get("_meta") or {}).get("relation", "substring")
    lines = ["Reasoning:"]
    for a, b in example["gold_doc_indices"]:
        wa, wb = docs[a - 1]["text"].split(), docs[b - 1]["text"].split()
        if rel == "substring":
            run = " ".join(_longest_word_run(wa, wb))
            lines.append(f'- Strings [{a}] and [{b}] share the run "{run}".')
        else:
            shared = ", ".join(sorted(set(wa) & set(wb)))
            lines.append(f"- Strings [{a}] and [{b}] share the words {shared}.")
    return "\n".join(lines)


def _build_redundancy_template_cot(example) -> str:
    """Quote both claims per gold pair and assert they state the same fact
    (the agreement mirror of the contradiction template CoT)."""
    docs = example["documents"]
    lines = ["Reasoning:"]
    for a, b in example["gold_doc_indices"]:
        lines.append(f'- Claim [{a}]: "{_quote_claim(docs[a - 1]["text"])}"')
        lines.append(f'  Claim [{b}]: "{_quote_claim(docs[b - 1]["text"])}"')
        lines.append("  These two claims state the same fact.")
    return "\n".join(lines)


def _build_cycle_template_cot(example) -> str:
    """Order each gold cycle's claims into a path A>B>C>A and narrate the loop."""
    docs = example["documents"]
    lines = ["Reasoning:"]
    for cyc in example["gold_doc_indices"]:
        edges = {}  # claim_id -> (source_name, target_name)
        for cid in cyc:
            toks = docs[cid - 1]["text"].split()
            edges[cid] = (toks[0], toks[-1])
        # follow source->target to order the loop
        order, used, cur = [], set(), edges[cyc[0]][0]
        while len(order) < len(cyc):
            nxt = next((c for c in cyc if c not in used and edges[c][0] == cur), None)
            if nxt is None:
                order = list(cyc)  # fallback: unordered
                break
            order.append(nxt); used.add(nxt); cur = edges[nxt][1]
        for cid in order:
            a, b = edges[cid]
            lines.append(f"- Claim [{cid}]: {a} > {b}.")
        names = [edges[c][0] for c in order]
        lines.append(f"  This forms a loop {' > '.join(names)} > {names[0]}, "
                     f"which is impossible.")
    return "\n".join(lines)


def _build_groups4_template_cot(example) -> str:
    """Quote each group member's value and state they are all within tolerance."""
    docs = example["documents"]
    X = (example.get("_meta") or {}).get("tolerance")
    lines = ["Reasoning:"]
    for grp in example["gold_doc_indices"]:
        parts = []
        for cid in grp:
            v = _safe_num(docs[cid - 1]["text"])
            parts.append(f"[{cid}]={v if v is not None else docs[cid - 1]['text']}")
        within = f"within {X} of each other" if X is not None else "all close"
        lines.append(f"- Expressions {', '.join(parts)} are {within}.")
    return "\n".join(lines)


def _is_textdiff_absence(example) -> bool:
    """True for the Gutenberg text-diff absence variant (Version A / Version B
    prose, no doc IDs, answer = first-four-words of each removed sentence)."""
    return (example.get("meta") or {}).get("format") == "textdiff"


def _tg_feature_phrase(meta) -> str:
    """Human-readable name of the textgroups feature (for CoT/query prose)."""
    feat = (meta or {}).get("feature", "nouns")
    if feat == "connector":
        return f'occurrences of the word "{(meta or {}).get("word", "and")}"'
    return {"nouns": "nouns", "verbs": "verbs",
            "adjectives": "adjectives"}.get(feat, feat)


def _build_textgroups_template_cot(example) -> str:
    """Quote each gold group's per-passage feature counts and show they sum to T."""
    meta = example.get("_meta") or {}
    counts = meta.get("counts") or []
    T = meta.get("target")
    W = meta.get("tolerance", 0) or 0
    phrase = _tg_feature_phrase(meta)
    lines = [f"Reasoning: each passage's value is its number of {phrase}."]
    for grp in example["gold_doc_indices"]:
        vals = [counts[c - 1] for c in grp]
        terms = " + ".join(f"[{c}]={counts[c - 1]}" for c in grp)
        tail = (f"within {W} of {T}" if W else f"exactly {T}")
        lines.append(f"- Passages {terms} sum to {sum(vals)} ({tail}).")
    return "\n".join(lines)


def _build_textgroups_scan_cot(example) -> str:
    """List EVERY passage's feature count (scales with N), then the gold groups
    whose values hit the target. Mirrors the groups4 `sort` CoT."""
    meta = example.get("_meta") or {}
    counts = meta.get("counts") or []
    T = meta.get("target")
    W = meta.get("tolerance", 0) or 0
    phrase = _tg_feature_phrase(meta)
    listing = ", ".join(f"[{i + 1}]={c}" for i, c in enumerate(counts))
    tail = (f"within {W} of {T}" if W else f"to {T}")
    lines = [f"Reasoning: counting the {phrase} in each passage: {listing}."]
    for grp in example["gold_doc_indices"]:
        terms = " + ".join(str(counts[c - 1]) for c in grp)
        lines.append(f"Passages {', '.join(map(str, grp))} give {terms} = "
                     f"{sum(counts[c - 1] for c in grp)}, summing {tail}.")
    return "\n".join(lines)


def _build_absence_template_cot(example) -> str:
    """Name and quote each removed item (single-[id] so the trailing 'Missing:'
    line is what the parser reads)."""
    docs = example["documents"]
    lines = ["Reasoning: scanning the original against the second version, these "
             "numbered items are absent below:"]
    for g in sorted(example["gold_doc_indices"]):
        lines.append(f'- [{g + 1}]: "{_quote_claim(docs[g]["text"])}"')
    return "\n".join(lines)


def _build_rerank_template_cot(example) -> str:
    """State which document(s) are relevant before the ranking line."""
    gold = sorted(example["gold_doc_indices"])
    lines = ["Reasoning:"]
    for g in gold:
        lines.append(f"- Document [{g + 1}] directly answers the question, so it "
                     f"ranks near the top.")
    return "\n".join(lines)


def _build_oolong_plan_cot(example) -> str:
    """Generic aggregation plan (the synthetic data has no per-item gold to
    enumerate, so this scaffolds the 'analyze every item then aggregate' strategy)."""
    tg = (example.get("_meta") or {}).get("task_group", "")
    plan = {
        "counting": "go through every item, determine its label, tally the counts, "
                    "then compare the totals.",
        "user": "go through every item, note its user id, tally per user, then find "
                "the most frequent.",
        "timeline": "go through the items in date order, track the per-period label "
                    "counts, then compare across periods.",
    }.get(tg, "analyze every item and aggregate the result before answering.")
    return f"Reasoning: to answer this, {plan}"


# ───────────────────────── long / scan CoT (scale with N) ──────────────────
# These walk EVERY item (not just gold), like the contradiction `enumerate` CoT,
# so chunked-family models get supervision proportional to context size and
# learn the scan-the-whole-corpus algorithm. CoT bodies use bare numbers or
# single-[id] brackets so the answer parsers still lock onto the final line.

def _build_strmatch_enumerate_cot(example) -> str:
    """Per-string verdict: matches which other string, or none."""
    n = len(example["documents"])
    partner = {}
    for a, b in example["gold_doc_indices"]:
        partner.setdefault(a, []).append(b)
        partner.setdefault(b, []).append(a)
    parts = [f"{i}: matches {', '.join(map(str, sorted(partner[i])))}" if i in partner
             else f"{i}: no match" for i in range(1, n + 1)]
    return "Reasoning: " + ", ".join(parts)


def _build_redundancy_enumerate_cot(example) -> str:
    """Per-claim verdict: paraphrase of which claim, or none (scales with N)."""
    n = len(example["documents"])
    partner = {}
    for a, b in example["gold_doc_indices"]:
        partner.setdefault(a, []).append(b)
        partner.setdefault(b, []).append(a)
    parts = [f"{i}: paraphrase of {', '.join(map(str, sorted(partner[i])))}" if i in partner
             else f"{i}: unique" for i in range(1, n + 1)]
    return "Reasoning: " + ", ".join(parts)


def _build_cycle_trace_cot(example) -> str:
    """List ALL edges (the full comparison graph), then follow the chain that
    closes each gold loop. Bare numbers keep the final 'Cycles:' JSON parseable."""
    docs = example["documents"]
    lines = ["Reasoning: the comparisons define these edges:"]
    for i, d in enumerate(docs, 1):
        toks = d["text"].split()
        lines.append(f"- Claim {i}: {toks[0]} > {toks[-1]}")
    for cyc in example["gold_doc_indices"]:
        edges = {c: (docs[c - 1]["text"].split()[0],
                     docs[c - 1]["text"].split()[-1]) for c in cyc}
        order, used, cur = [], set(), edges[cyc[0]][0]
        while len(order) < len(cyc):
            nxt = next((c for c in cyc if c not in used and edges[c][0] == cur), None)
            if nxt is None:
                order = list(cyc); break
            order.append(nxt); used.add(nxt); cur = edges[nxt][1]
        chain = " > ".join(f"{edges[c][0]} (claim {c})" for c in order)
        lines.append(f"Following the chain: {chain} > {edges[order[0]][0]} — a loop.")
    return "\n".join(lines)


def _build_groups4_sort_cot(example) -> str:
    """Sort every value, then point out the tight cluster (scales with N)."""
    docs = example["documents"]
    X = (example.get("_meta") or {}).get("tolerance")
    vals = [(i + 1, _safe_num(d["text"])) for i, d in enumerate(docs)]
    vals_sorted = sorted((v, i) for i, v in vals if v is not None)
    listing = ", ".join(f"{v} [{i}]" for v, i in vals_sorted)
    grp = sorted(example["gold_doc_indices"][0]) if example["gold_doc_indices"] else []
    gv = sorted(_safe_num(docs[c - 1]["text"]) for c in grp)
    return (f"Reasoning: sorting all values: {listing}.\n"
            f"The values {', '.join(map(str, gv))} (expressions "
            f"{', '.join(map(str, grp))}) all fall within {X}; every other value "
            f"is farther than {X} from its neighbours.")


def _build_absence_walk_cot(example) -> str:
    """Walk every numbered original item, marking present / ABSENT (scales with N).
    All brackets are before the final 'Missing:' line, which the parser anchors to."""
    n = len(example["documents"])
    gone = {g + 1 for g in example["gold_doc_indices"]}
    parts = [f"[{i}] ABSENT" if i in gone else f"[{i}] present" for i in range(1, n + 1)]
    return "Reasoning: checking each numbered item against the second version:\n" \
           + ", ".join(parts)


def _build_rerank_score_cot(example) -> str:
    """Per-candidate relevance verdict before the ranking (scales with pool size)."""
    n = len(example["documents"])
    gold = {g + 1 for g in example["gold_doc_indices"]}
    parts = [f"[{i}] relevant" if i in gold else f"[{i}] off-topic" for i in range(1, n + 1)]
    return "Reasoning: assessing each candidate's relevance to the question:\n" \
           + ", ".join(parts)


def _rerank_reference_order(example):
    """Reference relevance order as 1-indexed doc IDs, most → least relevant.

    If the example carries per-document cross-encoder scores (`ce_scores`, as
    emitted by generate_msmarco_trainhn_data.py), the CE score is treated as the
    relevance truth: scored docs are sorted by CE descending, then the unscored
    random-fill docs follow in their displayed order. Without CE scores we fall
    back to the binary scheme — gold docs first, then the rest in displayed order.
    """
    n = len(example["documents"])
    ce = example.get("ce_scores")
    if ce and any(s is not None for s in ce):
        scored = sorted((i for i in range(n) if ce[i] is not None),
                        key=lambda i: -ce[i])
        rest = [i for i in range(n) if ce[i] is None]
        order = scored + rest
    else:
        raise NotImplementedError(
            "DEPRECATED binary rerank format: example has no `ce_scores`, so the "
            "reference order would fall back to the old gold-first-then-displayed "
            "scheme. That format is disabled. Regenerate rerank with "
            "generate_msmarco_trainhn_data.py (CE-graded) instead of the old "
            "generate_msmarco_data.py."
        )
    return [i + 1 for i in order]


def _helmet_rerank_context_and_order(example):
    """(context, ordered_ids) for a HELMET-native rerank record.

    `example` has HELMET's schema: {qid, query, ctxs:[{id, title?, text, label,
    score?}]}. context = passages joined by "\\n\\n"; ordered_ids = the IDs in
    relevance order. We sort by the continuous `score` (raw cross-encoder logit)
    when present, NOT the coarse bucketed `label` — bucketing ties the gold with
    strong hard negs, and a label-only sort would let the display-order tiebreak
    rank a distractor above the gold. Falls back to `label` for HELMET-native
    files that carry no `score`."""
    ctxs = example["ctxs"]
    has_title = "title" in ctxs[0]
    context = "\n\n".join(
        helmet_rerank_passage(c["id"], c["text"],
                              c["title"] if has_title else None)
        for c in ctxs)
    key = "score" if "score" in ctxs[0] else "label"
    ordered = sorted(ctxs, key=lambda c: c[key], reverse=True)
    return context, [str(c["id"]) for c in ordered]


HELMET_RERANK_TOP_K = 10  # NDCG@10 cutoff — the target lists only the top-K ids


def build_helmet_rerank(example, top_k=HELMET_RERANK_TOP_K):
    """(prompt, target) for task='rerank_helmet'.

    Emits the RAW HELMET prompt (no alpaca wrapper) so the sequence is
    token-identical to HELMET eval: the prompt ends with "Ranking:" and the target
    continues with " ID3 > ID1 > ...". The PROMPT is verbatim HELMET (instruction
    still says "Include all documents") so the files stay drop-in for HELMET; the
    TARGET lists only the top-`top_k` ids by label — that is all NDCG@10 scores and
    all that fits HELMET's 200-token decode budget, so it's the better train signal.
    `top_k <= 0` reverts to the full permutation. A pre-formatted few-shot block may
    be supplied on the record as `_demos` (the eval loader sets it); default 0-shot.
    """
    context, ordered_ids = _helmet_rerank_context_and_order(example)
    if top_k and top_k > 0:
        ordered_ids = ordered_ids[:top_k]
    prompt = helmet_rerank_prompt(context, example["query"],
                                  example.get("_demos", ""))
    return prompt, " " + " > ".join(ordered_ids)


def _build_output(example, task, cot_mode="label", output_top_k=-1):
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
    if task == "qdmatch":
        # Ordered [query_id, doc_id] pairs, 1-based (matches the rendered item
        # numbers). NOT sorted — query index must come first.
        pairs = [[int(a), int(b)] for a, b in example.get("gold_pairs", [])]
        answer = _json.dumps(pairs)
        if cot_mode == "enumerate":
            # Long CoT that WALKS EVERY query (scales with M, like the
            # contradiction `enumerate` scaffold): state which document answers
            # each query, or "none". Body uses Q<n>/D<n> with NO square brackets
            # so the eval parser's greedy bracket span still isolates the final
            # JSON answer.
            by_q = {}
            for q, d in pairs:
                by_q.setdefault(q, []).append(d)
            lines = []
            for gi, item in enumerate(example.get("documents", []), start=1):
                if item.get("type") != "query":
                    continue
                if gi in by_q:
                    ds = " and ".join(f"D{d}" for d in sorted(by_q[gi]))
                    lines.append(f"Q{gi} -> {ds}")
                else:
                    lines.append(f"Q{gi} -> none")
            return "\n".join(lines) + "\n" + answer
        return answer
    if task == "xabsence":
        # Unmatched item IDs (1-based, matching the rendered [i] numbers),
        # scored by the absence set-F1.
        gold = sorted(int(g) + 1 for g in example.get("gold_doc_indices", []))
        return "Unmatched: " + ", ".join(f"[{g}]" for g in gold)
    if task == "matching_ngram":
        import json
        return json.dumps(example["gold_doc_indices"])
    if task == "strmatch":
        import json
        answer = json.dumps(example["gold_doc_indices"])
        if cot_mode == "template":
            return f"{_build_strmatch_template_cot(example)}\nMatching pairs: {answer}"
        if cot_mode == "enumerate":
            return f"{_build_strmatch_enumerate_cot(example)}\nMatching pairs: {answer}"
        return answer
    if task == "redundancy":
        import json
        answer = json.dumps(example["gold_doc_indices"])
        if cot_mode == "template":
            return f"{_build_redundancy_template_cot(example)}\nRedundant pairs: {answer}"
        if cot_mode == "enumerate":
            return f"{_build_redundancy_enumerate_cot(example)}\nRedundant pairs: {answer}"
        return answer
    if task == "cycle":
        import json
        answer = json.dumps(example["gold_doc_indices"])
        if cot_mode == "template":
            return f"{_build_cycle_template_cot(example)}\nCycles: {answer}"
        if cot_mode == "trace":
            return f"{_build_cycle_trace_cot(example)}\nCycles: {answer}"
        return answer
    if task == "groups4":
        import json
        answer = json.dumps(example["gold_doc_indices"])
        if cot_mode == "template":
            return f"{_build_groups4_template_cot(example)}\nGroups: {answer}"
        if cot_mode == "sort":
            return f"{_build_groups4_sort_cot(example)}\nGroups: {answer}"
        return answer
    if task == "textgroups":
        import json
        answer = json.dumps(example["gold_doc_indices"])
        if cot_mode == "template":
            return f"{_build_textgroups_template_cot(example)}\nGroups: {answer}"
        if cot_mode == "scan":
            return f"{_build_textgroups_scan_cot(example)}\nGroups: {answer}"
        return answer
    if task == "absence":
        if _is_textdiff_absence(example):
            # Gutenberg text-diff variant: answer is the first-four-words snippet
            # of each removed sentence, in order of occurrence, as a JSON list.
            import json
            snippets = example["answers"]
            answer = json.dumps(snippets, ensure_ascii=False)
            if cot_mode == "list":
                removed = (example.get("meta") or {}).get("removed_sentences") or []
                lines = "\n".join(f"- {s}" for s in removed)
                return f"The missing sentences are:\n{lines}\nAnswer: {answer}"
            return answer
        # gold_doc_indices are 0-indexed positions of removed items.
        ids = ", ".join(f"[{g + 1}]" for g in sorted(example["gold_doc_indices"]))
        answer = f"Missing: {ids}"
        if cot_mode == "template":
            return f"{_build_absence_template_cot(example)}\n{answer}"
        if cot_mode == "walk":
            return f"{_build_absence_walk_cot(example)}\n{answer}"
        return answer
    if task == "summarization":
        return example["answers"][0]
    if task == "ruler":
        # All required gold strings, comma-joined, behind the "Answer:" prefix
        # the instruction asks for. The ruler eval scores recall over them.
        return "Answer: " + ", ".join(str(a) for a in example["answers"])
    if task == "oolong":
        if cot_mode == "plan":
            return f"{_build_oolong_plan_cot(example)}\nAnswer: {example['answers'][0]}"
        return example["answers"][0]
    if task == "rerank":
        # Target ranking: most → least relevant, 1-indexed. Uses CE scores as the
        # relevance order when present, else gold-first (see _rerank_reference_order).
        order = _rerank_reference_order(example)
        if output_top_k and output_top_k > 0:
            order = order[:output_top_k]
        answer = "Ranking: " + ", ".join(f"[{i}]" for i in order)
        if cot_mode == "template":
            return f"{_build_rerank_template_cot(example)}\n{answer}"
        if cot_mode == "score":
            return f"{_build_rerank_score_cot(example)}\n{answer}"
        return answer
    if task == "mathmatch":
        import json
        answer = json.dumps(example["gold_doc_indices"])
        if cot_mode == "template":
            cot = _build_mathmatch_template_cot(example)
            return f"{cot}\nMatching pairs: {answer}"
        return answer
    if task == "contradiction":
        import json
        # gold_doc_indices stores the contradiction pairs as [[a, b], [c, d]]
        # These are already 1-indexed claim IDs
        answer = json.dumps(example["gold_doc_indices"])
        if cot_mode == "template":
            cot = _build_contradiction_template_cot(example)
            return f"{cot}\nContradicting pairs: {answer}"
        if cot_mode == "enumerate":
            cot = _build_contradiction_enumerate_cot(example)
            return f"{cot}\nContradicting pairs: {answer}"
        if cot_mode == "conflicts":
            cot = _build_contradiction_conflicts_cot(example)
            return f"{cot}\nContradicting pairs: {answer}"
        return answer
    if task == "reorder":
        import json
        # gold_order is already a list of 1-indexed display IDs in source order.
        answer = json.dumps(example["gold_order"])
        if cot_mode == "successor":
            cot = _build_reorder_successor_cot(example)
            return f"{cot}\nFinal answer: {answer}"
        if cot_mode == "anchor":
            cot = _build_reorder_anchor_cot(example)
            return f"{cot}\nFinal answer: {answer}"
        return answer
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
            elif src == "wiki_outlier_topic" and minn:
                if maj:  # simple v1: single majority article
                    cot = (f"Most passages are about {maj} and the outliers "
                           f"are about {minn}.")
                else:    # mixed v2: K majority articles in category_distribution
                    dist = meta.get("category_distribution") or {}
                    maj_titles = [t for t in dist if t != minn]
                    if maj_titles:
                        joined = ", ".join(maj_titles)
                        cot = (f"Most passages are about {joined} and the "
                               f"outliers are about {minn}.")
                    else:
                        cot = ""
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


def _build_questions_block(queries):
    """Format the question(s) section of the prompt."""
    if len(queries) == 1:
        return f"Question: {queries[0]}"
    return "\n".join(f"Question {i+1}: {q}" for i, q in enumerate(queries))


def _build_task_query(example, task, queries):
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
    if task == "qdmatch":
        return QDMATCH_INSTRUCTION
    if task == "xabsence":
        return XABSENCE_INSTRUCTION
    if task == "redundancy":
        return REDUNDANCY_INSTRUCTION
    if task == "absence":
        # original docs (context) → instruction (refers to "above"/"below") →
        # the modified second version carried in queries[0].
        return f"{ABSENCE_INSTRUCTION}\n\n{queries[0]}"
    if task == "matching_ngram":
        return MATCHING_NGRAM_INSTRUCTION
    if task == "mathmatch":
        return f"{MATHMATCH_INSTRUCTION}\n\n{queries[0]}"
    if task == "groups4":
        return f"{GROUPS4_INSTRUCTION}\n\n{queries[0]}"
    if task == "textgroups":
        return f"{TEXTGROUPS_INSTRUCTION}\n\n{queries[0]}"
    if task == "strmatch":
        return f"{STRMATCH_INSTRUCTION}\n\n{queries[0]}"
    if task == "cycle":
        return f"{queries[0]}\n\n{CYCLE_INSTRUCTION}"
    if task == "reorder":
        return REORDER_INSTRUCTION
    if task == "grouping":
        return f"{GROUPING_INSTRUCTION}\n\n{queries[0]}"
    if task == "grouping_labeled":
        return f"{GROUPING_LABELED_INSTRUCTION}\n\n{queries[0]}"
    if task == "outlier":
        return f"{OUTLIER_INSTRUCTION}\n\n{queries[0]}"
    if task == "ruler":
        # instruction (refers to "the text above") then the per-example question.
        instr = RULER_INSTRUCTIONS.get(_ruler_subtask(example),
                                       RULER_INSTRUCTION_DEFAULT)
        return f"{queries[0]}\n\n{instr}"
    # retrieval / cot_retrieval / qa / multi-qa: use the task instruction so
    # the task type is identifiable at the positioned slot, followed by the
    # per-example question(s).
    task_instruction = _get_instruction(example, task)
    return f"{task_instruction}\n\n{_build_questions_block(queries)}"


def build_prompt(example, task="retrieval", query_position="after",
                 use_titles=True, before_dummy=0, after_dummy=0,
                 use_alpaca=True, unified_prompt=False, cot_mode="label",
                 output_top_k=-1):
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
    # HELMET-native rerank record ({qid, query, ctxs:[...]}) — distinct schema,
    # raw HELMET prompt (no alpaca). Handle before touching documents/queries.
    # output_top_k overrides the target length; the unset default (-1) -> top-10.
    if task == "rerank_helmet":
        top_k = output_top_k if (output_top_k and output_top_k > 0) else HELMET_RERANK_TOP_K
        return build_helmet_rerank(example, top_k=top_k)

    # qdmatch always repeats the instruction both before the items AND right
    # before the response, so the model re-reads the (long) item list's task
    # and output format immediately before generating.
    if task == "qdmatch":
        query_position = "both"

    docs = example["documents"]
    queries = example["queries"]

    # Handle no-document (closed-book) case
    if not docs:
        instruction = (GENERIC_INSTRUCTION if unified_prompt
                       else _get_instruction(example, task, output_top_k))
        questions = _build_questions_block(queries)
        output = _build_output(example, task, cot_mode=cot_mode,
                               output_top_k=output_top_k)
        if use_alpaca:
            prompt = format_alpaca_prompt(instruction, questions)
        else:
            prompt = f"{instruction}\n\n{questions}\n"
        return prompt, output

    # Gutenberg text-diff absence: two flowing prose passages (Version A = the
    # full sentence segment held in `documents`, Version B = the kept sentences
    # in queries[0]), no doc IDs. Dedicated path so the generic numbered-doc
    # rendering never applies.
    if task == "absence" and _is_textdiff_absence(example):
        output = _build_output(example, task, cot_mode=cot_mode)
        version_a = " ".join(d["text"] for d in docs)
        version_b = queries[0]
        input_text = (f"Version A:\n\n{version_a}\n\n"
                      f"Version B:\n\n{version_b}\n\n"
                      f"{ABSENCE_GUTENBERG_INSTRUCTION}")
        if before_dummy > 0 or after_dummy > 0:
            input_text = insert_dummy_tokens(input_text, before_dummy, after_dummy)
        if use_alpaca:
            prompt = format_alpaca_prompt(GENERIC_INSTRUCTION, input_text)
        else:
            prompt = f"{GENERIC_INSTRUCTION}\n\n{input_text}\n"
        return prompt, output

    context = _format_documents(docs, task, use_titles=use_titles)
    output = _build_output(example, task, cot_mode=cot_mode,
                           output_top_k=output_top_k)

    # Contradiction and reorder always use the unified-style prompt: there is
    # no per-example query, so the task instruction itself plays the role of
    # the positioned ask and gets placed before/after/both relative to the
    # documents. The alpaca header is GENERIC_INSTRUCTION.
    force_unified = task in ("contradiction", "qdmatch", "xabsence", "redundancy", "absence", "matching_ngram", "mathmatch", "strmatch", "cycle", "groups4", "textgroups", "reorder", "ruler")

    # Unified path: every task shares the same structural prefill. The
    # task-specific ask lives in `query`, positioned relative to the docs.
    if unified_prompt or force_unified:
        query = _build_task_query(example, task, queries)
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
        input_text = f"{context}\n\n{queries[0]}"
        instruction = _get_instruction(example, task)
        if use_alpaca:
            prompt = format_alpaca_prompt(instruction, input_text)
        else:
            prompt = f"{instruction}\n\n{input_text}\n"
        return prompt, output

    questions = _build_questions_block(queries)

    if query_position == "before":
        input_text = f"{questions}\n\n{context}"
    elif query_position == "both":
        input_text = f"{questions}\n\n{context}\n\n{questions}"
    else:  # "after" (default)
        input_text = f"{context}\n\n{questions}"

    if before_dummy > 0 or after_dummy > 0:
        input_text = insert_dummy_tokens(input_text, before_dummy, after_dummy)

    instruction = _get_instruction(example, task, output_top_k)
    if use_alpaca:
        prompt = format_alpaca_prompt(instruction, input_text)
    else:
        prompt = f"{instruction}\n\n{input_text}\n"

    return prompt, output


def build_prompt_parts(example, task="retrieval", query_position="after",
                       use_titles=True, before_dummy=0, after_dummy=0,
                       cot_mode="label", output_top_k=-1):
    """Like build_prompt, but returns (instruction, input_text, output) separately.

    Used by scripts/train/train.py to convert unified JSONL to Axolotl-compatible
    alpaca format (cached under data/.cache/) when dispatching to axolotl training.
    """
    # HELMET rerank is trained on the RAW prompt (no alpaca split). Return an
    # empty instruction so any alpaca-wrap leaves the HELMET prompt intact-ish;
    # prefer the non-alpaca SFT path (build_prompt with use_alpaca=False) for it.
    if task == "rerank_helmet":
        top_k = output_top_k if (output_top_k and output_top_k > 0) else HELMET_RERANK_TOP_K
        prompt, output = build_helmet_rerank(example, top_k=top_k)
        return "", prompt, output

    # qdmatch repeats the instruction before the items and again before the
    # response (mirrors build_prompt).
    if task == "qdmatch":
        query_position = "both"

    docs = example["documents"]
    queries = example["queries"]
    output = _build_output(example, task, cot_mode=cot_mode,
                           output_top_k=output_top_k)

    # Contradiction/reorder use unified-style: generic header + task
    # instruction placed in the positioned query slot.
    if task in ("contradiction", "qdmatch", "xabsence", "matching_ngram", "mathmatch", "reorder"):
        instruction = GENERIC_INSTRUCTION
    else:
        instruction = _get_instruction(example, task, output_top_k)

    if not docs:
        return instruction, _build_questions_block(queries), output

    context = _format_documents(docs, task, use_titles=use_titles)

    if task in ("contradiction", "qdmatch", "xabsence", "matching_ngram", "mathmatch", "reorder"):
        query = _build_task_query(example, task, queries)
        if query_position == "before":
            input_text = f"{query}\n\n{context}"
        elif query_position == "both":
            input_text = f"{query}\n\n{context}\n\n{query}"
        else:  # "after"
            input_text = f"{context}\n\n{query}"
    elif task in ("grouping", "grouping_labeled", "outlier"):
        input_text = f"{context}\n\n{queries[0]}"
    else:
        questions = _build_questions_block(queries)
        if query_position == "before":
            input_text = f"{questions}\n\n{context}"
        elif query_position == "both":
            input_text = f"{questions}\n\n{context}\n\n{questions}"
        else:
            input_text = f"{context}\n\n{questions}"

    if before_dummy > 0 or after_dummy > 0:
        input_text = insert_dummy_tokens(input_text, before_dummy, after_dummy)

    return instruction, input_text, output
