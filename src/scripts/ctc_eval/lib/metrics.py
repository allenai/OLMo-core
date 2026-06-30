"""Shared metric computation for evaluation scripts.

Two families of metrics:
  1. QA metrics (exact_match, substring_match, token_f1): Compare predicted answer
     text against gold answers. Uses HELMET-compatible normalization (lowercase,
     remove articles/punctuation). All QA metrics support max_over_answers() for
     datasets with multiple valid gold answers.

  2. Retrieval metrics (retrieval_exact_match, retrieval_recall, retrieval_precision,
     retrieval_f1): Compare predicted document ID sets against gold sets. Used by
     the retrieval task where the model outputs document IDs like "[3], [7]".
"""

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """Lower text, remove punctuation/articles/extra whitespace. (HELMET-compatible)"""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def substring_match(pred: str, gold: str) -> bool:
    return normalize_answer(gold) in normalize_answer(pred)


def token_f1(pred: str, gold: str) -> float:
    """Token-level F1 score between predicted and gold answers.

    Tokenizes both strings (after normalization), computes precision/recall
    based on token overlap using Counter intersection, and returns the
    harmonic mean. This is the standard SQuAD-style F1 metric.
    """
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    # Counter intersection gives the minimum count of each shared token
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def max_over_answers(metric_fn, prediction: str, answers: list[str]):
    """Compute max metric score over all ground truth answers.

    Many datasets have multiple valid gold answers (e.g. KILT provides several
    aliases for each entity). This function evaluates the metric against each
    gold answer and returns the best score, following the convention used by
    SQuAD, HELMET, and KILT evaluations.
    """
    if isinstance(answers, str):
        answers = [answers]
    # Handle nested lists (some datasets wrap answers in an extra list layer)
    elif answers and isinstance(answers[0], list):
        answers = [a for sublist in answers for a in sublist]
    return max(metric_fn(prediction, gt) for gt in answers)


def retrieval_exact_match(predicted_ids: set[int], gold_ids: set[int]) -> bool:
    """True if predicted set exactly equals gold set."""
    return predicted_ids == gold_ids


def retrieval_recall(predicted_ids: set[int], gold_ids: set[int]) -> float:
    """Fraction of gold documents that were retrieved."""
    if not gold_ids:
        return 1.0
    return len(predicted_ids & gold_ids) / len(gold_ids)


def retrieval_precision(predicted_ids: set[int], gold_ids: set[int]) -> float:
    """Fraction of predicted documents that are gold."""
    if not predicted_ids:
        return 0.0
    return len(predicted_ids & gold_ids) / len(predicted_ids)


def retrieval_f1(predicted_ids: set[int], gold_ids: set[int]) -> float:
    """Harmonic mean of retrieval precision and recall."""
    p = retrieval_precision(predicted_ids, gold_ids)
    r = retrieval_recall(predicted_ids, gold_ids)
    if p + r == 0:
        return 0.0
    return (2 * p * r) / (p + r)


def parse_doc_ids(text: str) -> set[int]:
    """Extract document IDs from text like '[3], [7]' or 'Document [3]'."""
    return set(int(m) for m in re.findall(r'\[(\d+)\]', text))


def aggregate(results: list[dict], keys: list[str]) -> dict:
    """Average metric values across results."""
    if not results:
        return {}
    return {k: sum(r[k] for r in results) / len(results) for k in keys}
