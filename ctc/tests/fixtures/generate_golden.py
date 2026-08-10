"""
Regenerate ``golden_format.json`` by running the **pre-migration** implementation.

The ported modules in :mod:`ctc.format` must emit byte-identical output to the code that built
every shard we have already tokenized and every result we have already published. "Looks correct"
is not the bar; "same bytes" is. So the fixture is produced by importing the old package and is
then committed, which means the test keeps its meaning after the old tree is deleted -- at that
point the fixture is the only surviving record of what the original code did.

Run this only to add coverage for a case the fixture does not yet include::

    python ctc/tests/fixtures/generate_golden.py --old-repo /path/to/old/OLMo-core

**Never** run it to make a failing test pass. A diff here means the port changed behaviour, and
regenerating would erase the evidence rather than the bug.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Every branch of the old `_format_documents` task chain, plus the fallthrough. Text is chosen to
# exercise the formatting hazards: embedded blank lines (which reorder collapses), missing titles,
# and items whose `type`/`corpus` tag drives the rendering.
DOC_CASES = {
    "contradiction": [{"text": "The bridge opened in 1931."}, {"text": "The bridge opened in 1937."}],
    "redundancy": [{"text": "Sales rose 4%."}, {"text": "Revenue increased four percent."}],
    "cycle": [{"text": "A outranks B."}, {"text": "B outranks A."}],
    "absence": [{"text": "line one"}, {"text": "line two"}],
    "matching_ngram": [{"text": "red fox"}, {"text": "red fox"}],
    "mathmatch": [{"text": "3 * 7 + 1"}, {"text": "20 + 2"}],
    "groups4": [{"text": "1 + 1"}, {"text": "4 - 2"}],
    "textgroups": [{"text": "A short passage about birds."}, {"text": "Another about trains."}],
    "strmatch": [{"text": "alpha beta gamma"}, {"text": "beta gamma delta"}],
    "grouping": [
        {"title": "Attention", "text": "We propose a new architecture."},
        {"title": "", "text": "An untitled abstract."},
    ],
    "grouping_labeled": [{"title": "Optics", "text": "On lenses."}],
    "oolong": [{"text": "item: 1\nlabel: x"}, {"text": "item: 2\nlabel: y"}],
    "summarization": [{"text": "Paragraph one.\n\nParagraph two."}],
    "ruler": [{"text": "The magic number for foo is 42."}, {"text": "Filler sentence."}],
    "reorder": [
        {"text": "First para.\n\nSecond para of the same passage."},
        {"text": "A later passage."},
    ],
    "qdmatch": [
        {"type": "query", "text": "Who wrote it?"},
        {"type": "document", "text": "It was written by Ada."},
    ],
    "xabsence": [
        {"corpus": "A", "text": "Claim in A."},
        {"corpus": "B", "text": "Paraphrase in B."},
        {"text": "No corpus tag -- defaults to A."},
    ],
    # Default serializer: numbered for these, unnumbered for anything else.
    "retrieval": [{"title": "T1", "text": "body one"}, {"text": "untitled body"}],
    "cot_retrieval": [{"title": "T1", "text": "body one"}],
    "outlier": [{"title": "R", "text": "Five stars."}],
    "rerank": [{"title": "P", "text": "candidate passage"}],
    "qa": [{"title": "T1", "text": "body one"}, {"text": "untitled body"}],
    "an_unregistered_task": [{"title": "T1", "text": "body one"}],
}

# Both title settings matter: only the default serializer consults `use_titles`, and the port must
# preserve that it is ignored elsewhere.
USE_TITLES = (True, False)

PARSE_DOC_ID_CASES = [
    "[3], [7]",
    "Document [3]",
    "8]",  # primed continuation -- the bug that scored niah 0.16
    "Relevant Documents: [1], [2], [15]",
    "no ids here",
    "",
    "[ 4 ]",
    "12]",
]

OUTLIER_CASES = [
    ("Outliers: [2], [5]", 10),
    ("The majority are 5-star; outliers are 1-star.\nOutliers: [3]", 10),
    ("Outliers: [99]", 10),  # out of range -> None
    ("no outliers line, bare [4]", 10),
    ("", 10),
    ("Outlier: [1], [1], [2]", 10),  # singular header, duplicate id
]

PARTITION_CASES = [
    ('{"groups": [{"doc_ids": [1, 2]}, {"doc_ids": [3]}]}', 3),
    ('Here you go: {"groups": [{"doc_ids": [1]}]} and some trailing ramble {oops', 2),
    ('2, 3, 4]}, {"doc_ids": [1, 6]}]}', 6),  # begins mid-array
    ("[[1, 2], [3]]", 3),
    ("1 2\n3", 3),  # digit-scrape fallback
    ("", 3),
    ('{"groups": []}', 2),
]

PERMUTATION_CASES = [
    ("[3, 1, 2]", 3),
    ("The order is 2 3 1 overall.", 3),
    ("[1, 2]", 3),  # wrong length
    ("[1, 1, 2]", 3),  # not a permutation
    ("", 3),
]

QA_METRIC_CASES = [
    ("The Golden Gate Bridge", "golden gate bridge"),
    ("Answer: Paris.", "Paris"),
    ("a the an", ""),
    ("nothing alike", "completely different"),
    ("partial overlap here", "partial overlap"),
]

MAX_OVER_CASES = [
    ("Paris", ["London", "Paris"]),
    ("Paris", "Paris"),
    ("Paris", [["London"], ["Paris", "paris"]]),
]

RETRIEVAL_CASES = [
    ([1, 2], [1, 2]),
    ([1], [1, 2]),
    ([1, 2, 3], [1]),
    ([], [1]),
    ([1], []),
    ([], []),
]


def build(old_repo: Path) -> dict:
    """
    Import the pre-migration modules and record their output.

    :param old_repo: Checkout containing ``src/corpus_reasoning``.

    :returns: The fixture payload.

    :raises SystemExit: If the old package cannot be imported.
    """
    src = old_repo / "src"
    if not (src / "corpus_reasoning" / "lib" / "data_format.py").exists():
        raise SystemExit(f"no src/corpus_reasoning under {old_repo}")
    sys.path.insert(0, str(src))

    from corpus_reasoning.lib import data_format as old_df  # type: ignore
    from corpus_reasoning.lib import eval_tasks as old_et  # type: ignore
    from corpus_reasoning.lib import metrics as old_m  # type: ignore
    from corpus_reasoning.lib import prompts as old_p  # type: ignore

    # Constants: every module-level string. These ARE the training data.
    constants = {
        k: v
        for k, v in vars(old_p).items()
        if not k.startswith("_") and isinstance(v, (str, dict))
    }
    constants.pop("__doc__", None)

    documents = {
        f"{task}|use_titles={ut}": old_df._format_documents(docs, task, use_titles=ut)
        for task, docs in DOC_CASES.items()
        for ut in USE_TITLES
    }

    return {
        "_source": "corpus_reasoning.lib (pre-migration)",
        "prompt_constants": constants,
        "rerank_instruction": {str(k): old_p.rerank_instruction(k) for k in (-1, 0, 5, 10)},
        "documents": documents,
        "parse_doc_ids": {t: sorted(old_m.parse_doc_ids(t)) for t in PARSE_DOC_ID_CASES},
        "parse_outlier_ids": {
            f"{t}|{n}": old_et.parse_outlier_ids(t, n) for t, n in OUTLIER_CASES
        },
        "parse_partition": {
            f"{t}|{n}": old_et.parse_partition(t, n) for t, n in PARTITION_CASES
        },
        "partition_to_labels": {
            f"{t}|{n}": old_et.partition_to_labels(old_et.parse_partition(t, n) or [], n)
            for t, n in PARTITION_CASES
        },
        "parse_permutation": {
            f"{t}|{n}": old_et.parse_permutation(t, n) for t, n in PERMUTATION_CASES
        },
        "normalize_answer": {p: old_m.normalize_answer(p) for p, _ in QA_METRIC_CASES},
        "qa_metrics": {
            f"{p}|{g}": {
                "exact_match": old_m.exact_match(p, g),
                "substring_match": old_m.substring_match(p, g),
                "token_f1": old_m.token_f1(p, g),
            }
            for p, g in QA_METRIC_CASES
        },
        "max_over_answers": {
            f"{p}|{json.dumps(a)}": old_m.max_over_answers(old_m.token_f1, p, a)
            for p, a in MAX_OVER_CASES
        },
        "retrieval_metrics": {
            f"{pred}|{gold}": {
                "exact_match": old_m.retrieval_exact_match(set(pred), set(gold)),
                "recall": old_m.retrieval_recall(set(pred), set(gold)),
                "precision": old_m.retrieval_precision(set(pred), set(gold)),
                "f1": old_m.retrieval_f1(set(pred), set(gold)),
            }
            for pred, gold in RETRIEVAL_CASES
        },
    }


def main() -> int:
    """:returns: Process exit status."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--old-repo", required=True, type=Path, help="pre-migration checkout")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "golden_format.json")
    args = ap.parse_args()

    payload = build(args.old_repo)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    counts = {k: len(v) for k, v in payload.items() if isinstance(v, dict)}
    print(f"wrote {args.out}")
    for k, n in sorted(counts.items()):
        print(f"  {k}: {n} case(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
