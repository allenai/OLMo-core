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

# Prompt assembly. This is the highest-stakes surface in the port: build_prompt's output IS the
# text every shard was tokenized from, so a one-character difference makes new data incompatible
# with every existing checkpoint. Covered across tasks, all three query positions, and both alpaca
# settings. cot_mode is pinned to "none" throughout because the port drops CoT (see
# ctc/tasks/README.md), so "none" is the only mode whose behaviour must be preserved.
PROMPT_CASES = [
    ("contradiction", "after", True),
    ("contradiction", "after", False),
    ("contradiction", "before", True),
    ("contradiction", "both", True),
    ("retrieval", "after", True),
    ("retrieval", "before", False),
    ("qa", "after", True),
    ("outlier", "after", True),
    ("reorder", "after", True),
    ("grouping_labeled", "after", True),
    ("oolong", "after", True),
    ("absence", "after", True),
    ("xabsence", "after", True),
    ("qdmatch", "after", True),
    ("cycle", "after", True),
    ("strmatch", "after", True),
    ("redundancy", "after", True),
    ("mathmatch", "after", True),
    ("groups4", "after", True),
    ("textgroups", "after", True),
    # retrieval's instruction depends on the EXAMPLE, not just the task: single-gold vs multi-gold
    # vs multi-query select three different strings. All three shapes are pinned.
    ("retrieval_multigold", "after", True),
    ("retrieval_multiquery", "after", True),
]

#: One example per task, shaped like the unified JSONL the generators emit.
PROMPT_EXAMPLES = {
    "contradiction": {
        "documents": [{"text": "The bridge opened in 1931."}, {"text": "The bridge opened in 1937."}],
        "queries": ["Find contradicting claims."],
        "answers": [""],
        "gold_doc_indices": [[1, 2]],
        "source": "pubmed",
    },
    "retrieval": {
        "documents": [{"title": "T1", "text": "body one"}, {"text": "untitled body"}],
        "queries": ["Who built it?"],
        "answers": ["Ada"],
        "gold_doc_indices": [0],
        "source": "nq",
    },
    "qa": {
        "documents": [{"title": "T1", "text": "body one"}],
        "queries": ["Who built it?"],
        "answers": ["Ada"],
        "gold_doc_indices": [0],
        "source": "nq",
    },
    "outlier": {
        "documents": [{"title": "R1", "text": "Five stars."}, {"title": "R2", "text": "One star."}],
        "queries": ["Find the outliers."],
        "answers": [""],
        "gold_doc_indices": [1],
        "source": "amazon",
    },
    "reorder": {
        "documents": [{"text": "First para.\n\nSecond para."}, {"text": "A later passage."}],
        "queries": ["Restore the order."],
        "answers": [""],
        "gold_doc_indices": [0, 1],
        # Already 1-indexed display IDs in source order -- reorder's target is this list verbatim,
        # not derived from gold_doc_indices.
        "gold_order": [2, 1],
        "source": "gutenberg",
    },
    "grouping_labeled": {
        "documents": [
            {"title": "Attention", "text": "A new architecture."},
            {"title": "Optics", "text": "On lenses."},
        ],
        "queries": ["Group into 2 categories."],
        "answers": [""],
        # Clusters, not flat indices -- every document belongs to exactly one group.
        "gold_doc_indices": [[0], [1]],
        "cluster_labels": ["architectures", "optics"],
        "source": "arxiv",
    },
    "oolong": {
        "documents": [{"text": "item: 1\nlabel: x"}, {"text": "item: 2\nlabel: y"}],
        "queries": ["How many are labelled x? Answer with a number."],
        "answers": ["1"],
        "gold_doc_indices": [0],
        "source": "oolong",
    },
    "absence": {
        "documents": [{"text": "line one"}, {"text": "line two"}],
        "queries": ["line one"],
        "answers": [""],
        "gold_doc_indices": [1],
        "source": "absencebench",
    },
    "xabsence": {
        "documents": [
            {"corpus": "A", "text": "Claim in A."},
            {"corpus": "B", "text": "Paraphrase in B."},
        ],
        "queries": ["Find unmatched claims."],
        "answers": [""],
        "gold_doc_indices": [0],
        "source": "pubmed",
    },
    "qdmatch": {
        "documents": [
            {"type": "query", "text": "Who wrote it?"},
            {"type": "document", "text": "Written by Ada."},
        ],
        "queries": ["Match queries to documents."],
        "answers": [""],
        "gold_doc_indices": [[1, 2]],
        "source": "hpqa",
    },
    "cycle": {
        "documents": [{"text": "A outranks B."}, {"text": "B outranks A."}],
        "queries": ["Find cycles."],
        "answers": [""],
        "gold_doc_indices": [[1, 2]],
        "source": "synthetic",
    },
    "strmatch": {
        "documents": [{"text": "alpha beta gamma"}, {"text": "beta gamma delta"}],
        "queries": ["Find pairs sharing a run of 2 words."],
        "answers": [""],
        "gold_doc_indices": [[1, 2]],
        "source": "synthetic",
    },
    "redundancy": {
        "documents": [{"text": "Sales rose 4%."}, {"text": "Revenue increased four percent."}],
        "queries": ["Find redundant claims."],
        "answers": [""],
        "gold_doc_indices": [[1, 2]],
        "source": "pubmed",
    },
    "mathmatch": {
        "documents": [{"text": "3 * 7 + 1"}, {"text": "20 + 2"}],
        "queries": ["Find pairs whose values differ by at most 1."],
        "answers": [""],
        "gold_doc_indices": [[1, 2]],
        "source": "synthetic",
    },
    "groups4": {
        "documents": [{"text": "1 + 1"}, {"text": "4 - 2"}, {"text": "6 / 3"}],
        "queries": ["Find groups of 3 whose values are within 1 of each other."],
        "answers": [""],
        "gold_doc_indices": [[1, 2, 3]],
        "source": "synthetic",
    },
    "textgroups": {
        "documents": [
            {"text": "A short passage about birds."},
            {"text": "Another about trains."},
            {"text": "A third about rivers."},
        ],
        "queries": ["Find groups of 3 whose noun counts sum to 9."],
        "answers": [""],
        "gold_doc_indices": [[1, 2, 3]],
        "source": "synthetic",
    },
    # Same task, three instruction shapes. The `_task_of` map below routes these back to the real
    # task name when generating.
    "retrieval_multigold": {
        "documents": [{"title": "T1", "text": "body one"}, {"title": "T2", "text": "body two"}],
        "queries": ["Who built it?"],
        "answers": ["Ada"],
        "gold_doc_indices": [0, 1],  # >1 gold -> the multi-doc instruction
        "source": "hotpotqa",
    },
    "retrieval_multiquery": {
        "documents": [{"title": "T1", "text": "body one"}, {"title": "T2", "text": "body two"}],
        "queries": ["Who built it?", "When was it built?"],
        "answers": ["Ada", "1931"],
        "gold_doc_indices": [[0], [1]],  # >1 query -> the multi-query instruction
        "source": "hotpotqa",
    },
}

#: Fixture-key -> real task name, for keys that exist only to pin an example-dependent variant.
TASK_OF = {
    "retrieval_multigold": "retrieval",
    "retrieval_multiquery": "retrieval",
}

PAIR_CASES = [
    "[[1, 4], [3, 7]]",
    "1, 37], [6, 60], [35, 71]]",  # primed '[[' dropped -- the bug that read EM 0.60 vs true >0.9
    "[1, 4], [3, 7]]",  # primed outer '[' dropped
    "[]",
    "",
    "The pairs are (1, 4) and (3, 7).",
    "[[4, 1]]",  # unordered: contradiction pairs are sorted
    "not a pair anywhere",
    "[[1, 2, 3]]",  # wrong arity
]

QD_PAIR_CASES = [
    "[[1, 8], [3, 4]]",
    "[[8, 1]]",  # order-preserving: (query, doc) is NOT interchangeable
    "[]",
    "",
]

CYCLE_CASES = [
    "[[3, 8, 12]]",
    "[[3, 8, 12], [1, 4, 9]]",
    "[3, 8, 12]",  # a single flat cycle, not a list of cycles
    "[]",
    "",
    "[[8, 3, 12]]",  # unordered: cycles are normalized to sorted sets
    "[[3, 3, 8]]",  # duplicate id inside a cycle
    "[[5]]",  # length-1 is not a cycle
    "The cycle is [3, 8, 12] I think.",
    "no cycles anywhere",
]

ID_SET_CASES = [
    ("Missing: [3], [7]", 10),
    ("Unmatched: [3], [7]", 10),
    ("Missing: 3, 7", 10),  # bare ints, no brackets
    ("reasoning first\nMissing: [3]", 10),  # anchor takes the LAST occurrence
    ("Missing: [1]\nactually no\nMissing: [4]", 10),
    ("Missing: [99]", 10),  # out of range -> filtered
    ("Missing:", 10),  # explicit empty answer
    ("[]", 10),
    ("", 10),
    ("no anchor and no ids", 10),
]

SNIPPET_CASES = [
    '["Bob was not happy", "Jane felt sad that"]',
    'Bob was not happy", "Jane felt sad that"]',  # dropped leading '["' -- chunked failure mode
    '"Bob was not happy", "Jane felt sad that"]',  # dropped leading '[' only
    "[]",
    "",
    'Here you go: ["Bob was not happy"]',
    "not a list at all",
]

SNIPPET_NORM_CASES = [
    "  Bob was not happy  ",
    '"Bob was not happy"',
    "Bob was not happy indeed and more",  # keeps first 4 tokens only
    "...Bob was not happy!!!",
    "BOB WAS NOT HAPPY",
]

# Grouping and reorder primaries, checked against sklearn/scipy so the pure-Python versions in
# ctc.format.metrics can be trusted to match. Keeping the primaries dependency-free is what lets
# ctc.format stay importable without sklearn/scipy; ARI/NMI/spearman remain optional extras.
PAIRWISE_CASES = [
    ([[1, 2], [3]], [[1, 2], [3]], 3),
    ([[1, 2, 3]], [[1, 2], [3]], 3),
    ([[1], [2], [3]], [[1, 2], [3]], 3),
    ([[1, 2], [3, 4]], [[1, 3], [2, 4]], 4),
]

KENDALL_CASES = [
    ([1, 2, 3], [1, 2, 3]),
    ([3, 2, 1], [1, 2, 3]),
    ([1, 3, 2], [1, 2, 3]),
    ([2, 1, 4, 3], [1, 2, 3, 4]),
    ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
]

CYCLE_METRIC_CASES = [
    ([[3, 8, 12]], [[3, 8, 12]]),
    ([[3, 8]], [[3, 8, 12]]),  # partial: cycle-level 0, claim-level > 0
    ([], []),
    ([], [[3, 8, 12]]),
    ([[3, 8, 12]], []),
    ([[3, 8, 12], [1, 4]], [[3, 8, 12]]),
]

PAIR_METRIC_CASES = [
    ([[1, 4], [3, 7]], [[1, 4], [3, 7]]),
    ([[1, 4]], [[1, 4], [3, 7]]),
    ([[1, 4], [2, 9]], [[1, 4]]),
    ([], []),
    ([], [[1, 4]]),
    ([[1, 4]], []),
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


def _pairwise_reference(old_et) -> dict:
    """
    Pairwise precision/recall/F1 for grouping, computed exactly as ``_eval_grouping`` does.

    :param old_et: The pre-migration ``eval_tasks`` module, for ``partition_to_labels``.

    :returns: ``"pred|gold|n" -> metrics``.
    """
    from itertools import combinations

    out = {}
    for pred, gold, n in PAIRWISE_CASES:
        pl = old_et.partition_to_labels(pred, n)
        gl = old_et.partition_to_labels(gold, n)
        pp = {(i, j) for i, j in combinations(range(n), 2) if pl[i] == pl[j]}
        gp = {(i, j) for i, j in combinations(range(n), 2) if gl[i] == gl[j]}
        tp = len(pp & gp)
        p = tp / len(pp) if pp else 0.0
        r = tp / len(gp) if gp else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        out[f"{json.dumps(pred)}|{json.dumps(gold)}|{n}"] = {
            "pairwise_precision": p, "pairwise_recall": r, "pairwise_f1": f1,
        }
    return out


def _kendall_reference() -> dict:
    """
    Kendall tau from scipy, the reference the pure-Python implementation must match.

    Permutations have no ties, so scipy's tau-b reduces to tau-a and a plain concordant/discordant
    count is exact. Pinning it here is what makes that claim checkable rather than assumed.

    :returns: ``"pred|gold" -> tau``.
    """
    from scipy.stats import kendalltau

    return {
        f"{json.dumps(pred)}|{json.dumps(gold)}": float(kendalltau(pred, gold).correlation)
        for pred, gold in KENDALL_CASES
    }


def _build_prompts(old_df) -> dict:
    """
    Snapshot ``build_prompt`` for every case in :data:`PROMPT_CASES`.

    Collects failures instead of stopping at the first, so a missing field in a synthetic example
    is reported for every task at once rather than one run per fix.

    :param old_df: The pre-migration ``data_format`` module.

    :returns: ``"task|position|alpaca=X" -> [prompt, target]``.

    :raises SystemExit: If any case fails, listing all of them.
    """
    out, failures = {}, []
    for task, pos, alpaca in PROMPT_CASES:
        try:
            prompt, target = old_df.build_prompt(
                PROMPT_EXAMPLES[task],
                task=TASK_OF.get(task, task),
                query_position=pos,
                use_alpaca=alpaca,
                cot_mode="none",
            )
        except Exception as e:  # noqa: BLE001 -- all reported together below
            failures.append(f"  {task}|{pos}|alpaca={alpaca}: {type(e).__name__}: {e}")
            continue
        out[f"{task}|{pos}|alpaca={alpaca}"] = [prompt, target]
    if failures:
        raise SystemExit(
            "build_prompt failed for these cases -- the synthetic example is missing a field the "
            "task requires:\n" + "\n".join(failures)
        )
    return out


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

    from corpus_reasoning.eval import evaluate as old_ev  # type: ignore
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
        "prompts": _build_prompts(old_df),
        "parse_pairs": {t: old_ev.parse_pairs(t) for t in PAIR_CASES},
        "parse_cycles": {t: old_ev.parse_cycles(t) for t in CYCLE_CASES},
        "pairwise_f1": _pairwise_reference(old_et),
        "kendall_tau": _kendall_reference(),
        "parse_id_set": {
            f"{t}|{n}": (
                sorted(r) if (r := old_ev._parse_id_set(t, n)) is not None else None
            )
            for t, n in ID_SET_CASES
        },
        "parse_snippet_list": {t: old_ev._parse_snippet_list(t) for t in SNIPPET_CASES},
        "norm_snippet": {s: old_ev._norm_snippet(s) for s in SNIPPET_NORM_CASES},
        "cycle_metrics": {
            f"{json.dumps(p)}|{json.dumps(g)}": old_ev.cycle_metrics(p, g)
            for p, g in CYCLE_METRIC_CASES
        },
        "parse_qd_pairs": {t: old_ev.parse_qd_pairs(t) for t in QD_PAIR_CASES},
        "pair_metrics": {
            f"{json.dumps(p)}|{json.dumps(g)}": old_ev.pair_metrics(p, g)
            for p, g in PAIR_METRIC_CASES
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
