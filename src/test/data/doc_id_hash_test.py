"""Hashed document ids (``--doc-id-style hash``): the id draw is pinned, and on every id-bearing setA
spec the relabel touches exactly the document labels and the answer's ids, and inverts.

The SAME module is vendored in olmo-eval (``ctc_suite/doc_id_hash.py``) with the same pinned vector;
real-row token parity with the eval is ``recipes/eval/setA_audit/check_doc_marker_parity.py
--doc-id-style hash``.
"""

import json
import re

import pytest

from olmo_core.data.corpus_reasoning_prompts import build_prompt
from olmo_core.data.corpus_reasoning_prompts.doc_id_hash import (
    ID_MAX,
    ID_MIN,
    PINNED_EXAMPLE,
    PINNED_IDS,
    hash_doc_ids,
    relabel_answer,
    relabel_prompt,
    shows_doc_ids,
    unmap_ids,
)


def _docs(n, **extra):
    return [{"text": f"body text of item {i} with words", **extra} for i in range(n)]


#: spec -> a minimal unified example its renderer accepts (cot_mode none answers are id-only)
EXAMPLES = {
    "retrieval": {"documents": _docs(12), "queries": ["who?"], "gold_doc_indices": [3, 7]},
    "outlier": {
        "documents": _docs(9),
        "queries": ["find the outliers"],
        "gold_doc_indices": [0, 8],
    },
    "rerank": {
        "documents": _docs(6),
        "queries": ["q"],
        "gold_doc_indices": [2],
        "ce_scores": [1.0, None, 5.0, -3.0, None, 0.5],
    },
    "qdmatch": {
        "documents": [
            {"text": f"item {i}", "type": "query" if i % 2 == 0 else "doc"} for i in range(8)
        ],
        "queries": [],
        "gold_pairs": [[1, 2], [5, 8]],
    },
    "xabsence": {
        "documents": [{"text": f"passage {i}", "corpus": "A" if i < 5 else "B"} for i in range(9)],
        "queries": [],
        "gold_doc_indices": [0, 3],
        "orphan_side": "A",
        "num_unmatched": 2,
        "source": "xabsence_gutenberg",
    },
    "contradiction": {
        "documents": _docs(10),
        "queries": [],
        "gold_doc_indices": [[2, 9], [4, 5]],
    },
    "strmatch": {
        "documents": _docs(10),
        "queries": ["find pairs"],
        "gold_doc_indices": [[1, 4], [6, 10]],
    },
    "reorder": {
        "documents": [{"text": f"para {i}a\n\npara {i}b"} for i in range(7)],
        "queries": [],
        "gold_order": [3, 1, 7, 2, 6, 4, 5],
    },
    "grouping": {
        "documents": [{"text": f"abstract {i}", "title": f"Paper {i}"} for i in range(6)],
        "queries": ["group them"],
        "answers": [json.dumps({"groups": [{"doc_ids": [1, 4, 6]}, {"doc_ids": [2, 3, 5]}]})],
    },
}


def test_pinned_ids_and_range():
    assert hash_doc_ids(PINNED_EXAMPLE) == PINNED_IDS
    ex = {"documents": _docs(5000)}
    ids = hash_doc_ids(ex)
    assert len(set(ids)) == 5000 and all(ID_MIN <= i <= ID_MAX for i in ids)
    assert hash_doc_ids(json.loads(json.dumps(ex))) == ids  # deterministic
    assert hash_doc_ids({"documents": _docs(5000)[:-1] + [{"text": "x"}]}) != ids


@pytest.mark.parametrize("task", sorted(EXAMPLES))
@pytest.mark.parametrize("use_titles", [False, True])
def test_relabel_changes_only_ids_and_inverts(task, use_titles):
    ex = EXAMPLES[task]
    assert shows_doc_ids(ex, task)
    prompt, answer = build_prompt(
        ex, task=task, query_position="both", use_alpaca=False, cot_mode="none",
        use_titles=use_titles,
    )
    ids = hash_doc_ids(ex)
    hp = relabel_prompt(prompt, ex, task, ids)
    ha = relabel_answer(answer, ex, task, ids)
    n = len(ex["documents"])
    # every hashed id is shown exactly once, and nothing but the n labels changed
    for h in ids:
        assert len(re.findall(rf"(?<![0-9]){h}(?![0-9])", hp)) == 1
    back = hp
    for i, h in enumerate(ids):
        back = re.sub(rf"(?<![0-9]){h}(?![0-9])", str(i + 1), back)
    assert back == prompt
    for doc in ex["documents"]:
        assert doc["text"].replace("\n\n", "\n") in hp
    # the answer: same shape, every id mapped, and unmapping gives the sequential answer back
    assert re.sub(r"[0-9]+", "#", ha) == re.sub(r"[0-9]+", "#", answer)
    assert {int(x) for x in re.findall(r"[0-9]+", ha)} <= set(ids)
    assert unmap_ids(ha, ex, task, ids) == answer
    assert n >= 1


def test_unmap_unknown_ids_are_out_of_range():
    ex = EXAMPLES["retrieval"]
    ids = hash_doc_ids(ex)
    n = len(ex["documents"])
    got = unmap_ids(f"[{ids[4]}], [3], [{ID_MIN - 1}], [123456]", ex, "retrieval", ids)
    assert got == f"[5], [{n + 1}], [{n + 1}], [{n + 1}]"


def test_unnumbered_and_unknown_tasks():
    ex = {"documents": [{"text": "a || b"}], "queries": ["q"], "answers": ["7"]}
    assert relabel_prompt("p 7", ex, "oolong") == "p 7"
    assert relabel_answer("7", ex, "oolong") == "7"
    assert unmap_ids("7", ex, "oolong") == "7"
    textdiff = {"documents": _docs(3), "meta": {"format": "textdiff"}}
    assert not shows_doc_ids(textdiff, "absence")
    with pytest.raises(ValueError):
        shows_doc_ids(ex, "no_such_task")


def test_non_id_answer_is_refused():
    ex = EXAMPLES["outlier"]
    with pytest.raises(ValueError):
        relabel_answer("Most reviews are 5-star.\nOutliers: [1]", ex, "outlier")
    with pytest.raises(ValueError):
        relabel_answer("Outliers: [99]", ex, "outlier")  # out of range


def test_missing_label_raises():
    ex = EXAMPLES["contradiction"]
    prompt, _ = build_prompt(ex, task="contradiction", query_position="both", use_alpaca=False,
                             cot_mode="none")
    with pytest.raises(ValueError):
        relabel_prompt(prompt.replace("Claim 3: ", "Claim three: "), ex, "contradiction")


def test_segment_prompt_seq_is_default_and_hash_relabels():
    tokenizers = pytest.importorskip("transformers")
    try:
        tok = tokenizers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    except Exception:  # no hub access
        pytest.skip("tokenizer unavailable")
    from olmo_core.data.document_chunk_landmark import segment_prompt_to_chunks

    ex = EXAMPLES["contradiction"]
    kw = dict(query_position="both", cot_mode="none", chunk_by="document", include_answer=True)
    _, ids0, m0 = segment_prompt_to_chunks(tok, ex, "contradiction", **kw)
    _, ids1, m1 = segment_prompt_to_chunks(tok, ex, "contradiction", doc_id_style="seq", **kw)
    assert (ids0, m0) == (ids1, m1)
    _, ids2, m2 = segment_prompt_to_chunks(tok, ex, "contradiction", doc_id_style="hash", **kw)
    text = tok.decode(ids2)
    for h in hash_doc_ids(ex):
        assert f"Claim {h}: " in text
    # the labeled span may open with the template's empty think block; the target is the JSON
    answer = tok.decode([t for t, m in zip(ids2, m2) if m])
    assert json.loads(re.search(r"\[\[.*\]\]", answer).group()) == [
        [hash_doc_ids(ex)[a - 1], hash_doc_ids(ex)[b - 1]] for a, b in ex["gold_doc_indices"]
    ]
