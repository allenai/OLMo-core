"""
Rung -> document count, per task. The calibration table for the CTC suite.

A rung is a *token* budget; a generator takes a *document* count. This table is the bridge, and it
is per task because a contradiction claim is ~42 tokens while a BEIR SciFact abstract is ~365 -- the
same "8k" means 187 documents for one and 21 for the other.

**Getting this wrong is not a sizing nuisance, it is a mislabelled axis.** The pre-migration
contradiction ladder was fit against a filler pool that turned out to be 92-99.6% FEVER/wiki rather
than PubMed. Wikipedia trivia claims tokenize at ~22.8 tok/doc against PubMed's ~43, so re-running
the same document counts against the corrected pool overshot every rung by ~1.8x -- the file called
``rung_2048`` actually rendered to 3413 tokens, and ``32k`` to 61461. Every number plotted against
that axis was plotted against the wrong x.

So each row records how it was calibrated, and ``estimated`` rows are exactly the ones to re-measure
before quoting a length. The authority for un-ported tasks remains
``src/scripts/data/ctc_suite/BUILD_MATRIX.md`` in the pre-migration tree; rows land here as their
generators do. This module is the single source: contradiction's spec derives its
``CLAIMS_PER_RUNG`` -- and hence its ``extra["claims_per_rung"]`` -- from the row below rather than
declaring a copy. That ladder has already been wrong once, and two copies of it drift.

**Rungs are open-ended, not a closed set.** The table stops at 32k because that is what has been
tokenizer-measured, but any parseable rung label (``64k``, ``256k``, ``1m``, ``10m``, ...) resolves
to a document count by extrapolating the least-squares line through the task's own table
(:func:`fit_for`). An extrapolated count has, by definition, never been measured against the real
tokenizer at that length, so the build report flags it and the flag should follow the number. Two
hard limits survive extrapolation: :data:`CEILINGS`, for a corpus that arithmetic says cannot
supply the documents, and the generator itself, which fails loudly when its pool runs dry
(``50 consecutive rejections``) rather than quietly recycling material.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

from ..format import rungs as rung_util

__all__ = [
    "LADDERS",
    "CEILINGS",
    "SUPPLY_BOUNDED",
    "docs_for_rung",
    "fit_for",
    "is_extrapolated",
    "rungs_for",
    "max_rung",
]

#: task -> {rung label: documents per example}. Synthetic tasks tokenize 1.5-3x higher per
#: character than natural text, which is why their per-document estimates are so much lower than
#: the retrieval tasks'. ``oolong`` is the exception in kind rather than degree: its value is a
#: **token budget**, because its items are lines inside one document rather than documents.
LADDERS: Dict[str, Dict[str, int]] = {
    # ── pure synthetic ──
    # ~25-30 tok/claim. BUILD_MATRIX recommends capping at n1000 and letting 32k run slightly short.
    # ~15-20 tok/expression -- the shortest documents in the suite, hence the largest counts.
    # ~27 tok/string at str_len=10, tokenizer-MEASURED (2022/4080/8201/16459/33193 median prompt
    # tokens, every rung within 1.3% of its label). NOT the BUILD_MATRIX row-20 counts
    # (38/82/170/350/700), which that row itself flagged `synth x1.5-3 ... calibrate before
    # freezing n values`: the multiplier is not there. Re-measuring the SHIPPED
    # `eval_rungs/strmatch/rung_{2048,32768}.jsonl` gives 1119 and 18696 median tokens against
    # labels of 2048 and 32768 -- every shipped strmatch rung is ~0.56x its label, so a strmatch
    # point plotted at 32k was really a 19k point. The wiki vocabulary those files used and the
    # frozen wordlist here tokenize almost identically (26.7-29.5 vs 27.3-30.9 tok/doc), so this
    # row is a recalibration, not a consequence of the vocabulary swap.
    "strmatch": {"2k": 72, "4k": 149, "8k": 301, "16k": 606, "32k": 1216},
    # ~150 tok/passage: the feature must be spread densely over several sentences, so a textgroups
    # document is an order of magnitude longer than a short synthetic one at the same task shape.
    "textgroups": {"2k": 11, "4k": 24, "8k": 50, "16k": 103, "32k": 210},
    # ── the five in-distribution CTC-suite ladders ──
    # ~43 tok/claim, tokenizer-MEASURED at 1925/3933/8052/16074/32397 median tokens against a
    # PubMed-only filler pool. NOT the pre-migration `contra` row (40/88/190/385/765), which was
    # fit against a pool that was 92-99.6% FEVER/wiki: Wikipedia trivia claims tokenize at ~22.8
    # tok/doc, so re-running those counts against the corrected pool overshoots every rung by
    # ~1.8x (n=77 measured 3413 tokens, not 2048; n=1423 measured 61461, not 32768).
    "contradiction": {"2k": 44, "4k": 92, "8k": 187, "16k": 379, "32k": 762},
    # ~42 tok/claim, tokenizer-MEASURED (2044/3980/8099/16648/32897 median prompt tokens). Its own
    # BUILD_MATRIX row (17) was struck out when the task was dropped from the suite, so there is no
    # pre-migration ladder to inherit -- but the corpus is contradiction's, at the same document
    # shape, and the two fits agree to within 3% at every rung (46/95/193/390/784 vs 44/92/187/
    # 379/762), which is the cross-check that says the fit is measuring the corpus and not noise.
    "nq": {"2k": 11, "4k": 23, "8k": 48, "16k": 100, "32k": 200},  # ~160 tok/passage
    # ~113 tok/paragraph, tokenizer-MEASURED. NOT the BUILD_MATRIX row-2 counts (11/24/50/100/205),
    # which the 2026-07-19 "FIX2" recalibration found undershooting their labels by 0.64-0.69x: a
    # least-squares fit over the built rungs gave `tokens ~= 66.6 + 113.36 * n_docs`, and the
    # rebuilt 17/36/72 re-measured at 1954/4124/8240 median tokens (ratios 0.954/1.007/1.006). The
    # shipped `eval_rungs/hotpotqa/rung_{2048,4096,8192,16384}.jsonl` carry exactly 17/36/72/144
    # documents. 32k is the same fit extrapolated one rung past the shipped ladder.
    "hotpotqa": {"2k": 17, "4k": 36, "8k": 72, "16k": 144, "32k": 288},
    "outlier": {"2k": 14, "4k": 28, "8k": 57, "16k": 115, "32k": 220},  # ~140 tok/passage
    # ~100-160 tok/passage, the widest uncertainty band here: BUILD_MATRIX gives a range per rung
    # (13-18 / 25-38 / 50-78 / 100-158 / 200-315) and these are its midpoints. Re-measure before
    # quoting a rerank context length.
    "rerank": {"2k": 15, "4k": 30, "8k": 62, "16k": 125, "32k": 250},
    # TOKEN budgets, not document counts. The generator draws items until the budget is met.
    "oolong": {"2k": 2048, "4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768},
    # ── the absence family ──
    # ~76 tok/sentence, tokenizer-MEASURED. NOT BUILD_MATRIX row 18's ~20/sentence ->
    # {90,180,360,720,1440}: that estimate charges each sentence once, and an absence prompt
    # carries the whole corpus TWICE -- numbered as Version A, then again inside the second
    # version. The shipped `absence_eval_gutenberg_n{10,50,200}_k3.jsonl` measure 548/3117/14790
    # median Qwen3 tokens, giving `tokens ~= -412 + 75.7 * n_docs`, so the estimated ladder
    # overshoots by ~3.4x and the staged `n1440` file is a ~109k-token file labelled 32k.
    "absence": {"2k": 32, "4k": 60, "8k": 114, "16k": 222, "32k": 438},
    # ~33 tok/claim, tokenizer-MEASURED, and ODD on purpose: an example is 2P+k documents, so an
    # even rung with the default k=3 would round down to one document under its label. NOT
    # BUILD_MATRIX row 22's P18/P39/P81/P165/P333 (39/81/165/333/669 documents at an estimated ~95
    # tok/pair); the shipped `xabsence_eval_pubmed_p{8,18,48}_k3.jsonl` measure 772/1394/3424
    # median tokens at 19/39/99 documents, giving `tokens ~= 120 + 33.3 * n_docs`.
    "xabsence": {"2k": 59, "4k": 119, "8k": 243, "16k": 489, "32k": 981},
    # ── the ladders whose rungs are drawn independently (gold covers every document) ──
    # ~151 tok/passage, tokenizer-MEASURED over THIS generator's own output rather than over the
    # shipped files: a 96-example build at n=14/58/233 measures 2178/8890/35333 median prompt
    # tokens, giving `tokens ~= 84 + 151.3 * n_docs`. The shipped
    # `reorder_gutenberg100w_n{5,20,50}_eval_500.jsonl` fit 140.5/passage with their whitespace
    # collapsed, i.e. ~7% lower -- the ported chunker closes a passage on whole SENTENCES once it
    # reaches 100 words, so it overshoots the target by the length of the closing sentence
    # (measured median 112 words, p95 143). Both fits bracket BUILD_MATRIX row 24
    # (12/27/57/116/234).
    # NOTE the ANSWER is what binds at the top of this ladder, not the prompt: the target is a
    # permutation of n ids at ~4.5 tokens each, so 32k needs ~980 decode tokens. That is why the
    # spec raises max_new_tokens to 2048; at the pre-migration 1024 the 32k target does not fit and
    # every example at that rung parses as None.
    "reorder": {"2k": 13, "4k": 27, "8k": 54, "16k": 108, "32k": 216},
    # ~183 tok/abstract (title + a 120-word abstract), tokenizer-MEASURED over this generator's own
    # output: 2210/8215/32184 median prompt tokens at n=11/44/175, giving
    # `tokens ~= 187 + 182.8 * n_docs`. The shipped `openalex_grouping_n{20,100}_levels_eval_*`
    # files fit 187.0/abstract, i.e. the same corpus measured the same way. BUILD_MATRIX row 14
    # estimated ~180 tok/doc net of a ~300-token overhead allowance, landing on 9/20/42/85/170.
    "grouping_labeled": {"2k": 10, "4k": 21, "8k": 44, "16k": 89, "32k": 178},
    # ITEMS (M queries + N documents), NOT the `q` of BUILD_MATRIX rows 21a/21b -- an example
    # renders 2q items and both the ladder and `ctc.data.build.shrink` measure `len(documents)`.
    # ~88 tok/item, tokenizer-MEASURED over this generator's own output: 2135/8206/33002 median
    # prompt tokens at n=21/92/374, giving `tokens ~= 231 + 87.58 * n_docs`. The shipped
    # `qdmatch_eval_nq_q{20,50,100,250}_*_separate.jsonl` files fit 87.06/item -- the same number,
    # which is expected since both draw NQ gold passages -- and that is also BUILD_MATRIX's
    # "~175 per (query+doc) unit". Its q9/q20/q42/q87/q178 (= 18/40/84/174/356 items) sit ~5% low
    # because that row also charged a ~300-token query/answer overhead.
    "qdmatch_nq": {"2k": 21, "4k": 44, "8k": 91, "16k": 184, "32k": 372},
    # HotpotQA paragraphs run a little longer than NQ's 100-word DPR chunks, but the shipped hpqa
    # files were built at the same item counts as the NQ ones and no separate fit exists; measure
    # before quoting a hpqa context length.
    "qdmatch_hpqa": {"2k": 21, "4k": 44, "8k": 91, "16k": 184, "32k": 372},
    # ── the four held-out (OOD) ladders ──
    "fiqa": {"2k": 4, "4k": 9, "8k": 19, "16k": 40, "32k": 80},  # ~400 tok/post
    "scifact": {"2k": 5, "4k": 10, "8k": 21, "16k": 43, "32k": 88},  # ~365 tok/abstract
    "outlier_review": {"2k": 20, "4k": 40, "8k": 80, "16k": 160, "32k": 320},  # ~100 tok/review
}

#: How each row was arrived at. ``estimated`` means an offline per-document token estimate, not a
#: measurement against the real tokenizer -- re-measure before quoting a context length.
CALIBRATION: Dict[str, str] = {
    "strmatch": (
        "measured (Qwen3 tokenizer, frozen wordlist); REPLACES BUILD_MATRIX row 20, whose "
        "38/82/170/350/700 renders to ~0.56x of every rung label"
    ),
    "textgroups": "estimated",
    "contradiction": "measured (Qwen3 tokenizer, PubMed-only filler pool)",
    "nq": "estimated (BUILD_MATRIX row 1)",
    "hotpotqa": (
        "measured 2k/4k/8k (FIX2 recalibration, 1954/4124/8240 median tokens); 16k/32k from the "
        "same fitted 66.6 + 113.36*n, 16k built and shipped, 32k extrapolated"
    ),
    "outlier": "estimated (BUILD_MATRIX row 11; matches the shipped n14-220 files)",
    "rerank": "estimated, wide (BUILD_MATRIX rows 9/10 give a range; these are its midpoints)",
    "oolong": "exact (the value IS the token budget the generator fills)",
    "absence": (
        "measured (Qwen3 tokenizer over the shipped gutenberg n10/n50/n200 files, fitted "
        "-412.3 + 75.74*n); replaces BUILD_MATRIX row 18's ~3.4x-overshooting estimate"
    ),
    "xabsence": (
        "measured (Qwen3 tokenizer over the shipped pubmed p8/p18/p48 files, fitted "
        "120.1 + 33.31*n); replaces BUILD_MATRIX row 22's ~1.4x-overshooting estimate"
    ),
    "reorder": (
        "measured (Qwen3 tokenizer over a 96-example build of THIS generator at n=14/58/233, "
        "fitted 83.8 + 151.31*n); brackets BUILD_MATRIX row 24 with the shipped-file fit"
    ),
    "grouping_labeled": (
        "measured (Qwen3 tokenizer over a 96-example build of THIS generator at n=11/44/175, "
        "fitted 186.8 + 182.82*n); agrees with the shipped openalex n20/n100 files at 187.0/doc"
    ),
    "qdmatch_nq": (
        "measured (Qwen3 tokenizer over a 96-example build of THIS generator at n=21/92/374 "
        "ITEMS, fitted 230.6 + 87.58*n); agrees with the shipped nq q20-q250 files at 87.06/item"
    ),
    "qdmatch_hpqa": (
        "estimated: the qdmatch_nq fit, reused. The shipped hpqa files were built at the same item "
        "counts but were never separately measured -- HotpotQA paragraphs are not DPR chunks"
    ),
    "fiqa": "estimated (BUILD_MATRIX row 8)",
    "scifact": "estimated (BUILD_MATRIX row 7)",
    "outlier_review": "estimated (BUILD_MATRIX row 12)",
}


#: Tasks whose corpus arithmetic FORBIDS a rung, however patient the build: ``task -> (highest
#: buildable rung, why)``. These are refusals, not warnings, because the generator would otherwise
#: spin through its pool and die on the rejection limit with a message that reads like a transient
#: problem. Tasks merely *likely* to exhaust supply at some unmeasured point belong in
#: :data:`SUPPLY_BOUNDED` instead.
CEILINGS: Dict[str, Tuple[str, str]] = {
    # The frozen-suite roster (olmo-eval ctc_suite) documents this cap: 4,000 recoverable HotpotQA
    # units are the whole labeled universe, and a 512k example needs ~7k distinct units.
    "qdmatch_hpqa": ("256k", "4,000 labeled HotpotQA units exist and a 512k example needs ~7k"),
    # The BEIR SciFact corpus is 5,183 abstracts at ~365 tok each; a 2m example needs ~5.7k
    # distinct documents, more than exist.
    "scifact": ("1m", "the BEIR SciFact corpus is 5,183 abstracts and a 2m example needs ~5.7k"),
    # Measured against the generator: every document consumes ~9.8 distinct words from the frozen
    # 20,045-word vocabulary (planting rejects reuse), and n=2131 (56k) asks for 20,976.
    "strmatch": (
        "48k",
        "the frozen 20,045-word vocabulary caps ~1.9k documents at ~9.8 words each",
    ),
}

#: Tasks whose supply is bounded by a structure no table can price: ``task -> what bounds it``.
#: Unlike :data:`CEILINGS` these are not refused up front, because the bound depends on what the
#: corpus happens to contain -- the build instead fails loudly when the generator cannot draw. The
#: note is surfaced by ``ctc-data list`` so the failure is expected rather than diagnosed.
SUPPLY_BOUNDED: Dict[str, str] = {
    "absence": (
        "an example is one contiguous run of sentences from a single Gutenberg book, so the "
        "longest book in the pool bounds the rung"
    ),
    "reorder": (
        "an example is consecutive ~100-word passages of a single Gutenberg book, so the longest "
        "book in the pool bounds the rung"
    ),
    "rerank": (
        "every document must carry a cross-encoder score, so the per-query scored fill drawn at "
        "load time bounds the rung; raise the loader's fill size to go longer"
    ),
    "qdmatch_nq": "bounded by the ~79k labeled NQ-open queries (two items each); ~10m is the limit",
    "hotpotqa": "bounded by the benchmark's own 10-paragraph distractor sets plus mined negatives",
}

#: Structural constraints an extrapolated count must respect, applied after rounding. ``xabsence``
#: is the only current case: an example is 2P+k documents (k=3 by default), so an even count would
#: silently round the ladder down one pair below its label.
_CONSTRAIN: Dict[str, str] = {"xabsence": "odd"}


@lru_cache(maxsize=None)
def fit_for(task: str) -> Tuple[float, float]:
    """
    The least-squares line ``tokens = a + b * docs`` through the task's calibrated table rows.

    Fitted over the table rather than declared beside it so the two can never disagree: the fit
    IS the table, extended. For ``oolong``, whose rung values are already token budgets, the fit
    degenerates to the identity, which is exactly right.

    :param task: Task name.

    :returns: ``(a, b)`` -- intercept and tokens per document.

    :raises KeyError: If the task has no ladder.
    """
    if task not in LADDERS:
        raise KeyError(
            f"no rung ladder for {task!r}; have {', '.join(sorted(LADDERS))}. Un-ported tasks keep "
            "their ladder in the pre-migration BUILD_MATRIX.md."
        )
    points = [(docs, rung_util.parse_rung(label)) for label, docs in LADDERS[task].items()]
    n = len(points)
    mean_docs = sum(d for d, _ in points) / n
    mean_tokens = sum(t for _, t in points) / n
    slope = sum((d - mean_docs) * (t - mean_tokens) for d, t in points) / sum(
        (d - mean_docs) ** 2 for d, _ in points
    )
    return mean_tokens - slope * mean_docs, slope


def is_extrapolated(task: str, rung: str) -> bool:
    """
    :param task: Task name.
    :param rung: Rung label.

    :returns: True when the count for this rung comes from the fit rather than a calibrated table
        row -- i.e. when it has never been measured against the tokenizer and the build report
        should say so.
    """
    ladder = {rung_util.normalize(k) for k in LADDERS[task]}
    return rung_util.normalize(rung) not in ladder


def rungs_for(task: str) -> List[str]:
    """
    :param task: Task name.

    :returns: Its rung labels, ascending by context length.

    :raises KeyError: If the task has no ladder.
    """
    if task not in LADDERS:
        raise KeyError(
            f"no rung ladder for {task!r}; have {', '.join(sorted(LADDERS))}. Un-ported tasks keep "
            "their ladder in the pre-migration BUILD_MATRIX.md."
        )
    return rung_util.sort_rungs(LADDERS[task])


def docs_for_rung(task: str, rung: str) -> int:
    """
    Documents per example at a rung -- calibrated where the table has a row, extrapolated from
    :func:`fit_for` anywhere else.

    Extrapolation is deliberate, not a fallback: the ladder must reach any budget a corpus can
    supply (the shipped suite runs to 1M, and the synthetics arbitrarily far), and a closed rung
    set would make every new length a code change. The cost is that an extrapolated count was
    never measured against the tokenizer, which is why :func:`is_extrapolated` exists and the
    build report flags such rungs.

    :param task: Task name.
    :param rung: Rung label, in any accepted spelling (``"32k"``, ``"32768"``, ``"10m"``).

    :returns: Documents per example at that rung (for ``oolong``: the token budget itself).

    :raises KeyError: If the task is unknown.
    :raises ValueError: If the label does not parse, the rung is above the task's
        :data:`CEILINGS` entry, or it extrapolates below one document.
    """
    if task not in LADDERS:
        raise KeyError(
            f"no rung ladder for {task!r}; have {', '.join(sorted(LADDERS))}. Un-ported tasks keep "
            "their ladder in the pre-migration BUILD_MATRIX.md."
        )
    ladder = {rung_util.normalize(k): v for k, v in LADDERS[task].items()}
    label = rung_util.normalize(rung)
    if label in ladder:
        return ladder[label]

    tokens = rung_util.parse_rung(label)
    if task in CEILINGS:
        top, reason = CEILINGS[task]
        if tokens > rung_util.parse_rung(top):
            raise ValueError(
                f"{task} cannot be built past {top}: {reason}. Requested rung {label} is a "
                "request the corpus arithmetic cannot honour, not a missing table row."
            )
    intercept, slope = fit_for(task)
    docs = round((tokens - intercept) / slope)
    if _CONSTRAIN.get(task) == "odd" and docs % 2 == 0:
        docs -= 1
    if docs < 1:
        raise ValueError(
            f"rung {label} extrapolates to {docs} document(s) for {task}; the smallest useful "
            f"rung is around {format_min_rung(task)}"
        )
    return docs


def format_min_rung(task: str) -> str:
    """
    :param task: Task name.

    :returns: The task's shortest calibrated rung label, for error messages.
    """
    return rungs_for(task)[0]


def max_rung(task: str) -> str:
    """
    :param task: Task name.

    :returns: Its longest rung -- where a nested eval ladder is built before being shrunk down.
    """
    return rungs_for(task)[-1]
