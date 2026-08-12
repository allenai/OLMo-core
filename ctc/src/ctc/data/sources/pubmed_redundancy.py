"""
PubMed redundancy pairs: a real claim, an LLM paraphrase of it, and the hard negatives.

The mirror image of :mod:`ctc.data.sources.pubmed`. Gold is ``(S, S')`` where ``S`` is a real
PubMed sentence and ``S'`` restates the *same* fact; contradiction's gold is the same shape with
``S'`` denying it. Only one sentence of a gold pair is model-written, for the same reason: a corpus
where every planted sentence is LLM prose is findable by style rather than by meaning.

**The hard negatives are the point of this task, not a refinement of it.** They are pairs of
distinct claim sentences from *one* abstract -- same study, same entities, same vocabulary,
different fact -- kept only when a judge confirms they are NOT redundant. Without them "which two
sentences are about the same thing?" answers the task, and every long-context claim made from it
would be a claim about topic matching.

.. warning::
   **Lexical overlap still separates gold from the hard negatives, and it was measured.** Over the
   6807 gold pairs recovered from the shipped pre-migration files, gold word-Jaccard runs a median
   of 0.441 against 0.105 for the shipped hard negatives, and on
   ``redundancy_eval_pubmed_both_n100_k3_hn6.jsonl`` the maximum-overlap pair is a gold pair in
   **61 of 100** examples by Jaccard (50 of 100 by raw shared-word count, which is what
   :func:`ctc.data.audit.overlap_pair_is_gold` uses) against a 0.001 chance baseline. The
   ``--mode both`` default is why: half its gold is ``simple``, a light reword that leaves most
   words in place.

   Two levers are applied here and both were available pre-migration. ``mode`` defaults to
   ``subtle`` -- restate the fact sharing as few words as possible -- which is the same choice the
   contradiction port made for the same reason (a near-duplicate tell makes the pair findable by
   string match). And ``max_overlap`` rejects a gold pair whose word-Jaccard exceeds it, the
   near-duplicate backstop contradiction already carries at 0.5.

   Together they move the probe from **0.500 to 0.400** at a matched n=100, and neither removes the
   effect, because it is not a construction defect: a paraphrase of a sentence shares its content,
   so lexical similarity is a blunt version of the criterion rather than an alternative to it.
   :func:`ctc.data.audit.overlap_pair_is_gold` therefore runs on redundancy as an **accepted**
   probe, with the number recorded, exactly as ``mathmatch`` records ``closest_pair_is_gold``. What
   must not happen is the number being forgotten and a redundancy result being read as long-context
   reasoning when a bag-of-words baseline gets most of the way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .pubmed import clean_response, is_claim_sentence, load_abstracts, word_jaccard

__all__ = [
    "RedundantPair",
    "HardNegativePair",
    "RedundancyPool",
    "load_pool",
    "mine_pairs",
    "build_hardneg_pool",
    "PARAPHRASE_SIMPLE",
    "PARAPHRASE_SUBTLE",
    "JUDGE_PROMPT",
]


@dataclass(frozen=True)
class RedundantPair:
    """
    One gold redundancy.

    :param claim: The real PubMed sentence.
    :param paraphrase: The model-written sentence stating the same fact.
    :param abstract_id: Which abstract ``claim`` came from. Load-bearing: fillers may not be drawn
        from it, because another sentence of the same abstract may restate the same finding and
        would then be an unlabelled second gold pair.
    :param mode: ``"simple"`` or ``"subtle"``, for the provenance record.
    """

    claim: str
    paraphrase: str
    abstract_id: str
    mode: str = "subtle"


@dataclass(frozen=True)
class HardNegativePair:
    """
    One planted non-redundant pair: two claim sentences from the same abstract.

    :param first: One claim sentence.
    :param second: The other.
    :param abstract_id: The abstract both came from.
    :param overlap: Their word-Jaccard, kept so the pool can be ordered by it without recomputing.
    """

    first: str
    second: str
    abstract_id: str
    overlap: float = 0.0


@dataclass(frozen=True)
class RedundancyPool:
    """
    :param pairs: Gold pairs, in a fixed shuffled order. Example *i* uses pairs *i*..*i+K*.
    :param hardnegs: Planted non-redundant pairs, consumed the same way.
    :param fillers: ``abstract_id -> sentences``, the distractor supply.
    :param provenance: What produced the pairs, copied into every example's ``_meta``.
    """

    pairs: Tuple[RedundantPair, ...]
    hardnegs: Tuple[HardNegativePair, ...]
    fillers: Dict[str, Tuple[str, ...]]
    provenance: Dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.pairs)

    def for_split(self, split: str) -> "RedundancyPool":
        """
        Slice off this split's gold and hard-negative pairs; fillers are shared.

        Both halves are split, not just gold: a hard-negative pair is a *question* too -- a model
        that saw "these two sentences from abstract 4711 are not redundant" in training has been
        told the answer for that pair, and the eval example containing it is no longer held out.

        :param split: ``"train"`` or ``"eval"``.

        :returns: A pool holding only that split's pairs.

        :raises ValueError: On an unknown split name.
        """
        if split not in ("train", "eval"):
            raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
        gold_cut = max(1, len(self.pairs) // 10)
        hard_cut = max(1, len(self.hardnegs) // 10)
        if split == "eval":
            pairs, hardnegs = self.pairs[:gold_cut], self.hardnegs[:hard_cut]
        else:
            pairs, hardnegs = self.pairs[gold_cut:], self.hardnegs[hard_cut:]
        return RedundancyPool(
            pairs=pairs, hardnegs=hardnegs, fillers=self.fillers, provenance=self.provenance
        )


def build_hardneg_pool(
    fillers: Dict[str, Tuple[str, ...]], rng, *, order: str = "overlap"
) -> List[HardNegativePair]:
    """
    Candidate non-redundant pairs: two claim sentences from one abstract.

    One pair per abstract, as pre-migration -- taking several would concentrate the hard negatives
    in whichever abstracts happen to be long, and an example's hard negatives are supposed to come
    from as many different studies as its fillers do.

    :param fillers: ``abstract_id -> sentences``.
    :param rng: Seeded RNG.
    :param order: How to choose the pair within an abstract. ``"overlap"`` takes the pair with the
        highest word-Jaccard, which is the hardest one available and narrows the lexical gap the
        module warning measures; ``"random"`` reproduces the pre-migration shuffle-and-take-two.

    :returns: The candidates, shuffled.

    :raises ValueError: On an unknown ``order``.
    """
    if order not in ("overlap", "random"):
        raise ValueError(f"order must be 'overlap' or 'random', got {order!r}")
    pairs: List[HardNegativePair] = []
    for abstract_id, sentences in fillers.items():
        claims = [s for s in sentences if is_claim_sentence(s)]
        if len(claims) < 2:
            continue
        if order == "random":
            rng.shuffle(claims)
            first, second = claims[0], claims[1]
        else:
            # Capped: an abstract with 30 claim sentences would otherwise cost 435 comparisons for
            # one kept pair, and the long tail contributes nothing the first few sentences do not.
            head = claims[:12]
            first, second = max(
                ((a, b) for i, a in enumerate(head) for b in head[i + 1 :]),
                key=lambda pair: word_jaccard(*pair),
            )
        pairs.append(
            HardNegativePair(first, second, abstract_id, round(word_jaccard(first, second), 4))
        )
    rng.shuffle(pairs)
    return pairs


PARAPHRASE_SIMPLE = """Lightly paraphrase the following biomedical sentence so it states exactly the same fact, changing only a few words (synonyms, minor reordering). Keep all numbers, entities, and the finding identical in meaning.

Return only the paraphrased sentence, no preamble or quotes.

Original: {sentence}
Paraphrase:"""

#: The default. Gold that shares most of its words with the original is findable by string match,
#: which is the whole content of the warning at the top of this module.
PARAPHRASE_SUBTLE = """Restate the following biomedical sentence so it conveys exactly the same fact and finding, but shares as few words as possible with the original. Rename entities (synonyms / alternative names / expand or contract abbreviations), use different verbs, a different sentence opening, and a different structure. Do NOT change any number, direction, magnitude, or the underlying finding — the two sentences must remain mutually entailing (a domain expert would say "these say the same thing").

Return only the restated sentence, no preamble or quotes.

Original: {sentence}
Restated:"""

JUDGE_PROMPT = """Sentence A: {a}
Sentence B: {b}

Do these two sentences state the SAME fact or finding — i.e. one is a paraphrase or restatement of the other (mutually entailing)? Two sentences about the same topic that assert DIFFERENT facts are NOT the same. A direct contradiction is NOT the same.

Answer with exactly one word: YES or NO."""


def mine_pairs(
    claims: Sequence[Tuple[str, str]],
    candidates: Sequence[HardNegativePair],
    *,
    mode: str = "subtle",
    model: str = "gemini-2.5-flash",
    base_url: Optional[str] = None,
    max_concurrent: int = 32,
    max_overlap: float = 0.5,
    validity_filter: bool = True,
    cache_dir: Optional[str] = None,
) -> Tuple[List[RedundantPair], List[HardNegativePair], Dict[str, object]]:
    """
    Write one paraphrase per claim, then throw away the ones that are not paraphrases -- and clear
    the hard negatives through the same judge, in the opposite direction.

    Both directions are needed and they fail differently. An unchecked paraphrase may have drifted
    into a different claim (a false *positive* in gold). An unchecked same-abstract pair may
    genuinely restate one fact twice (a false *negative*: real gold, unlabelled, scored as a model
    error every time it is found).

    :param claims: ``(sentence, abstract_id)`` gold sources.
    :param candidates: Same-abstract candidate hard negatives, from :func:`build_hardneg_pool`.
    :param mode: ``"simple"``, ``"subtle"`` (default) or ``"both"`` -- alternating, as
        pre-migration.
    :param model: Model id.
    :param base_url: OpenAI-compatible endpoint(s), comma-separated for a local fleet.
    :param max_concurrent: In-flight requests.
    :param max_overlap: Reject a gold pair whose word-Jaccard exceeds this. ``1.0`` disables it.
    :param validity_filter: Run the judge. Off is only right for a smoke test.
    :param cache_dir: Response cache directory.

    :returns: ``(gold, hardnegs, provenance)``. The provenance carries every rejection count,
        because "how many survived" is what says whether a build is trustworthy.
    """
    from ..llm import ChatClient

    client = ChatClient(
        base_url or "", model=model, max_concurrent=max_concurrent, cache_dir=cache_dir
    )
    modes = [
        ("simple" if i % 2 == 0 else "subtle") if mode == "both" else mode
        for i in range(len(claims))
    ]
    templates = {"simple": PARAPHRASE_SIMPLE, "subtle": PARAPHRASE_SUBTLE}
    responses = client.run(
        [templates[m].format(sentence=s) for (s, _), m in zip(claims, modes)],
        temperature=0.7,
        max_tokens=200,
    )

    written: List[Optional[str]] = []
    unusable = overlapping = 0
    for (sentence, _), response in zip(claims, responses):
        cleaned = clean_response(response)
        if not cleaned or cleaned.strip() == sentence.strip():
            written.append(None)
            unusable += 1
        elif max_overlap < 1.0 and word_jaccard(sentence, cleaned) > max_overlap:
            written.append(None)
            overlapping += 1
        else:
            written.append(cleaned)

    not_redundant = 0
    if validity_filter:
        live = [i for i, text in enumerate(written) if text is not None]
        verdicts = client.run(
            [JUDGE_PROMPT.format(a=claims[i][0], b=written[i]) for i in live],
            temperature=0.0,
            max_tokens=8,
        )
        for i, verdict in zip(live, verdicts):
            if not (verdict or "").strip().upper().startswith("YES"):
                written[i] = None
                not_redundant += 1

    gold = [
        RedundantPair(sentence, paraphrase, abstract_id, item_mode)
        for (sentence, abstract_id), paraphrase, item_mode in zip(claims, written, modes)
        if paraphrase is not None
    ]

    hardnegs = list(candidates)
    actually_redundant = 0
    if validity_filter and candidates:
        verdicts = client.run(
            [JUDGE_PROMPT.format(a=c.first, b=c.second) for c in candidates],
            temperature=0.0,
            max_tokens=8,
        )
        hardnegs = [
            candidate
            for candidate, verdict in zip(candidates, verdicts)
            if not (verdict or "").strip().upper().startswith("YES")
        ]
        actually_redundant = len(candidates) - len(hardnegs)

    provenance: Dict[str, object] = {
        "pairs": "mined",
        "mode": mode,
        "candidates": len(claims),
        "kept": len(gold),
        "dropped_unusable": unusable,
        "dropped_overlap": overlapping,
        "dropped_not_redundant": not_redundant,
        "hardneg_candidates": len(candidates),
        "hardneg_kept": len(hardnegs),
        "hardneg_dropped_redundant": actually_redundant,
        **client.describe(),
    }
    return gold, hardnegs, provenance


def load_pool(
    *,
    pairs_path: Optional[str] = None,
    num_abstracts: int = 20_000,
    num_mined_pairs: int = 30_000,
    seed: int = 42,
    mode: str = "subtle",
    model: str = "gemini-2.5-flash",
    base_url: Optional[str] = None,
    max_concurrent: int = 32,
    max_overlap: float = 0.5,
    validity_filter: bool = True,
    hardneg_order: str = "overlap",
    cache_dir: str = "data/.cache/ctc_llm",
) -> RedundancyPool:
    """
    Build the redundancy pool: gold pairs, hard negatives, and the PubMed filler supply.

    Two ways in, and the first is the one to prefer:

    * ``pairs_path`` -- a JSONL of already-mined pairs. Rows are ``{"kind": "gold", "claim": ...,
      "paraphrase": ..., "abstract_id": ...}`` and ``{"kind": "hardneg", "a": ..., "b": ...,
      "abstract_id": ...}``. No model, no network beyond the abstract download, exactly
      reproducible. Mining costs a serving run and its output is data in its own right;
      ``debug/ctc_strmatch_redundancy_port/harvest_redundancy_pairs.py`` recovers 6807 gold and
      13456 hard-negative pairs from the shipped pre-migration files, which is the corpus the
      published redundancy numbers were computed on.
    * otherwise, mine fresh via :mod:`ctc.data.llm`, writing through a response cache.

    ``max_overlap`` is applied on **both** paths, so a pairs file mined before it existed is still
    filtered on the way in rather than silently reintroducing near-duplicate gold.

    :param pairs_path: Pre-mined pairs JSONL; skips the model entirely.
    :param num_abstracts: PubMed abstracts to pool for claims, hard negatives and fillers.
    :param num_mined_pairs: Gold pairs to mine when mining. Named apart from the per-example
        ``num_pairs`` so one CLI flag namespace can carry both.
    :param seed: Abstract sample, pair order, and mode assignment.
    :param mode: ``"simple"``, ``"subtle"`` (default) or ``"both"``; see the module warning.
    :param model: Model id for mining.
    :param base_url: OpenAI-compatible endpoint(s).
    :param max_concurrent: In-flight requests.
    :param max_overlap: Word-Jaccard above which a gold pair is dropped as a near duplicate.
    :param validity_filter: LLM-judge gold and hard negatives. Only applies while mining.
    :param hardneg_order: ``"overlap"`` or ``"random"``; see :func:`build_hardneg_pool`.
    :param cache_dir: Response cache.

    :returns: The pool.

    :raises ValueError: If neither ``pairs_path`` nor ``base_url`` is given -- there is no third way
        to obtain a paraphrase.
    """
    import json
    import random

    claims, fillers = load_abstracts(num_abstracts, seed)
    rng = random.Random(seed)

    if pairs_path:
        gold: List[RedundantPair] = []
        hardnegs: List[HardNegativePair] = []
        with open(pairs_path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("kind") == "hardneg":
                    first, second = row["a"], row["b"]
                    hardnegs.append(
                        HardNegativePair(
                            first,
                            second,
                            str(row.get("abstract_id", "")),
                            round(word_jaccard(first, second), 4),
                        )
                    )
                else:
                    gold.append(
                        RedundantPair(
                            row["claim"],
                            row["paraphrase"],
                            str(row.get("abstract_id", "")),
                            row.get("mode", mode),
                        )
                    )
        provenance: Dict[str, object] = {
            "pairs": "file",
            "path": pairs_path,
            "gold_in_file": len(gold),
            "hardneg_in_file": len(hardnegs),
        }
        if not hardnegs:
            # A gold-only file is legal but changes the task: without planted non-redundant pairs
            # the corpus has no lexically-similar decoys at all.
            hardnegs = build_hardneg_pool(fillers, rng, order=hardneg_order)
            provenance["hardnegs"] = "rebuilt from the abstract pool (none in the pairs file)"
    else:
        if not base_url:
            raise ValueError(
                "mining redundancy pairs needs an OpenAI-compatible endpoint: pass base_url=..., "
                "or pass pairs_path=... to reuse pairs mined earlier"
            )
        rng.shuffle(claims)
        candidates = build_hardneg_pool(fillers, rng, order=hardneg_order)
        gold, hardnegs, provenance = mine_pairs(
            claims[:num_mined_pairs],
            candidates,
            mode=mode,
            model=model,
            base_url=base_url,
            max_concurrent=max_concurrent,
            max_overlap=max_overlap,
            validity_filter=validity_filter,
            cache_dir=cache_dir,
        )

    if max_overlap < 1.0:
        kept = [p for p in gold if word_jaccard(p.claim, p.paraphrase) <= max_overlap]
        provenance["gold_dropped_near_duplicate"] = len(gold) - len(kept)
        gold = kept
    provenance["max_overlap"] = max_overlap
    provenance["hardneg_order"] = hardneg_order

    # A stream of its own for the order, as the contradiction pool does: changing how many pairs
    # were mined must not reshuffle the ones already in an eval set.
    order_rng = random.Random(seed + 1)
    order_rng.shuffle(gold)
    order_rng.shuffle(hardnegs)
    return RedundancyPool(
        pairs=tuple(gold), hardnegs=tuple(hardnegs), fillers=fillers, provenance=provenance
    )
