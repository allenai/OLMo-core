"""
Claim/paraphrase twins over PubMed: the substrate for ``xabsence``.

One pool entry is a fact stated twice -- once as the real PubMed sentence, once as a rewrite that
shares as few words with it as possible. ``xabsence`` puts the originals in corpus A and the
rewrites in corpus B, so *almost* every claim has a twin across the boundary and the few that do
not are the answer.

.. warning::
   **The shipped pre-migration xabsence data is solvable by word overlap alone, and this is where
   the fix lives.** Ranking every document by its best word-Jaccard against the other corpus and
   naming the ``k`` lowest recovers **0.867 / 0.727 / 0.453** of the orphans in
   ``xabsence_eval_pubmed_p{8,18,48}_k3.jsonl`` against chance baselines of 0.158 / 0.077 / 0.030.
   Nothing about the task's *meaning* is needed: a paraphrase still shares a quarter of its words
   with its source, while two unrelated PubMed claims share about a tenth, so the orphan is simply
   the document with no lexically close counterpart. Two knobs close it, and both are here:

   * **``max_overlap`` at 0.22, not the pre-migration 0.3.** Measured on the staged pool, twin
     overlap has median 0.250 while the *best non-twin* an orphan can find has median 0.147 -- the
     gap the heuristic reads. Filtering at 0.22 removes the twins wide enough to be spotted.
   * **A lexical decoy per orphan** (:meth:`ParaphrasePool.decoys`): the matched pairs whose
     cross-corpus form is nearest to the orphan are forced into the same example, so the orphan
     also has a lexically close counterpart and only meaning separates them.

   Simulated against the staged pool at the 2k rung (``P=28``), the probe goes 0.602 -> 0.287 with
   decoys alone, and 0.287 -> 0.167 once ``max_overlap`` is also tightened to 0.22 (threshold
   0.251). The residual falls further with pool size -- 0.470 at 120 entries, 0.290 at 659 -- which
   is the same undersized pool BUILD_MATRIX blocker B3 is about.

.. note::
   **``mode="exact"`` is deliberately the string-matchable variant.** The pre-migration tree added
   it to close a *different* leak: when orphans were always inserted in original-claim form, a
   B-side orphan was the one non-paraphrase sitting among paraphrases and a trained 4B read it off
   that single document's style (recall 0.98 on B-side orphans vs 0.08 on A-side). Making the twin
   a byte-identical copy removes every per-document cue by construction. The ported generator fixes
   the same leak the *other* way -- an orphan is rendered in the form its own corpus uses
   (:mod:`ctc.tasks.xabsence.generate`) -- so ``mode="paraphrase"`` remains the semantic variant, and
   ``exact`` is the **suite's declared mode** (decided 2026-08-20): the task is verbatim-twin
   absence at scale -- mechanically checkable, needs no model, and its pool grows to every
   claim sentence PubMedQA supplies. Its lexical probe scores 1.0 by construction,
   which is not an acceptable finding for a suite row.

Three ways in, in order of preference:

* ``pool_path`` -- a JSONL of already-mined ``{"claim": ..., "paraphrase": ...}`` entries. No model,
  no network, exactly reproducible, and ``max_overlap`` is re-applied on read so an old pool mined
  at a looser threshold is filtered rather than trusted. The pre-migration pool lives at
  ``/scratch/users/prasann/corpus-reasoning/data/xabsence_pool_pubmed.jsonl``.
* ``mode="paraphrase"`` -- mine fresh through :mod:`ctc.data.llm`, writing through a response cache.
* ``mode="exact"`` -- no model at all; see the note above before grading anything on it.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .pubmed import clean_response, is_claim_sentence, load_abstracts, word_jaccard

__all__ = [
    "ParaphrasePair",
    "ParaphrasePool",
    "PARAPHRASE_PROMPT",
    "mine_paraphrases",
    "nearest_across_forms",
    "load_pool",
]

#: The pre-migration paraphrase prompt, verbatim. It is what produced the staged pool, so changing
#: it changes the corpus rather than just the wording of a request.
PARAPHRASE_PROMPT = (
    "Paraphrase the following biomedical claim. Restate the SAME fact with "
    "DIFFERENT wording and sentence structure — change as many words as you can "
    "while preserving the exact meaning. Do not add or remove information. "
    "Output only the paraphrase.\n\nClaim: {claim}\n\nParaphrase:"
)

_WORD = re.compile(r"[a-z0-9]+")

#: A token appearing in more than this share of entries carries no topical information and only
#: costs postings-list time, so it is skipped when gathering decoy candidates.
_DF_CAP = 0.05

#: Candidates scored per entry when finding its nearest cross-form neighbours. The postings lists
#: are visited rarest-token-first, so the cap keeps the search O(N) rather than O(N^2) on a pool
#: large enough to matter -- 30k entries is 900M pairs, which is not a build step.
_MAX_CANDIDATES = 400


def _tokens(text: str) -> Set[str]:
    """:param text: Any document text. :returns: Its lowercased alphanumeric token set."""
    return set(_WORD.findall(text.lower()))


def nearest_across_forms(
    sources: Sequence[str], destinations: Sequence[str], *, top: int
) -> Tuple[Tuple[int, ...], ...]:
    """
    For each source text, the destination entries whose text it overlaps most.

    Used to give an orphan a *lexical* counterpart in the other corpus that is not a semantic one.
    An inverted index rather than all-pairs: the pool this has to work on is tens of thousands of
    entries, where the quadratic version is not a build step but an afternoon.

    :param sources: Texts to find neighbours for, e.g. every entry's original.
    :param destinations: Texts to search, e.g. every entry's paraphrase. Same length and order as
        ``sources``, since index ``i`` in each is the same pool entry.
    :param top: Neighbours per source.

    :returns: One tuple of destination indices per source, best first, never containing the
        source's own index -- that is the twin, which is exactly what an orphan must not have.

    :raises ValueError: If the two sequences differ in length, which would silently pair each entry
        with someone else's twin.
    """
    if len(sources) != len(destinations):
        raise ValueError(
            f"sources and destinations must be parallel, got {len(sources)} and {len(destinations)}"
        )
    dst_tokens = [_tokens(text) for text in destinations]
    postings: Dict[str, List[int]] = defaultdict(list)
    for i, tokens in enumerate(dst_tokens):
        for token in tokens:
            postings[token].append(i)
    cap = max(1, int(len(destinations) * _DF_CAP))

    out: List[Tuple[int, ...]] = []
    for i, text in enumerate(sources):
        src = _tokens(text)
        # Rarest tokens first: a five-token biomedical entity narrows the field far faster than
        # "patients", and the candidate cap then bites on the useful end of the list.
        rare = sorted(
            (t for t in src if len(postings.get(t, ())) <= cap), key=lambda t: len(postings[t])
        )
        candidates: Set[int] = set()
        for token in rare:
            candidates.update(postings[token])
            if len(candidates) >= _MAX_CANDIDATES:
                break
        candidates.discard(i)
        scored = sorted(
            ((len(src & dst_tokens[j]) / max(1, len(src | dst_tokens[j])), j) for j in candidates),
            reverse=True,
        )
        out.append(tuple(j for _, j in scored[:top]))
    return tuple(out)


@dataclass(frozen=True)
class ParaphrasePair:
    """
    One fact, stated twice.

    :param original: The real PubMed sentence. Goes into corpus A.
    :param paraphrase: Its rewrite. Goes into corpus B.
    """

    original: str
    paraphrase: str


@dataclass(frozen=True)
class ParaphrasePool:
    """
    :param pairs: Twins in a fixed shuffled order.
    :param provenance: What produced them, copied into every example's ``_meta``.
    """

    pairs: Tuple[ParaphrasePair, ...]
    provenance: Dict[str, object] = field(default_factory=dict)
    #: Lazily-filled ``side -> per-entry neighbour indices``. Not part of the pool's identity: it is
    #: derived from ``pairs``, and it is per-split because a decoy has to come from the same split.
    _neighbours: Dict[str, Tuple[Tuple[int, ...], ...]] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __len__(self) -> int:
        return len(self.pairs)

    def decoys(self, side: str, index: int, count: int) -> List[int]:
        """
        Entries whose corpus-``side`` counterpart is lexically nearest to entry ``index``.

        :param side: ``"A"`` if the orphan will be rendered as an original (so its counterparts are
            the other corpus's paraphrases), ``"B"`` if it will be rendered as a paraphrase.
        :param index: The orphan's pool index.
        :param count: How many decoys are wanted.

        :returns: Pool indices, best first; possibly fewer than ``count`` on a small pool.

        :raises ValueError: On a side other than ``"A"`` or ``"B"``.
        """
        if side not in ("A", "B"):
            raise ValueError(f"side must be 'A' or 'B', got {side!r}")
        table = self._neighbours.get(side)
        if table is None:
            originals = [pair.original for pair in self.pairs]
            paraphrases = [pair.paraphrase for pair in self.pairs]
            # An A-side orphan is an original, and the documents it is compared against are the
            # other corpus's paraphrases -- so the neighbour search runs original -> paraphrase.
            src, dst = (originals, paraphrases) if side == "A" else (paraphrases, originals)
            table = nearest_across_forms(src, dst, top=8)
            self._neighbours[side] = table
        return list(table[index][:count])

    def for_split(self, split: str) -> "ParaphrasePool":
        """
        Slice off this split's twins.

        Splitting by *entry* rather than by example is what makes the split real here: an example is
        a random subset of the pool, so two examples drawn from one pool share most of their
        documents and a per-example split would put an eval example's own claims into training.

        :param split: ``"train"`` or ``"eval"``.

        :returns: A pool holding only that split's entries, with an empty neighbour cache -- the
            indices are different and a decoy must come from the split it will be placed in.

        :raises ValueError: On an unknown split name.
        """
        cut = max(1, len(self.pairs) // 10)
        if split == "eval":
            kept = self.pairs[:cut]
        elif split == "train":
            kept = self.pairs[cut:]
        else:
            raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
        return ParaphrasePool(pairs=kept, provenance=self.provenance)


def mine_paraphrases(
    claims: Sequence[str],
    *,
    model: str,
    base_url: str,
    max_concurrent: int = 25,
    max_overlap: float = 0.22,
    cache_dir: Optional[str] = None,
) -> Tuple[List[ParaphrasePair], Dict[str, object]]:
    """
    Rewrite every claim, then throw away the rewrites that are still near-copies.

    :param claims: Real PubMed claim sentences.
    :param model: Model id.
    :param base_url: OpenAI-compatible endpoint(s), comma-separated for a local fleet.
    :param max_concurrent: In-flight requests.
    :param max_overlap: Reject a rewrite whose word-Jaccard with the original exceeds this. The one
        knob that decides whether the task is semantic or lexical; see the module warning.
    :param cache_dir: Response cache directory.

    :returns: ``(pairs, provenance)``. ``provenance`` carries every rejection count, because "how
        many survived" is the number that says whether a pool is worth building on.
    """
    from ..llm import ChatClient

    client = ChatClient(base_url, model=model, max_concurrent=max_concurrent, cache_dir=cache_dir)
    responses = client.run(
        [PARAPHRASE_PROMPT.format(claim=claim) for claim in claims],
        temperature=0.7,
        max_tokens=200,
    )

    pairs: List[ParaphrasePair] = []
    unusable = overlapping = 0
    for claim, response in zip(claims, responses):
        rewritten = clean_response(response)
        if not rewritten or rewritten.strip() == claim.strip():
            unusable += 1
        elif word_jaccard(claim, rewritten) > max_overlap:
            overlapping += 1
        else:
            pairs.append(ParaphrasePair(claim, rewritten))
    provenance: Dict[str, object] = {
        "pool": "mined",
        "candidates": len(claims),
        "kept": len(pairs),
        "dropped_unusable": unusable,
        "dropped_overlap": overlapping,
        **client.describe(),
    }
    return pairs, provenance


def load_pool(
    *,
    pool_path: Optional[str] = None,
    mode: str = "paraphrase",
    num_abstracts: int = 8_000,
    num_claims: int = 30_000,
    seed: int = 42,
    model: str = "Qwen2.5-14B-Instruct",
    base_url: Optional[str] = None,
    max_concurrent: int = 25,
    max_overlap: float = 0.22,
    cache_dir: str = "data/.cache/ctc_llm",
) -> ParaphrasePool:
    """
    Build the ``xabsence`` twin pool.

    :param pool_path: A JSONL of already-mined ``{"claim", "paraphrase"}`` entries. Skips the model
        entirely and is the reproducible path; ``max_overlap`` is still applied, so a pool mined at
        a looser threshold is filtered on read rather than taken on trust.
    :param mode: ``"paraphrase"`` (mine rewrites, needs ``base_url``) or ``"exact"`` (the twin is a
        byte-identical copy -- no model, and see the module note before grading anything on it).
    :param num_abstracts: PubMed abstracts to pool claim sentences from.
    :param num_claims: Claim sentences to take. One example needs ``num_pairs + num_unmatched``
        distinct entries, so a pool much smaller than that times the example count means every
        example is mostly the same documents.
    :param seed: Abstract sample, claim order and final pool order.
    :param model: Model id for mining.
    :param base_url: OpenAI-compatible endpoint(s) for mining.
    :param max_concurrent: In-flight requests.
    :param max_overlap: Near-copy rejection threshold; the module warning is why it is 0.22.
    :param cache_dir: Response cache.

    :returns: The pool.

    :raises ValueError: On an unknown ``mode``, or when mining was asked for with no endpoint.
    """
    import json

    if pool_path:
        with open(pool_path, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        raw = [ParaphrasePair(r["claim"], r["paraphrase"]) for r in rows]
        pairs = [p for p in raw if word_jaccard(p.original, p.paraphrase) <= max_overlap]
        provenance: Dict[str, object] = {
            "pool": "file",
            "path": pool_path,
            "entries": len(raw),
            "dropped_overlap": len(raw) - len(pairs),
        }
    else:
        claims, _ = load_abstracts(num_abstracts, seed)
        # `load_abstracts` returns every sentence; only the self-contained claim-bearing ones can
        # be paraphrased into a twin a reader could match back, so filter with the same predicate
        # contradiction uses rather than a second, divergent one.
        texts: List[str] = []
        seen = set()
        for sentence, _abstract_id in claims:
            if sentence not in seen and is_claim_sentence(sentence):
                seen.add(sentence)
                texts.append(sentence)
        random.Random(seed).shuffle(texts)
        texts = texts[:num_claims]

        if mode == "exact":
            pairs = [ParaphrasePair(text, text) for text in texts]
            provenance = {"pool": "exact-copy", "entries": len(pairs)}
        elif mode == "paraphrase":
            if not base_url:
                raise ValueError(
                    "mining xabsence paraphrases needs an OpenAI-compatible endpoint: pass "
                    "base_url=..., or pass pool_path=... to reuse a pool mined earlier"
                )
            pairs, provenance = mine_paraphrases(
                texts,
                model=model,
                base_url=base_url,
                max_concurrent=max_concurrent,
                max_overlap=max_overlap,
                cache_dir=cache_dir,
            )
        else:
            raise ValueError(f"mode must be 'paraphrase' or 'exact', got {mode!r}")

    # A duplicate original or a duplicate rewrite makes gold ambiguous: an "orphan" whose text also
    # appears as some other entry's twin does have a match in the other corpus, so the label is
    # wrong. Deduplicate on both sides before the pool is ever sampled from.
    unique: List[ParaphrasePair] = []
    seen_text: Set[str] = set()
    for pair in pairs:
        if pair.original in seen_text or pair.paraphrase in seen_text:
            continue
        seen_text.add(pair.original)
        seen_text.add(pair.paraphrase)
        unique.append(pair)
    provenance = {**provenance, "unique_entries": len(unique), "max_overlap": max_overlap}

    order = list(range(len(unique)))
    random.Random(seed + 1).shuffle(order)
    return ParaphrasePool(pairs=tuple(unique[i] for i in order), provenance=provenance)
