"""
PubMed claim sentences, and the LLM perturbation that turns one into a contradicting pair.

Gold is ``(S, S')`` where ``S`` is a real PubMed sentence and ``S'`` is a minimally-perturbed
version that cannot be true alongside it. **Only one sentence per gold pair is model-written**;
every other document in the corpus is real PubMed text, which is what stops the pair being findable
by style rather than by meaning.

.. warning::
   **Fillers must come from the same corpus as gold.** A domain-agnostic glob once pulled FEVER and
   Wikipedia claims into PubMed contradiction evals -- 92% of fillers at 2k rising to 99.6% at 32k,
   against gold that was entirely PubMed. "Find the contradicting pair" then collapses to "find the
   biomedical sentences". The effect was to *depress* scores rather than flatter them (it is a
   train/eval domain shift, not a shortcut), and worst at the long end: the 32k rung went 0.335 ->
   0.559 once rebuilt. This pool is single-corpus by construction, which is the structural fix.

.. warning::
   **Do not scale contradiction by recombining gold pairs against a global filler pool.** Pooling
   gold pairs against globally-sampled fillers instead of each example's co-sampled distractors
   dropped full-attention train F1 from 0.934 to 0.585, because the nearest non-gold pair's
   similarity crashed 0.372 -> 0.163 and "pick the two most similar claims" solves the task.
   Contamination checks, pair-reuse checks and gold-pair-distance matching all passed; none asked
   whether it was still the same task. Scale by drawing more fillers per example, which is what
   this pool does.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "ClaimPair",
    "PubMedPool",
    "is_claim_sentence",
    "split_sentences",
    "load_pool",
    "mine_pairs",
    "clean_response",
    "word_jaccard",
    "sample_fillers",
    "CONTRADICTION_TYPES",
]

#: A sentence must assert something measurable to be worth contradicting.
CLAIM_TOKEN = re.compile(
    r"\b(increased?|decreased?|reduced?|elevated|higher|lower|greater|"
    r"fewer|more|less|associated|correlated|induced?|inhibited?|"
    r"prevented?|caused?|significant(ly)?|improved?|worsened?|"
    r"\d+(?:\.\d+)?%?|\d+-fold)\b",
    re.IGNORECASE,
)

#: Anaphora and discourse markers mean the sentence is not self-contained: "It reduced mortality"
#: cannot be contradicted without the antecedent, so the pair would be unjudgeable.
LEADING_BAD = re.compile(
    r"^(It|This|These|Those|They|He|She|We|Our|Their|Such|However|"
    r"Moreover|Furthermore|Also|Additionally|Thus|Therefore|Hence|"
    r"Consequently|Although|Though|Instead|Meanwhile|Finally|First|"
    r"Second|Third|In\s+contrast|In\s+addition|In\s+summary|"
    r"In\s+conclusion)\b",
    re.IGNORECASE,
)

#: Figure/table/citation references point outside the sentence.
DANGLING = re.compile(r"\(Fig\.?|\(Table|\[\d+\]|\bet\s+al\.|supplement", re.IGNORECASE)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_sentences(text: str) -> List[str]:
    """
    :param text: An abstract chunk.

    :returns: Its sentences. Deliberately the strict split (a capital must follow), because a
        looser one turns "et al." and "vs." into sentence boundaries and shreds the claims.
    """
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s and s.strip()]


def is_claim_sentence(sentence: str) -> bool:
    """
    Whether a sentence is self-contained and claim-bearing enough to perturb.

    :param sentence: Candidate sentence.

    :returns: True when it can carry a contradiction on its own.
    """
    sentence = sentence.strip()
    if not sentence.endswith(".") or "?" in sentence:
        return False
    if not 8 <= len(sentence.split()) <= 40:
        return False
    if LEADING_BAD.match(sentence) or DANGLING.search(sentence):
        return False
    if not CLAIM_TOKEN.search(sentence):
        return False
    return bool(sentence[0].isupper())


@dataclass(frozen=True)
class ClaimPair:
    """
    One gold contradiction.

    :param claim: The real PubMed sentence.
    :param contradiction: The model-written sentence that conflicts with it.
    :param abstract_id: Which abstract ``claim`` came from. Load-bearing: fillers are drawn only
        from *other* abstracts, so no filler can restate the fact the contradiction denies.
    :param mode: Which perturbation style produced it, for the provenance record.
    """

    claim: str
    contradiction: str
    abstract_id: str
    mode: str = "realistic"


@dataclass(frozen=True)
class PubMedPool:
    """
    :param pairs: Gold pairs, in a fixed shuffled order. Example *i* uses pairs *i*..*i+K*.
    :param fillers: ``abstract_id -> sentences``, the distractor supply.
    :param provenance: What produced the pairs, copied into every example's ``_meta``.
    """

    pairs: Tuple[ClaimPair, ...]
    fillers: Dict[str, Tuple[str, ...]]
    provenance: Dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.pairs)

    def for_split(self, split: str) -> "PubMedPool":
        """
        Slice off this split's gold pairs; fillers are shared.

        :param split: ``"train"`` or ``"eval"``.

        :returns: A pool holding only that split's pairs.

        :raises ValueError: On an unknown split name.
        """
        cut = max(1, len(self.pairs) // 10)
        if split == "eval":
            kept = self.pairs[:cut]
        elif split == "train":
            kept = self.pairs[cut:]
        else:
            raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
        return PubMedPool(pairs=kept, fillers=self.fillers, provenance=self.provenance)


def load_abstracts(num_abstracts: int, seed: int) -> Tuple[List[Tuple[str, str]], Dict]:
    """
    Read PubMedQA and split it into claim candidates and a filler supply.

    :param num_abstracts: Abstracts to sample.
    :param seed: Which abstracts.

    :returns: ``(claims, fillers)`` where ``claims`` is ``(sentence, abstract_id)`` and ``fillers``
        is ``abstract_id -> sentences`` (every sentence, claim-bearing or not).

    :raises ImportError: If ``datasets`` is missing; install ``ctc[sources]``.
    """
    import random

    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - depends on the install
        raise ImportError("PubMed loading needs `datasets`: pip install './ctc[sources]'") from e

    dataset = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
    order = list(range(len(dataset)))
    random.Random(seed).shuffle(order)

    claims: List[Tuple[str, str]] = []
    fillers: Dict[str, Tuple[str, ...]] = {}
    for i in order[:num_abstracts]:
        sentences: List[str] = []
        for chunk in dataset[i]["context"]["contexts"]:
            if chunk:
                sentences.extend(split_sentences(chunk))
        if not sentences:
            continue
        abstract_id = str(i)
        fillers[abstract_id] = tuple(sentences)
        claims.extend((s, abstract_id) for s in sentences if is_claim_sentence(s))
    return claims, fillers


def load_pool(
    *,
    pairs_path: Optional[str] = None,
    num_abstracts: int = 20_000,
    num_mined_pairs: int = 30_000,
    seed: int = 42,
    mode: str = "realistic",
    model: str = "gemini-2.5-flash",
    base_url: Optional[str] = None,
    max_concurrent: int = 25,
    max_overlap: float = 0.5,
    validity_filter: bool = True,
    cache_dir: str = "data/.cache/ctc_llm",
) -> PubMedPool:
    """
    Build the contradiction pool: claim pairs plus the PubMed filler supply.

    Two ways in, and the first is the one to prefer:

    * ``pairs_path`` -- a JSONL of already-mined pairs (``claim``, ``contradiction``,
      ``abstract_id``). No model, no network beyond the abstract download, exactly reproducible.
      Mining is expensive and its output is data in its own right, so it is worth keeping.
    * otherwise, mine fresh via :mod:`ctc.data.llm`, writing through a response cache.

    :param pairs_path: Pre-mined pairs JSONL; skips the model entirely.
    :param num_abstracts: PubMed abstracts to pool for claims and fillers.
    :param num_mined_pairs: Gold pairs to mine when mining. Named apart from the per-example
        ``num_pairs`` on purpose: one CLI flag namespace covers both, and two parameters
        sharing a name means one of them can never be set.
    :param seed: Abstract sample, pair order, and perturbation-type assignment.
    :param mode: ``"realistic"`` (recommended), ``"simple"``, ``"subtle"`` or ``"both"``. See
        :mod:`ctc.tasks.contradiction.sources.pubmed` for what each does and why ``realistic`` is
        the default.
    :param model: Model id for mining.
    :param base_url: OpenAI-compatible endpoint(s); a local vLLM fleet may be comma-separated.
    :param max_concurrent: In-flight requests.
    :param max_overlap: Reject a mined pair whose word-Jaccard exceeds this -- the near-duplicate
        backstop that stops "the original with one word changed" being solvable by string match.
    :param validity_filter: LLM-judge each pair and drop the ones that are paraphrases rather than
        contradictions. On by default; roughly a quarter of raw outputs fail it.
    :param cache_dir: Response cache.

    :returns: The pool.
    """
    import json
    import random

    claims, fillers = load_abstracts(num_abstracts, seed)

    if pairs_path:
        with open(pairs_path, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        pairs = [
            ClaimPair(
                r["claim"], r["contradiction"], str(r.get("abstract_id", "")), r.get("mode", mode)
            )
            for r in rows
        ]
        provenance: Dict[str, object] = {"pairs": "file", "path": pairs_path}
    else:
        if not base_url:
            raise ValueError(
                "mining contradiction pairs needs an OpenAI-compatible endpoint: pass "
                "base_url=..., or pass pairs_path=... to reuse pairs mined earlier"
            )
        rng = random.Random(seed)
        rng.shuffle(claims)
        pairs, provenance = mine_pairs(
            claims[:num_mined_pairs],
            mode=mode,
            seed=seed,
            model=model,
            base_url=base_url,
            max_concurrent=max_concurrent,
            max_overlap=max_overlap,
            validity_filter=validity_filter,
            cache_dir=cache_dir,
        )

    order = list(range(len(pairs)))
    random.Random(seed + 1).shuffle(order)
    return PubMedPool(pairs=tuple(pairs[i] for i in order), fillers=fillers, provenance=provenance)


def word_jaccard(a: str, b: str) -> float:
    """
    Token-set overlap, used to reject near-duplicate gold pairs.

    :param a: One sentence.
    :param b: The other.

    :returns: Jaccard overlap of lowercased alphanumeric tokens. **1.0 on an empty input**, so an
        empty generation is rejected by an overlap filter rather than sailing through it -- the
        pre-migration tree had two copies of this function that disagreed on exactly that, one
        returning 1.0 and the other 0.0, inside an overlap-*reject* filter.
    """
    wa = set(re.findall(r"[a-z0-9]+", (a or "").lower()))
    wb = set(re.findall(r"[a-z0-9]+", (b or "").lower()))
    if not wa or not wb:
        return 1.0
    return len(wa & wb) / len(wa | wb)


def sample_fillers(
    fillers: Dict[str, Sequence[str]],
    exclude_abstracts: Sequence[str],
    exclude_texts: Sequence[str],
    count: int,
    rng,
) -> List[str]:
    """
    Draw filler sentences from abstracts that contributed no gold to this example.

    :param fillers: ``abstract_id -> sentences``.
    :param exclude_abstracts: Abstracts this example took gold from. PubMed's scale means two
        unrelated abstracts almost never state the same fact, but the *same* abstract very well
        might restate the one the contradiction denies -- which would be an unlabelled second gold
        pair, i.e. a false negative.
    :param exclude_texts: Texts already placed.
    :param count: How many are needed.
    :param rng: Seeded RNG.

    :returns: Up to ``count`` sentences.
    """
    banned = set(exclude_abstracts)
    seen = set(exclude_texts)
    ids = [key for key in fillers if key not in banned]
    rng.shuffle(ids)
    out: List[str] = []
    for abstract_id in ids:
        if len(out) >= count:
            break
        for sentence in fillers[abstract_id]:
            if len(out) >= count:
                break
            if sentence in seen:
                continue
            out.append(sentence)
            seen.add(sentence)
    return out


# ── pair mining: the one step in this package that needs a model ────────────────────────────────

SIMPLE_PROMPT = """Flip the polarity or direction of the following biomedical sentence to produce a direct, obvious contradiction. Preserve as much wording as possible — change only the minimal polarity-bearing words (verbs, adjectives, or insert a negation).

Return only the flipped sentence, with no preamble, quotes, or explanation.

Examples:
Original: Aspirin reduced the risk of myocardial infarction by 23%.
Flipped: Aspirin increased the risk of myocardial infarction by 23%.

Original: The novel compound was effective against resistant bacterial strains.
Flipped: The novel compound was ineffective against resistant bacterial strains.

Original: Vitamin D supplementation was associated with lower all-cause mortality.
Flipped: Vitamin D supplementation was associated with higher all-cause mortality.

Original: {sentence}
Flipped:"""

#: v3, tuned on a 25-sentence dev set. Pulls mean word overlap of (S, S') from ~0.28 to ~0.13 --
#: the overlap of a *non*-contradicting same-abstract pair -- so a word-overlap heuristic can no
#: longer separate positives from negatives. About 76% of outputs are genuine contradictions; the
#: validity filter drops the rest.
SUBTLE_PROMPT = """Your task: write ONE biomedical sentence that genuinely CONTRADICTS the sentence below, while sharing almost no words with it.

Step 1 (think, do not output): identify the single concrete factual element in the original you will contradict — a number, magnitude, duration, scope/population, or comparator. Your new sentence must describe the SAME underlying finding, so that the two sentences cannot both be true.

Step 2 (output): write the contradicting sentence so that:
- It clearly conflicts with the original on that one element — a domain expert reading both would say "these disagree." It must NOT be a mere paraphrase.
- It shares as few words as possible with the original: rename every entity (synonyms, alternative names, expand/contract abbreviations), use different verbs, a different sentence opening, and a different structure.
- It stays a plausible biomedical sentence.
- No polarity flips ("increased"<->"decreased"), no inserted "not".

Return only the contradicting sentence, no preamble or quotes.

Original: {sentence}
Contradicting:"""

#: The ``realistic`` menu. A polarity flip (simple) or a numeric tweak (subtle) leaves a
#: near-duplicate tell that makes the pair findable by string matching rather than by reasoning;
#: drawing a contradiction TYPE per pair and demanding a full rephrase removes it.
CONTRADICTION_TYPES = (
    (
        "direction",
        "the original reports an effect in one direction (e.g. increase, benefit, protection); yours "
        "reports the OPPOSITE direction for the same relationship, phrased as a different study "
        "would — do NOT just insert 'not' or swap a single antonym",
    ),
    (
        "magnitude",
        "yours reports a substantially different effect size for the same relationship (e.g. a "
        "large/clinically-meaningful effect vs a negligible one), with a different number where "
        "natural",
    ),
    (
        "number",
        "yours states a different concrete count, sample size, dose, threshold, or measured value "
        "that is logically incompatible with the original",
    ),
    (
        "scope",
        "yours changes the population, subgroup, or setting so the claims become incompatible (e.g. "
        "the original's finding is contradicted within a group it explicitly covered)",
    ),
    (
        "temporal",
        "yours reports a conflicting timing, duration, or follow-up window (e.g. an effect the "
        "original places at one timepoint is reported absent or reversed at that timepoint)",
    ),
    (
        "significance",
        "yours flips the statistical conclusion (significant<->not significant, "
        "independent<->dependent, correlated<->uncorrelated) using different numbers, p-values, and "
        "wording",
    ),
    (
        "comparator",
        "yours reverses which group, treatment, or condition is higher/better relative to the "
        "comparator in the original",
    ),
)

REALISTIC_PROMPT = """You are building a benchmark for detecting contradictions in biomedical literature. Given a real sentence from a paper, write ONE sentence that a DIFFERENT study might report which genuinely CONTRADICTS the original — both cannot be true of the same underlying finding.

Make the contradiction of this kind: {type_desc}.

Requirements:
- A domain expert reading both sentences would say "these disagree." It must NOT be a paraphrase (same meaning) or an unrelated fact.
- Write it as natural prose from a different paper: rename entities (synonyms, alternative names, expand or contract abbreviations), use different verbs, a different opening, and a different structure. Share as FEW words as possible with the original — it must not look like the original with one word changed.
- Keep it a specific, plausible biomedical claim (retain concrete numbers/units where natural).

Return only the contradicting sentence — no preamble, quotes, or explanation.

Original: {sentence}
Contradicting sentence:"""

JUDGE_PROMPT = """Sentence A: {a}
Sentence B: {b}

Do these two sentences make factually INCOMPATIBLE claims about the same finding — i.e. they genuinely contradict, both cannot be true? A mere paraphrase (same meaning) or two unrelated facts are NOT a contradiction.

Answer with exactly one word: YES or NO."""


def clean_response(text: Optional[str]) -> Optional[str]:
    """
    Normalise one model output into a usable sentence.

    :param text: Raw generation.

    :returns: The sentence, or ``None`` when it is unusable -- empty, under four words, or nothing
        but a wrapper the model added despite instructions.
    """
    if not text:
        return None
    text = text.strip()
    if text.startswith(('"', "'")) and text.endswith(('"', "'")):
        text = text[1:-1].strip()
    text = re.sub(r"^(Flipped|Modified|Contradicting):\s*", "", text, flags=re.IGNORECASE)
    text = text.splitlines()[0].strip() if text else text
    if not text or len(text.split()) < 4:
        return None
    return text


def _prompt_for(sentence: str, mode: str, rng) -> Tuple[str, str]:
    if mode == "simple":
        return "simple", SIMPLE_PROMPT.format(sentence=sentence)
    if mode == "subtle":
        return "subtle", SUBTLE_PROMPT.format(sentence=sentence)
    name, description = CONTRADICTION_TYPES[rng.randrange(len(CONTRADICTION_TYPES))]
    return name, REALISTIC_PROMPT.format(type_desc=description, sentence=sentence)


def mine_pairs(
    claims: Sequence[Tuple[str, str]],
    *,
    mode: str = "realistic",
    seed: int = 42,
    model: str = "gemini-2.5-flash",
    base_url: Optional[str] = None,
    max_concurrent: int = 25,
    max_overlap: float = 0.5,
    validity_filter: bool = True,
    cache_dir: Optional[str] = None,
) -> Tuple[List[ClaimPair], Dict[str, object]]:
    """
    Write one contradicting sentence per claim, then throw away the ones that are not.

    Three filters, and each closes a way the data looked right and was not:

    * an unusable or echoed generation (:func:`clean_response`);
    * word-Jaccard above ``max_overlap`` -- "the original with one word changed" is solvable by
      string match, not by reasoning;
    * an LLM judge, which drops paraphrases and shifts to a non-conflicting timepoint or subgroup
      where both sentences could be true. Roughly a quarter of raw ``realistic`` outputs fail it.

    :param claims: ``(sentence, abstract_id)`` candidates.
    :param mode: ``"realistic"``, ``"simple"``, ``"subtle"`` or ``"both"`` (half simple, half
        subtle).
    :param seed: Contradiction-type assignment, so a rerun asks for the same types.
    :param model: Model id.
    :param base_url: OpenAI-compatible endpoint(s), comma-separated for a local fleet.
    :param max_concurrent: In-flight requests.
    :param max_overlap: Overlap rejection threshold.
    :param validity_filter: Run the judge.
    :param cache_dir: Response cache directory.

    :returns: ``(pairs, provenance)``. ``provenance`` records the model and every rejection count,
        because "how many pairs survived" is the number that says whether a build is trustworthy.
    """
    import random

    from ..llm import ChatClient

    rng = random.Random(seed)
    modes = []
    for i in range(len(claims)):
        modes.append(("simple" if i % 2 == 0 else "subtle") if mode == "both" else mode)

    types, prompts = [], []
    for (sentence, _), item_mode in zip(claims, modes):
        name, prompt = _prompt_for(sentence, item_mode, rng)
        types.append(name)
        prompts.append(prompt)

    client = ChatClient(
        base_url or "", model=model, max_concurrent=max_concurrent, cache_dir=cache_dir
    )
    responses = client.run(prompts, temperature=0.7, max_tokens=200)

    candidates: List[Optional[str]] = []
    failed = overlapping = 0
    for (sentence, _), response in zip(claims, responses):
        cleaned = clean_response(response)
        if not cleaned or cleaned.strip() == sentence.strip():
            candidates.append(None)
            failed += 1
        elif max_overlap < 1.0 and word_jaccard(sentence, cleaned) > max_overlap:
            candidates.append(None)
            overlapping += 1
        else:
            candidates.append(cleaned)

    rejected = 0
    if validity_filter:
        live = [i for i, c in enumerate(candidates) if c is not None]
        verdicts = client.run(
            [JUDGE_PROMPT.format(a=claims[i][0], b=candidates[i]) for i in live],
            temperature=0.0,
            max_tokens=10,
        )
        for i, verdict in zip(live, verdicts):
            if not (verdict or "").strip().upper().startswith("YES"):
                candidates[i] = None
                rejected += 1

    pairs = [
        ClaimPair(sentence, contradiction, abstract_id, kind)
        for (sentence, abstract_id), contradiction, kind in zip(claims, candidates, types)
        if contradiction is not None
    ]
    provenance: Dict[str, object] = {
        "pairs": "mined",
        "mode": mode,
        "candidates": len(claims),
        "kept": len(pairs),
        "dropped_unusable": failed,
        "dropped_overlap": overlapping,
        "dropped_not_a_contradiction": rejected,
        **client.describe(),
    }
    return pairs, provenance
