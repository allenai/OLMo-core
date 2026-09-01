"""Generate PubMed synthetic-perturbation contradiction data.

Each example is a corpus of N sentences drawn from PubMed abstracts, with K
gold contradiction pairs hidden among fillers. Each gold pair is:

    (S, S')  where S is a REAL PubMed sentence and S' is a minimally-perturbed
             version of S produced by an LLM that factually contradicts S.

Only one sentence per gold pair is LLM-generated. Everything else in the
corpus is real PubMed text. This avoids the common pitfall where long
LLM-written passages are identifiable by surface-level stylistic cues.

Two perturbation modes (set with `--mode`):

  simple — direct polarity/direction flip with high word overlap.
           "Aspirin reduced MI risk by 23%" -> "Aspirin increased MI risk by 23%"

  subtle — change a single numeric/scope element so the new sentence is
           factually incompatible with the original but requires close
           reading to spot.
           "30% reduction in mortality" -> "3% reduction in mortality"

Both can be generated in one run with `--mode both` (half of gold pairs each).

Candidate S is filtered with heuristics to keep only self-contained,
claim-bearing sentences (length, presence of a claim-carrying token, no
leading anaphora/discourse markers, no dangling citations).

False-negative control: filler sentences are sampled from PubMed abstracts
DIFFERENT from any gold-source abstract in the same example. PubMed's scale
(>20M abstracts) means unrelated abstracts almost never state the same fact
that S' denies.

`gold_doc_indices` stores contradiction pairs as 1-indexed sentence-ID
pairs `[[a, b], ...]` — the same format consumed by
`scripts/lib/data_format.py:_build_output(task="contradiction")`.

Usage:
    python scripts/data/generate_pubmed_contradiction_data.py \\
        --num-docs 100 --num-contradictions 3 --mode both \\
        --num-train 2000 --num-eval 300
"""

import argparse
import random
import re
from collections import defaultdict

from datasets import load_dataset
from tqdm import tqdm

from corpus_reasoning.lib.io import load_jsonl, print_dataset_stats, save_jsonl

# Deferred: scripts.lib.llm_request_client pulls in openai, which the train
# conda env lacks. Expansion mode doesn't need the LLM, so import lazily in
# generate_perturbations() instead.


# ─────────────────────────── sentence filtering ────────────────────────────

CLAIM_TOKEN = re.compile(
    r"\b(increased?|decreased?|reduced?|elevated|higher|lower|greater|"
    r"fewer|more|less|associated|correlated|induced?|inhibited?|"
    r"prevented?|caused?|significant(ly)?|improved?|worsened?|"
    r"\d+(?:\.\d+)?%?|\d+-fold)\b",
    re.IGNORECASE,
)

LEADING_BAD = re.compile(
    r"^(It|This|These|Those|They|He|She|We|Our|Their|Such|However|"
    r"Moreover|Furthermore|Also|Additionally|Thus|Therefore|Hence|"
    r"Consequently|Although|Though|Instead|Meanwhile|Finally|First|"
    r"Second|Third|In\s+contrast|In\s+addition|In\s+summary|"
    r"In\s+conclusion)\b",
    re.IGNORECASE,
)

DANGLING = re.compile(r"\(Fig\.?|\(Table|\[\d+\]|\bet\s+al\.|supplement", re.IGNORECASE)


def is_claim_sentence(s):
    """Keep only self-contained, claim-bearing sentences."""
    s = s.strip()
    if not s.endswith("."):
        return False
    if "?" in s:
        return False
    words = s.split()
    if not (8 <= len(words) <= 40):
        return False
    if LEADING_BAD.match(s):
        return False
    if DANGLING.search(s):
        return False
    if not CLAIM_TOKEN.search(s):
        return False
    # Exclude fragments starting with lowercase or non-alpha
    if not s[0].isupper():
        return False
    return True


# ────────────────────────── PubMed sentence pool ───────────────────────────


def load_pubmed_pool(num_abstracts, seed):
    """Load PubMed abstracts from PubMedQA; return (claim_pool, filler_pool).

    claim_pool: list of (sentence, abstract_id) — filtered self-contained,
                claim-bearing sentences (used as perturbation sources).
    filler_pool: dict abstract_id -> list of any sentences (any claim or
                non-claim) from that abstract. Used for random fillers, with
                abstract_id tracked so we can exclude gold-source abstracts.
    """
    print(f"Loading PubMedQA (pqa_artificial, {num_abstracts} abstracts)...")
    ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[:num_abstracts]

    claim_pool = []
    filler_pool = {}
    for i in tqdm(indices, desc="Filtering sentences"):
        ex = ds[i]
        chunks = ex["context"]["contexts"]
        aid = str(i)
        sents = []
        for chunk in chunks:
            if not chunk:
                continue
            sents.extend(re.split(r"(?<=[.!?])\s+(?=[A-Z])", chunk.strip()))
        valid = [s.strip() for s in sents if s and s.strip()]
        if not valid:
            continue
        filler_pool[aid] = valid
        for s in valid:
            if is_claim_sentence(s):
                claim_pool.append((s, aid))

    print(f"  Abstracts: {len(filler_pool)}, " f"claim-bearing sentences: {len(claim_pool)}")
    return claim_pool, filler_pool


# ───────────────────────────── LLM perturbation ────────────────────────────

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


# v3 prompt — tuned on a 25-sentence dev set (scripts/debug/dev_subtle_overlap.py).
# Pulls mean word overlap of (S, S') from ~0.28 down to ~0.13, matching the
# overlap of a non-contradicting same-abstract sentence pair, so a word-overlap
# heuristic can no longer separate positives from negatives. ~76% of outputs
# are genuine contradictions; the validity filter (--validity-filter, on by
# default) drops the rest.
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


# v4 "realistic" mode — the most realistic/challenging generator. Instead of a
# naive polarity flip (simple) or a numeric-only tweak (subtle), it draws a
# contradiction TYPE per gold pair from the menu below and asks for a fully
# rephrased contradiction of that type. The rephrasing requirement + the
# explicit "do NOT insert 'not'/swap one word" guard kill the near-duplicate
# tell that made simple-mode pairs solvable by string matching rather than
# reasoning. Pairs are still validity-judged AND overlap-filtered (--max-overlap).
CONTRADICTION_TYPES = [
    (
        "direction",
        "the original reports an effect in one direction (e.g. increase, benefit, "
        "protection); yours reports the OPPOSITE direction for the same "
        "relationship, phrased as a different study would — do NOT just insert "
        "'not' or swap a single antonym",
    ),
    (
        "magnitude",
        "yours reports a substantially different effect size for the same "
        "relationship (e.g. a large/clinically-meaningful effect vs a negligible "
        "one), with a different number where natural",
    ),
    (
        "number",
        "yours states a different concrete count, sample size, dose, threshold, or "
        "measured value that is logically incompatible with the original",
    ),
    (
        "scope",
        "yours changes the population, subgroup, or setting so the claims become "
        "incompatible (e.g. the original's finding is contradicted within a group "
        "it explicitly covered)",
    ),
    (
        "temporal",
        "yours reports a conflicting timing, duration, or follow-up window (e.g. an "
        "effect the original places at one timepoint is reported absent or reversed "
        "at that timepoint)",
    ),
    (
        "significance",
        "yours flips the statistical conclusion (significant<->not significant, "
        "independent<->dependent, correlated<->uncorrelated) using different "
        "numbers, p-values, and wording",
    ),
    (
        "comparator",
        "yours reverses which group, treatment, or condition is higher/better "
        "relative to the comparator in the original",
    ),
]

REALISTIC_PROMPT = """You are building a benchmark for detecting contradictions in biomedical literature. Given a real sentence from a paper, write ONE sentence that a DIFFERENT study might report which genuinely CONTRADICTS the original — both cannot be true of the same underlying finding.

Make the contradiction of this kind: {type_desc}.

Requirements:
- A domain expert reading both sentences would say "these disagree." It must NOT be a paraphrase (same meaning) or an unrelated fact.
- Write it as natural prose from a different paper: rename entities (synonyms, alternative names, expand or contract abbreviations), use different verbs, a different opening, and a different structure. Share as FEW words as possible with the original — it must not look like the original with one word changed.
- Keep it a specific, plausible biomedical claim (retain concrete numbers/units where natural).

Return only the contradicting sentence — no preamble, quotes, or explanation.

Original: {sentence}
Contradicting sentence:"""


# Judge prompt for the validity filter — drops generated pairs that aren't a
# genuine contradiction (paraphrases, or a shift to a non-conflicting
# timepoint/subgroup where both sentences could be true).
JUDGE_PROMPT = """Sentence A: {a}
Sentence B: {b}

Do these two sentences make factually INCOMPATIBLE claims about the same finding — i.e. they genuinely contradict, both cannot be true? A mere paraphrase (same meaning) or two unrelated facts are NOT a contradiction.

Answer with exactly one word: YES or NO."""


def word_jaccard(a, b):
    """Token-set Jaccard overlap of two sentences (lowercased alnum tokens).

    Used to reject near-duplicate gold pairs (the verbatim-one-word-flip tell):
    a realistic contradiction from a different paper shares few tokens.
    """
    wa = set(re.findall(r"[a-z0-9]+", (a or "").lower()))
    wb = set(re.findall(r"[a-z0-9]+", (b or "").lower()))
    if not wa or not wb:
        return 1.0
    return len(wa & wb) / len(wa | wb)


def clean_response(text):
    if not text:
        return None
    text = text.strip()
    # Strip surrounding quotes the model may add despite instructions
    if text.startswith(('"', "'")) and text.endswith(('"', "'")):
        text = text[1:-1].strip()
    # Some models echo "Flipped:" / "Modified:" — strip that too
    text = re.sub(r"^(Flipped|Modified|Contradicting):\s*", "", text, flags=re.IGNORECASE)
    # Keep only the first line if the model generated multiple
    text = text.splitlines()[0].strip() if text else text
    if not text or len(text.split()) < 4:
        return None
    return text


def generate_perturbations(
    sentences, modes, model, max_concurrent, base_url=None, max_overlap=1.0, type_seed=42
):
    """Generate S' for each (sentence, mode) pair via ParallelResponsesClient.

    Returns list of S' aligned with `sentences` (same length). None for
    entries where the API failed, the response was unusable, or (when
    max_overlap < 1) the word-Jaccard with S exceeds max_overlap — the
    near-duplicate backstop for the 'realistic' mode.

    base_url: local vLLM OpenAI endpoint (routes any non-Gemini/OpenAI model
    name to the local server). None -> use the API client / env default.
    """
    from corpus_reasoning.lib.llm_request_client import ParallelResponsesClient

    client = ParallelResponsesClient(
        max_concurrent=max_concurrent,
        use_cache=True,
        local_base_url=base_url,
    )
    # For 'realistic' mode, assign a contradiction type per pair (deterministic).
    trng = random.Random(type_seed)
    chosen_types = []
    prompts = []
    for s, m in zip(sentences, modes):
        if m == "realistic":
            tname, tdesc = CONTRADICTION_TYPES[trng.randrange(len(CONTRADICTION_TYPES))]
            chosen_types.append(tname)
            prompts.append(REALISTIC_PROMPT.format(type_desc=tdesc, sentence=s))
        elif m == "simple":
            chosen_types.append("simple")
            prompts.append(SIMPLE_PROMPT.format(sentence=s))
        else:
            chosen_types.append("subtle")
            prompts.append(SUBTLE_PROMPT.format(sentence=s))

    print(
        f"Perturbing {len(prompts)} sentences with {model}"
        f"{' @ '+base_url if base_url else ''}..."
    )
    responses = client.run(
        model=model,
        prompts=prompts,
        temperature=0.7,
        max_output_tokens=200,
    )
    out = []
    n_fail = n_overlap = 0
    for s, r in zip(sentences, responses):
        if not r.get("success", True):
            out.append(None)
            n_fail += 1
            continue
        cleaned = clean_response(r.get("response") or "")
        if not cleaned or cleaned.strip() == s.strip():
            out.append(None)
            n_fail += 1
        elif max_overlap < 1.0 and word_jaccard(s, cleaned) > max_overlap:
            out.append(None)
            n_overlap += 1
        else:
            out.append(cleaned)
    print(
        f"  {len(out) - n_fail - n_overlap} usable perturbations, "
        f"{n_fail} failures/dupes, {n_overlap} rejected for overlap>{max_overlap}"
    )
    return out


def filter_valid_contradictions(sentences, perturbations, model, max_concurrent, base_url=None):
    """LLM-judge each (S, S') pair; null out the ones that aren't real
    contradictions (paraphrases, or a shift to a non-conflicting
    timepoint/subgroup). Returns perturbations with rejects set to None.
    """
    from corpus_reasoning.lib.llm_request_client import ParallelResponsesClient

    client = ParallelResponsesClient(
        max_concurrent=max_concurrent, use_cache=True, local_base_url=base_url
    )

    idx = [i for i, p in enumerate(perturbations) if p is not None]
    prompts = [JUDGE_PROMPT.format(a=sentences[i], b=perturbations[i]) for i in idx]
    print(f"Validity filter: judging {len(prompts)} candidate contradictions...")
    responses = client.run(
        model=model,
        prompts=prompts,
        temperature=0.0,
        max_output_tokens=10,
    )

    out = list(perturbations)
    n_rejected = 0
    for i, r in zip(idx, responses):
        verdict = (r.get("response") or "").strip().upper()
        if not r.get("success", True) or not verdict.startswith("YES"):
            out[i] = None
            n_rejected += 1
    print(
        f"  {len(prompts) - n_rejected}/{len(prompts)} kept as genuine "
        f"contradictions, {n_rejected} rejected"
    )
    return out


# ───────────────────────────── example builder ─────────────────────────────


def build_example(pairs_info, filler_pool, num_docs, rng):
    """Build one example with K (S, S') contradiction pairs.

    pairs_info: list of (S, S', source_aid) for this example.
    filler_pool: dict aid -> list of sentences.

    Fillers are sampled only from abstracts NOT used as gold sources in this
    example. `gold_doc_indices` records 1-indexed positions of (a, b) pairs.
    """
    statements = []
    pair_indices = []
    gold_aids = {info[2] for info in pairs_info}

    for S, S_prime, _ in pairs_info:
        a, b = len(statements), len(statements) + 1
        statements.extend([S, S_prime])
        pair_indices.append((a, b))

    filler_aids = [aid for aid in filler_pool if aid not in gold_aids]
    rng.shuffle(filler_aids)

    need = num_docs - len(statements)
    seen = set(statements)
    for aid in filler_aids:
        if need <= 0:
            break
        for s in filler_pool[aid]:
            if need <= 0:
                break
            if s in seen:
                continue
            statements.append(s)
            seen.add(s)
            need -= 1

    if len(statements) < num_docs:
        raise RuntimeError(
            f"Ran out of filler sentences; got {len(statements)}/{num_docs}. "
            f"Increase --pool-abstracts."
        )

    order = list(range(len(statements)))
    rng.shuffle(order)
    old_to_new = {old: new + 1 for new, old in enumerate(order)}

    documents = [{"text": statements[order[i]]} for i in range(len(order))]
    gold_pairs = sorted(sorted([old_to_new[a], old_to_new[b]]) for a, b in pair_indices)

    return {
        "documents": documents,
        "queries": [],
        "answers": [],
        "gold_doc_indices": gold_pairs,
        "source": "pubmed_perturbation",
    }


# ─────────────────────── expansion of existing examples ───────────────────


def expand_example(example, filler_pool, num_docs, rng):
    """Resize an existing n=N0 example to `num_docs` claims.

    Preserves the gold pair texts (so a CoT generated downstream remains
    valid) and reshuffles positions so gold claims don't stay put across
    sizes — otherwise a model could learn "gold is always at position X."

    Direction is determined by `num_docs` vs the example's current size:
      - num_docs >= N0  (expand): keep all originals, add fresh fillers
        sampled from `filler_pool`.
      - num_docs <  N0  (shrink): drop random non-gold distractors until
        the corpus has `num_docs` claims; `filler_pool` is unused.

    The minimum feasible size is 2 * number of gold pairs (every gold claim
    must keep its slot).

    Args:
        example: dict with {documents, gold_doc_indices, ...} at some size N0.
        filler_pool: dict aid -> list of sentences. Only read on expansion.
        num_docs: target claim count.
        rng: random.Random — example-local so shuffles stay deterministic.
    """
    orig_docs = example["documents"]
    N0 = len(orig_docs)
    min_docs = 2 * len(example["gold_doc_indices"])
    assert (
        num_docs >= min_docs
    ), f"num_docs={num_docs} < 2*K={min_docs}; every gold slot must be kept."

    # Snapshot gold texts — referenced by 1-indexed positions in orig_docs.
    # Used for both (a) remapping gold_doc_indices after shuffle and (b)
    # protecting gold docs from being dropped in the shrink path.
    gold_texts = [
        [orig_docs[a - 1]["text"], orig_docs[b - 1]["text"]] for a, b in example["gold_doc_indices"]
    ]
    gold_text_set = {t for pair in gold_texts for t in pair}

    if num_docs >= N0:
        # ── Expand: keep all originals, draw fresh fillers from the pool ──
        # De-dupe by raw text so we never introduce a filler that exactly
        # matches any existing claim. A literal copy elsewhere would silently
        # poison the label.
        seen = {d["text"] for d in orig_docs}
        need = num_docs - N0
        filler_aids = list(filler_pool.keys())
        rng.shuffle(filler_aids)
        new_fillers = []
        for aid in filler_aids:
            if need <= 0:
                break
            for s in filler_pool[aid]:
                if need <= 0:
                    break
                if s in seen:
                    continue
                new_fillers.append({"text": s})
                seen.add(s)
                need -= 1
        if len(new_fillers) < num_docs - N0:
            raise RuntimeError(
                f"Ran out of filler sentences; only added "
                f"{len(new_fillers)}/{num_docs - N0}. Increase --pool-abstracts."
            )
        combined = orig_docs + new_fillers
    else:
        # ── Shrink: keep all gold claims + a random subset of distractors ──
        gold_docs = [d for d in orig_docs if d["text"] in gold_text_set]
        non_gold_docs = [d for d in orig_docs if d["text"] not in gold_text_set]
        keep_non_gold = num_docs - len(gold_docs)
        kept = rng.sample(non_gold_docs, keep_non_gold)
        combined = gold_docs + kept

    order = list(range(len(combined)))
    rng.shuffle(order)
    shuffled = [combined[i] for i in order]

    # Remap gold positions by text. Claim texts are unique within the
    # example (dedupe above, plus generator invariant), so text->pos is
    # well-defined.
    text_to_pos = {d["text"]: i + 1 for i, d in enumerate(shuffled)}
    new_pairs = sorted(sorted([text_to_pos[ta], text_to_pos[tb]]) for ta, tb in gold_texts)

    out = dict(example)
    out["documents"] = shuffled
    out["gold_doc_indices"] = new_pairs
    return out


def run_expansion(args):
    """Resize existing n=N0 JSONL data to --num-docs.

    Expand (num_docs > N0) draws fresh fillers from PubMed; shrink
    (num_docs < N0) just subsets the existing non-gold distractors and
    skips the pool load.
    """
    # Decide whether we actually need the filler pool. Shrink-only invocations skip the PubMed
    # dataset load entirely.
    #
    # This used to peek at the FIRST ROW only. Source files are mixed-size (the shipped realistic
    # pool spans n=50..950), so a file whose first row already held >= --num-docs claims set
    # need_pool=False, and the first SMALLER row downstream then hit the expand path with an empty
    # pool and died on "Ran out of filler sentences; only added 0/N". Take the minimum over every
    # row instead: one row shorter than the target is enough to need fillers.
    need_pool = False
    for src_path in [args.expand_from_train, args.expand_from_eval]:
        if src_path:
            sizes = [len(ex["documents"]) for ex in load_jsonl(src_path)]
            if sizes and args.num_docs > min(sizes):
                need_pool = True
                break

    if need_pool:
        # Load fresh filler pool with a distinct seed so abstracts are
        # disjoint from the original n=N0 pool — minimizes the chance a new
        # filler comes from the same abstract as a gold source sentence.
        _, filler_pool = load_pubmed_pool(args.pool_abstracts, args.filler_pool_seed)
    else:
        filler_pool = {}

    K = args.num_contradictions  # only used in the output filename tag
    for split_label, src_path in [
        ("train", args.expand_from_train),
        ("eval", args.expand_from_eval),
    ]:
        if not src_path:
            continue
        src_examples = load_jsonl(src_path)
        print(
            f"\nExpanding {split_label}: {len(src_examples)} examples "
            f"from {src_path} -> num_docs={args.num_docs}"
        )

        out_examples = []
        for i, ex in enumerate(tqdm(src_examples, desc=f"  expand {split_label}")):
            # Per-example deterministic seed so shuffles are reproducible AND
            # identical across different target sizes — the only thing that
            # varies between n=250/500/1000 is the filler count.
            ex_rng = random.Random(args.seed * 1_000_003 + i)
            out_examples.append(
                expand_example(
                    ex,
                    filler_pool,
                    args.num_docs,
                    ex_rng,
                )
            )

        tag = f"pubmed_{args.mode}_n{args.num_docs}_k{K}"
        path = f"{args.output_dir}/contradiction_{split_label}_{tag}.jsonl"
        save_jsonl(path, out_examples)
        print_dataset_stats(out_examples, split_label.capitalize(), path)


# ───────────────────────────────── main ────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="Generate PubMed synthetic-perturbation contradiction data"
    )
    ap.add_argument(
        "--num-docs",
        type=int,
        default=100,
        help="Total sentences (documents) per example. Ignored in "
        "the FRESH path when --num-docs-min/--num-docs-max are "
        "set (a per-example length is sampled instead).",
    )
    ap.add_argument(
        "--num-docs-min",
        type=int,
        default=None,
        help="FRESH path only: sample a per-example length "
        "uniformly in [min, max] (continuous-length data, "
        "like the other long-context tasks). ~43 Qwen "
        "tokens/doc: 50->2k, 190->8k, 385->16k, 765->32k. "
        "Requires --num-docs-max; falls back to fixed "
        "--num-docs when unset.",
    )
    ap.add_argument(
        "--num-docs-max",
        type=int,
        default=None,
        help="FRESH path only: upper bound for per-example length " "sampling. See --num-docs-min.",
    )
    ap.add_argument(
        "--num-contradictions",
        type=int,
        default=3,
        help="Number of contradicting pairs per example",
    )
    ap.add_argument("--num-train", type=int, default=2000)
    ap.add_argument("--num-eval", type=int, default=300)
    ap.add_argument(
        "--mode",
        choices=["simple", "subtle", "both", "realistic"],
        default="both",
        help="Perturbation style. 'both' splits gold pairs evenly "
        "between simple+subtle. 'realistic' (recommended) draws "
        "a contradiction TYPE per pair and fully rephrases — no "
        "near-duplicate tells.",
    )
    ap.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Local vLLM OpenAI-compatible endpoint (e.g. "
        "http://127.0.0.1:8765/v1). Routes a local --model "
        "(e.g. Qwen2.5-14B-Instruct) instead of a paid API.",
    )
    ap.add_argument(
        "--max-overlap",
        type=float,
        default=1.0,
        help="Reject a gold pair if word-Jaccard(S, S') exceeds this "
        "(realistic-mode near-duplicate backstop). Try 0.5.",
    )
    ap.add_argument(
        "--pool-abstracts",
        type=int,
        default=20000,
        help="How many PubMed abstracts to load as the sentence "
        "pool. Must supply enough claim sentences and "
        "fillers for all examples.",
    )
    ap.add_argument("--model", type=str, default="gemini-2.5-flash")
    ap.add_argument("--max-concurrent", type=int, default=25)
    ap.add_argument("--output-dir", type=str, default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--no-validity-filter",
        dest="validity_filter",
        action="store_false",
        help="skip the LLM judge that drops generated pairs which "
        "are not genuine contradictions (on by default)",
    )
    ap.set_defaults(validity_filter=True)
    # Expansion mode — keeps gold pairs from an existing n=N0 dataset and
    # adds distractors to reach --num-docs. Useful for scaling studies where
    # the task is held fixed while context size varies.
    ap.add_argument(
        "--expand-from-train",
        type=str,
        default="",
        help="Expansion mode: existing train JSONL whose gold "
        "pairs are preserved while fillers are added to "
        "reach --num-docs. Skips the LLM perturbation step.",
    )
    ap.add_argument(
        "--expand-from-eval",
        type=str,
        default="",
        help="Expansion mode: same as --expand-from-train but " "for the eval split.",
    )
    ap.add_argument(
        "--filler-pool-seed",
        type=int,
        default=43,
        help="Seed for the filler PubMed pool in expansion "
        "mode. Defaults to 43 so it's disjoint from the "
        "original generator's seed=42 pool.",
    )
    args = ap.parse_args()

    # Expansion mode short-circuits: no LLM calls, reuse existing golds.
    if args.expand_from_train or args.expand_from_eval:
        run_expansion(args)
        return

    # Per-example continuous length sampling (FRESH path). When both bounds are
    # set we sample nd in [min, max] per example; otherwise use fixed num_docs.
    sample_len = args.num_docs_min is not None and args.num_docs_max is not None
    if (args.num_docs_min is None) != (args.num_docs_max is None):
        raise ValueError("--num-docs-min and --num-docs-max must be set together")
    min_docs = 2 * args.num_contradictions
    if sample_len:
        assert args.num_docs_min <= args.num_docs_max, "--num-docs-min must be <= --num-docs-max"
        assert (
            args.num_docs_min >= min_docs
        ), f"--num-docs-min must hold all gold pair slots (>= {min_docs})"
    else:
        assert min_docs <= args.num_docs, "num_docs must hold all gold pair slots"

    rng = random.Random(args.seed)
    claim_pool, filler_pool = load_pubmed_pool(args.pool_abstracts, args.seed)

    total_examples = args.num_train + args.num_eval
    K = args.num_contradictions
    total_pairs = total_examples * K

    if len(claim_pool) < total_pairs:
        raise RuntimeError(
            f"Only {len(claim_pool)} claim sentences; need {total_pairs}. "
            f"Increase --pool-abstracts."
        )

    rng.shuffle(claim_pool)
    selected = claim_pool[:total_pairs]

    # Assign modes per gold pair
    if args.mode in ("simple", "subtle", "realistic"):
        modes = [args.mode] * total_pairs
    else:
        modes = []
        for _ in range(total_examples):
            half = K // 2
            ex_modes = ["simple"] * half + ["subtle"] * (K - half)
            rng.shuffle(ex_modes)
            modes.extend(ex_modes)

    sentences = [s for s, _ in selected]
    perturbations = generate_perturbations(
        sentences,
        modes,
        args.model,
        args.max_concurrent,
        base_url=args.base_url,
        max_overlap=args.max_overlap,
        type_seed=args.seed,
    )

    if args.validity_filter:
        perturbations = filter_valid_contradictions(
            sentences,
            perturbations,
            args.model,
            args.max_concurrent,
            base_url=args.base_url,
        )

    # Group into examples, skipping failed pairs. If an example loses a pair,
    # refill from the claim pool leftovers (best effort; else raise).
    pair_tuples = []
    for (s, aid), s_prime, m in zip(selected, perturbations, modes):
        if s_prime is None:
            continue
        pair_tuples.append((s, s_prime, aid, m))

    if len(pair_tuples) < total_pairs:
        print(
            f"  Note: {total_pairs - len(pair_tuples)} pairs dropped due "
            f"to API failures; examples may get fewer pairs."
        )

    # Slice into examples of K pairs each
    examples = []
    cursor = 0
    for split_label, count in [("train", args.num_train), ("eval", args.num_eval)]:
        split_ex = []
        ndesc = f"n{args.num_docs_min}-{args.num_docs_max}" if sample_len else f"n{args.num_docs}"
        for _ in tqdm(range(count), desc=f"Building {split_label} {ndesc} k{K}"):
            if cursor + K > len(pair_tuples):
                break
            picked = pair_tuples[cursor : cursor + K]
            cursor += K
            info = [(s, sp, aid) for s, sp, aid, _ in picked]
            # Continuous length: sample a per-example doc count when bounds are
            # set, else the fixed --num-docs. build_example already supports
            # arbitrary num_docs.
            nd = rng.randint(args.num_docs_min, args.num_docs_max) if sample_len else args.num_docs
            split_ex.append(build_example(info, filler_pool, nd, rng))
        examples.append((split_label, split_ex))

    ntag = f"n{args.num_docs_min}-{args.num_docs_max}" if sample_len else f"n{args.num_docs}"
    tag = f"pubmed_{args.mode}_{ntag}_k{K}"
    for split_label, split_ex in examples:
        if not split_ex:
            continue
        path = f"{args.output_dir}/contradiction_{split_label}_{tag}.jsonl"
        save_jsonl(path, split_ex)
        print_dataset_stats(split_ex, split_label.capitalize(), path)


if __name__ == "__main__":
    main()
