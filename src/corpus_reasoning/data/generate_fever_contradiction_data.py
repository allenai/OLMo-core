"""Generate FEVER contradiction detection data in unified structured format.

Each example is a corpus of N numbered claims with K contradicting pairs hidden
among filler claims. Contradiction pairs come straight from FEVER's REFUTES
rows: each REFUTES row has a (claim, evidence_sentence) tuple where the claim
is asserted false and the cited Wikipedia sentence is the truth that refutes
it. The two cannot both be true, so they contradict by construction — no proxy
heuristics, no NLI re-scoring needed.

Each "document" in the unified format is a single short statement (either a
FEVER claim or a Wikipedia sentence):
    {"text": "Joey Graceffa was born in July."}
    {"text": "Joseph Michael Graceffa (born May 16, 1991) is an American ..."}

The example is structured into topical *clusters* by default:

  - Each gold contradiction pair anchors a **gold cluster** containing the
    REFUTES pair plus NEI claims about the same Wikipedia page. NEI claims
    are by definition unverifiable against Wikipedia, so they're much less
    likely to directly state the truth that the REFUTES claim denies. (They
    aren't 100% safe — broad-scope REFUTES claims like "X does not have an
    acting career" can still be contradicted by NEI claims that mention X's
    acting at all — but the false-negative rate is much lower than if we
    used SUPPORTS evidence sentences here.)

  - **Decoy clusters** are added on random non-gold Wikipedia pages to fill
    the remaining context. Decoys can safely contain SUPPORTS (claim,
    evidence) tuples in addition to NEI claims, because their content is
    about a different entity than any gold pair, so cross-page false
    negatives are vanishingly rare. The SUPPORTS pairs in decoys are useful
    "fake pair" distractors — they look syntactically like contradiction
    pairs but are consistent, forcing the model to verify the semantic
    relation rather than just match topics. Without decoys, gold clusters
    would form dense topical islands and the model could shortcut on "find
    the dense cluster"; decoys make K + D islands all look the same, only
    K of which contain contradictions.

  - Any leftover slots are filled from a flat random pool (NEI claims +
    SUPPORTS evidence sentences) so the budget always lands at exactly N.

Hard distractors are ON by default. To revert to plain random fillers (the
older simpler behavior), pass `--hard-nei-per-pair 0 --decoy-support-pairs 0
--no-decoys`.

`gold_doc_indices` stores the contradiction pairs as a list of 1-indexed
claim-ID pairs (`[[a, b], [c, d], ...]`). This matches the format expected by
`scripts/lib/data_format.py:_build_output(task="contradiction")`.

HF dataset: https://huggingface.co/datasets/copenlu/fever_gold_evidence
  train:      228,277 (claim, evidence) rows  (~115k SUPPORTS, ~47k REFUTES, ~66k NEI)
  validation:  15,935 rows
  test:        16,039 rows

The canonical `fever/fever` HF dataset is no longer loadable (script-based,
removed in newer `datasets` versions); `copenlu/fever_gold_evidence` is the
parquet mirror used here.

Usage:
    # Default: hard distractors + decoys filling the remainder
    python scripts/data/generate_fever_contradiction_data.py \\
        --num-claims 100 --num-contradictions 3 \\
        --num-train 5000 --num-eval 500

    # Plain random fillers only (older behavior)
    python scripts/data/generate_fever_contradiction_data.py \\
        --num-claims 100 --num-contradictions 3 \\
        --hard-nei-per-pair 0 --decoy-support-pairs 0 --no-decoys
"""

import argparse
import random
import re
from collections import defaultdict

from datasets import load_dataset
from tqdm import tqdm

from corpus_reasoning.lib.io import save_jsonl, print_dataset_stats


def _clean_wiki_text(text):
    """Undo PTB-style tokenization artifacts in FEVER evidence sentences.

    FEVER evidence sentences are stored with PTB-tokenized punctuation
    (-LRB-/-RRB-, ``/'', spaced commas, etc.). Clean them up so the model
    sees natural prose that matches the FEVER claim style.
    """
    if not text:
        return text
    text = (text
            .replace("-LRB-", "(").replace("-RRB-", ")")
            .replace("-LSB-", "[").replace("-RSB-", "]")
            .replace("-LCB-", "{").replace("-RCB-", "}")
            .replace("``", '"').replace("''", '"'))
    # Remove space before common punctuation
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = text.replace(" 's", "'s").replace(" n't", "n't")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _first_evidence_sentence(evidence):
    """Pick the first non-empty cleaned sentence from a FEVER evidence field."""
    if not evidence:
        return None
    for triple in evidence:
        if not triple or len(triple) < 3:
            continue
        sent = _clean_wiki_text(triple[2])
        if sent:
            return sent
    return None


def _build_pools(ds):
    """Walk a FEVER split and build everything needed for example construction.

    Returns a dict with:
        pairs:              list of (false_claim, true_sentence, page) — each
                            is a guaranteed contradiction pair anchored to a
                            specific Wikipedia page (used for topic lookup).
        nei_by_page:        page -> list of NEI claim texts citing that page.
                            Used for hard NEI distractors and decoy clusters.
        sup_pairs_by_page:  page -> list of (claim, evidence_sentence) tuples
                            for SUPPORTS rows citing that page. Each tuple
                            adds 2 docs when used as a distractor.
        random_fillers:     flat mixed pool of NEI claims + SUPPORTS evidence
                            sentences. Used to top up any leftover slots.
        all_pages:          set of all pages with at least one NEI or
                            SUPPORTS row, for decoy sampling.
    """
    pairs = []
    seen_pairs = set()
    nei_by_page = defaultdict(list)
    sup_pairs_by_page = defaultdict(list)
    nei_all = set()
    sup_sentences_all = set()
    seen_nei = set()
    seen_sup_pair = set()

    for ex in ds:
        label = ex.get("label")
        claim = (ex.get("claim") or "").strip()
        if not claim:
            continue

        ev = ex.get("evidence") or []
        page = ev[0][0] if (ev and ev[0] and ev[0][0]) else None

        if label == "NOT ENOUGH INFO":
            nei_all.add(claim)
            if page and claim not in seen_nei:
                nei_by_page[page].append(claim)
                seen_nei.add(claim)
            continue

        sentence = _first_evidence_sentence(ev)
        if not sentence or sentence == claim:
            continue

        if label == "SUPPORTS":
            sup_sentences_all.add(sentence)
            if page:
                sig = (claim, sentence)
                if sig not in seen_sup_pair:
                    sup_pairs_by_page[page].append((claim, sentence))
                    seen_sup_pair.add(sig)
        elif label == "REFUTES":
            if not page:
                continue
            sig = (claim, sentence)
            if sig in seen_pairs:
                continue
            seen_pairs.add(sig)
            pairs.append((claim, sentence, page))

    all_pages = set(nei_by_page) | set(sup_pairs_by_page)
    return {
        "pairs": pairs,
        "nei_by_page": dict(nei_by_page),
        "sup_pairs_by_page": dict(sup_pairs_by_page),
        "random_fillers": list(nei_all) + list(sup_sentences_all),
        "all_pages": all_pages,
    }


def _gather_nei_only(page, hard_nei, pools, seen_text, rng, slot_budget):
    """Pull up to `hard_nei` NEI claims about `page`.

    Used to populate gold clusters. NEI claims are by construction unverifiable
    against Wikipedia, so they're much less likely to directly contradict the
    gold REFUTES claim than SUPPORTS evidence sentences (which state the
    truth that the REFUTES claim denies). They are not 100% safe for very
    broad REFUTES claims, but the false-negative rate is far lower.
    """
    out = []
    if slot_budget <= 0:
        return out
    nei_pool = pools["nei_by_page"].get(page, [])
    if hard_nei > 0 and nei_pool:
        candidates = [c for c in nei_pool if c not in seen_text]
        rng.shuffle(candidates)
        for c in candidates[:hard_nei]:
            if len(out) >= slot_budget:
                break
            out.append(c)
            seen_text.add(c)
    return out


def _gather_decoy_cluster(page, hard_nei, decoy_sup, pools, seen_text, rng,
                          slot_budget):
    """Build a decoy cluster on `page`: NEI claims + SUPPORTS (claim, evidence)
    tuples.

    Decoys live on non-gold Wikipedia pages, so SUPPORTS evidence sentences
    here cannot accidentally refute any gold REFUTES claim — the cross-page
    contradiction risk is essentially zero. SUPPORTS pairs are still useful
    here as "fake pair" distractors that look syntactically like contradiction
    pairs but are consistent, forcing the model to verify the relation
    rather than just match topics.
    """
    out = []
    if slot_budget <= 0:
        return out

    nei_pool = pools["nei_by_page"].get(page, [])
    if hard_nei > 0 and nei_pool:
        candidates = [c for c in nei_pool if c not in seen_text]
        rng.shuffle(candidates)
        for c in candidates[:hard_nei]:
            if len(out) >= slot_budget:
                break
            out.append(c)
            seen_text.add(c)

    sup_pool = pools["sup_pairs_by_page"].get(page, [])
    if decoy_sup > 0 and sup_pool:
        candidates = [(c, s) for c, s in sup_pool
                      if c not in seen_text and s not in seen_text]
        rng.shuffle(candidates)
        for c, s in candidates[:decoy_sup]:
            if len(out) + 2 > slot_budget:
                break
            out.extend([c, s])
            seen_text.update([c, s])

    return out


def _build_example(contradiction_pairs, pools, num_claims, hard_nei, decoy_sup,
                   use_decoys, rng):
    """Build a single unified-format example with K contradiction pairs.

    Layout (before shuffle):
      [gold pair 1] [NEI x N1]
      [gold pair 2] [NEI x N2]
      ...
      [decoy cluster 1: NEI + SUPPORTS pairs] [decoy cluster 2: ...] ...
      [random fillers ...]

    Note: gold clusters contain only NEI distractors (no SUPPORTS pairs),
    because SUPPORTS evidence sentences from the same Wikipedia page often
    state the truth that the gold REFUTES claim denies, which would create
    unlabelled contradiction pairs (false negatives). SUPPORTS pairs are
    still used in decoy clusters where the cross-page risk is negligible.

    Final list is shuffled, and `gold_doc_indices` records the 1-indexed
    positions of each (a, b) pair.
    """
    statements = []
    pair_indices = []
    seen_text = set()
    gold_pages = set()

    # ── Gold clusters: pair + NEI distractors only ──
    for false_claim, true_sentence, page in contradiction_pairs:
        if false_claim in seen_text or true_sentence in seen_text:
            continue
        a, b = len(statements), len(statements) + 1
        statements.extend([false_claim, true_sentence])
        seen_text.update([false_claim, true_sentence])
        pair_indices.append((a, b))
        gold_pages.add(page)
        budget = num_claims - len(statements)
        cluster = _gather_nei_only(
            page, hard_nei, pools, seen_text, rng, budget,
        )
        statements.extend(cluster)

    if len(statements) > num_claims:
        raise RuntimeError(
            f"Gold clusters total {len(statements)} > num_claims={num_claims}; "
            f"reduce --hard-nei-per-pair, or increase --num-claims"
        )

    # ── Decoy clusters: NEI + SUPPORTS pairs on non-gold pages ──
    if use_decoys and len(statements) < num_claims:
        decoy_pages = [p for p in pools["all_pages"] if p not in gold_pages]
        rng.shuffle(decoy_pages)
        for page in decoy_pages:
            budget = num_claims - len(statements)
            if budget <= 0:
                break
            cluster = _gather_decoy_cluster(
                page, hard_nei, decoy_sup, pools, seen_text, rng, budget,
            )
            if cluster:
                statements.extend(cluster)

    # ── Random fillers: top up any leftover slots ──
    if len(statements) < num_claims:
        num_random = num_claims - len(statements)
        random_pool = pools["random_fillers"]
        candidates = rng.sample(
            random_pool, min(num_random * 3, len(random_pool))
        )
        picked = [c for c in candidates if c not in seen_text][:num_random]
        while len(picked) < num_random:
            picked.append(rng.choice(random_pool))
        statements.extend(picked)
        seen_text.update(picked)

    # ── Shuffle and recover 1-indexed gold pair positions ──
    order = list(range(len(statements)))
    rng.shuffle(order)
    old_to_new = {old: new + 1 for new, old in enumerate(order)}

    documents = [{"text": statements[order[i]]} for i in range(len(order))]
    gold_pairs = sorted(
        sorted([old_to_new[a], old_to_new[b]]) for a, b in pair_indices
    )

    return {
        "documents": documents,
        "queries": [],
        "answers": [],
        "gold_doc_indices": gold_pairs,
        "source": "fever",
    }


def _generate_split(label, count, pools, num_claims, k, hard_nei, decoy_sup,
                    use_decoys, rng):
    if count == 0:
        return []
    pairs = pools["pairs"]
    if len(pairs) < k:
        raise RuntimeError(
            f"{label}: only {len(pairs)} contradiction pairs available, "
            f"need ≥{k} per example"
        )
    examples = []
    cursor = 0
    for _ in tqdm(range(count), desc=f"FEVER {label.lower()} n{num_claims} k{k}"):
        if cursor + k > len(pairs):
            rng.shuffle(pairs)
            cursor = 0
        picked = pairs[cursor:cursor + k]
        cursor += k
        examples.append(_build_example(
            picked, pools, num_claims, hard_nei, decoy_sup, use_decoys, rng,
        ))
    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Generate FEVER contradiction detection data in unified format"
    )
    parser.add_argument("--num-claims", type=int, default=100,
                        help="Total claims (documents) per example")
    parser.add_argument("--num-contradictions", type=int, default=3,
                        help="Number of contradicting pairs per example "
                             "(2 * this many slots are 'gold')")
    parser.add_argument("--num-train", type=int, default=5000)
    parser.add_argument("--num-eval", type=int, default=500)
    parser.add_argument("--hard-nei-per-pair", type=int, default=15,
                        help="Topic-matched NEI claims per CLUSTER (used in "
                             "both gold and decoy clusters), capped by what "
                             "the page actually has. Set to 0 to disable.")
    parser.add_argument("--decoy-support-pairs", type=int, default=2,
                        help="SUPPORTS (claim, evidence) tuples per DECOY "
                             "cluster only. Each tuple adds 2 documents that "
                             "look like a contradiction pair but are actually "
                             "consistent. NOT used in gold clusters because "
                             "SUPPORTS evidence sentences from the same page "
                             "as a REFUTES claim often state the truth that "
                             "the REFUTES claim denies, creating false-"
                             "negative pairs. Set to 0 to disable.")
    parser.add_argument("--no-decoys", action="store_true",
                        help="Disable decoy clusters. By default, after gold "
                             "clusters are placed, decoy clusters are added "
                             "on random non-gold pages until the context is "
                             "filled, defeating the topical-island shortcut.")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    assert args.num_contradictions * 2 <= args.num_claims, \
        "num_claims must hold all contradiction claims plus some fillers"

    use_decoys = not args.no_decoys
    rng = random.Random(args.seed)

    print("Loading copenlu/fever_gold_evidence...")
    train_ds = load_dataset("copenlu/fever_gold_evidence", split="train")
    eval_ds = load_dataset("copenlu/fever_gold_evidence", split="validation")
    print(f"  train rows: {len(train_ds)}, validation rows: {len(eval_ds)}")

    print("\nBuilding pools from REFUTES / SUPPORTS / NEI rows...")
    train_pools = _build_pools(train_ds)
    eval_pools = _build_pools(eval_ds)
    print(f"  Train: {len(train_pools['pairs'])} pairs, "
          f"{len(train_pools['nei_by_page'])} NEI pages, "
          f"{len(train_pools['sup_pairs_by_page'])} SUPPORTS pages, "
          f"{len(train_pools['random_fillers'])} random fillers")
    print(f"  Eval:  {len(eval_pools['pairs'])} pairs, "
          f"{len(eval_pools['nei_by_page'])} NEI pages, "
          f"{len(eval_pools['sup_pairs_by_page'])} SUPPORTS pages, "
          f"{len(eval_pools['random_fillers'])} random fillers")

    # FEVER's dev split has only ~16k rows total — its pools may be too
    # thin to fill an n=1000 example without heavy reuse. Pool with train
    # data if so (gold pairs are still split-disjoint, only fillers overlap).
    if len(eval_pools["random_fillers"]) < args.num_claims * 4:
        merged = defaultdict(list)
        for src in (eval_pools["nei_by_page"], train_pools["nei_by_page"]):
            for k_, v in src.items():
                merged[k_].extend(v)
        eval_pools["nei_by_page"] = dict(merged)
        merged = defaultdict(list)
        for src in (eval_pools["sup_pairs_by_page"], train_pools["sup_pairs_by_page"]):
            for k_, v in src.items():
                merged[k_].extend(v)
        eval_pools["sup_pairs_by_page"] = dict(merged)
        eval_pools["random_fillers"] = list(set(
            eval_pools["random_fillers"] + train_pools["random_fillers"]
        ))
        eval_pools["all_pages"] = (set(eval_pools["nei_by_page"])
                                   | set(eval_pools["sup_pairs_by_page"]))
        print(f"  Eval pools extended with train pools "
              f"({len(eval_pools['random_fillers'])} random fillers)")

    rng.shuffle(train_pools["pairs"])
    rng.shuffle(eval_pools["pairs"])

    k = args.num_contradictions
    tag = f"fever_n{args.num_claims}_k{k}"

    for label, count, pools in [
        ("Train", args.num_train, train_pools),
        ("Eval", args.num_eval, eval_pools),
    ]:
        examples = _generate_split(
            label, count, pools, args.num_claims, k,
            args.hard_nei_per_pair, args.decoy_support_pairs,
            use_decoys, rng,
        )
        if not examples:
            continue
        suffix = "train" if label == "Train" else "eval"
        path = f"{args.output_dir}/contradiction_{suffix}_{tag}.jsonl"
        save_jsonl(path, examples)
        print_dataset_stats(examples, label, path)


if __name__ == "__main__":
    main()
