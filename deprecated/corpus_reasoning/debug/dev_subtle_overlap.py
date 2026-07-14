"""Dev loop: iterate the subtle-contradiction prompt to lower word overlap.

Goal: make a subtle contradiction S' share as few words with the source S as
a same-abstract non-contradicting pair does (Jaccard ~0.12), so a word-overlap
heuristic can't separate positives from negatives.

Runs every candidate prompt below on the SAME fixed dev set of source
sentences and prints, per prompt: mean/median Jaccard(S, S') and a few
examples (so we can eyeball that S' is still a real contradiction).

Targets (measured on the current data):
    subtle positives  ~0.283   <- what we want to lower
    same-abstract neg ~0.123   <- target
    random neg        ~0.061

Run (corpus-reasoning-eval env, needs the LLM client + datasets):
    python scripts/debug/dev_subtle_overlap.py
"""

import re
import numpy as np

from corpus_reasoning.data.generate_pubmed_contradiction_data import (
    load_pubmed_pool, clean_response,
)
from corpus_reasoning.lib.llm_request_client import ParallelResponsesClient

N_DEV = 25
DEV_SEED = 999

# ── candidate prompts; add/edit and rerun ──────────────────────────────────
PROMPTS = {
    "v0_current": """Write a biomedical sentence that factually contradicts the one below on one specific element (a number, magnitude, duration, scope qualifier, or comparator), BUT uses clearly different wording and sentence structure. The two sentences should read as if they came from two different abstracts — a reader must actually understand both to detect the contradiction, because surface form alone won't reveal it.

Requirements:
- Factually incompatible on ONE specific element — the two sentences cannot both be true.
- Paraphrase aggressively: reorder clauses, swap verbs, use synonyms, change sentence opening. Do NOT preserve the original structure.
- Preserve biomedical register and plausibility.
- Do NOT use polarity flips ("increased"↔"decreased") or insert "not".
- Length should be similar to the original (±50%).

Return only the contradicting sentence, with no preamble, quotes, or explanation.

Original: {sentence}
Contradicting:""",

    "v3_specific_and_valid": """Your task: write ONE biomedical sentence that genuinely CONTRADICTS the sentence below, while sharing almost no words with it.

Step 1 (think, do not output): identify the single concrete factual element in the original you will contradict — a number, magnitude, duration, scope/population, or comparator. Your new sentence must describe the SAME underlying finding, so that the two sentences cannot both be true.

Step 2 (output): write the contradicting sentence so that:
- It clearly conflicts with the original on that one element — a domain expert reading both would say "these disagree." It must NOT be a mere paraphrase.
- It shares as few words as possible with the original: rename every entity (synonyms, alternative names, expand/contract abbreviations), use different verbs, a different sentence opening, and a different structure.
- It stays a plausible biomedical sentence.
- No polarity flips ("increased"<->"decreased"), no inserted "not".

Return only the contradicting sentence, no preamble or quotes.

Original: {sentence}
Contradicting:""",

    "v4_fewshot": """Your task: write ONE biomedical sentence that genuinely CONTRADICTS the sentence below, while sharing almost no words with it.

Step 1 (think, do not output): identify the single concrete factual element in the original you will contradict — a number, magnitude, duration, scope/population, or comparator. Your new sentence must describe the SAME underlying finding, so that the two sentences cannot both be true.

Step 2 (output): write the contradicting sentence so that:
- It clearly conflicts with the original on that one element — a domain expert reading both would say "these disagree." It must NOT be a mere paraphrase, and must NOT just describe a different timepoint/subgroup (that would let both be true).
- It shares as few words as possible with the original: rename every entity (synonyms, alternative names, expand/contract abbreviations), use different verbs, a different opening, a different structure.
- It stays a plausible biomedical sentence.
- No polarity flips ("increased"<->"decreased"), no inserted "not".

Examples (note: low word overlap, but each pair genuinely cannot both be true):
Original: Treatment reduced infarct size to 4.8% versus 25.8% in controls.
Contradicting: Necrotic tissue in the intervention arm differed negligibly from untreated animals, registering 24.1% against 25.8%.

Original: The antibiotic did not significantly affect cytokine levels.
Contradicting: Interleukin and tumour necrosis factor concentrations rose markedly once the drug was administered.

Original: Clinical benefit was confined to patients older than 65.
Contradicting: Therapeutic response among the same trial population was concentrated in adults below 45.

Return only the contradicting sentence, no preamble or quotes.

Original: {sentence}
Contradicting:""",
}

JUDGE_PROMPT = """Sentence A: {a}
Sentence B: {b}

Do these two sentences make factually INCOMPATIBLE claims about the same finding — i.e. they genuinely contradict, both cannot be true? A mere paraphrase (same meaning) or two unrelated facts are NOT a contradiction.

Answer with exactly one word: YES or NO."""


def toks(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def jaccard(a, b):
    A, B = toks(a), toks(b)
    return len(A & B) / len(A | B) if (A | B) else 0.0


def main():
    claim_pool, _ = load_pubmed_pool(400, DEV_SEED)
    import random
    rng = random.Random(DEV_SEED)
    rng.shuffle(claim_pool)
    dev = [s for s, _ in claim_pool[:N_DEV]]
    print(f"dev set: {len(dev)} source sentences\n")

    client = ParallelResponsesClient(max_concurrent=25, use_cache=False)

    for name, tmpl in PROMPTS.items():
        prompts = [tmpl.format(sentence=s) for s in dev]
        responses = client.run(model="gemini-2.5-flash", prompts=prompts,
                                temperature=0.7, max_output_tokens=200)
        kept = []
        for s, r in zip(dev, responses):
            if not r.get("success", True):
                continue
            sp = clean_response(r.get("response") or "")
            if not sp or sp.strip() == s.strip():
                continue
            kept.append((s, sp))

        # validity judge: is each (S, S') a genuine contradiction?
        judge_prompts = [JUDGE_PROMPT.format(a=s, b=sp) for s, sp in kept]
        judgments = client.run(model="gemini-2.5-flash", prompts=judge_prompts,
                               temperature=0.0, max_output_tokens=10)
        valid = []
        for (s, sp), j in zip(kept, judgments):
            verdict = (j.get("response") or "").strip().upper().startswith("YES")
            valid.append(verdict)

        all_ov = np.array([jaccard(s, sp) for s, sp in kept])
        valid_ov = np.array([jaccard(s, sp) for (s, sp), v in zip(kept, valid) if v])
        n_valid = sum(valid)
        print(f"=== {name} ===")
        print(f"  usable {len(kept)}/{len(dev)}   "
              f"valid contradictions {n_valid}/{len(kept)} "
              f"({n_valid/max(len(kept),1):.0%})")
        print(f"  overlap all   : mean={all_ov.mean():.3f} median={np.median(all_ov):.3f}")
        if len(valid_ov):
            print(f"  overlap valid : mean={valid_ov.mean():.3f} median={np.median(valid_ov):.3f}")
        for (s, sp), v in list(zip(kept, valid))[:5]:
            print(f"  jac={jaccard(s,sp):.3f}  judge={'VALID' if v else 'INVALID'}")
            print(f"   S : {s}")
            print(f"   S': {sp}")
        print()


if __name__ == "__main__":
    main()
