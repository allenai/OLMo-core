# PubMed multi-claim contradiction — a from-scratch rebuild of the contradiction setting

**Status 2026-07-22:** FULL MIXED BUILD COMPLETE (PubMed + arXiv), 20,000 train + 2,500 eval across 5 rungs, all decontamination audits PASS. Validated end-to-end at 2k (4B full-attn: CE 0.626->0.082, eval f1 0.949 +/- 0.010, eval_size=500). Generator lives at
`src/corpus_reasoning/data/generate_pubmed_multiclaim.py`; pilots, review pages and tests in
`debug/pubmed_multiclaim/`.

## Why a rebuild

The old pipeline (`generate_pubmed_contradiction_data.py`) perturbs a *real* PubMed sentence S into a
contradicting S'. Two structural problems follow:

1. **Word overlap.** S' is a rewrite of S, so the gold pair is far more lexically similar to each
   other than to any filler.
2. **Provenance asymmetry.** S is human-written and S' LLM-written, so style separates the halves of
   every gold pair.

Measured: a trivial word-overlap baseline (predict the K most similar pairs, content-word Jaccard)
scores **set-F1 0.809** on `contradiction_eval_pubmed_both_n20_k3`. **The old task is ~80% solvable
without reading anything.**

## What the new setting is

Every corpus item is a 2–3 sentence, LLM-written claim. Per abstract the model emits up to 4
*orthogonal* claims as role-labelled sentence fields (`assertion` / `support` / `detail`, joined into
one paragraph — never split). Each claim then gets a contradicting claim of a drawn type
(direction / magnitude / number / scope / temporal / significance / comparator).

Both halves of a gold pair are LLM-written from the same prompt family and multi-sentence, so neither
cue above survives.

Phases (each re-runnable, responses cached to disk):
`split-abstracts` → `claims` → `contradictions` (+ conflict matrix) → `assemble`.

## ⚠ Known limitation: lexical leakage is n-DEPENDENT (accepted 2026-07-21, deferred fix)

The new setting more than halves the leak, but does not remove it — and the residue is **not uniform
across the ladder**:

| n (docs) | pairs | string-baseline F1 | chance | ratio |
|---|---|---|---|---|
| 21 (2k) | 210 | **0.497** | 0.0143 | 35× |
| 46 (4k) | 1,035 | 0.306 | 0.0029 | 105× |
| 97 (8k) | 4,656 | 0.267 | 0.0006 | 414× |
| 197 (16k) | 19,306 | 0.108 | 0.0002 | 697× |
| 397 (32k) | 78,606 | **0.036** | 0.00004 | 946× |

Absolute leakage falls 14× across the ladder (good), but **the gradient runs along the same axis the
CTC experiment varies.** A model exploiting lexical overlap gains a lot at 2k and nothing at 32k —
the same shape predicted for a full-vs-chunked attention gap. Treat any short-rung result with that
in mind.

**Root cause:** the contradiction is required to hold subject, outcome, population and timepoint
fixed (necessary for a genuine contradiction), which forces entity overlap, while distractors come
from other abstracts.

**Deferred fix — paraphrase distractors.** Generate a paraphrase of some claims (high overlap,
*not* contradictory) and place them as decoys, so the top-similarity pairs are decoys rather than
gold. Same trick `generate_xabsence_data.py` uses. Attacks the short rungs, where the leak lives.
Explicitly deferred by the user; do it before drawing conclusions about the 2k–8k rungs.

## Design decisions that cost a failed attempt each

- **Sentence structure must be schema, not instruction.** Asking for "2–3 sentences" produced ONE
  long sentence 19/20 times. Emitting `assertion`/`support`/`detail` as separate JSON fields fixed
  it (15/20 three-sentence).
- **Few-shot examples leak.** A BAD/GOOD pair drawn from an abstract *in the pilot set* made the
  model copy the example verbatim into all 4 of that abstract's claims. Use an unrelated domain.
- **Cross-claim constraints need per-item assignment.** Qwen3-8B ignored "vary the support
  sentences" (boilerplate overlap 0.74); assigning each claim an evidence *kind* by position fixed it
  (0.44).
- **Hold the subject fixed.** The commonest generation failure was answering a claim about
  insulin/fidarestat with one about pioglitazone/metformin, or "at 12 months" with "at nine months" —
  a different question, hence no contradiction. Validity went 16/20 → 20/20 once pinned.
- **Sibling conflicts are a placement CONSTRAINT, not a defect.** 127 of 197 unrecoverable pairs
  (65% of all waste) failed only because the contradiction also conflicted with a sibling claim —
  yet such a pair is fine in any example that omits that sibling. Recording conflicts and honouring
  them at assembly lifted usable yield **51% → 94%**.

## Verify your verifiers (two checks silently did nothing)

Both are now regression-tested; re-run after any model or prompt change.

- `debug/pubmed_multiclaim/test_verifier.py` — plants a known contradicting pair as *distractors*.
  The first `verify_example` only checked GOLD items against the rest and **missed it entirely**.
- `debug/pubmed_multiclaim/test_coherence.py` — three hand-verified incoherent claims (e.g. "linked
  to **higher** incidence … odds ratio **0.62**"). The LLM coherence judge scored **2/5**; a
  deterministic direction-vs-ratio check scores **5/5** and is free.

## Traps hit while wiring this up

- **`run_ctc_local.sbatch` silently reuses stale data.** It stages to
  `$ROOT/data/$(basename DATA_SRC)` with `cp -n` (no-clobber). A shard dir named
  `contradiction_train` collided with the existing CTC-suite dataset, so the new data was **never
  copied** and training ran on the old 40,957-token corpus. It only crashed because `--seq-len 2048`
  was incompatible. **Always give a new dataset a unique basename.**
- **Tokens per claim is 81.6, not the ~55 estimated** (Qwen3.5 tokenizer). The first conversion
  dropped **all 4,000** examples. Correct ladder: **21 / 46 / 97 / 197 / 397** claims per rung.
  ~17% still exceed 2048 at n=21 — size n for <5% drops on the real build.
- **Claim-vs-claim conflicts are NOT negligible** (~12% of conflict edges). A proposed 21% saving by
  skipping them would have silently mislabeled examples.
- **vLLM `--data-parallel-size` fails here** (`DP Coordinator process failed to report ZMQ addresses
  within timeout=120s`). Use N independent single-GPU replicas + client-side round-robin
  (`--base-url` accepts a comma-separated list); see `debug/pubmed_multiclaim/serve_replicas.sbatch`.
- **Qwen3-8B needs `--no-think`**, or the reasoning block consumes the whole token budget before any
  JSON is emitted (3/5 abstracts produced nothing).

## Sizing

Target for the full build: **20,000 train + 2,500 eval** examples over 5 rungs (4,000 / 500 each),
K=3 gold pairs, from ~7,000 abstracts (~5,500 train / ~1,500 eval).

**Train and eval are disjoint at the ABSTRACT level** — splitting any finer leaks, because an eval
gold claim's sibling could become a train distractor while sharing its entities and numbers.
`debug/pubmed_multiclaim/audit_decontamination.py` verifies the built corpora on normalised document
text and exits non-zero on any leak.

## Full build (2026-07-22) — what actually shipped

Sources: PubMed (`pqa_artificial`) + arXiv (`gfissore/arxiv-abstracts-2021`), 7,000 abstracts,
split disjoint at abstract level (5,500 train / 1,500 eval).

| | claims | domain mix | usable pairs |
|---|---|---|---|
| train | 20,792 | arxiv 9,768 / biomed 11,024 | 19,648 (95%) |
| eval | 5,996 | arxiv 3,028 / biomed 2,968 | 5,680 (95%) |

Rungs (n from CLT sizing on the measured **78.4 +/- 16.2 tok/claim**, targeting <5% over-length
drops — NOT an estimate; the first attempt guessed 55 tok/claim and lost all 4,000 examples):

| rung | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|
| n claims | 20 | 45 | 96 | 199 | 406 |
| predicted drop | 3.6% | 2.2% | 2.3% | 2.8% | 3.4% |
| train / eval | 4000/500 | 4000/500 | 4000/500 | 4000/500 | 4000/500 |

⚠ **Mid-rung leakage is WORSE in the full build than the pilot predicted** (accepted by the user
2026-07-22):

| n | pilot | full build |
|---|---|---|
| ~20 | 0.497 | **0.514** |
| ~45 | 0.306 | **0.411** |
| ~96 | 0.267 | **0.383** |

Mechanism: siblings (the hard negatives) are capped at ~9 per example (3 per gold abstract), so their
share collapses as n grows — 24% of items at 2k, 6% at 8k, **2% at 32k**. Meanwhile fillers are drawn
from a 5,500-abstract two-domain pool and are therefore *less* similar to each other than in the
small pilot, which makes the gold pair stand out more. Cross-domain diversity made the task
lexically easier, via the filler pool rather than the topical-pruning route anticipated; the 85%
within-domain filler bias (`--same-domain-frac`) did not prevent it.

Paraphrase distractors remain the fix and would scale with n (unlike siblings), but were explicitly
deferred. Keep this in mind before drawing conclusions from the 4k/8k rungs.

## arXiv specifics

- `--source arxiv`; abstracts are shorter (median ~109 words), so `--min-words 150` keeps only ~26%.
- Category mix is physics/math-heavy (cond-mat, astro-ph, math, hep-*). This is physical-sciences
  breadth, NOT social science or humanities — OpenAlex would be needed for that (repo blocker B2).
- Domain profiles (`DOMAIN_PROFILES`) give each source its own field name, assertion definition,
  evidence kinds and subject-fixing examples. Forcing the clinical schema (effect sizes, cohorts,
  p-values) onto theory abstracts makes the model invent evidence.
- ⚠ **LaTeX breaks JSON parsing.** `$p\bar{p}$`, `$\sqrt{s}$`, `$OSp(2\Omega/12,R)$` are illegal
  JSON escapes: **301 of 302 claim failures at the full build were this, ~11% of arXiv abstracts and
  ~0% of PubMed**. Fixed by `repair_json_escapes()`. The affected responses are CACHED, so re-running
  the claims phase recovers them for zero LLM calls (top-up not yet applied to the shipped build).

## Traps hit while wiring this up (continued)

- **A patch that reported success but did nothing.** `DOMAIN_PROFILES`/`profile()` were never
  inserted while all five call sites were, leaving the module importable but broken at runtime,
  because the patch script printed "patched" without asserting its anchor matched. Regression test:
  `debug/pubmed_multiclaim/test_prompts.py` formats both prompts for every domain.
- **NFS lag between login node and compute node.** A file written from radagast was briefly invisible
  on sneetches, surfacing as a confusing `ImportError`. Re-run before believing it.
- **cubbins is out of disk** (`/data` 100%, `/tmp` 100%). nvcc/flashinfer JIT then fails with
  "No space left on device" buried under WARNING lines while vLLM appears hung at "Padding mamba page
  size". Point `HOME`, the four JIT cache dirs, AND `TMPDIR` at `/var/tmp`. Deleting 1.74T of
  intermediate checkpoints did NOT free space — ZFS snapshots retained it (`USEDSNAP` 1.74T).
