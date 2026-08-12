# CTC suite: data generation

Reference for the paper's data appendix. Every task in the corpus-reasoning suite, how its examples
are built, how gold is guaranteed unique, and what prevents a model from scoring well without doing
the task.

## 1. Common structure

Every task emits the same JSON object: a list of `documents`, a list of `queries`, a list of
`answers`, and a gold structure indexing into the documents. A task is therefore fully described by
three choices — what a document is, what the query asks, and what shape the gold takes — and the
prompt template, parser and metric follow from the task name alone.

**Gold shapes.** Four, across the suite: a *set of document ids* (retrieval, outlier, absence,
grouping), a *set of pairs* (contradiction, mathmatch, strmatch, redundancy), a *set of variable-size
groups* (cycle, groups4, textgroups), and a *permutation* (reorder). Two index bases are in use and
each is a per-task property: the pair and group families are 1-based, matching the item numbers the
prompt renders; the retrieval family is 0-based. This is declared once per task and checked against
the generated data, because an index-base disagreement is silent — correct answers score zero,
uniformly, and read as a modelling result.

**The context ladder.** Each task has a ladder of context budgets (2k / 4k / 8k / 16k / 32k tokens),
realised as a per-task document count, because documents differ by an order of magnitude in length
(a contradiction claim is ~42 tokens; a BEIR SciFact abstract is ~365). The rung→count table is
calibrated per task and is the paper's x-axis.

**Eval sets are nested, not regenerated.** One canonical set of 500 examples is built at the longest
rung, and shorter rungs are derived by *removing distractors in place*, keeping every gold document
and preserving relative order. So all five rungs grade the same 500 questions with identical gold
text, and each rung's documents are a subset of the next one's. Regenerating per rung would confound
"the context got longer" with "the questions changed", which is precisely the comparison being made.

**Training sets** are spread uniformly over the same document counts (20k examples, 4k per rung), so
a model is never measured in a length regime it has not seen.

## 2. Per-task construction

Documents-per-example at 2k / 4k / 8k / 16k / 32k.

| Task | Documents are | Gold | Uniqueness guarantee | n per rung |
|---|---|---|---|---|
| **contradiction** | PubMed claim sentences | the contradicting pair | one LLM-validated perturbed claim per example; fillers drawn from the same PubMed pool | 44/92/187/379/762 |
| **NQ** | Wikipedia 100-word passages | the answer-bearing passage | NQ-open gold; 10% BM25 hard negatives, cross-encoder filtered | 11/23/48/100/200 |
| **HotpotQA** | Wikipedia paragraphs | the two supporting paragraphs | HotpotQA distractor split's own labels | 11/24/50/100/205 |
| **BEIR SciFact / FiQA** | abstracts / finance posts | the relevant document | BEIR qrels; negatives BM25-mined and CE-scored | 5/10/21/43/88 · 4/9/19/40/80 |
| **MS MARCO rerank** | passages | full relevance ranking | qrels plus cross-encoder scores; ranked by graded gain | ~15→~300 |
| **outlier** | Wikipedia passages on shared topics | the off-topic passage(s) | K topics sampled disjointly; the outlier comes from a topic no other document shares | 14/28/57/115/220 |
| **grouping (OpenAlex)** | paper abstracts | the field clusters | OpenAlex field labels; cluster count sampled capacity-aware, so grouping granularity does not covary with N | 10/21/43/88/176 |
| **oolong** | synthetic log/dialogue lines | per-question answer | from the OOLONG benchmark; length varied by a token budget rather than an item count | token knob |
| **reorder** | 100-word Gutenberg chunks | the true reading order | chunks are contiguous in one book, so the permutation is unique | 12/27/57/116/234 |
| **absence** | sentences from a Gutenberg passage | the deleted sentences | two versions of one passage differing by K sentences | ~90→~1440 |
| **xabsence** | paraphrase pairs | the unpaired item | LLM-built paraphrase pool with an overlap-rejection filter | 18/39/81/165/333 |
| **strmatch** | Wikipedia sentences | the shared-n-gram pair | one planted n-gram; every other pair checked to share none | 38/82/170/350/700 |
| **qdmatch** | queries and documents interleaved | the query→document pairing | derived from single-query retrieval data, so each pairing is already labelled | q9/q20/q42/q87/q178 |
| **mathmatch** | arithmetic expressions | pairs within tolerance X | K base values drawn >3X apart, each given a partner within X, all other values >X from everything | 48/105/220/450/900 |
| **cycle** | comparative claims ("A > B") | the claim sets forming a cycle | see §3 | 60/130/270/550/1100 |
| **groups4** | arithmetic expressions | groups of G within tolerance X | see §3 | 100/210/440/900/1800 |
| **textgroups** | short passages | triples whose feature counts sum to T | K disjoint triples planted; a distractor is admitted only if it completes no near-target triple with any pair already placed, so exactness holds at any N without a C(N,3) scan | 11/24/50/103/210 |

The four synthetic tasks (mathmatch, cycle, groups4, textgroups) need no corpus and no model: they
are generated from a seed alone, which makes them exactly reproducible and lets the gold set be
*proved* unique rather than validated.

## 3. Uniqueness in the synthetic tasks

Two constructions are worth stating in full, because the guarantee is not obvious and because the
first version of each was exploitable.

**cycle.** All entities share one random total order. Every background claim runs strictly forward in
that order, so the background alone is acyclic and any cycle must use a planted backward edge. Each
cycle's L entities occupy *consecutive* ranks, and no background claim may have both endpoints inside
one such block. A forward path from a block's bottom to its top is then trapped inside the block —
rank only increases, and no outside entity ranks between the endpoints — so the planted chain is the
unique completion of each backward edge, and there are exactly K cycles.

**groups4.** For a set of numbers, "all pairwise distances ≤ X" is equivalent to "fits inside one
window of width X". So capping each new value's within-X neighbour count at G−2 before insertion caps
every window's occupancy at G−1 afterwards, and no accidental G-clique can form. Cluster centres are
spaced >2X apart so planted groups cannot merge.

## 4. Controls against shortcuts

Both synthetic constructions above were rebuilt after measurement showed the originals were solvable
without the intended reasoning:

- **cycle** originally drew background claims from a pool disjoint from the cycle entities, which
  pinned each gold entity's claim frequency at exactly 2 while background frequency grew with N. Gold
  was recoverable as "the rarest names", and the shortcut *strengthened* with context length — the
  scaling axis was partly measuring its own artifact. Cycle entities are now full participants in
  background sampling, subject only to the same-block exclusion the uniqueness proof requires.
- **groups4** originally placed every distractor >X from everything, so a single close pair
  identified gold at any N without ever finding a G-clique. Distractors may now cluster up to G−2.

Both are now regression-tested by probes that run on every build: the gold-entity frequency gap, and
whether the closest pair is gold. Two generic probes also run for every task — whether gold
concentrates in any decile of the document list, and whether gold is recoverable as the longest or
shortest document. (mathmatch is a deliberate exception: its distractors *are* isolated, so at K=1 the
closest pair is gold by construction. Its difficulty is intended to come from N and from evaluating
the arithmetic, and this is recorded rather than treated as a defect.)

## 5. Contamination controls

- **Train and eval are drawn from independent, separately-keyed RNG streams**, so the eval set at a
  given seed does not change when the training-set size changes.
- **Two levels of overlap are rejected at build time.** A shared *example* is duplication; a shared
  *gold fingerprint* — the same claim pair or cycle with different filler around it — is the subtler
  case, since the corpora differ and every surface check passes while the eval question is still in
  training. Both are rejected during generation and re-checked in the audit, which reports how many
  draws were discarded.
- **Filler provenance is pinned per task.** A domain-agnostic glob previously pulled FEVER and
  Wikipedia claims into PubMed contradiction evals (92% of fillers at 2k, 99.6% at 32k) against gold
  that was entirely PubMed, turning "find the contradicting pair" into "find the biomedical
  sentences". Note this *depressed* scores rather than flattering them — it was a train/eval domain
  shift — and worst at the long end: the 32k rung went 0.335 → 0.559 once rebuilt. Filler pools are
  now specified by manifest.
- **Eval sets are 500 examples.** At f1≈0.70 the binomial standard error is ±0.021 and at f1≈0.95 it
  is ±0.010; run-to-run seed variation adds more. Any smaller eval set is flagged inline with its
  size and error bar.

## 6. Status

Ported into the current pipeline: **13 ladders**, the names `ctc-data list` prints — the four
synthetic tasks (mathmatch, cycle, groups4, textgroups), the five in-distribution ones
(contradiction, nq, outlier, rerank, oolong) and the four held-out ones (fiqa, scifact,
outlier_review, contra_fever). Only the four synthetic ones have byte-level parity fixtures against
the pre-migration generators; the corpus-backed ones are validated structurally, because they cannot
run without their corpora. The other tasks in the table above have **no generator in this tree** —
their constructions are described from the pre-migration scripts and are unchanged by the port. See
`records/data-generator-port.md`.

Whether a rung→document-count row is an offline per-document estimate or a tokenizer measurement is
recorded per task in `CALIBRATION` (`ctc/src/ctc/data/ladders.py:71`): contradiction and
contra_fever are `measured`, oolong is `exact` (its rung value *is* the token budget), and the other
ten are `estimated` and should be re-measured before being quoted as context lengths. The
contradiction row's measured medians are 1925 / 3933 / 8052 / 16074 / 32397.
