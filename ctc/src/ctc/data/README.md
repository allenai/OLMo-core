# `ctc.data` — generating task data

JSONL in, JSONL out. Tokenizing into olmo-core training shards is *not* here (it writes olmo-core's
format, so it lives on the training side in `src/scripts/ctc/`). Task JSONL is the boundary.

## One command per task

```bash
ctc-data list                                              # what exists, what it takes
ctc-data build --task <task> --out DIR                     # train + the full eval ladder
ctc-data build --task <task> --split eval --out DIR        # just the ladder
ctc-data audit --task <task> --dir DIR                     # re-check data already written
```

`--task` takes the same names `ctc-eval --task` does, so a build and a results row always match.

| task | graded by | corpus | one command |
|---|---|---|---|
| **the five in-distribution ladders** | | | |
| `contradiction` | contradiction | PubMed (PubMedQA) + mined claim pairs | `ctc-data build --task contradiction -C pairs_path=PAIRS.jsonl --out DIR` |
| `nq` | retrieval | NQ-open + `wikipedia-dpr-100w` BM25 + CE | `ctc-data build --task nq --out DIR` |
| `outlier` | outlier | wiki100w article pool (pickle) | `ctc-data build --task outlier -C cache=POOL.pkl --out DIR` |
| `rerank` | rerank | MS MARCO + SBERT hard negatives + CE scores | `ctc-data build --task rerank --out DIR` |
| `oolong` | oolong | `oolongbench/oolong-synth` + a tokenizer | `ctc-data build --task oolong --out DIR` |
| **further in-distribution corpora**, graded by a spec that already exists | | | |
| `hotpotqa` | retrieval | HotpotQA `distractor` (bridge) — 2 gold/question, the benchmark's own distractors as hard negatives, CE-ranked | `ctc-data build --task hotpotqa --out DIR` |
| `absence` | absence | Project Gutenberg (`sedthh/gutenberg_english`) — a window of N sentences, K deleted in a second copy. Needs the punkt model; **rungs are built independently**, see below | `ctc-data build --task absence --out DIR` |
| `xabsence` | xabsence | PubMed claim twins. Default `mode=exact` (the suite's declared construction: the twin is a byte-identical copy — no model, and the pool grows to every claim sentence PubMedQA supplies). The semantic variant stays behind `-C mode=paraphrase -C base_url=...` / `-C pool_path=...` | `ctc-data build --task xabsence --out DIR` |
| `reorder` | reorder | Project Gutenberg — N consecutive ~100-word passages of one book, shuffled. Same corpus as `absence`; **rungs are built independently**, see below | `ctc-data build --task reorder --out DIR` |
| `qdmatch_nq` / `qdmatch_hpqa` | qdmatch | A *transform*, not a corpus: it pools prepared retrieval queries, so `-C path=RETRIEVAL.jsonl` reads a built unified-retrieval file and needs no network at all | `ctc-data build --task qdmatch_nq -C path=nq_train_*.jsonl --out DIR` |
| `grouping_labeled` | grouping_labeled | OpenAlex **compact** JSONL (the ~300 GB works snapshot is not fetched here); pass a second, year-restricted file as `eval_path` or the temporal split is too thin at coarse levels | `ctc-data build --task grouping_labeled -C path=COMPACT.jsonl -C eval_path=EVAL.jsonl --out DIR` |
| **held-out-corpus ladders** — suite rows; since 2026-08-20 they train in-domain like every other row (one model per task × arm). Their older role as OOD probes of the 5-task *mixed* models survives as protocol (never feed them to a mix you will score them on), not as a build refusal | | | |
| `fiqa` | retrieval | BEIR FiQA + BM25 + CE | `ctc-data build --task fiqa --out DIR` |
| `scifact` | retrieval | BEIR SciFact + BM25 | `ctc-data build --task scifact --out DIR` |
| `outlier_review` | outlier | Amazon-Reviews-2023 | `ctc-data build --task outlier_review --out DIR` |
| `contra_fever` | contradiction | FEVER (`copenlu/fever_gold_evidence`) | `ctc-data build --task contra_fever --split eval --out DIR` |
| **pure synthetic** — no corpus, no network | | | |
| `cycle` | cycle | — | `ctc-data build --task cycle --out DIR` |
| `groups4` | groups4 | — | `ctc-data build --task groups4 --out DIR` |
| `mathmatch` | mathmatch | — | `ctc-data build --task mathmatch --out DIR` |
| `textgroups` | textgroups | — | `ctc-data build --task textgroups --out DIR` |

A build writes `DIR/<task>/train.jsonl` plus one `DIR/<task>/eval_<rung>.jsonl` per rung, and
**refuses to write if the audit fails** (`--force` overrides, and says so in the output).

Corpus loading needs the extras: `pip install './ctc[sources]'` for the HF datasets,
`./ctc[gen]` for the cross-encoder and the OOLONG tokenizer, plus `pyserini` for anything that
mines BM25 negatives. A bare install still builds all four synthetic tasks and grades everything.
**Or skip the extras entirely and build from a seed pool — see the next section.**

## Seed pools: any build in about a minute, on a bare install

Everything a build can need a GPU, a Lucene index, an LLM endpoint or a multi-gigabyte download
*for* happens inside the corpus loader; everything after the pool — gold placement, distractor
draws, rung laddering, the audit — is pure Python and fast. A **seed pool** is that loader's
output, serialized (`ctc.data.seeds`, gzipped two-line JSONL, explicit per-pool codecs — loading
one executes nothing but `json.loads` and whitelisted constructors). Build from one and the
expensive half has already happened:

```bash
ctc-data build --task nq --pool auto --out DIR          # fetch the published pool from the Hub
ctc-data build --task nq --pool nq.seed.jsonl.gz --out DIR       # or a local file
ctc-data pool export --task nq --out seeds/             # publish side: the expensive load, once
ctc-data pool info seeds/nq.seed.jsonl.gz               # ladder + provenance header
```

`--pool auto` fetches `<task>.seed.jsonl.gz` from the HF dataset repo named in
`ctc.data.seeds.DEFAULT_REPO` (override with `$CTC_SEED_POOL_REPO`) and caches it locally, so
repeat builds are offline. Three properties worth knowing:

- **A seeded build is THE SAME build.** The pool is everything a generator reads, so the same
  `(--seed, config)` produces identical examples from the live loader and from the file — the
  tests assert example equality per ladder, not just schema validity.
- **The rung ladder stays open-ended.** The seed fixes the corpus, not the ladder: any rung label
  (`64k`, `1m`, `10m`, ...) still works, subject to the same `CEILINGS`/supply bounds as a live
  build. A 20k-example train set or a laddered eval at any scale is minutes of pure assembly.
- **Corpus `-C` parameters are refused alongside `--pool`** — they were consumed at export time
  and accepting one here would label the output as built with a setting that had no effect.
  Build-side `-C` parameters (`num_pairs`, `num_docs`, ...) still apply.
- **The published contradiction pool holds 60,342 pairs**, which at the default `num_pairs=3`
  supplies ~18k train examples; `--train 20000` hits the rejection limit at the last rung. Pass
  `--train 18000` (measured: 18k train + the 5-rung eval ladder in ~4 minutes; an eval-only
  ladder in seconds).

The seed file records the ladder it was exported for, and `build` refuses a mismatch (an `nq`
pool fed to the `fiqa` ladder would build plausible data for the wrong ladder). Export scripts
and provenance for the published pools live in `debug/ctc_seed_pools/`.

### Changing a parameter

`-C KEY=VALUE`, repeatable, routed automatically to the generator or to the corpus loader.
`ctc-data list` prints both sets per task. A key neither side accepts is an error, not a shrug —
silently ignoring a typo builds data at the default size and labels it as what was asked for.

```bash
ctc-data build --task contradiction -C num_pairs=5 -C num_abstracts=50000 --out DIR
ctc-data build --task nq -C hard_frac=0.1 -C ce_filter=true --out DIR
```

## What the layer is made of

```
ctc/data/
  generators/base.py   the registry: ladder name -> Generator (task, corpus, build_example, ...)
  build.py             train + eval ladder, the contamination guards, the rung loop
  ladders.py           rung label -> documents (or, for oolong, tokens) per example
  audit.py             integrity + shortcut checks; build refuses to write past a failure
  schema.py            one Example constructor and the first real validator this pipeline has had
  gold.py              place_gold / remap, with `base` a REQUIRED argument
  io.py                load_jsonl / save_jsonl -- one implementation, explicit utf-8
  sources/             one module per corpus; the ONLY code that touches the network
  llm.py               stdlib chat client, for contradiction's pair mining
```

A generator declares two things: where its raw material comes from, and how to build **one**
example from a seeded RNG. Everything above that — how many, which rungs, train vs eval,
deduplication, auditing — is shared, because those are the decisions that must not be re-litigated
per task. The pre-migration tree let each generator own its `main()` and ended up with five
train/eval splitters using two different eval fractions and two different roundings.

## Rungs beyond 32k — up to 10M+ tokens per example

The calibrated table stops at 32k, but the ladder does not: **any parseable rung label works**
(`64k`, `256k`, `1m`, `10m`, ...). A label past the table resolves its document count by
extrapolating the least-squares line through the task's own calibrated rows — the same
calibration that produced the shipped 64k–1M suite ladders (the fit gives contradiction 1528
documents at 64k against the independently shipped file's 1525).

```bash
# a 5-rung ultra-long eval ladder, one command, ~a minute for a synthetic task
ctc-data build --task cycle --split eval --rungs 64k,256k,1m,4m,10m \
    --eval-size 125 --allow-small-eval --out /data/ctc/xlong
```

Four things to know at this scale:

- **Extrapolated rungs are flagged in the build report** and have never been measured against the
  tokenizer at that length. Measure the built prompts before quoting one as a context length.
- **500 examples of a 10M-token corpus is ~65 GB of JSONL per rung.** The shipped suite holds 125
  examples at ≥256k for exactly this reason; `--eval-size 125 --allow-small-eval` is the intended
  spelling, and the size + error bar must follow every number quoted from such a file.
- **Corpora run out, and the failure is loud.** `ladders.CEILINGS` refuses rungs the corpus
  arithmetic provably cannot honour (`qdmatch_hpqa` past 256k, `scifact` past 1m, `strmatch` past
  48k — its frozen vocabulary caps ~1.9k documents). Tasks in `ladders.SUPPLY_BOUNDED` (absence,
  reorder, rerank, ...) are bounded by what their corpus happens to contain — a single book's
  length, the per-query scored fill — and `ctc.data.supply` now refuses those up front too, from
  the **loaded pool's own arithmetic** ("the 10m rung needs 138,425 per example; this pool
  supplies at most 1,367 sentences in the longest prose run"), in milliseconds instead of a
  rejection-limit error minutes into the draw. The bound is generous, so a pass only means "not
  provably hopeless"; oolong is exempt because it under-fills rather than fails. The four pure synthetics (`cycle`, `groups4`, `mathmatch`, `textgroups`) scale
  arbitrarily; all are O(N) per example (~10–20 s per 10M-token example).
- **Two tasks' answers grow with the rung** (`reorder`, `grouping_labeled`) — see below; at 1m a
  reorder target alone is ~30k tokens, which is a decode-budget problem before it is a data one.

## Things worth knowing before you build

**Rung labels are token budgets; the table is per task.** A contradiction claim is ~43 tokens and a
BEIR SciFact abstract ~365, so the same "8k" is 187 documents for one and 21 for the other. Rows
marked `estimated` in `ladders.CALIBRATION` come from an offline per-document estimate — re-measure
before quoting one as a context length. Contradiction's row is the *corrected* one (44/92/187/379/762);
the pre-migration ladder was fit against a filler pool that turned out to be 92–99.6 % FEVER/wiki
rather than PubMed and overshoots every rung by ~1.8×.

**Eval ladders are nested, and a ladder is one build, not an accretion of files.** The shrink is
chained — each rung drops distractors from the next-longer rung's rows, which is what makes every
rung's documents a subset of the next one's (the audit checks it). The chain means the rung *set*
is part of the draw: building `2k,8k` and building `2k,4k,8k` give different `2k` files. Build the
whole ladder you want in one command; don't extend it file by file.

**Four tasks reach nesting differently.** Most shrink one canonical set
built at the longest rung. `outlier` cannot — dropping random distractors can shrink a majority
topic below the outlier count, and then the question has two correct answers and one label — so it
builds every rung of a row at once, fixing the outlier and growing the majority. `xabsence` cannot
either — dropping half of a matched pair leaves its partner unmatched, i.e. a correct answer the
label does not list — so it drops whole *pairs* instead, which is safe. `oolong` cannot nest at
all, because its gold is recomputed over whichever items were drawn; its rungs are built
independently and both the build report and the audit say so. **`absence` is the same case**: its
second version is rendered text inside `queries[0]`, so it is a function of the whole corpus and no
resize survives it — its rung-to-rung deltas carry eval-set resampling noise. So are `reorder` and
`grouping_labeled`, for one shared reason: **their gold covers every document**, a permutation in
one case and a partition in the other, so there is no distractor for the shrink to drop and a
shorter rung is a different answer, not a smaller one. `qdmatch` *does* nest — but only in the
`separate` layout, which is what keeps every gold pair's query id below its document id, and hence
what makes the shrink's within-group sort a no-op instead of a pair-swapper.

**`contra_fever` refuses to produce training data.** It is not a suite row; it exists solely to
probe the 5-task mixed models on an unseen corpus, and that refusal is not a warning: by the time
a warning is noticed the checkpoint is trained and the OOD column means nothing.

**NQ's defaults are deliberately not the old ones.** `hard_frac=0.1` with the CE gold filter **on**.
The pre-migration generator defaulted to 1.0 with the filter off, which silently reproduced the
retired 98 %-hard pipeline; every current NQ number was measured on the 10 % + CE pipeline.

**Contradiction needs a model once.** Its gold pairs are `(real PubMed sentence, model-written
sentence that contradicts it)`. Mine them once with `-C base_url=...`, keep the resulting JSONL, and
pass `-C pairs_path=...` on every later build — the pairs are data in their own right and the
build is then exactly reproducible.

**Two tasks' answers grow with the rung, and both budgets are sized against a measurement.**
`reorder`'s target is a permutation of *n* ids (~4.5 Qwen3 tokens each) and `grouping_labeled`'s is
one labelled group per cluster, which at the finest concept level is nearly one per document. Both
overran the pre-migration `max_new_tokens` at their 32k rung, and the resulting truncation parses as
*no answer at all* — a uniform zero at the longest rung, which reads exactly like a long-context
collapse. If either ladder is lengthened, re-measure the target, not just the prompt.

**Eval sets are 500 examples.** `build_eval` refuses smaller. SciFact's real ladder is 299 (the
entire test split with qrels), which is below the floor and must be quoted with its size and error
bar inline.

## Testing

`ctc/tests/data/` runs with no GPU, no network and no weka. Corpus-backed generators are exercised
against fixture pools from `ctc/tests/fixtures/pools.py`: a pool is a plain dataclass and only its
loader touches the Hub, which is exactly what the `sources/` seam exists for. The four synthetic
generators additionally have byte-level golden fixtures captured from the pre-migration tree — a
failure there means the port changed behaviour, and the fix is the port, not the fixture.
