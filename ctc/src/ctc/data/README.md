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
| **the four held-out (OOD) ladders** — eval only | | | |
| `fiqa` | retrieval | BEIR FiQA + BM25 + CE | `ctc-data build --task fiqa --split eval --out DIR` |
| `scifact` | retrieval | BEIR SciFact + BM25 | `ctc-data build --task scifact --split eval --out DIR` |
| `outlier_review` | outlier | Amazon-Reviews-2023 | `ctc-data build --task outlier_review --split eval --out DIR` |
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

## Things worth knowing before you build

**Rung labels are token budgets; the table is per task.** A contradiction claim is ~43 tokens and a
BEIR SciFact abstract ~365, so the same "8k" is 187 documents for one and 21 for the other. Rows
marked `estimated` in `ladders.CALIBRATION` come from an offline per-document estimate — re-measure
before quoting one as a context length. Contradiction's row is the *corrected* one (44/92/187/379/762);
the pre-migration ladder was fit against a filler pool that turned out to be 92–99.6 % FEVER/wiki
rather than PubMed and overshoots every rung by ~1.8×.

**Eval ladders are nested, and three tasks reach that differently.** Most shrink one canonical set
built at the longest rung. `outlier` cannot — dropping random distractors can shrink a majority
topic below the outlier count, and then the question has two correct answers and one label — so it
builds every rung of a row at once, fixing the outlier and growing the majority. `oolong` cannot
nest at all, because its gold is recomputed over whichever items were drawn; its rungs are built
independently and both the build report and the audit say so.

**The held-out ladders refuse to produce training data.** Not a warning: by the time a warning is
noticed the checkpoint is trained and the whole OOD column means nothing.

**NQ's defaults are deliberately not the old ones.** `hard_frac=0.1` with the CE gold filter **on**.
The pre-migration generator defaulted to 1.0 with the filter off, which silently reproduced the
retired 98 %-hard pipeline; every current NQ number was measured on the 10 % + CE pipeline.

**Contradiction needs a model once.** Its gold pairs are `(real PubMed sentence, model-written
sentence that contradicts it)`. Mine them once with `-C base_url=...`, keep the resulting JSONL, and
pass `-C pairs_path=...` on every later build — the pairs are data in their own right and the
build is then exactly reproducible.

**Eval sets are 500 examples.** `build_eval` refuses smaller. SciFact's real ladder is 299 (the
entire test split with qrels), which is below the floor and must be quoted with its size and error
bar inline.

## Testing

`ctc/tests/data/` runs with no GPU, no network and no weka. Corpus-backed generators are exercised
against fixture pools from `ctc/tests/fixtures/pools.py`: a pool is a plain dataclass and only its
loader touches the Hub, which is exactly what the `sources/` seam exists for. The four synthetic
generators additionally have byte-level golden fixtures captured from the pre-migration tree — a
failure there means the port changed behaviour, and the fix is the port, not the fixture.
