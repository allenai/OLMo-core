---
license: odc-by
pretty_name: CTC seed pools
---

# CTC seed pools

Serialized **corpus pools** for the CTC long-context suite's data generators
([`allenai/OLMo-core`, branch `prasann/ctc`](https://github.com/allenai/OLMo-core/tree/prasann/ctc),
pip package `ctc/`). Each file is the output of the one build step that needs heavy machinery — a
GPU cross-encoder, a pyserini/Lucene index, an LLM mining run, or a multi-gigabyte download —
captured once, so that **anyone can build train and eval data at any context scale (2k to 10M+
tokens per example) on a bare `pip install`**: no GPU, no Java, no API key.

```bash
git clone -b prasann/ctc https://github.com/allenai/OLMo-core && cd OLMo-core
pip install ./ctc

ctc-data build --task contradiction --pool auto --train 18000 --out DIR   # ~4 min: 18k train
                                                                          # + a nested 5-rung
                                                                          # 500-example eval ladder
ctc-data build --task nq --pool auto --split eval --rungs 2k,32k --out DIR    # seconds
ctc-data build --task contradiction --pool auto --split eval --rungs 64k,1m \
    --eval-size 125 --allow-small-eval --out DIR                          # rungs extrapolate
                                                                          # beyond the 32k table
```

`--pool auto` downloads `<task>.seed.jsonl.gz` from this repo (cached locally by
`huggingface_hub`; repeat builds are offline). A seeded build is **the same build**: the pool is
everything a generator reads, so identical `(--seed, config)` gives identical examples from the
live loader and from the file — asserted per ladder by the package's tests.

## Format

Gzipped two-line JSONL: a header (`format: ctc-seed-pool-v1`, the ladder the pool was exported
for, provenance) and one payload line. Loading executes nothing but `json.loads` and whitelisted
dataclass constructors — no pickle. `ctc-data build` refuses a pool exported for a different
ladder, and `ctc-data pool info FILE` prints the header.

## What each pool contains, and what it saved

| file | expensive part captured | notes |
|---|---|---|
| `contradiction.seed.jsonl.gz` | LLM-mined claim/contradiction pairs (60,342, recovered losslessly from the audited 20k train build) + PubMed filler abstracts | pairs are consumed, never reused: k=3 caps train at ~18k examples — pass `--train 18000` |
| `redundancy.seed.jsonl.gz` | LLM-mined paraphrase pairs (4,477) + LLM-judged same-abstract hard negatives + fillers | supply-bounded like contradiction: ~1.3k train examples at the default k=3 |
| `nq.seed.jsonl.gz` | BM25 hard negatives from the 21M-passage `wikipedia-dpr-100w` Lucene index + GPU cross-encoder gold filter | 10% hard-negative regime, CE filter on; 9,093 distinct queries — a 20k train build reuses queries with fresh distractor draws and says so ("pool wraps") in its report |
| `hotpotqa.seed.jsonl.gz` | GPU cross-encoder ranking of the benchmark's distractors | bridge questions, 2 gold each; 25k queries |
| `rerank.seed.jsonl.gz` | MS MARCO mined hard negatives + a cross-encoder score for **every** document (25k queries) | the graded-ordering reference. Cannot wrap: fill is pre-drawn and scored per query, so distinct examples need distinct queries |
| `fiqa.seed.jsonl.gz` / `scifact.seed.jsonl.gz` | BEIR corpus + locally-built Lucene index + CE margin filter | **eval-only** ladders; `build` refuses `--split train` |
| `outlier.seed.jsonl.gz` | full scan of the 21M-passage wiki100w index into an article pool (2.2 GB) | largest file; expect a slow first load |
| `outlier_review.seed.jsonl.gz` | Amazon-Reviews-2023 streaming sample | eval-only |
| `contra_fever.seed.jsonl.gz` | FEVER gold-evidence restructuring | eval-only |
| `oolong.seed.jsonl.gz` | OOLONG-synth pull + per-item token counts (Qwen3 tokenizer) | |
| `absence.seed.jsonl.gz` / `reorder.seed.jsonl.gz` | Project Gutenberg (~11 GB) + punkt sentence segmentation into prose runs / passage streams | |
| `grouping_labeled.seed.jsonl.gz` | OpenAlex compact projection (52k papers + 31k year-restricted eval fetch) | the ~300 GB works snapshot, pre-reduced |
| `qdmatch_nq.seed.jsonl.gz` / `qdmatch_hpqa.seed.jsonl.gz` | projected from the nq / hotpotqa pools above | |
| `xabsence.seed.jsonl.gz` | LLM-mined paraphrase-twin pool | ⚠ 659 pairs only — seeds small builds; large rungs need a bigger mining run |

The four pure-synthetic ladders (`cycle`, `groups4`, `mathmatch`, `textgroups`) need no pool —
they build from a seed integer alone.

Pools are corpus material only (no rendered prompts, no eval answers beyond the corpora's own
annotations). Underlying sources carry their own licenses: PubMedQA, Natural Questions, HotpotQA,
MS MARCO, BEIR (FiQA/SciFact), FEVER, Amazon-Reviews-2023, OOLONG-synth, Project Gutenberg,
OpenAlex, Wikipedia (DPR 100-word split).
