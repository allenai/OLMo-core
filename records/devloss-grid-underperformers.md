# Dev-loss grid — audit of every underperforming cell (2026-09-21)

Grid: `research/token_routing/devloss_grid/` (driver `ctc_devloss_grid.py`, artifact "Dev-Loss Scheme Grid"). Frozen dense
4B suite checkpoints, teacher-forced CE on the rendered gold answer, rows 16/16/8 at 2k/8k/32k (⚠ small
by design — every Δ below carries its paired SE). "Underperforms" = ΔCE vs full > 0.05 for `gold_fl20`
(gold documents whole + first_last 20 % of every other body, cent_cmean slot), `gold_fl64`, or
`gold_fl20p8` (same + a uniform 8-token real prefix per document).

Verdict codes: **(a)** construction / mapping / checkpoint bug · **(b)** artifact of the rule on that
task's document shape · **(c)** genuinely load-bearing everywhere · **(d)** small-sample noise.

Method per cell: per-row CE across schemes (`per_row` in the JSON), marker-span vs document count,
gold set actually kept (`kept_docs`), the raw row's gold/answer fields, and — where the numbers did not
explain themselves — the exact compacted token stream decoded on CPU (`inspect_construction.py`,
tiny random model, same code path as the GPU driver).

## Two findings that apply to the whole grid, not one cell

1. **`gold_fl64` was not compressing on six columns.** Its compaction is 0.94–1.00 on niah,
   strmatch, contradiction, absence, obliq and oolong (documents ≤ 64 tokens are kept whole by
   construction, markers included), so its "parity" there is trivial. `gold_fl20` is the first real
   compression test on those tasks; read the K=64 rows only on tasks with long documents
   (outlier_amzn 0.55, textgroups 0.32–0.66, fiqa 0.34–0.51, reorder 0.45–0.49, cpt80 0.30).
2. **A pure-fraction budget pools the document id on short documents.** With ~15–25-token documents,
   20 % first_last is 2+2 tokens: the stream reads `String ▮ favorable rasch` / `Document [ ▮ Froome.`
   — the `[N]`/`N:` id is gone for every pooled document. Tasks that answer with ids (niah, strmatch,
   contradiction, qdmatch, textgroups) lose the id→position map even though the gold documents are
   whole and their own ids visible; the damage grows with document count (strmatch 0.00 → 0.27 → 0.79
   at 38/170/700 docs). `fl20p8` (fixed 8-token real prefix per document, uniform across tasks) is
   the test of this explanation, and at 2k it closes every id-answer gap: `gold_fl20p8` niah 0.000 (from
   +1.07), contradiction +0.000, strmatch 0.000, qdmatch_hpqa 0.000, outlier +0.001, nq −0.001,
   msmarco −0.005 — at compaction 0.34–0.65 (docs are short, so 8 real tokens is a large fraction on
   those columns). 8k/32k cells were still landing when this was written; see the grid artifact.

## Cells

### niah @2k/8k/32k — gold_fl20 +1.07 / +1.60 / +2.86 — **(b)**
Gold index 0-based and correct (docs[7] is the contradicting statement; 40 spans / 40 docs); the gold
document is kept whole (`⟨Document [8]: Death occurred within 2 days …⟩`). Per-row CE is spread
(0.4–4.0), not a few catastrophes. Neighbours render as `Document ▮ Side.` — ids pooled. `gold_fl64`
= 0.000 but at compaction 0.99 (nothing pooled). Fix confirmed: `gold_fl20p8` @2k = 0.000 at ×0.62.

### strmatch @8k/32k — gold_fl20 +0.27 / +0.79 (2k: 0.00) — **(b)**
10-word strings, gold pairs share a 3-word run; gold 1-based nested → 6 docs kept whole (verified:
`String 59: dns definitively …` real, answer `[[59, 146], [64, 153], [94, 129]]`). Pooled rows are
`String ▮ oxford buildup` — id gone. CE rises with the number of pooled strings (38 → 170 → 700),
exactly the id-map failure; `gold_fl64` = 0 is trivial (compaction 1.00). `gold_fl20p8` @2k = 0.000 at
×0.65; 8k/32k pending.

### contradiction @32k — gold_fl20 +0.14 ± 0.08 (8 rows; 2k/8k ≤ 0.02) — **(b)**, mild
762 claims of ~20 words; 6 gold whole. Per-row 0.01–0.55 vs full ~0; same id-map mechanism
(answers are claim-id pairs), only visible at the 32k document count. `gold_fl64` trivial (0.94).
Decoded row 2 (762 spans / 762 docs): all 6 gold claims real with their ids (`⟨Claim 131: …⟩`, answer
`[[131, 251], [249, 401], [254, 606]]`), so the mapping is right. `gold_fl20p8` @2k +0.000; 32k pending.

### textgroups @8k/32k — gold_fl20 +0.08 / +0.16; gold_fl64 +0.13 / +0.24 — **(c)**
"Find every group of 3 passages whose value (adjective count / count of 'with' …) is equal": the value
of EVERY passage must be computed, so any per-document budget removes evidence; gold whole only gives
the answer's own passages. Gold 1-based nested verified (doc texts match the counted feature). Δ is
uniform across rows (0.3–0.4 vs 0.25–0.35 full), not noise. gold_fl64 being worse than gold_fl20 is
inside the SEs (0.05/0.04 at 32k). Not fixable by a selection rule; a trained reader is the only route.

### absence @2k/8k — gold_fl20 +0.87 / +1.24; gold_fl64 8k +0.05 (0.98 compaction, 3 rows) — **(c)**
Version-A-vs-version-B sentence diff: every sentence must be checked against the other version, so
pooling 80 % of each sentence destroys the comparison. Gold (0-based) verified, 3 gold whole, per-row
uniform 0.2–2.2. `fl64` ≈ full because sentences are ≤ 64 tokens (trivial). Only a whole-document
rule can work here (rand33 fails too: +3.7), i.e. the task is O(N) load-bearing by construction.

### reorder @2k/8k — fl20 = gold_fl20 +0.46 / +0.73; fl64 +0.28 / +0.48 — **(c)**
No gold subset (order IS the answer, every chunk load-bearing); uniform per-row 0.8 vs 0.1. Expected.

### outlier_amzn @2k/8k/32k — gold_fl20 +0.15 / +0.15 / +0.14; gold_fl64 +0.06 / +0.10 — **(c)**, mild
Reviews have no explicit rating/category field — the attribute is inferred from the review text, and
the *majority* attribute must be established from the non-outlier documents, which are the pooled
ones. Gold (0-based, the 4 outliers) verified against the answer `Outliers: [1], [2], [12], [18]`;
20 spans / 20 docs; kept 4/20. Per-row: rows where full is already high stay high (0.35→0.59,
0.45→0.67) plus a few 0→0.4. `gold_fl20p8` @2k improves it to +0.065 (headers/opening sentence carry
sentiment). Genuine but small.

### rerank @32k — gold_fl20 +0.14 ± 0.01; gold_fl64 +0.07 — **(c)**, uniform
Target is a top-10 ranking of 375 passages (full CE 1.1 — the model is unsure even with full text);
gold = qrel ∪ CE-positive docs (1–4) whole. Ranking the other 365 needs their text. Per-row Δ is
identical on every row (1.25 vs 1.10): a clean, small, real cost.

### fiqa @32k — gold_fl20 +0.11 ± 0.05; gold_fl64 +0.15 ± 0.10 (8 rows) — **(d)** with a (c) tail
Per-row: 5 of 8 rows within ±0.05 of full; rows 2 and 4 carry the Δ (0.23→0.58, 1.69→2.0) and
full CE itself is 0–1.7 (fiqa is hard). gold_fl64 > gold_fl20 is inside noise. 2k/8k are at parity
(+0.05 / −0.09). fiqa's "7 spans for 8 documents" warning (one empty document in r2k row 29) is a
data quirk the driver drops; 0 rows dropped at 32k.

### obliq @32k — gold_fl20 +0.08 ± 0.02 — **(d)/(a)**: checkpoint mismatch
Uniform +0.05–0.1 per row on a baseline of CE 0.9–1.0: the checkpoint is `ctc-4b-obliq_retrieval-full`
(the VOID retrieval build) scored on the `obliq_twitter` row, so the model is already at chance-ish
on full text. The Δ is real but the column measures the wrong model; needs the twitter-trained
checkpoint (weka) before it says anything about selection.

### oolong @2k — gold_fl20 = fl20 +0.07 ± 0.05 — **(c)/(d)**
No gold (degenerate → fl20). 14 of 16 rows identical to full; rows 3 and 10 (0.11→0.86, 0.19→0.65)
are aggregate questions whose fields sit in the pooled 80 % of the lines. 8k/32k at parity
(+0.04 / 0.00). Noise-level at this eval size, mechanism (c).

### xabsence @2k/8k/32k — Δ −0.26 … +0.11, full CE 2.1–5.9 — **(a)**: wrong checkpoint
`k0 ≈ fl20 ≈ full` on every row: the model ignores the context entirely, and full-text CE of 2.5 on an
answer like `Missing: [7], [8], [19]` means it cannot do the task at all. The only local 4B xabsence
checkpoint (`ctc-xabsence-full-4b`, step1500, 2026-07-24, shard `shards_xabsence_combined`) predates
the EXACT ladder and is the PARAPHRASE-era run ([[xabsence-exact-vs-paraphrase-split]]); the staged
rows are EXACT (discriminator: 16/16 rows, `len(A&B) == num_pairs`). Rendering and gold (0-based
unmatched docs, 3 kept whole) are correct. **Column void** until the EXACT-trained checkpoint is
pulled from weka; every xabsence cell is noise around an incapable model.

### cpt80 @2k/8k/32k — fl20 (= gold_fl20) +0.14 / +0.07 / +0.08; fl64 +0.14 / +0.09 / +0.09 — **(c)**
Construction re-read and correct: first 80 % of the document in 512-token marker-wrapped blocks,
loss on the last 20 % (from tail[1], since tail[0] is predicted from the last `<|doc_end|>` that every
compaction pools); the tail itself is plain text. Per-row Δ is uniform (+0.05 … +0.2, k0 > fl > full on
every row), rand33 (whole random blocks) ≈ fl20 at similar compaction. The base has repaired marker
embeddings but never trained with them, so `full` also carries a marker every 512 tokens — the same
for every scheme, so paired Δ is fair, but absolute CE is slightly inflated. Real cost of ~0.08 nats
at ×0.3 for a frozen base; the trained CPT arms (separate experiment) are the right test.

## Summary table

| cell | Δ (gold_fl20) | verdict | one line |
|---|---|---|---|
| niah 2k/8k/32k | +1.07/+1.60/+2.86 | b | ids pooled on one-sentence docs; gold whole & correct |
| strmatch 8k/32k | +0.27/+0.79 | b | same; grows with pooled-string count; fl64 trivial (×1.00) |
| contradiction 32k | +0.14 | b (mild) | same at 762 claims; fl64 trivial (×0.94) |
| textgroups 8k/32k | +0.08/+0.16 | c | every passage's value is needed |
| absence 2k/8k | +0.87/+1.24 | c | sentence diff needs every sentence |
| reorder 2k/8k | +0.46/+0.73 | c | no gold; order is the answer |
| outlier_amzn all | +0.15 | c (mild) | majority attribute lives in pooled reviews; p8 halves it |
| rerank 32k | +0.14 | c (mild) | ranking 365 non-gold passages |
| fiqa 32k | +0.11±0.05 | d | 2 of 8 rows; 2k/8k parity |
| obliq 32k | +0.08 | d/a | retrieval-trained ckpt on the twitter row |
| oolong 2k | +0.07±0.05 | c/d | 2 of 16 rows; 8k/32k parity |
| xabsence all | ±0.2 | **a** | PARAPHRASE checkpoint on EXACT rows; column void |
| cpt80 all | +0.07–0.14 | c | frozen base; uniform per row; construction verified |

No driver bug was found; no cell needs a re-run for correctness. Two columns need a different
checkpoint (xabsence EXACT, obliq twitter) before they are interpretable.
