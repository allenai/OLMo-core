# Can a dense model read a POOLED outlier out of its soft-token slot? (eval-side probe)

**Date** 2026-09-14 · **Status** ANSWERED — canonical ds64 2k run complete, replicated on a second checkpoint + corpus (§4) · **Branch** `prasann/landmark`
**Driver** `debug/pooled_kv/outlier_probe/outlier_slot_probe.py`

## 1. The question this answers

The gold-blind soft-token outlier arms split in two:

* `kvgb50` (gold-blind, keep 1/2, **no** header) reaches dense parity — 0.871 / 0.188 / 0.066 at
  2k / 8k / 16k, against dense 0.911 / 0.483 / 0.153 (eval_size 500/rung).
* `xhdr00/17/33` (gold-blind **+** `Document [N]:` headers real) sit **exactly on the `k/n`
  uniform-guess floor** at every rung, every keep, every budget.

`debug/ds64/xhdr_collapse_diagnosis.md` establishes there is no code bug and argues the collapse is
a *shortcut*: header-real makes every document id copyable, so "emit three ids from the visible
list" is already loss-optimal and the gradient never has to read the slots. That argument leaves the
decisive question open:

* **(A) optimisation** — the slot *does* carry enough signal to name the odd document out, and
  gradient descent simply found a cheaper policy first. Then the fix is training-side.
* **(B) hard limit** — the slot does *not* carry that signal, the header-real target really is
  unjustifiable, and no training trick recovers it; only a richer slot would.

Nothing about the training runs can separate these, because in both worlds the training loss is
flat. A **frozen dense model** can: if a model that was never trained on slots can still pick out a
*pooled* outlier from its mean-embedding slot, the information is there.

## 2. Constructions

All on the ds64 outlier rung files with a **ds64 dense (full-attention-trained) checkpoint held
fixed**; only the slot construction changes.

| # | name | keep set | header real | training arm it stands for |
|---|---|---|---|---|
| 1 | `full` | every doc real | — | dense reference |
| 2 | `goldonly` | gold docs only | no | the 2026-09-08 "parity gold-only" point |
| 3 | `gb50` | gold-blind, p = 1/2 | no | **kvgb50** (the parity arm) |
| 4 | `gb50h` | gold-blind, p = 1/2 | yes | the would-be **xhdr50** / `xh2k50` |
| 5 | `gb17h` | gold-blind, p = 1/6 | yes | **xhdr17** |
| 6 | `gb00h` | keep 0 | yes | **xhdr00** |
| 7 | `gb00` | keep 0 | no | pure slots (ids not in context at all) |
| 8 | `gb00h_swap` | keep 0 | yes | **CONTROL**: gold docs' slot vectors swapped with random non-gold slots' |

Fidelity notes:

* the header path is the trainer's own `mark_doc_headers_free` reached by setting
  `header_stop_id = 5491`, `header_stop_count = 1` on the model's pooled-soft-token config — not the
  older probe's `--prefix-real` chunk-id patch, so it is bit-for-bit `--st-header-stop-id 5491
  --st-header-stop-count 1`;
* the gold-blind keep is the trainer's own draw (`resolve_keep_docs` with `holder=None`), i.e. the
  `--st-gold-blind --st-keep-prob p` path;
* gold comes straight from the sidecar. It is **not** `make_fingerprint_keep_docs_fn(..., n_random_frac=0.0)`:
  that helper still forces `max(1, round(frac * n_non_gold))` == **one random non-gold document real**,
  so the 2026-09-08 "gold-only" probe point was really *gold + 1 random*. `goldonly` here is exactly
  the gold documents.

## 3. Metrics

Answer-position CE, top-1 agreement with `full`, KL(full ‖ soft), teacher-forced exact match, and
set-F1 / exact-set-match over the k document ids from **free greedy generation**. Two metrics carry
the argument:

* **`R@gold_pooled`** — over every gold document that was POOLED in its row, the fraction the
  model's generated id set recovered. This is literally "can the frozen dense model pick out an
  outlier it can only see as one mean-embedding vector?".
* **`tf_id1`** — teacher-forced correctness of the FIRST id. `tf_f1` is leaky (teacher forcing puts
  the earlier true ids in the prefix); the first id is predicted from a prefix with no id in it.

`pred_pooled` vs `base_pooled` says whether the model's picks are biased toward the real documents
(the mixture effect) or track the pooled/real base rate.

## 4. Results

### Run B — the canonical pairing (COMPLETE): `ds64-outlier-dense-u64M` on the ds64 2k rung

`outlier_lengthmix/eval_rungs/outlier/rung_2048.jsonl`, **eval_size = 240** (⚠ < 500; binomial SE
≈ 0.032 at f1 ≈ 0.5), free generation on the first 64 rows (192 gold documents). 13.4
documents/row, k = 3, so the **uniform-guess floor is `k/n` = 0.224** — the same floor the xhdr
training arms sat on.

| condition | CE | CE(digits) | top1=full | KL | tf_f1 | tf_id1 | **genF1** | genEM | **R@gold_pooled** | **R@gold_real** | pred_pooled | base_pooled | compaction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `full` | 0.010 | 0.035 | 1.000 | 0.000 | 0.991 | 0.975 | **0.979** | 0.97 | — | 0.979 (192) | — | 0.00 | 1.000 |
| `goldonly` | 0.194 | 0.595 | 0.947 | 0.184 | 0.737 | 0.237 | 0.318 | 0.25 | — | 0.318 (192) | 0.682 | 0.78 | 0.284 |
| `gb50` (=kvgb50) | 0.429 | 1.605 | 0.892 | 0.420 | 0.549 | 0.375 | 0.453 | 0.09 | **0.126** (95) | **0.732** (97) | 0.224 | 0.49 | 0.542 |
| `gb50h` (=xh2k50) | 0.405 | 1.523 | 0.895 | 0.395 | 0.558 | 0.454 | 0.555 | 0.23 | **0.264** (91) | **0.812** (101) | 0.429 | 0.49 | 0.563 |
| **`gb50h_swap`** (control) | 0.404 | 1.519 | 0.897 | 0.394 | 0.568 | 0.450 | 0.562 | 0.23 | **0.242** (91) | 0.842 (101) | 0.421 | 0.49 | 0.563 |
| `gb17h` (=xhdr17) | 0.504 | 1.869 | 0.844 | 0.495 | 0.286 | 0.237 | 0.323 | 0.00 | **0.227** (154) | **0.711** (38) | 0.727 | 0.83 | 0.256 |
| `gb00h` (=xhdr00) | 0.462 | 1.698 | 0.834 | 0.454 | 0.204 | 0.233 | 0.229 | 0.02 | **0.229** (192) | — | 1.000 | 1.00 | 0.105 |
| `gb00` (pure slots) | 0.587 | 1.889 | 0.834 | 0.579 | 0.205 | 0.233 | 0.276 | 0.02 | 0.276 (192) | — | 1.000 | 1.00 | 0.071 |
| **`gb00h_swap`** (control) | 0.457 | 1.677 | 0.837 | 0.448 | 0.217 | 0.233 | 0.229 | 0.02 | **0.229** (192) | — | 1.000 | 1.00 | 0.105 |

This is the decisive table.

* **Every header-real construction lands on the guess floor for pooled gold**: `gb00h` 0.229,
  `gb17h` 0.227, `gb50h` 0.264, against `k/n` = **0.224** (SE 0.030 at 192 documents, 0.046 at 91).
  This is exactly where the `xhdr00/17/33` *training* arms landed, reproduced here in a model that
  was never trained that way — the floor is a property of the construction, not of the optimiser.
* **The swap controls change nothing.** `gb00h` → `gb00h_swap`: 0.229 → 0.229 on pooled recall, and
  CE 0.462 → 0.457, CE(digits) 1.698 → 1.677, KL 0.454 → 0.448, tf_id1 0.233 → 0.233.
  `gb50h` → `gb50h_swap`: 0.264 → 0.242 (SE 0.046), genF1 0.555 → 0.562, CE 0.405 → 0.404. Replacing
  the gold documents' slot vectors with random non-gold documents' is undetectable in the output.
* **Real body vs pooled body is the whole signal**: 0.732 / 0.812 / 0.711 recall on gold whose body
  stayed REAL, against 0.126 / 0.264 / 0.227 when it was pooled — a 3–6x gap in the *same rows*.
* **The generations are the position prior, verbatim.** All **12 of 12** dumped `gb00h` pooled-gold
  rows emit `Outliers: [1], [2], [3]`, regardless of the true ids (`[7,10,12]`, `[8,9,11]`,
  `[6,12,14]`, …). One row, whose gold really was `[1,2,3]`, scores 1.0 by coincidence.
* The header's apparent *help* (`gb50` genF1 0.453 → `gb50h` 0.555) is a mixture effect on the
  REAL-gold half (0.732 → 0.812) plus floor-rate credit on the pooled half — and the swap control
  says none of it comes from the slots.

### Run A — replicate on a second checkpoint and corpus (complete)

`lmx-full-mixs160M-4b` (dense) on `outlier_wiki100w_n22_k3_eval_600.jsonl`,
eval_size = 200 rows (⚠ < 500; binomial SE ≈ 0.035 at f1 ≈ 0.5), free generation on the first 64
rows (192 gold documents). Mean 21.4 documents/row, k = 3, so the uniform-guess floor for both
set-F1 and gold recall is `k/n` = 0.140. A different checkpoint AND a different corpus from Run B,
which is what makes it a useful replicate.

| condition | CE | **CE(digits)** | top1=full | KL | tf_f1 | **tf_id1** | **genF1** | genEM | **R@gold_pooled** | **R@gold_real** | pred_pooled | base_pooled | compaction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `full` | 1.811 | **0.092** | 1.000 | 0.000 | 0.955 | 0.875 | **0.979** | 0.953 | — | 0.979 (192) | — | 0.00 | 1.000 |
| `goldonly` | 1.398 | 0.330 | 0.911 | 0.321 | 0.865 | 0.620 | 0.510 | 0.438 | — | 0.510 (192) | 0.487 | 0.86 | 0.184 |
| `gb50` (=kvgb50) | 2.186 | 1.941 | 0.869 | 0.516 | 0.431 | 0.355 | 0.357 | 0.078 | **0.062** (97) | **0.642** (95) | 0.187 | 0.50 | 0.521 |
| `gb50h` (=xh2k50) | 1.996 | 1.631 | 0.876 | 0.459 | 0.508 | 0.340 | 0.402 | 0.062 | **0.105** (95) | **0.691** (97) | 0.366 | 0.49 | 0.554 |
| `gb17h` (=xhdr17) | 1.857 | 2.151 | 0.796 | 0.721 | 0.216 | 0.090 | 0.234 | 0.000 | **0.154** (162) | **0.667** (30) | 0.698 | 0.83 | 0.241 |
| `gb00h` (=xhdr00) | 1.498 | 1.747 | 0.788 | 0.791 | 0.179 | 0.135 | 0.094 | 0.000 | **0.094** (192) | — | 1.000 | 1.00 | 0.084 |
| `gb00` (pure slots) | 1.970 | 1.785 | 0.708 | 0.772 | 0.159 | 0.135 | 0.016 | 0.000 | 0.016 (192) | — | 1.000 | 1.00 | 0.048 |
| **`gb00h_swap`** (control) | 1.505 | 1.765 | 0.786 | 0.794 | 0.160 | 0.135 | **0.099** | 0.000 | **0.099** (192) | — | 1.000 | 1.00 | 0.084 |

(counts in parentheses are the number of gold documents the metric averages over; the per-document
SE is ≈ 0.021 at 192 documents and ≈ 0.03–0.05 at 95–97.)

Four things, in order of how much they decide the question.

**1. The control fires exactly.** `gb00h` 0.094 vs `gb00h_swap` 0.099 on R@gold_pooled — a 0.005
difference against an SE of 0.021, and the same on CE (1.498 / 1.505), CE(digits) (1.747 / 1.765),
KL (0.791 / 0.794), top-1 (0.788 / 0.786) and tf_id1 (0.135 / 0.135). **Replacing the gold
documents' slot vectors with random non-gold documents' changes nothing at all**, so the model is
not reading slot content in this construction. Both sit *at or below* the 0.140 guess floor.

**2. Real body → found; pooled body → not found, by ~7x.** In the two mixed constructions the split
is unambiguous: `gb50` recovers 0.642 of the gold documents whose body stayed REAL and 0.062 of
those that were pooled; `gb50h` 0.691 vs 0.105. Header-real moves the pooled number by +0.04 — i.e.
from "below the floor" to "at the floor" — while `pred_pooled` jumps 0.187 → 0.366, which is the
copying, not the finding: with ids visible the model spreads its guesses over the pooled documents
too, and gets floor-rate credit for it.

**3. The generations are a position prior, not a slot read.** Of the 12 dumped `gb00h` pooled-gold
rows, **11 emit `Outliers: [1], [2], [3]`** (the twelfth emits `[1], [15], [20]`), and the topic
sentence is a hallucinated constant — the same `The Great Gatsby (1974 film)` / `2016 Summer
Olympics – Women\'s 10 metre air pistol` recycled across rows whose actual corpora differ. The
model is generating from the prior with no document content reaching it. This is the "copy three
ids from the visible list with the empirical position prior" policy the xhdr diagnosis predicted,
seen directly.

**4. `gb00` (no header) is the impossibility control and behaves like one**: 0.016, below every
other cell, because a pooled document's id is nowhere in the context.

### Run B, 8k rung (INTERIM: snapshot at row 25/240, 75 gold documents)

`ds64-outlier-dense-u64M` on `rung_8192.jsonl`. 56.3 documents/row, so the guess floor is
`k/n` = **0.053**. Job `01M2HFQCS5T08MNNJP1TD35PHP`, still running (⚠ eval_size 25 so far —
per-document SE ≈ 0.03 at 75, ≈ 0.08 at 37–38; treat as a shape check, not a number to quote):

| condition | CE | CE(digits) | genF1 | **R@gold_pooled** | **R@gold_real** | compaction |
|---|---|---|---|---|---|---|
| `full` | 0.042 | 0.121 | 0.880 | — | 0.880 (75) | 1.000 |
| `goldonly` | 0.291 | 0.785 | 0.027 | — | 0.027 (75) | 0.076 |
| `gb50` | 0.697 | 2.077 | 0.165 | **0.000** (38) | **0.324** (37) | 0.493 |
| `gb50h` | 0.596 | 1.731 | 0.312 | **0.000** (38) | **0.622** (37) | 0.525 |
| **`gb50h_swap`** | 0.608 | 1.768 | 0.312 | **0.000** (38) | **0.622** (37) | 0.525 |
| `gb17h` | 0.767 | 2.225 | 0.072 | 0.049 (61) | 0.143 (14) | 0.210 |
| `gb00h` | 0.682 | 1.967 | 0.080 | **0.080** (75) | — | 0.063 |
| `gb00` | 0.736 | 2.096 | 0.067 | 0.067 (75) | — | 0.023 |
| **`gb00h_swap`** | 0.684 | 1.972 | 0.080 | **0.080** (75) | — | 0.063 |

`gb50h` and `gb50h_swap` are **bit-for-bit identical on every recall column** (0.000 / 0.622 /
genF1 0.312), and `gb00h` matches its swap at 0.080. Pooled-gold recall is 0.000–0.080 against a
0.053 floor at every keep. Note `goldonly` collapses to 0.027 here even though the gold bodies are
REAL — at 8k, pooling 53 of 56 documents destroys the *comparison* the task needs, which is the
same message from the other side.

### Run C, ~8k rung on the replicate checkpoint (INTERIM: snapshot at row 75/200, 144 gold documents in the generation split)

Same checkpoint, `outlier_wiki100w_n55_k3_eval_600.jsonl` (n = 55 documents/row, so the `k/n`
guess floor drops to **0.055**). This run also carries `gb50h_swap`. Snapshot at row 75/200
(sneetches `3549488`, still running; ⚠ eval_size 75 so far, per-document SE ≈ 0.018 at 144
documents, ≈ 0.03–0.06 at 66–78):

| condition | CE | CE(digits) | genF1 | **R@gold_pooled** | **R@gold_real** | compaction |
|---|---|---|---|---|---|---|
| `full` | 1.777 | 0.181 | 0.762 | — | 0.729 (144) | 1.000 |
| `goldonly` | 1.366 | 0.369 | 0.403 | — | 0.403 (144) | 0.078 |
| `gb50` | 2.351 | 2.308 | 0.228 | **0.000** (66) | **0.410** (78) | 0.505 |
| `gb50h` | 2.241 | 2.116 | 0.233 | **0.044** (68) | **0.382** (76) | 0.534 |
| **`gb50h_swap`** | 2.245 | 2.133 | 0.254 | **0.059** (68) | 0.408 (76) | 0.534 |
| `gb17h` | 2.272 | 2.897 | 0.062 | 0.049 (123) | 0.143 (21) | 0.219 |
| `gb00h` | 1.691 | 2.161 | 0.049 | **0.049** (144) | — | 0.063 |
| `gb00` | 2.161 | 2.169 | 0.021 | 0.021 (144) | — | 0.024 |
| **`gb00h_swap`** | 1.702 | 2.190 | 0.049 | **0.049** (144) | — | 0.063 |

Same shape, four times longer. `gb00h` and its swap control are **identical** on pooled-gold recall
(0.049 / 0.049) and within 0.01–0.03 on CE (1.691 / 1.702), CE(digits) (2.161 / 2.190), KL
(0.889 / 0.898) and top-1 (0.717 / 0.716) — and both sit on the 0.055 floor. `gb50h`'s swap control
moves pooled recall *up* (0.044 → 0.059, inside noise) rather than down. The real-vs-pooled gap
holds and widens with length: 0.38–0.41 real against 0.000–0.059 pooled. The second control behaves
like the first — whatever the header-real constructions recover, they do not recover it from slot
content.

### Caveat Run A also produced: answer-CE parity on outlier does NOT mean the answer is right

`full` has CE 1.811 but CE **on the digit tokens** of 0.092 — this corpus wraps the ids in a prose
topic sentence, and that prose is ~95% of the mean answer CE. So the CE column ranks constructions
almost backwards (`gb00h` 1.498 "beats" `full` 1.811). `goldonly` is the sharp case: CE 1.398, i.e.
*better than full*, and yet genF1 0.510 against full's 0.979 — it gets **half** the answers wrong.
The 2026-09-08 handoff's "nq and outlier are at parity with gold-only" is an answer-CE statement
(outlier 1.201 vs full 1.206); on this corpus the same construction loses half its generation
accuracy. Any outlier claim built on mean answer CE should be re-checked against CE(digits) or a
generation metric.

## 4b. Verdict

**(B) — not readable, with one honest qualification.** On outlier, a pooled document's
mean-embedding slot carries **no** usable signal about whether that document is the odd one out, at
least as far as a dense-trained model can use it. The evidence is three-fold, mutually reinforcing,
and it reproduces across two checkpoints, two corpora and three lengths. The pooled-gold recall of
every header-real construction sits **on the `k/n` uniform-guess floor** — on the canonical ds64 2k
run, `gb00h` 0.229 / `gb17h` 0.227 / `gb50h` 0.264 against a floor of 0.224, which is exactly where
the `xhdr*` *training* arms landed. **Swapping the gold documents' slot vectors with random non-gold
ones is undetectable**: 0.229 → 0.229 on `gb00h`, 0.264 → 0.242 (SE 0.046) on `gb50h`, with CE, KL,
top-1 and tf_id1 all moving by ≤ 0.02. And the greedy output on pooled-gold rows is a constant
`Outliers: [1], [2], [3]` — **12 of 12** dumped rows on the ds64 run, whatever the true ids were.
Meanwhile the same model, in the same rows, recovers 0.71–0.84 of the gold documents whose **body**
stayed real — a 3–6x gap that is the whole distance between "there is signal here" and "there is
not". So the `xhdr*` training collapse is **not** a
pure optimisation shortcut that a smarter recipe would fix: the target on a gold-blind, header-real,
pooled-gold row really is unjustifiable from the context, and the model at chance is the model being
right. (The xhdr diagnosis\'s mechanism is confirmed — id-copying *is* what happens — but its
implication "the question is still worth answering if you remove the confound" is not supported:
remove the confound and there is nothing underneath.)

The qualification: this measures what a **dense-trained** model can extract, and it never trained on
slots. In principle a trained reader could learn to use a mean-embedding vector better. Two things
argue it would not get far. First, the 2026-09-08 finding that **the representation is exhausted** —
the default soft token already equals the oracle mean-K/V slot, the fitted log-mass slot, and 8
slots/doc, within noise, at every keep on every task. Second, outlier is a *comparative* judgement
over all n documents ("which one is unlike the rest"), which needs per-document detail that survives
averaging; contradiction and nq are *retrieval* judgements, where a coarse topical match is enough,
and those are exactly the tasks where slots reach parity.

**Recommendation.** Do not spend more on header-real outlier arms, and do not "fix" them with
gold-forcing or id-renumbering — those address a confound that is not the binding constraint.
`kvgb50` (gold-blind, keep 1/2, **no** header) stays the outlier arm: it is at dense parity and its
supervision is content-grounded by construction, because a pooled document\'s id is not in the
context at all. If the comparative-judgement question is worth reopening, the thing to change is the
**slot**, not the keep policy or the header: G > 1 slots per document (the multi-landmark path), or a
trained summarizer whose output is not a mean — and the cheapest test of that is this same probe,
with the richer slot substituted, looking for `R@gold_pooled` to lift off the `k/n` floor.

## 5. Runs

**Run A (done, the table in §4).** sneetches job `3549422`, log
`/net/sneetches/data/prasann/outlier_probe/local_3549422.out`, JSON
`/net/sneetches/data/prasann/outlier_probe/local_2k.json`, launcher
`/scratch/users/prasann/outlier_probe/run_sneetches_local.sbatch`, run from a detached worktree at
the pushed commit (`/scratch/users/prasann/outlier_probe/wt`, commit `185107aa7` — it therefore has
every condition except `gb50h_swap`, which was added in `deb1d1e20`).
Checkpoint `/data/prasann/dense_ckpts/lmx-full-mixs160M-4b-.../model_and_optim` (the dense outlier
checkpoint the 2026-09-08 probe used); rows `outlier_wiki100w_n22_k3_eval_600.jsonl` from
`/scratch/users/prasann/cpt_data/eval500_v2/outlier/`.

**Runs B (Beaker).** The canonical pairing — the ds64 dense arm `ds64-outlier-dense-u64M` on the
ds64 rung files `outlier_lengthmix/eval_rungs/outlier/rung_*.jsonl`, 240 rows, 64 generation rows,
all 9 conditions. They write JSON to
`/weka/oe-training-default/ai2-llm/checkpoints/prasanns/_eval_results/outlier_slot_probe/` and print
the same table to stdout, so `beaker job logs <job>` is enough to read them.

| rung | cluster | experiment | job | state |
|---|---|---|---|---|
| 2k | ceres/saturn | `01M2HFPGBB3ZMV7MMJ10K625N5` | `01M2HFPGEX76K2FGPDWE0XXQ5J` | **DONE — the §4 headline table**; JSON `.../outlier_slot_probe_ds64-outlier-dense-u64M_2k.json` |
| 8k | ceres/saturn | `01M2HFQCNE55V2QAQQ1GDA1C5Y` | `01M2HFQCS5T08MNNJP1TD35PHP` | running |
| 8k | jupiter | `01M2HFG92C4TSM882PN51FNYGF` | `01M2HFG962162KWQ8G3N7Y4QCR` | queued |
| 16k | jupiter | `01M2HFH8ZN0WY4ERP66J5E12GY` | `01M2HFH93673KY57QD5A2FE22N` | queued |
| 2k | jupiter | `01M2HFFDXCKYVX9A0RYD4WX5BV` | `01M2HFFE0WT654RME3CFZZKMNP` | cancelled (duplicate of the ceres 2k) |

⚠ **Infra note**: jupiter took > 1.5 h to schedule a 1-GPU **urgent, unallocated** job and still has
not scheduled these; the ceres/saturn pair started in ~15 min. Spec verified correct (urgent, 1 GPU,
`/weka/oe-training-default` mounted, budget `ai2/oe-other`, workspace `ai2/flex2`). If a probe is
time-sensitive, launch it with `--cluster "ai2/ceres-cirrascale,ai2/saturn-cirrascale"` — both carry
the `storage:weka` tag, so the weka mount works there too.

**Runs C (queued locally, the longer local rungs).** sneetches `3549488` (n=55 docs, ~8k tokens) and
`3549489` (n=110, ~17k), same checkpoint, 200 rows / 48 generation rows, logs
`/data/prasann/outlier_probe/local_n{55,110}_<jobid>.out`.

**Reproduce (1 GPU):**

```
python debug/flop_scaling/beaker_bench_launch.py --cluster ai2/jupiter-cirrascale-2 \
  --script debug/pooled_kv/outlier_probe/outlier_slot_probe.py \
  --extra "--rung 2k --rows 240 --gen-rows 64 --dump-rows 12 --ckpt-name ds64-outlier-dense-u64M --work /tmp/probe_work"
```

## 6. Traps hit building this

* **`rope.forward`'s `position_ids` branch sizes its absolute-position sin/cos buffer from the
  CACHE, never from the positions it is handed** (`rope.py:563-571`). A compacted row is short but
  carries ORIGINAL positions, and free generation walks positions past the row's own length, so
  `index_select` reads past the end. CUDA reports it asynchronously, as
  `vectorized gather kernel index out of bounds` raised inside an unrelated `silu` — nowhere near
  the cause. Fix: warm every RoPE module to `max(row) + gen budget` before the loop. Any probe that
  *generates* on the compacted path needs this; the existing `eval_side_slot_probe.py` does not
  generate, so it never hit it.
* The probe compacts once itself (to map answer positions to compacted columns) and the forward
  compacts again. If those two ever disagree the gathered columns go out of bounds the same silent
  way, so the compaction is memoized on the exact input tensor + config.
* `make_fingerprint_keep_docs_fn(..., n_random_frac=0.0)` does **not** mean "gold only" (see §2).
* Running a GPU job straight out of this working tree is unsafe while another session is editing
  `src/olmo_core/` — use a detached `git worktree` at the commit you pushed.
