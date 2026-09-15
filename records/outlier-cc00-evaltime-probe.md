# Does the ds64 `cc00` checkpoint have EVAL-TIME loss parity? (trained-arm probe)

**Date** 2026-09-15 · **Status** RUNNING — jobs launched, numbers pending · **Branch** `prasann/landmark`
**Driver** `debug/pooled_kv/outlier_probe/outlier_slot_probe.py --trained-parity`
**Context** `records/ds64-overnight-2026-09-14.md` (the cc00 family, log entries from 09-15 02:30),
`records/outlier-slot-probe.md` (the frozen-dense eval-side probe this extends).

## 1. The check that was missing

Every ladder eval in this campaign feeds a checkpoint **full real text**. A soft arm trained at
keep 0 has **never seen a real document body** — its whole training distribution is
`header + one slot per document`. So its ladder number silently mixes two very different failures:

* the model reads slots fine and the ladder is measuring **eval-time distribution shift**, or
* **slot reading itself does not scale** with document count.

They call for opposite fixes (exposure/curriculum vs. a different slot or readout), and nothing in
the ladder number separates them. `cc00` forced the issue: `ds64-outlier-cc00-b128f3-u128M`
(cent_cmean slot, headers real, keep 0, 233 PF) scores **f1 0.80 at 2k, 0.17 at 8k, 0.05 at 16k**
on the ladder — a 2k win that evaporates exactly as document count grows.

This record answers it by scoring **that same checkpoint** twice per rung.

## 2. The reusable check (ship first, run second)

`--trained-parity` is now a flag of the outlier probe and should be run on **every** new soft
checkpoint before its ladder number is believed. One command, one checkpoint load, all rungs:

```bash
python debug/pooled_kv/outlier_probe/outlier_slot_probe.py --trained-parity \
  --ckpt-name ds64-outlier-cc00-b128f3-u128M --rungs 2k,8k,16k,32k --rows 240 --gen-rows 32
```

It prints, per rung: `CE_full CE_soft dCE | CEdig_full CEdig_soft dCEdig | F1_full F1_soft dF1 |
compaction`, with `eval_size` on every row.

* the construction comes from the run name's **arm field** (`cc00`) through the `ARMS` registry,
  which mirrors `debug/ds64/launch_ds64.py:ARM_EXTRA`. For an unregistered arm pass it verbatim as
  the trainer's own flags with **commas for spaces** (so the string survives a launcher that
  splits argv on whitespace):
  `--construction --st-gold-blind,--st-keep-prob,0.0,--st-header-stop-id,5491,--st-header-stop-count,1,--st-slot-mode,cent_cmean`
* the `cent_cmean` **stop set is rebuilt from the arm's own TRAINING shard**, exactly as
  `train_ctc_suite.build_slot_stop_set` does — not from the eval rows. Verified bit-identical for
  this run: the probe logged `1244 ids from 20001129 tokens of .../ds64/shards/outlier_u128M`
  against the training job's (`01M2J9WD14KWT9ABPDEBMF5BS1`)
  `1244 ids from 20001129 tokens of token_ids_part_000000.npy`, same most-frequent list.
* `--trained-parity` implies `--no-reset-projector`, so a trained checkpoint keeps its own
  `pooled_projector`. The log prints `max|w_out|`; for a `detach_soft_kv` arm it is **0.0**, i.e.
  the projector is the identity and the slot IS the (content-only, row-centred) mean embedding.
  On this checkpoint: `max|w_out| = 0.000e+00`.
* every rung is scored in one process, so the 4B checkpoint loads once (~15 s off weka).

**Reading it.** Parity means `dCEdig ≈ 0` **and** `dF1 ≈ 0`. A large `dCEdig` with the SOFT side
better is distribution shift. Both sides weak is not parity — it is two ways of being wrong, and
the `F1_full` column is what stops you calling that a success.

## 3. Conditions

All on the ds64 outlier rung files `outlier_lengthmix/eval_rungs/outlier/rung_*.jsonl`, one
checkpoint held fixed, only the input construction changing.

| # | name | what it is | why |
|---|---|---|---|
| A | `full` | every document real, full attention | **what the ladder eval does** — reproduces the ladder f1 and anchors everything |
| B | `arm` | headers real + `cent_cmean` slot per document, keep 0 | **the checkpoint's own TRAINING construction** |
| C | `gb00h` | same, but the PLAIN mean slot | is the trained reader specific to `cent_cmean`? |
| D | `ccgold` | headers real + slots everywhere, **gold documents real** | oracle: does seeing the gold body help or confuse a keep-0-trained model? |

Metrics per condition: answer CE, **CE on the answer's digit tokens** (the ids carry the decision;
mean answer CE is mostly prose — see `records/outlier-slot-probe.md` §4 caveat), free-generation
set-F1 over the k ids, `R@gold_pooled` / `R@gold_real`, and compaction.

## 4. Results

### 4.1 `ds64-outlier-cc00-b128f3-u128M` (233 PF, the ladder's best cc00 point)

Ladder reference for this checkpoint (`results/ds64/results.csv`): **2k 0.801 · 8k 0.168 · 16k 0.046
· 32k 0.011**, mean 0.207.

**2k rung — COMPLETE.** `rung_2048.jsonl`, **eval_size = 240** (⚠ < 500; binomial SE ±0.032 at
f1 ≈ 0.5). Free generation on the first 48 rows = **144 gold documents** (⚠ per-document SE ≈ 0.036;
genF1 is a 48-row mean, SE ≈ 0.06). 14 documents/row, k = 3, so the uniform-guess floor is
`k/n` ≈ **0.22**.

| | condition | CE | **CE(digits)** | top1=full | KL | tfID1 | **genF1** | **R@gold_pooled** | **R@gold_real** | compaction |
|---|---|---|---|---|---|---|---|---|---|---|
| **A** | `full` (real text = the ladder) | 0.071 | **0.264** | 1.000 | 0.000 | 0.792 | **0.750** | — | 0.750 (144) | 1.000 |
| **B** | `arm` = the cc00 TRAINING construction | 0.112 | **0.421** | 0.957 | 0.093 | 0.688 | **0.778** | **0.778** (144) | — | **0.105** |
| **C** | `gb00h` = same, PLAIN mean slot | 0.399 | 1.534 | 0.839 | 0.291 | 0.237 | 0.243 | 0.243 (144) | — | 0.105 |
| **D** | `ccgold` = gold bodies REAL (oracle) | 0.214 | 0.791 | 0.924 | 0.114 | 0.167 | 0.326 | — | 0.326 (144) | 0.310 |

Three things, at 2k:

* **A ≈ B: eval-time parity holds at 2k.** ΔCE **+0.041**, ΔCE(digits) **+0.157**, ΔgenF1
  **+0.028** (inside the ±0.06 generation SE) — at **9.5× compaction**. The probe's `full` genF1
  0.750 also sits near the ladder's 0.801, so the setup reproduces the ladder.
* **C says the trained reader is specific to `cent_cmean`.** The identical keep-0, header-real
  construction with the PLAIN mean slot collapses to genF1 0.243 — i.e. **onto the `k/n` = 0.22
  guess floor**, exactly where `xhdr00` and the frozen-dense probe's `gb00h` sat. Swapping only the
  slot construction is worth **+0.535 genF1** on a model trained for it. The 2026-09-15 fast2k
  screen (cc00 0.482 vs xhdr00 0.239) is reproduced here on the trained 128M checkpoint, larger.
* **D says the oracle HURTS.** Giving the model the gold documents' real bodies while every other
  document stays a slot drops it from 0.778 to **0.326** — worse than its own construction by
  0.45, on rows where the answer is *more* visible. A keep-0-trained reader is not merely
  indifferent to real bodies; a real body next to slots actively breaks it. This is the same
  direction the (A) gap will take at longer rungs.

**8k rung — COMPLETE.** `rung_8192.jsonl`, **eval_size = 240** (⚠ < 500; SE ±0.032). Generation on
the first 48 rows = **144 gold documents** (⚠ per-document SE ≈ 0.036; genF1 is a 48-row mean,
SE ≈ 0.06). ~56 documents/row, k = 3, so the uniform-guess floor is `k/n` ≈ **0.053**.

| | condition | CE | **CE(digits)** | top1=full | KL | tfID1 | **genF1** | **R@gold_pooled** | **R@gold_real** | compaction |
|---|---|---|---|---|---|---|---|---|---|---|
| **A** | `full` (real text = the ladder) | 0.225 | **0.623** | 1.000 | 0.000 | 0.167 | **0.229** | — | 0.229 (144) | 1.000 |
| **B** | `arm` = the cc00 TRAINING construction | 0.346 | **0.988** | 0.920 | 0.128 | 0.100 | **0.125** | 0.125 (144) | — | **0.062** |
| **C** | `gb00h` = same, PLAIN mean slot | 0.637 | 1.859 | 0.801 | 0.336 | 0.054 | 0.069 | 0.069 (144) | — | 0.062 |
| **D** | `ccgold` = gold bodies REAL (oracle) | 0.544 | 1.560 | 0.872 | 0.279 | 0.021 | 0.007 | — | 0.007 (144) | 0.114 |

**The 2k parity is gone by 8k, and it is the SOFT side that lost it**: ΔCE **+0.121**,
ΔCE(digits) **+0.365**, ΔgenF1 **−0.104** (against an SE ≈ 0.06, so real). The ordering
C < B < A is preserved — `cent_cmean` is still worth ~1.8× the plain slot (0.125 vs 0.069) — but
2× the guess floor is not a working reader. `ccgold` has collapsed to 0.007, **below the floor**:
at 56 documents the oracle is actively harmful.

**16k rung — COMPLETE.** `rung_16384.jsonl`, **eval_size = 240** (⚠ < 500; SE ±0.032). Generation on
the first 32 rows = **96 gold documents** (⚠ per-document SE ≈ 0.028; genF1 is a 32-row mean,
SE ≈ 0.05). ~111 documents/row, k = 3, so the uniform-guess floor is `k/n` ≈ **0.027**.

| | condition | CE | **CE(digits)** | top1=full | KL | **genF1** | **R@gold_pooled** | **R@gold_real** | compaction |
|---|---|---|---|---|---|---|---|---|---|
| **A** | `full` (real text = the ladder) | 0.332 | **0.877** | 1.000 | 0.000 | **0.083** | — | 0.083 (96) | 1.000 |
| **B** | `arm` = the cc00 TRAINING construction | 0.480 | **1.297** | 0.888 | 0.164 | **0.062** | 0.062 (96) | — | **0.056** |
| **C** | `gb00h` = same, PLAIN mean slot | 0.730 | 2.007 | 0.795 | 0.311 | 0.042 | 0.042 (96) | — | 0.056 |
| **D** | `ccgold` = gold bodies REAL (oracle) | 0.701 | 1.902 | 0.841 | 0.315 | 0.010 | — | 0.010 (96) | 0.082 |

**There is no parity at 16k, and the soft side is the WORSE one**: ΔCE **+0.149**, ΔCE(digits)
**+0.421**, ΔgenF1 −0.021. Both sides are within a whisker of the 0.027 guess floor (0.083 and
0.062), so this is the "two ways of being wrong" case, not a hidden competence. The dumped
generations say the same thing directly — on 16k pooled-gold rows the model emits scattered,
unrelated ids (`true=[6,29,64] pred=[11,17,107]`, `true=[17,31,97] pred=[11,12,14]`,
`true=[22,38,108] pred=[10,24,81]`), under BOTH inputs.

**The parity table, `cc00-u128M`** (printed by `--trained-parity`; eval_size 240/rung):

```
 rung eval_size |  CE_full  CE_soft      dCE | CEdig_full CEdig_soft   dCEdig |  F1_full  F1_soft     dF1 | compact
   2k       240 |    0.071    0.112   +0.042 |      0.264      0.421   +0.157 |    0.750    0.778  +0.028 |   0.105
   8k       240 |    0.225    0.346   +0.121 |      0.623      0.988   +0.365 |    0.229    0.125  -0.104 |   0.062
  16k       240 |    0.332    0.480   +0.149 |      0.877      1.297   +0.421 |    0.083    0.062  -0.021 |   0.056
```

**32k — running** (job C); `cc00-u32M` (D/E) queued.

## 5. Interpretation

_(pending)_

## 6. Runs

1-GPU Beaker, urgent, workspace `ai2/flex2`, budget `ai2/oe-other`, clusters
`ai2/ceres-cirrascale,ai2/saturn-cirrascale,ai2/jupiter-cirrascale-2`, reading the checkpoints off
weka. JSON lands in
`/weka/oe-training-default/ai2-llm/checkpoints/prasanns/_eval_results/outlier_slot_probe/outlier_slot_probe_<ckpt>_<rung>_trained.json`
and the same table goes to stdout, so `beaker job logs <job>` is enough to read a run.

| # | checkpoint | rungs | rows | gen rows | conditions | experiment | job |
|---|---|---|---|---|---|---|---|
| A | cc00-u128M | 2k, 8k | 240 | 48 | full, arm, gb00h, ccgold | `01M2K33J74VYC6AMEDR6MVSC3Y` | `01M2K33JAQA5020KY44823DWGS` |
| B | cc00-u128M | 16k | 240 | 32 | full, arm, gb00h, ccgold | `01M2K34GB0HQ9WT6Q8WA643BRF` | `01M2K34GEFKQGFCSZW1P58BQM9` |
| C | cc00-u128M | 32k | 120 | 16 | full, arm, gb00h, ccgold | `01M2K35DEVE4SR8S387FB9DN48` | `01M2K35DN8PVACVB7FBMENP9S9` |
| D | cc00-u32M | 2k, 8k, 16k | 240 | 24 | full, arm | `01M2K366ZSVKFT0BVE9Q4S5XCQ` | `01M2K3676JT9GXSZPCKB5YJV71` |
| E | cc00-u32M | 32k | 120 | 16 | full, arm | `01M2K379XDN0SD6A8T8CCGTGR1` | `01M2K37A2MYAGTT96MG3WCGK1X` |

⚠ **eval_size is below 500 on every rung** (240 at 2k/8k/16k, 120 at 32k). Binomial SE on a
right/wrong metric at f1 ≈ 0.5 is **±0.032** at 240 rows and **±0.046** at 120; generation metrics
average over the first `gen rows` only, so their SE is larger still and is quoted with each table.

Three earlier submissions (`01M2K2XT3R9QSGAHEYQBFKQ62G`, `01M2K2YMH5F613M1Z9S47T35HD`,
`01M2K2ZR1P83N57D185TFC86AY`) were cancelled two minutes in: they ran the first pushed commit, in
which `--trained-parity` *replaced* `--conditions` instead of adding to it, so they would have run
only A and B. Fixed in `6b48341e7`.
