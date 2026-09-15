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

_(pending — jobs below)_

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
