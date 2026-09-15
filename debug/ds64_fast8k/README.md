# fast8k — the screening loop moved to the rung where outlier actually fails

`debug/ds64_fast2k/` screens a soft-token recipe in under an hour and is decisive about whether a
recipe can **learn** the task. It is blind to the one thing outlier actually fails at. Measured on
the ds64 ladder (`records/ds64-overnight-2026-09-14.md`, 09-15):

| arm | 2k rung | 8k rung |
|---|---|---|
| `cc00` (keep 0, header real, `cent_cmean` slot) | 0.80 | **0.17** |
| `kv33` (gold + 1/3 random real) | 0.890 | 0.018 @32k |

A 2k screen calls both a success. **fast8k moves the whole loop up one rung**, so the failure is
inside the screen: training rows come from the ds64 **8k** pool (n ≈ 56 documents) and the decisive
eval is the 8k rung.

## What is held fixed vs fast2k, and what deliberately differs

Same task, same generator, same ds64 pools, the same repaired Qwen3.5-4B base
(`q35-4b-base-markerfix`), the same `--st-*` flag strings, the same unpacked matched-step
construction, the same FLOP meter, the same eval files the ds64 ladder uses. `launch_fast8k.py`
imports fast2k's arm vocabulary and helpers rather than copying them, and `collect_fast8k.py`
imports fast2k's log fetch / CE-curve parse / log-FLOP interpolation verbatim.

| | fast2k | fast8k |
|---|---|---|
| data | ds64 **2k** rung pool | ds64 **8k** rung pool (n ≈ 56 docs) |
| seq-len | 4096 | **PLACEHOLDER_SEQLEN** (unpacked; shard `max_example_len` = PLACEHOLDER_MAXLEN) |
| rows/step | 32 | **16** — an 8k row carries ~4× the tokens, so 16 rows ≈ the same real tokens/step |
| budgets | 2M / 4M / 8M | **8M / 16M / 20M** nominal = PLACEHOLDER_ROWS rows |
| eval | 2k rung × 500 | **8k rung × 500 (decisive)** + 2k × 500 (continuity) |
| primary metric | ladder f1 + CE floor | **eval-time CE parity** (below), then f1 |
| GPUs | 2 | 2 (see below) |

**Why the 8k pool caps the budgets.** `POOL_8K` in `debug/ds64/build_ds64_data_beaker.sh` is 2700
rows, so 20M nominal (2441 rows) is the largest budget that fits. `compose_fast8k.py` **SKIPS** a
budget that does not fit rather than truncating it — truncation would quietly break the "same rows,
same order, same steps" guarantee the matched budget rests on.

**2 GPUs, not 4.** `model_scale=4b` resolves to `shard_degree=world_size` + FULL activation
checkpointing (`train_ctc_suite.resolve_activation_checkpointing`), so a 2-rank FSDP job holds half
of an 8 GB bf16 parameter set, its grads and its ~32 GB fp32 AdamW state, and 2 × PLACEHOLDER_SEQLEN
checkpointed tokens per micro-batch is far below the 4 GPU × 65536 the ds64 ladder already runs.
Memory is not the binding constraint; queue turnaround is, which is why the cluster list is
`ceres,saturn,jupiter` with ceres first.

**Every arm at micro 2, including dense.** The 09-15 FLOP audit (`debug/ds64/flop_audit/`) found
that `microbatch_sort_pad_id` + `compact_pooled_rows` make a soft arm's measured FLOPs depend on
the micro-batch size, so any pair that will be compared at matched FLOPs must run at identical
gpus/micro. Dense is batching-independent, so putting it on micro 2 too costs nothing and removes
the confound.

## THE PRIMARY METRIC IS EVAL-TIME CE PARITY, NOT f1

`launch_fast8k.py parity` runs the **same trained checkpoint on two inputs** — full real text (what
the ladder eval feeds it) and its own soft construction — via
`debug/pooled_kv/outlier_probe/outlier_slot_probe.py --trained-parity`, and reports `CE_full`,
`CE_soft`, `dCE`, the same on the answer **digits** only, and free-generation set-F1 both ways.

A ladder f1 alone cannot tell **"this checkpoint cannot do outlier"** from **"this checkpoint can do
outlier but the compaction throws the answer away"**, and those two call for opposite fixes
(exposure vs. a better slot/readout). `dCEdig` does tell them apart.

> **Parity = `dCEdig ≈ 0` AND `dF1 ≈ 0`.**
> ⚠ A large `dCEdig` with a small `|dF1|` on a **weak** `F1_full` is not parity — it is two ways of
> being wrong. Always read `dF1` beside `F1_full`.

`PARITY_ARMS` scores every arm under `cc00` (everything pooled — the common maximal-compaction
reference, so `dCE` is comparable across arms and budgets) plus its own keep rate. The constructions
are passed as the trainer's own flags (`--construction`), which is what lets an arm with no
registered `ARMS` entry be scored at all.

⚠ The cpi constructions used at parity time are the **gold-blind** analogues of the training
policy: `parse_construction` refuses `--st-keep-frac` because the probe implements the gold-blind
`keep_prob` path, not the gold-sidecar policies. That is the right thing to measure anyway — at eval
there is no gold sidecar, so "keep p of the bodies real, blind" is the construction a cpi arm would
actually be deployed under.

## The arms

`cpi<p>` — **gold always POOLED, p of the NON-gold bodies real** (`--st-keep-mode
gold_pooled_random`, this repo's new keep mode, `src/olmo_core/nn/attention/gold_grad_mask.py`). It
is the middle ground between the two regimes the 2k screen separated:

* **`cc00` (keep 0) reads the slots** — 0.482 vs a 0.239 same-FLOP control — because with no real
  body anywhere, guessing an id is exactly chance and the slots are the only gradient left. But a
  model that has never seen a real body scores 0.17 at 8k;
* **`cc03` / `cc08` / `cc17` (1/36, 1/12, 1/6 of bodies real, gold-BLIND) all revert to id-guessing**,
  because a gold body that happens to be real still pays off.

Pooling **gold** removes the payoff without removing real text: a visible id is never an answer, so
copying one earns exactly chance, while the real non-gold bodies keep the model in the distribution
it meets at eval, where every document is real.

| arm | construction |
|---|---|
| `dense` | full attention, unpacked (the anchors — three budgets) |
| `cc00` | keep 0, header real, `cent_cmean` slot (the 2k winner, the arm that dies at 8k) |
| `cpi17` / `cpi33` / `cpi50` | gold pooled, 1/6 / 1/3 / 1/2 of non-gold bodies real, header, `cent_cmean` |
| `kvgb50` | gold-blind keep 1/2, plain mean slot, no header (the ladder's parity arm — reference) |
| `cc00-P1` → `kvgb50-warm-P2` | **two-phase**: `cc00` for the first 85% of the 16M budget, then keep-1/2 real bodies for the last 15% |

The two-phase arm needs **no trainer change**: `<arm>-warm` loads another fast8k run's exported
weights through `--base-checkpoint`, and the trainer loads it with `load_optim_state=False` /
`load_trainer_state=False`, so phase 2 gets a fresh LR schedule and a fresh budget and phase 1's
flags do not carry over. `compose_fast8k.py` gives P1 and P2 **disjoint** windows of the 16M rows
(and asserts the disjointness), so phase 2 is not a second epoch on phase 1's data.

## Two CE floors, and the cpi arms need the second one

`ce_floor = ln C(n, k) / answer_tokens` is what a model that learned only the output **format**
cannot beat — parking exactly there is the xhdr-collapse signature, readable ~3 minutes into a run.

But a `cpi<p>` arm pools **every** gold document, so it can learn the *training-only* regularity
"the answer is among the pooled documents" and guess inside that smaller set:

```
ce_floor_pooled = ln C(k + (1-p)(n-k), k) / answer_tokens
```

At n = 56, k = 3, p = 1/2 that is ~0.72 × the nominal floor. **Read a cpi arm's CE descent against
`ce_floor_pooled`, not `ce_floor`.** The shortcut transfers to nothing — the eval has no pooled
subset — so it can only flatter the training CE, never the score. `collect_fast8k.py` computes both
and uses the right one for the `FLOOR` flag.

## Run it

```bash
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python

# 1. data (once per task) -- tokenize-only gantry job, no GPU
TASK=outlier bash debug/ds64_fast8k/build_fast8k_data_beaker.sh

# 2. cache the shard stats (CE floor, n/k, max_example_len, train-vs-eval overlap) from that log
$PY debug/ds64_fast8k/collect_fast8k.py --stats-from-job <data-experiment-id>

# 3. train.  ALWAYS git push first -- gantry runs the PUSHED commit, --allow-dirty ships nothing.
#    F8K_SEQ_LEN comes from step 2's reported max_example_len.
export F8K_SEQ_LEN=PLACEHOLDER_SEQLEN
PYTHONPATH=src $PY debug/ds64_fast8k/launch_fast8k.py --arms dense --budgets 8M,16M,20M launch
PYTHONPATH=src $PY debug/ds64_fast8k/launch_fast8k.py --arms cc00,cpi17,cpi33,cpi50,kvgb50 --budgets 16M launch

# 4. poll (blocking, read-only -- never pkill)
bash debug/ds64_fast8k/poll_fast8k.sh <ex-id> [<ex-id> ...]

# 5. score: the 8k + 2k ladder rungs, then the eval-time CE parity (the primary signal)
PYTHONPATH=src $PY debug/ds64_fast8k/launch_fast8k.py eval
PYTHONPATH=src $PY debug/ds64_fast8k/launch_fast8k.py parity

# 6. collect
$PY debug/ds64_fast8k/collect_fast8k.py
```

State lives in `state.json`; every launch also appends to `LAUNCH_LEDGER.tsv`. Per-launch stdout is
kept in `launch_logs/`; Beaker logs are cached in `logs/` **only once a job is finalized** (a mid-run
cache freezes a partial FLOP meter — ds64 trap 6).

## Reading the signals

`collect_fast8k.py` writes `results.csv` (one row per run) and `parity_rows.csv` (one row per run ×
construction × rung), and prints a table.

* **`dCEd` / `gF1_f` / `gF1_s`** — the parity block. See the box above.
* **`f1_8k`** — the screen's verdict. `f1_2k` is carried only for continuity with the fast2k table.
* **`matched_flop_delta`** — `f1_8k` minus the dense f1 interpolated **in log FLOPs** to this arm's
  measured `flop_meter/actual_pflops`. `extrapolated = True` means the arm sits outside the measured
  dense range, where ds64 has already been burned once (dense at ~195 PF was projected 0.07–0.15 and
  measured 0.317). Treat an extrapolated delta as a hypothesis, which is why three dense anchors are
  always launched and a new arm's budget should sit between them.
* **FLOPs come from `flop_meter/actual_pflops`**, never `throughput/total petaflops` — the latter
  charges every token the dense per-token cost and reads ~10× high on a compacted arm.

**Reporting.** `eval_size` is 500 per rung. The binomial SE is ≈0.021 at f1 0.70 and ≈0.010 at 0.95;
outlier's per-example set-F1 is not Bernoulli, so read the SE as a resolution rather than a test.
Never quote `n` for an eval-set size — `n` is corpus size, and here it is ≈56.

## Train/eval disjointness

The build job runs `debug/ds64_fast2k/fast2k_stats.py` **twice per shard**, against the 8k rung and
the 2k rung, fingerprinting each example by its document texts + gold indices. A nonzero
`example_overlap` prints a loud refusal line. Measured: see the Results section.

## What fast8k still cannot see

32k and 64k. An 8k screen is a much better proxy for the ladder than a 2k one, but `kv33` fell apart
between 8k and 32k, so **a fast8k win is still a licence to spend a full ladder, never a substitute
for one.** The parity mode's `--rungs 8k,32k` partly covers this: a 32k parity check needs no
training, only a forward pass, so the length axis is visible in the diagnosis even when it is not in
the screen.

---

## Results

PLACEHOLDER_RESULTS
