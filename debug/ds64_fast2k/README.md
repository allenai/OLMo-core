# fast2k — a <1 h screening loop for soft-token training recipes

The ds64 campaign screens a soft-token (pooled-doc KV) recipe by training a full short-heavy
2k–64k ladder and scoring five rungs: **4–20 GPU-hours per arm**. `fast2k` does the same
comparison on **2k rows only**, at seq-len 4096 and 32 rows/step: **~20–40 minutes and well under
1 GPU-hour per arm**, so a recipe can be killed or promoted the same afternoon it is invented.

## Why 2k is enough to *screen* (and what it cannot tell you)

In the linear-cost regime compaction saves the same **fraction** of FLOPs at any context length,
so *accuracy vs FLOPs at a matched budget* is already well posed at 2k. A recipe that cannot beat
dense at matched FLOPs on 2k rows is not going to start doing so at 32k.

⚠ **What fast2k cannot see is LENGTH GENERALISATION, which is exactly where ds64's outlier arms
died.** `kv33` scores 0.890 at 2k and 0.018 at 32k (`records/ds64-handoff.md` §1) — a 2k screen
would have called it a near-parity success. So:

> **A fast2k win is a licence to spend a full ladder. It is never a substitute for one, and a
> fast2k number must never be quoted as a campaign result.**

Conversely a fast2k *loss* is decisive: an arm that cannot learn the task at 2k has nothing left
to generalise.

## What is held fixed vs ds64, and what deliberately differs

Same task, same generator, the **same 2k rung pool** the ds64 short-heavy mix is drawn from, the
same repaired Qwen3.5-4B base (`q35-4b-base-markerfix`), the same `--st-*` flag strings, and the
same 2k eval file the ds64 ladder's 2k rung uses.

| | ds64 | fast2k |
|---|---|---|
| data | short-heavy mix 2k…56k | **2k rung only** |
| seq-len | 65536 | **4096** (shard `max_example_len` = 2949) |
| packing | dense packed, soft padded | **every arm unpacked** |
| rows/step | 8 packed / 128 padded | **32, every arm** |
| budgets | 16M…128M tokens | **2M / 4M / 8M** nominal tokens = 977 / 1953 / 3906 examples |
| eval | 5 rungs × 500 | **2k rung only × 500** |

**Every arm trains unpacked on purpose.** ds64 packs dense and pads the soft arms, so their
rows/step and step counts differ and the budget has to be reconciled in tokens — which is where
the `--max-tokens`-below-one-step trap lives (`records/ds64-handoff.md` §1). Here dense and soft
see the **same rows in the same order for the same number of steps**, so the budget is matched by
construction and the only variable is the compaction. Padding then costs wall-clock but never
FLOPs: `FlopMeterCallback` is wired with `pad_id`, so it charges **non-pad** tokens.

**2 GPUs, not 1.** A 4B model's AdamW state alone is ~32 GB fp32 on top of ~8 GB bf16 params and
~8 GB grads; a single 80 GB rank has almost nothing left for activations. FSDP over 2 ranks halves
params+optim and a 2-GPU slot still backfills a packed cluster quickly. `F2K_NGPU=1` if you want
to try. Keep `GLOBAL_BATCH % (micro * ngpu) == 0`.

## Run it

```bash
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python

# 1. data (once per task) -- tokenize-only gantry job, no GPU, ~90 s
TASK=outlier bash debug/ds64_fast2k/build_fast2k_data_beaker.sh

# 2. cache the shard stats (CE floor, n/k, train-vs-eval overlap) from that job's log
$PY debug/ds64_fast2k/collect_fast2k.py --stats-from-job <data-experiment-id>

# 3. train.  ALWAYS git push first -- gantry runs the PUSHED commit.
$PY debug/ds64_fast2k/launch_fast2k.py --arms dense --budgets 2M,4M,8M launch   # the anchors
$PY debug/ds64_fast2k/launch_fast2k.py --arms kvgb50,xhdr17,xhdr50 --budgets 4M launch

# 4. poll, then eval the finished runs on the 2k rung only (500 rows)
$PY debug/ds64_fast2k/launch_fast2k.py status
$PY debug/ds64_fast2k/launch_fast2k.py eval

# 5. collect
$PY debug/ds64_fast2k/collect_fast2k.py
```

State lives in `state.json`; every launch also appends to `LAUNCH_LEDGER.tsv` (same columns as
`debug/ds64/LAUNCH_LEDGER.tsv`). Per-launch stdout is kept in `launch_logs/`, Beaker logs are
cached in `logs/` **only once a job is finalized** (a mid-run cache freezes a partial FLOP meter —
ds64 trap 6).

## Adding an arm

Add one entry to `ARM_EXTRA` in `launch_fast2k.py` — the key is the arm name, the value is the
verbatim `--st-*` flag string. **Use the same name and the same flags as
`debug/ds64/launch_ds64.py`** so a fast2k screen and a ds64 ladder of that name are the same recipe
at two lengths. Set `ARM_MICRO[arm]` if the default micro-batch of 2 does not fit.

Two outlier-specific gotchas, both already encoded:

* the header stop id is **5491** (`']:'`), not 25 — Qwen3.5 fuses the bracket and the colon into
  one token, so a bare `':'` never occurs in an outlier header and `mark_doc_headers_free` would
  silently fall back to its 32-token cap, giving away a quarter of the compaction
  (`records/ds64-handoff.md` §8);
* **never pair `--st-header-stop-id` with `--st-gold-blind` when the answer is drawn from the
  header** (outlier, nq, rerank): every id becomes real text including for pooled docs, so the row
  hands the model a copyable-but-unjustifiable target. That is what `xhdr*` is, and screening it
  here is the point — see `debug/ds64/xhdr_collapse_diagnosis.md`.

### Warm-start / two-phase arms

`WARM_FROM` maps `<arm>-warm` to `(recipe, source run)`. The trainer loads `--base-checkpoint` with
`load_optim_state=False` and `load_trainer_state=False`, so phase 2 gets a **fresh LR schedule and
a fresh budget** and phase 1's flags do not carry over — that is the whole two-phase mechanism, no
extra trainer support needed. The source must have finished (its `model_and_optim` export is
written after `fit()`).

## Reading the signals

`collect_fast2k.py` writes `results.csv` and prints a table.

**`ce_floor` = `ln C(n, k) / answer_tokens`** — the cross-entropy a model that has learned only the
output FORMAT cannot beat. For outlier's 2k rung: n = 14 documents, k = 3 golds, 15.1 answer
tokens, so `ln 364 / 15.1 = 0.390`, and the matching uniform-guess f1 is `3/14 = 0.214`.
`at_floor_step10` fires when the step-10 CE is within `--floor-tol` (default 0.03) of it. That is
the exact signature of the xhdr collapse — "sample k ids from the visible list" — and it is
readable about three minutes into a run instead of after a 500-row eval. **A run that fires it is
dead. A run that does not still has to clear the matched-FLOP bar.**

**`matched_flop_delta`** = this arm's `f1_2k` minus the dense f1 interpolated **in log FLOPs** to
this arm's measured `flop_meter/actual_pflops`. Positive = beats dense at the same compute.
`extrapolated = True` means the arm sits outside the measured dense range, where ds64 has already
been burned once: dense at ~195 PF was projected at 0.07–0.15 from its log-linear slope and
actually measured **0.317**. Treat an extrapolated delta as a hypothesis. That is why the sweep
always launches **three** dense anchors, and why a new arm's budget should sit between them.

**FLOPs come from `flop_meter/actual_pflops`**, never the trainer's `throughput/total petaflops` —
the latter charges every token the dense per-token cost and reads ~10x high on a compacted arm.

**Reporting.** `eval_size` is 500 (the rung file holds 600; `--max-test 500`). The binomial SE is
≈0.021 at f1 0.70 and ≈0.010 at 0.95; outlier's per-example set-F1 is not Bernoulli, so read the SE
as a resolution rather than a test. Never quote `n` for an eval-set size (`n` is corpus size — the
2k rung's n is 14).

## Train/eval disjointness

The build job runs `fast2k_stats.py`, which fingerprints each example by its document texts + gold
indices and compares the training shard against the 2k eval rung. Measured 2026-09-14:
**0 shared examples and 0 shared documents** across all three budgets (train n = 14 docs/example,
eval n = 13.4). It re-runs on every build; a nonzero `example_overlap` prints a loud refusal line.

## In-loop downstream eval — deliberately NOT wired

`src/olmo_core/train/callbacks/evaluator_callback.py` runs loss/downstream evaluators, but nothing
in `train_ctc_suite.py` wires a **generative** task eval, and outlier is graded by generating an id
list. Adding one would mean a generation loop inside the training step — heavy enough to dominate a
30-step run. The in-loop signal is therefore the per-step train CE (logged every step to the
console and to wandb) against `ce_floor`; the accuracy signal is the final 2k-rung eval.

---

## First sweep, 2026-09-14 (outlier, Qwen3.5-4B) — the loop validated against ds64

7 arms, **7.8 GPU-hours total** (train + eval, job start → finalize), 20–42 min turnaround each
(train 11–34 min + an 8-min eval). Dense anchors at three budgets; soft arms at the middle one.
`eval_size = 500`, binomial SE ≈0.021 at f1 0.70 and ≈0.010 at 0.95.

| run | steps | PF (meter) | ×dense | CE@10 | CE final | f1 2k | Δ vs dense @ same FLOPs |
|---|---|---|---|---|---|---|---|
| dense 2M | 30 | 48.5 | 1.00 | 0.090 | 0.053 | 0.954 | — (anchor) |
| dense 4M | 61 | 98.4 | 1.00 | 0.128 | 0.020 | 0.987 | — (anchor) |
| dense 8M | 122 | 196.9 | 1.00 | 0.093 | 0.013 | 0.996 | — (anchor) |
| **kvgb50** 4M | 61 | 56.3 | 0.57 | 0.377 | **0.264** | **0.882** | −0.079 |
| **xhdr50** 4M | 61 | 57.4 | 0.58 | 0.387 | **0.275** | **0.895** | −0.067 |
| **xhdr17** 4M | 61 | 27.3 | 0.28 | 0.435 | **0.408** ⬅ FLOOR | **0.234** | −0.693 * |
| **xhdr17-warm** 4M | 61 | 27.3 | 0.28 | 0.409 | **0.386** ⬅ FLOOR | **0.236** | −0.691 * |

`ce_floor` = 0.390, uniform-guess f1 = 3/14 = 0.214. `*` = extrapolated below the measured dense
range.

**1. Both signals separate the known-good arm from the known-collapsed one.** `kvgb50` drives CE to
0.264, well under the 0.390 floor, and scores 0.882; `xhdr17` never leaves the floor (0.408) and
scores 0.234 — i.e. 3/14, the uniform guess. The CE verdict is available at the end of a 15-minute
training job, before any eval.

**2. It reproduces ds64's own 2k rung at ~1/14 the compute.** ds64 measured `xhdr17` 2k = 0.23 and
`kvgb50` 2k ≈ 0.87 after 780+ PF ladder arms; fast2k gets 0.234 and 0.882 at 27–56 PF.

**3. New: header-real does NOT collapse at keep 1/2.** `xhdr50` (header-real, gold-blind, keep 1/2)
lands at 0.895 / CE 0.275 — statistically indistinguishable from its header-free twin `kvgb50`
(0.882 / 0.264) at the same FLOPs. So the xhdr collapse is a **low-keep** phenomenon, not an
unconditional property of pairing `--st-header-stop-id` with `--st-gold-blind`: with half the bodies
real there is enough content supervision that the copyable id is not the cheapest policy.

**4. New: the collapse is NOT a basin/optimization artifact.** `xhdr17-warm` starts from the
*finished* `kvgb50` checkpoint — a model that already solves the task from content — and still falls
all the way back to the guess policy (0.236, CE 0.386). A content-grounded initialisation does not
save keep-1/6 header-real.

**5. At 2k, dense is near ceiling (0.954 → 0.996), so no soft arm beats it.** Every matched-FLOP
delta is negative. That is consistent with ds64's verdict that outlier is at best a parity task
(dense's own scaling curve is ~6× steeper there than on the tasks compaction wins). ⚠ It also means
2k has little accuracy headroom on **this** task — for a task where dense saturates, read the
fast2k screen as a *kill filter* (does the recipe learn at all, and at what FLOP ratio) rather than
as a matched-FLOP contest. A task whose dense 2k score sits well below 1.0 gives a sharper delta.

⚠ All of the above is **2k only**. `kvgb50` and `xhdr50` look equivalent here; ds64's ladder is what
decides whether either survives to 32k.
