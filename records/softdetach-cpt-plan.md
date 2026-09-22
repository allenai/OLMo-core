# Soft-detached pooling for continued pretraining — plan (2026-09-21)

**Ask (Prasann):** CPT runs "similar to one of amandab's" that use soft-detaching with ≥4×
compaction (random chunks are fine) — does it reach the same dev loss at the same FLOP budget?
Then, the same day: **screen strategies at small cost first, pick 1–2, then spend the 4B runs**;
"you can use the dev loss setup to measure this."

## Instrument: the frozen-base dev-loss harness

`debug/devloss_grid/ctc_devloss_grid.py --task cpt80` scores a frozen checkpoint (the
marker-repaired Qwen3.5-4B base) on held-out long documents: the first 80% of a document is cut
into 512-token pseudo-documents wrapped in `<|doc_start|>/<|doc_end|>`, the last 20% is the loss
region, rungs 2k/8k/32k. Every strategy is a `SCHEMES` entry that runs through
`Transformer._compact_pooled_soft_tokens`, so all strategies land on one axis: **tail-20% CE vs
compaction** (plus a per-scheme `sel_cost` for selectors that need an extra pass).

Strategies (all gold-blind, `cent_cmean` slot, detached K/V+GDN):

| scheme | rule | compaction on 512-token blocks | selection cost |
|---|---|---|---|
| `rand20` | 20% of pseudo-docs whole, rest one slot each (random-chunk k0) | ×0.20 | 0 |
| `fl20` | first 10% + last 10% of every block | ×0.20 | 0 |
| `first64` / `first128` | first K of every block | ×0.13 / ×0.25 | 0 |
| `rule20` | token-feature saliency (`DEFAULT_KEEP_TOKEN_WEIGHTS`), 20% per block | ×0.20 | 0 |
| `grad20` | **oracle**: top-20% per block by ‖∂CE(tail)/∂embedding‖ | ×0.20 | ~3 fwd-equiv (fwd+bwd), uses the label |
| `attnrow20` | layer-3 attention mass from the last 32 prompt positions sets each block's budget (20% total), first-k inside | ×0.20 | 1 fwd (4/32 if truncated at layer 3) |
| `k0` | everything pooled (floor) | ×0.20 on cpt80 (tail is real) | 0 |

Library additions for this: fractional `keep_token_k` (0<k<1 = per-document fraction,
`mark_doc_topk_tokens_free`), and a `custom` keep rule (`mark_positions_free`) that takes a
caller-built keep mask — how `grad20`/`attnrow20` inject their selections.

Runs: `debug/devloss_grid/run_grid_local.sbatch` with `SCHEMES=… RES_DIR=…/results_screen`
(jsteinhardt preemptive_high; also over the 18 CTC rows so the cheap ones merge into the CTC grid).
Only a strategy that NEEDS training (a trainable compressive-landmark / learned slot) gets a
training job, and the smallest one: train the slot layer on the frozen base briefly, then score with
the same driver.

## The 4B CPT runs (after screening picks 1–2)

Recipe copied from `src/scripts/train/memexpress/cpt/Qwen3.5-4B-dense-dolma3longmino.py`
(Qwen3.5-4B hybrid, 64k rows, dolma3_longmino sample at
`checkpoints/amandab/dolma3_longmino_mix_sample15B_qwen3_5`, no CP, no YaRN), run through
`train_ctc_suite.py` (`--variant full|softtoken`) so the soft-token machinery, the FLOP meter and
the ds64 Beaker path are reused verbatim: **same rows/step for every arm (8 × 64k = 520k
tokens/step)**, budgets 32M/64M/128M tokens as `--max-tokens` prefixes of one 128M shard,
`flop_meter/actual_pflops` as the x-axis (trap: a budget below one step's tokens is a 1-step run).

Data (`cpt/softdetach/build_cpt_shard*.{py,sh}`): CPT streams carry no document markers, so each 64k
row is cut into 127 pseudo-documents of 510 body tokens + the marker pair (65,024 tokens + EOS);
loss mask on body tokens only. Dense trains on the same marker-wrapped shard. The dev shard comes
from the last source part, never read by training.

Arms so far (`launch_softdetach_cpt.py`): `dense`; `sd20` = `--st-gold-blind --st-keep-prob 0.2`
(20% of blocks whole, 80% → one slot; ~×0.20); `sfl20` = `--st-keep-prob 0 --st-keep-token-rule
first_last --st-keep-token-k 0.2` (~×0.20). The screening decides which 1–2 replace/join these.

Dev loss (`eval_cpt_devloss.py`, one GPU per checkpoint, gantry wrapper): on the held-out shard,
full-row CE and tail-20% CE under FULL attention (the ds64 convention: compression is a training
saving) and, for soft arms, under their own construction (is it also an inference saving).

## Traps
- `--st-keep-token-k` was `type=int` in the trainer; fractional K needed `float` (done).
- The base must be the marker-repaired one (`q35-4b-base-markerfix`), CLAUDE.md "REPAIR THE BASE
  CHECKPOINT FIRST".
- Held-out text for the harness (`geodesic-research/dolma3_longmino_mix_500k_sample`) is a sample
  of the same mix as the training tokens on weka; overlap with the 15B tokenized sample is not
  excluded. The tokenized dev shard (last source part) is the clean held-out set for the trained
  arms.

## Status 2026-09-21 21:55 PDT — screening done, 12 CPT arms launched

**Screening verdict (cpt80, frozen base, tail-20% CE, `results_screen/`):** at ×0.36 every
training-free rule sits at ΔCE +0.06–0.08 at 8k/32k (fl20 +0.071/+0.078, first128 +0.069/+0.065,
attnrow20 +0.059/+0.077, grad20 +0.073 at 8k) — the gradient oracle buys nothing over first/last
tokens, so *which* tokens are kept is not the lever on pretraining text; the pooled remainder is
what is unreadable. Random-chunk pooling (rand20/k0, ×0.20–0.25) is +0.15–0.24 at 8k, +0.06–0.20 at
32k. rule20 is the worst rule (+0.13). → the 4B runs test whether TRAINING closes the gap.

**Launched (wave 2, commit e71683bbd; wave 1 died on a SyntaxError in 0fbd7ba60):** 4 arms × 3
budgets, 4 GPUs each, jupiter urgent unallocated, wandb group
https://wandb.ai/prasanns-allen-institute-for-ai/memory-networks/groups/sdcpt-q35-4b —
`dense`, `sd20` (random 20% of blocks whole, rest one slot, ×~0.2), `sfl20` (first_last 20% per
block, ×~0.36), `lslot20` (sd20 geometry, slot NOT detached, backbone frozen: 13.1M trainable
projector params) at 32M/64M/128M tokens (62/124/247 steps of 8×65k rows). sd20 runs ~3.5 s/step
on 4 GPUs. Experiment ids in `LAUNCH_LEDGER.tsv`.

⚠ **train/CE of the soft arms is scaled by the surviving-label fraction** — `loss_div_factor`
(`train_module.py:358`, `batch_num_tokens_for_loss`) is counted BEFORE compaction and the pooled
blocks' labels are dropped, so sd20 logs ~0.2 × the true per-token CE (0.45 vs dense 2.3). Adam
cancels the gradient scale (up to eps / the skip-step statistic); do not read the wandb train CE
across arms — the dev-loss eval is the comparison. (ds64 never hit this: its loss tokens were the
FREE answer span, which always survives.)

**Eval:** `eval_sweep.sh` submits `eval_cpt_devloss_beaker.sh` per run (waits inside the job for
the checkpoint); results → weka `softdetach_cpt/devloss/<run>.json`, `EVAL_LEDGER.tsv`.

## First trained numbers (2026-09-21 22:30 PDT; dev = 32 held-out 64k rows, CE nats/token)

| run | actual PF | dense-priced PF | full-attn dev CE | tail-20% CE | own construction CE @ ×comp |
|---|---|---|---|---|---|
| dense-32M | 1518 | 1518 | 1.283 | 1.308 | — |
| sd20-32M | 193 (0.13×) | 1518 | 1.291 | 1.319 | 1.533 @ 0.19 |
| sd20-64M | 383 (0.13×) | 3011 | **1.279** | 1.309 | 1.513 @ 0.19 |
| sfl20-32M | 172 (0.11×) | 1518 | 1.308 | 1.334 | 1.787 @ 0.20 |
| sfl20-64M | 340 (0.11×) | 3011 | 1.295 | 1.323 | 1.758 @ 0.20 |

Read: at **0.25× dense-32M's FLOPs** (sd20-64M, 383 vs 1518 PF) the random-chunk soft arm reaches
dense-32M's full-attention dev CE (1.279 vs 1.283); sfl20 trails sd20 by ~0.015. The compression is
a *training* saving only: scored under its own construction the model is +0.23 (sd20) / +0.46
(sfl20) worse — the ds64 pattern. Pending: dense-64M/128M, sd20/sfl20-128M and the three lslot20
evals (learned slot), which decide whether the own-construction gap can be closed. The base's CE on
this dev set is not yet measured (add `dense-0M` = the raw base through `eval_cpt_devloss.py`).
`collect_devloss.py` builds the table from the Beaker logs (`softdetach_devloss.csv`).

## Result table (2026-09-21 22:50 PDT; dense-128M pending) — `softdetach_devloss.csv`

| arm | 32M (PF · full CE · own CE) | 64M | 128M |
|---|---|---|---|
| dense | 1518 · 1.283 · — | 3011 · 1.256 · — | 5998 · pending |
| sd20 (random 20% of blocks whole, rest one slot) | 193 · 1.291 · 1.533 | 383 · 1.279 · 1.513 | 763 · 1.272 · 1.502 |
| sfl20 (first_last 20% of every block) | 172 · 1.308 · 1.787 | 340 · 1.295 · 1.758 | 678 · 1.288 · 1.740 |
| lslot20 (slot trained, backbone frozen) | 193 · 1.320 · 1.616 | 383 · 1.320 · 1.614 | 763 · 1.320 · 1.595 |

Frozen base = 1.320 (lslot20's full-attention column). **sd20-64M (383 PF) reaches dense-32M's
dev CE (1.279 vs 1.283) at 0.25× the FLOPs; sd20-128M (763 PF) is below it at 0.50×.** Random
whole-block pooling beats the first_last-20% rule at every budget (−0.015), the opposite of the
frozen-base screen — with training, the model learns to use the slot of a fully pooled block better
than it learns to read partial blocks. The saving is training-only: under its own construction sd20
is +0.23 above full attention and the projector-only learned slot (lslot20) does not close it
(1.60 vs sd20's 1.50 with a trained backbone).

Screening on the CTC rows (`screen_table.py --task ctc`): no training-free rule is near parity at
×0.2–0.3 without gold (best ΔCE +0.6, oracle included); `first128` (+0.05–0.07) only because
suite documents are short (×0.85).

## Downstream CTC-bench as an alternative metric (launched 2026-09-21 23:20 PDT)

Prasann: do downstream numbers rank the arms the way dev loss does? 24 Beaker jobs (jupiter,
urgent, unallocated): {base, dense 16M/32M/64M/128M, sd20 64M/128M, sfl20 64M} × {contra, nq,
oolong}, the standard `run_q4b_beaker_multirung_eval.py` path with **`--prompt-format raw`** (these
are raw CPT checkpoints, no SFT), `--query-position both`, v2_clean bundle, 2k–32k figure rungs only
(no xlong/YaRN/OOD — deliberate, see the ledger header), eval_size 500/rung. Ledger:
`records/eval_launches/2026-09-21_sdcpt-q35-4b_ctcbench.yaml` (+ rows in `EVAL_LEDGER.tsv`);
results → weka `softdetach_cpt/ctc_eval/<run>/`; table via
`python src/scripts/train/memexpress/cpt/softdetach/ctc_eval_ledger.py collect .../ctc_eval_wave.tsv`
(reads the `[ladder:task@rung] f1=` lines from the Beaker logs → `ctc_eval_results.json`).

Smoke (nq, 40 rows ⚠ SE ≈ 0.07): dense-64M 0.65 / 0.325 / 0.125 / 0.0 and sd20-64M 0.80 / 0.35 /
0.15 / 0.0 at 3k/8k/16k/32k — raw-format output is parseable (no format collapse at short rungs),
sd20 ≥ dense at every rung, and BOTH arms read 0.0 at 32k: check generations/parse rate at 32k
before reading that as capability (raw prompt + 64-token greedy continuation at 32k is the likely
artifact; `native-eval-repetition-loop-bug`). Table + dev-loss agreement: fill in when the wave lands.

## FINAL results (2026-09-21 23:25 PDT) — all 18 runs evaluated

Dev = 32 held-out 64k rows (`cpt_dev`, source part 28). CE = full-attention dev CE (nats/token);
ΔCE = **paired** per-row delta vs dense-32M (SE 0.001–0.007); own = the arm's own construction.
PF = `flop_meter/actual_pflops` (dense-priced PF = the dense column at the same token budget).

| tokens | dense: PF · CE | sd20 (random 20% of blocks whole): PF · CE · own | sfl20 (first_last 20%): PF · CE · own | lslot20 (slot only trained): PF · CE · own |
|---|---|---|---|---|
| 4M | 196 · 1.314 | — | — | — |
| 8M | 392 · 1.303 | — | — | — |
| 16M | 759 · 1.296 | 97 · 1.309 · 1.560 | — | — |
| 32M | 1518 · 1.283 | 193 · 1.291 · 1.533 | 172 · 1.308 · 1.787 | 193 · 1.320 · 1.616 |
| 64M | 3011 · 1.256 | 383 · **1.279** · 1.513 | 340 · 1.295 · 1.758 | 383 · 1.320 · 1.614 |
| 128M | 5998 · 1.245 | 763 · **1.272** · 1.502 | 678 · 1.288 · 1.740 | 763 · 1.320 · 1.595 |

Frozen base = 1.320. **Matched-FLOP, against MEASURED dense anchors:** sd20 is ~0.024 nats better
than dense at every equal-FLOP point — 193 PF: 1.291 vs dense-4M 1.314; 383 PF: 1.279 vs dense-8M
1.303; 763 PF: 1.272 vs dense-16M 1.296 (paired SE ≤0.005). Equivalently **sd20 reaches dense's
dev loss with ~4× fewer FLOPs** (sd20-64M 383 PF = dense-32M 1518 PF: 1.279 vs 1.283, Δ −0.004 ±
0.004; sd20-128M 763 PF beats dense-32M by −0.011 ± 0.005 at 0.5×, but not dense-64M). sfl20 is
only ~0.007 better than dense at equal FLOPs; the learned-slot-only arm never moves the backbone
(its full-attention CE is the base) and improves its own construction by just 0.02 over three
budgets. **Random whole-block pooling with a detached, untrained slot is the winning arm.**

The saving is training-only: under its own compressed input sd20 is +0.23 above full attention
(1.50 vs 1.27) — the ds64 pattern — so this buys cheaper CPT, not cheaper inference.

Screening on the frozen base predicted the opposite ranking (rand20 worst among the ×0.2–0.36
rules, fl20/attnrow20 best): training changes what the model can read out of a slot, so the frozen
screen ranks *inference* usability, not *training* value. Use the frozen harness to pick the
inference-side rule; use a short trained ladder (these 4-GPU runs are 5–20 min each) for the
training-side one.

Artifacts: `softdetach_devloss.csv` (collector), weka `softdetach_cpt/devloss/<run>.json`,
`LAUNCH_LEDGER.tsv` / `EVAL_LEDGER.tsv`, wandb group
https://wandb.ai/prasanns-allen-institute-for-ai/memory-networks/groups/sdcpt-q35-4b.
