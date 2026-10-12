# CTC figures: settings audit and unification options

Written 2026-08-19. Covers every CTC figure currently in `paperdraft2` plus the two tables, what
experimental settings each one selects, and what it would cost to put them on a common task set.

Source of truth for the coverage numbers: `paperdraft/figures/data/*.csv`,
`debug/ctc_final_suite/suite_table.json`, and `visualizations/make_family_figure.py`. The matrix in
§3 was computed from those files, not read off the README.

## 1. What is actually in the live draft

`paperdraft2` includes seven CTC figures and two tables. All but one sit in
`4_contradiction_case_study.tex`:

| Label | File | Section |
|---|---|---|
| CTC-PENALTY | `ctc_figure2.pdf` | 1_introduction |
| CTC-GRID | `ctc_figure.pdf` | 4_contradiction_case_study |
| CTC-RETENTION | `ctc_lengthgen_retention.pdf` | 4 |
| CTC-SCALE-ROW | `ctc_scale_row.pdf` | 4 |
| CTC-SCALE-BARS | `ctc_scale_bars.pdf` | 4 |
| CTC-FAMILY | `ctc_family_figure.pdf` | 4 |
| CTC-QDMATCH-COMPACT | `ctc_qdmatch_compact.pdf` | 4 |
| CTC-SUITE-HIGH / LOW | `ctc_suite_{high,low}.pdf` | appendix |
| TABLE-DATASTATS | `table_datastats` | appendix |
| TABLE-MASKMIX | `table_maskmix` | appendix (`app:mask-mix-ablation` refers to it) |

Not in the draft, still maintained: CTC-SCALE, CTC-LENGTHGEN (the 4-panel version; only panel (b)
ships as CTC-RETENTION), CTC-QDMATCH (the full version), CTC-FAMILY-BARS, both PASTEL recolours.

## 2. Settings selected per figure

Everything is Qwen3.5-4B, full vs chunked-mix, 2k–32k unless the row says otherwise.

| Figure | Model(s) | Tasks | Rungs | Arms |
|---|---|---|---|---|
| CTC-GRID | Qwen3.5-4B | 22 in the Average; 4 drawn as panels (outlier scale-k, reorder, contradiction, textgroups) | 2k–32k | full, chunked-mix |
| CTC-PENALTY | Qwen3.5-4B | same 22 | 2k–32k bucketed | full, chunked-mix |
| CTC-RETENTION | Qwen3.5-4B | 14 (dense only) | 32k/64k/128k | **full only** |
| CTC-LENGTHGEN | Qwen3.5-4B | 14; panel (c) qdmatch NQ is the **only** task with a chunked arm past 32k; panel (d) outlier fix-k on a doc-count axis | 32k/64k/128k | full; chunked for 1 task |
| CTC-SCALE-ROW | Qwen3.5 **0.8B/2B/4B** | 4: hpqa, fiqa, contradiction, qdmatch NQ | 2k–32k | full, chunked-mix |
| CTC-SCALE-BARS | Qwen3.5 0.8B/2B/4B | 4: hpqa, fiqa, **reorder**, qdmatch NQ | **3 rungs** (8k/16k/32k; reorder 4k/8k/16k) | **full only** |
| CTC-FAMILY | **4 families** (Qwen3.5-4B, OLMo-3-7B, Olmo-Hybrid-7B, Llama-3.2-3B) | 2: contradiction, hotpotqa | 2.5k/2k–**16k** | full, chunked-mix |
| CTC-QDMATCH-COMPACT | Qwen3.5-4B | 4 corpora × {N, N²} as a ratio | 2k–32k | full, chunked-mix |
| CTC-SUITE-HIGH/LOW | Qwen3.5-4B | all 22, split by class | 2k–32k | full, chunked-mix |
| TABLE-MASKMIX | Qwen3.5-4B | 5 (→7, see §4) | 2k–32k | full, chunked-mix, **pure-chunked** |

## 3. Coverage matrix — which task is measured on which axis

Five axes: SUITE (the 2k–32k grid), SCALE (0.8/2/4B), LENGTHGEN (past 32k), FAMILY (4 model
families), MASKMIX (the pure-chunked control).

```
task              class  SUITE  SCALE            LENGTHGEN     MASKMIX  FAMILY
contra_real       N2     yes    0.8b/2b/4b       full only     yes      yes     <-- only task on all five
hpqa              N      yes    0.8b/2b/4b       full only     --       yes
fiqa              N      yes    0.8b/4b (no 2B)  --            yes      --
qdmatch_nq        N2     yes    0.8b/2b/4b       full+chunked  --       --
reorder           N2     yes    0.8b(full)/2b/4b --            yes      --
nq                N      yes    --               full only     pending  --
outlier_scalek    NM     yes    --               full only     yes      --
outlier_fixk      N      yes    --               full only     yes(new) --
qdmatch_fiqa      N2     yes    --               --            yes      --
(13 others)              yes    --               partial       --       --
```

**Contradiction is the only task measured on every axis.** That is why it carries §4 — but it also
means every cross-axis claim in the paper rests on one task unless a reader stitches figures
together across different task sets.

Three specific inconsistencies worth naming:

1. **CTC-SCALE-ROW and CTC-SCALE-BARS are both in §4** and are the same experiment rendered twice
   with different settings: different 4th task (contradiction vs reorder), 5 rungs vs 3, both arms
   vs full-only. The swap is deliberate and documented — contradiction is monotone in scale and
   never fails, so it shows no scale floor — but a reader sees two model-scaling figures that
   disagree about which tasks matter.
2. **FAMILY is the binding constraint at 2 tasks**, and one of its six series is missing
   (Olmo-Hybrid never trained on HotpotQA; its retrieval-family run was `qdmatch_hpqa`, a different
   class). Its ladder also stops at 16k while every other figure runs to 32k.
3. **LENGTHGEN is effectively a dense-only figure.** Only qdmatch NQ has a chunked arm past 32k, so
   the retention claim is about full attention, not about chunking.

## 4. Mask-mixing: the new outlier fix-k result changes the table's claim

Launched 2026-08-19, the pure-chunked control for `outlier_fixedM` (suite row `outlier_fixk`,
class **N**) on the checkpoint that trained on saturn 2026-08-15:

| rung | dense | chunked-mix | pure-chunked | gain | gain SE |
|---|---|---|---|---|---|
| 2048 | 0.8798 | 0.8306 | 0.2300 | +0.601 | 0.025 |
| 4096 | 0.9312 | 0.9306 | 0.1140 | +0.817 | 0.018 |
| 8192 | 0.9483 | 0.9672 | 0.0540 | +0.913 | 0.013 |
| 16384 | 0.9695 | 0.9608 | 0.0200 | +0.941 | 0.011 |

Mean gain **+0.818**, deepest rung **+0.941**. eval_size 500 at every rung, parse_rate 1.0.

The 32k rung was not run: `build_suite_table.py` drops `("outlier_fixk", 32768)` and flags the
chunked-mix number there as stale, so there is no valid baseline to subtract from.

**This carries the † flat-CE marker.** Extracted from the training job's log
(`01M02CZSB82MH6SXDYHE5EE3Q1`, 2500 logged steps): CE falls 1.313 → 0.677 inside the first decile
and is then flat — 0.724, 0.723, 0.765, 0.705, 0.713, 0.682, 0.727, 0.634, 0.726. Same signature as
contradiction (1.175 → 1.071) and reorder, not like fiqa (0.48 → 0.065) or qdmatch_fiqa
(0.57 → 0.27). So this is a failure to fit, and per the table's own doctrine it must **not** be
upgraded into a claim about what the chunked mask can represent.

**Consequence for the table's headline.** It currently reads "Mixing is worth ~0 on $O_T(N)$
retrieval (+0.011) and +0.505/+0.791 on outlier/contradiction", i.e. the CTC class predicts the
gain. With outlier fix-k in, that framing does not survive:

```
task            class  mean gain   pure-chunked CE
fiqa            N      +0.011      0.48 -> 0.065   (fits)
qdmatch_fiqa    N2     +0.031      0.57 -> 0.27    (fits)
reorder         N2     +0.182      flat            (dagger, floor-limited)
outlier_scalek  NM     +0.505      descends
contra_real     N2     +0.791      flat            (dagger)
outlier_fixk    N      +0.818      flat            (dagger)   <-- NEW, class N
```

The split is not N vs N². It is **whether the pure-chunked run fit its training data at all**.
With NQ added, the mask-mixing gain is close to MONOTONE in the pure-chunked arm's final CE, and
CTC class has no predictive power at all:

```
task            class  mean gain   pure-chunked final CE
nq              N      +0.002      1.82 -> 0.062   fits
fiqa            N      +0.011      0.48 -> 0.065   fits
qdmatch_fiqa    N2     +0.031      0.57 -> 0.27    fits
reorder         N2     +0.182      flat            (dagger, floor-limited)
outlier_scalek  NM     +0.505      descends (curve not in purechunk_ce_curves.json)
contra_real     N2     +0.791      1.175 -> 1.071  flat (dagger)
outlier_fixk    N      +0.818      ~0.70 flat      (dagger)
```

Class N spans +0.002 to +0.818; class N2 spans +0.031 to +0.791. The two orderings are unrelated.
Rewrite the claim along the fitting axis, or quote the per-task numbers without the class
generalisation. ⚠ Note the honest caveat this creates: a large "gain" on a task whose pure-chunked
arm never fit is not evidence about what the mask can represent, so the three dagger rows should not
be read as "mask-mixing rescues hard tasks" — they say "mask-mixing rescues runs that otherwise fail
to optimise", which is a different and weaker claim.

An NQ pure-chunked run (`ctc-4b-nq-purechunk-08191654`, exp `01M0E76P7B4BDTQ4NPY9A4Z1ZM`,
seq_len 33280) was launched the same day and gives a second class-N point — importantly one on a
*retrieval* task, which is what the original claim was actually about. **It finished exit 0 and it
FIT**: CE 1.822 -> 0.062 (last-25 mean), min 0.0022, i.e. the fiqa pattern (0.48 -> 0.065), not the
flat-CE pattern. Under the fitting-not-class reading above, that predicts a mask-mixing gain near
zero.

**CONFIRMED 2026-08-19.** Full ladder, eval_size 500/rung, precision==recall at every rung (the
repetition-loop signature is precision collapsing below recall — absent here), `patch_debug`
applied==calls at every rung (so the chunked mask really reached the model; `calls=0` would be the
silently-unmasked bug):

| rung | dense | chunked-mix | pure-chunked | gain | gain SE |
|---|---|---|---|---|---|
| 2048 | 0.904 | 0.964 | 0.972 | -0.008 | 0.011 |
| 4096 | 0.930 | 0.942 | 0.932 | +0.010 | 0.015 |
| 8192 | 0.864 | 0.896 | 0.886 | +0.010 | 0.020 |
| 16384 | 0.856 | 0.886 | 0.884 | +0.002 | 0.020 |
| 32768 | 0.860 | 0.882 | 0.886 | -0.004 | 0.020 |

**Mean gain +0.002**, every rung inside one SE of zero. The prediction was recorded before the
numbers landed.

⚠ **NQ's ladder is MIXED and must stay that way.** Both suite arms were graded on `eval_rungs` at
2k/4k/8k and `eval_rungs_ext32k` at 16k/32k. The S3 default tree `ctc_eval_rungs/nq` now carries the
ext32k rebuild at the deep rungs (verified byte-size-identical against cubbins:
rung_16384 = 35,023,306 B, rung_32768 = 70,366,380 B), so the default `RUNG_TREE` reproduces the
baseline's ladder exactly. Do not "fix" this to a single tree.

## 5. Unification options

### Option A — cheapest coherent fix, no new training
Drop one of CTC-SCALE-ROW / CTC-SCALE-BARS. They are the same experiment and both are in §4. Keep
SCALE-ROW (5 rungs, both arms, so it says something about masks) or keep SCALE-BARS (shows the
minimum-viable-scale floor, which SCALE-ROW cannot). Cost: zero. Benefit: removes the most visible
disagreement in the section.

### Option B — unify on the CTC-SCALE-ROW four
Core = **hpqa, fiqa, contradiction, qdmatch NQ** (2 low, 2 high; already the SCALE-ROW panel set).
Missing cells, each one training run plus its eval ladder:

| axis | missing | runs |
|---|---|---|
| MASKMIX | hpqa, qdmatch NQ | 2 |
| FAMILY | fiqa × 3 families, qdmatch NQ × 3 families, Olmo-Hybrid hpqa | 7 |
| LENGTHGEN | fiqa (no rungs past 32k at all) | ladder build + evals |
| SCALE | fiqa @ 2B | 1 |

≈10 training runs. The FAMILY column is 7 of them and is the reason this is expensive.

### Option C — the corpus × CTC-order spine (most principled)
The paper already has this design in CTC-QDMATCH: the **same three corpora** (NQ, HotpotQA, FiQA)
asked an $O_T(N)$ question (retrieval) and an $O_T(N^2)$ one (qdmatch), so the CTC order is the only
thing that varies. Making that the spine of every figure would let one task set carry the whole
section, with contradiction retained as the $O_T(N^2)$ flagship.

Cost is higher than B (SCALE and LENGTHGEN each cover only one of the six), but it is the only
option where the task set itself encodes the paper's central variable.

### ⚠ Oolong regrade (2026-08-19) changes the low-CTC pool
The hybrid-vs-full oolong row was an eval artifact (repetition loops scored by a grader with no stop
condition); regraded it is at PARITY, not a 2-4x collapse. See
`results/ctc_suite/olmo3_vs_hybrid.md` note ¶ and [[native-eval-repetition-loop-bug]]. Two
consequences for this document: oolong is a *better* low-CTC candidate than the earlier draft
implied, and the `oolong` class conflict flagged in §3 resolves in favour of **N** (the parity
result is what an O(N) task should look like under a hybrid backbone).

### Recommendation
Do A now — it is free and fixes a real reader-facing contradiction. Then do B's MASKMIX column
(2 runs: hpqa, qdmatch NQ) since the mask-mixing table is the axis whose claim is currently
*wrong* (§4) and 2 runs would put its 4 tasks on the same set as SCALE-ROW. Treat the FAMILY
column as out of scope unless the family argument is being expanded — it is 7 runs to unify and
its current 2-task version already supports the claim it makes.

Leave LENGTHGEN as an explicitly dense-only figure and say so in the caption rather than paying to
add chunked arms past 32k.
