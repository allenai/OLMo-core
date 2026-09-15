# ds64 data-scaling campaign — handoff (state as of 2026-09-09 ~09:00 PDT)

Soft-token (pooled-document) training vs dense, on short-heavy 2k–64k mixes, Qwen3.5-4B, all on
Beaker. Goal (Prasann, 2026-09-08 evening): **is soft-KV training Pareto-optimal against dense
attention** on curves of accuracy vs training tokens / FLOPs / wall-clock, and **is the realised
speedup what the compaction ratio promises**? 27B is queued behind the 4B grid.

Plan + running status log: `records/ds64-scaling-plan.md`. This file is the "read me first".

**Update 2026-09-09 ~09:15.** Gen-3 relaunched clean: all 29 arms are up across all four tasks
(contradiction 5, oolong 8, nq 8, outlier 8), jobs `created`/`started`, no repeat of the
unpushed-commit trap. Two things changed underneath the numbers:
- `collect_ds64.py` now reads the FLOP meter itself and fills `flops_meter` / `dense_pflops` /
  `actual_over_dense` in `results/ds64/results.csv`, so the matched-FLOP table is reproducible from
  the CSV instead of by hand (§6). It also no longer caches a *mid-run* training log — it did, which
  froze partial FLOP and wall-clock readings for 81 of 88 runs.
- **First wall-clock read on gen-3, and length-sorting did NOT close the gap** (§2).

Live dashboard (regenerate with `python debug/ds64/make_pareto_artifact.py`, redraws from the CSV as
runs land): https://claude.ai/code/artifact/db31e725-e63b-48a7-b5a8-7af01b89038a

---

## 1. The headline so far

**Contradiction: the header-real soft token at keep 1/3 (`hdr33`) BEATS dense at matched training
FLOPs** — the first clean matched-compute win on this task (the 2026-09-02 study had it at
"parity to slight loss", 1.21x). Measured FLOP meter, mean f1 over 2k/8k/16k/32k/64k
(eval_size 500/rung; the 64k contradiction rung skips 19/500 rows over the 70k generation limit,
identically for every arm):

| arm | 16M tokens | 32M tokens | 64M tokens |
|---|---|---|---|
| dense | 0.763 @ 758 PF | 0.868 @ 1539 PF | 0.922 @ 3056 PF |
| hdr33 (headers real, gold + 1/3 docs) | 0.660 @ 277 PF | 0.773 @ 521 PF | **0.839 @ 1005 PF** |
| hdr17 (… + 1/6) | 0.441 @ 299 PF | 0.649 @ 549 PF | 0.750 @ 1076 PF |
| hdr08 (… + 1/12) | 0.259 @ 236 PF | 0.379 @ 436 PF | 0.504 @ 853 PF |
| runs08 (neighbour runs, 1/12) | 0.266 @ 153 PF | 0.402 @ 287 PF | 0.492 @ 571 PF |
| runs03 (neighbour runs, 1/36) | 0.221 @ 134 PF | 0.296 @ 262 PF | 0.309 @ 521 PF |

Interpolating the dense curve in log-FLOPs, `hdr33-64M` is **+0.034 above dense at the same
1005 PF** (0.839 vs 0.805) — this one comparison sits INSIDE the measured dense range and is the
trustworthy one. `hdr33-32M` (+0.066) and `hdr33-16M` (+0.046) look better still but both require
*extrapolating* dense below its cheapest measured point (758 PF).
**→ First job for the next agent: dense anchors at 4M and 8M** (see §5), exactly as the earlier
FLOP-scaling study had to do. Without them the cheap end of the curve is not evidence.

**Keep ratio is the load-bearing knob and eval-side parity did NOT transfer.** The 2026-09-08
eval-side probe found keep 1/36 reproduces full attention on held-out rows; in *training* it
collapses (hdr03 mean f1 0.11 at 16M, runs03 0.22) because a tiny real set teaches "the answer is
in a real document". Accuracy is monotone in keep at every budget, and only 1/3 tracks dense.
Contradiction's honest compaction is therefore ~0.33x FLOPs, not the 0.09x the probe suggested.

**KEEP-MODE RULE, CORRECTED 2026-09-14 — "gold-forcing leaks the answer on id-answer tasks" is
WRONG, and the 2026-09-02 report that said so was confounded by the gold off-by-one.**

We ran nq BOTH ways in ds64 with the gold index FIXED. Gold-forced wins at every budget:

| nq budget | dense | kv33 (forced) | kvgb (blind) | blind - forced |
|---|---|---|---|---|
| 16M | 0.844 | 0.792 | 0.746 | -0.045 |
| 32M | 0.865 | 0.849 | 0.821 | -0.028 |
| 64M | 0.907 | 0.878 | 0.825 | -0.054 |
| 128M | 0.915 | 0.887 | 0.850 | -0.038 |

`records/flop-scaling-report-2026-09-02.md` reported the opposite ("nq 1/6: 0.603 forced vs 0.728
blind") — but its *forced* nq/outlier arms carried the gold-index off-by-one
([[gold-sidecar-index-base-bug]]), so they kept the NEIGHBOUR document. That comparison was
blind-vs-broken, not blind-vs-forced. **Do not reuse its keep-mode recommendation.**

**The rule that actually holds: ask whether keeping gold real supplies EVIDENCE or leaks the LABEL.**

| task | what the gold doc holds | forcing it real |
|---|---|---|
| nq | the answer TEXT the model must read | **helps** (+0.04): blind pools the answer 2/3 of the time and those examples teach nothing |
| contradiction | the contradicting claim PAIR | **required**: blind at keep 1/6 shows both golds 1 time in 36 -> 0.053 |
| outlier | nothing but its own IDENTITY (it is the odd one out) | **catastrophic**: supplies a cue that replaces the skill; ~70x collapse |
| oolong | no gold subset at all | n/a, blind by construction |

So gold-forcing is not a "shortcut" in general — it is the supervision signal wherever the gold
document carries content the task requires. It only becomes a shortcut when the gold document's
*identity* is the label, which is true of outlier alone among these four.

**Consequence: nq should stay gold-FORCED (`kv33`).** The `kvgb` nq arms launched 2026-09-12 on the
old recommendation are a useful control, not an upgrade — do not promote them.

**OUTLIER RESOLVED (2026-09-13): gold-blind + keep 1/2 reaches matched-FLOP PARITY, and parity is
the ceiling. Dense anchors at 4M/8M are measured, so this is no longer extrapolation.**

Dense outlier, 4-rung mean, with the new cheap-end anchors:
`4M 0.317@196 | 8M 0.353@391 | 16M 0.400@780 | 32M 0.564@1582 | 64M 0.754@3168 | 128M 0.817@6334`

Soft arms vs dense at matched FLOPs — every delta <= 0:

| arm | f1 @ PF | dense@same | delta |
|---|---|---|---|
| kvgb50 32M | 0.350 @ 374 | 0.351 | **-0.001** |
| kvgb50 64M | 0.388 @ 761 | 0.398 | **-0.010** |
| kvgb50 16M | 0.285 @ 190 | 0.316 | -0.031 |
| kvgb 32M | 0.287 @ 309 | 0.341 | -0.054 |
| kvgb 64M | 0.279 @ 626 | 0.385 | -0.106 |
| kvgb50 128M | 0.430 @ 1569 | 0.562 | -0.131 |
| kvgb 128M | 0.356 @ 1298 | 0.518 | -0.161 |

Ranking of the levers: **keep 1/2 > keep 1/3, and the compression-mixing curriculum does nothing**
(kvgbmix 0.284 vs kvgb 0.287 at 32M — identical). Gold-blind is what rescued the task
(0.005 -> 0.350, ~70x); higher keep closed the rest of the way to parity.

⚠ **TRAP: `--max-tokens` below ONE step's tokens silently gives a 1-STEP run, not an error — and
those runs produce the LARGEST apparent matched-FLOP wins.** Soft arms do 128 rows x 65536 = **8.4M
tokens/step**, so the 4M and 8M soft budgets both collapsed to step 1 and came out byte-identical
(6.4 PF, f1 0.195/0.196). Sitting far left of the dense curve, they scored +0.055 / +0.064 at
"matched FLOPs" — the best deltas in the whole outlier ladder, and pure artefact (a barely-trained
model versus dense extrapolated 30x below its measured floor). Dense is immune because it runs
8 rows/step = 524k tokens, so 4M/8M map correctly to 8/16 steps.
**Rule: a budget below (rows/step x seq_len) is not expressible for that arm.** For the soft arms
that floor is ~8.4M tokens; to go below it, cut `--global-batch` too.
`make_pareto_artifact.py` now drops any run with `steps <= 2`.

⚠ **The extrapolation was wrong by a lot, in the flattering direction.** Before the anchors, dense at
~195 PF was projected at 0.07-0.15 from its log-linear slope; it actually scores **0.317**. Acting on
that projection would have produced a large claimed cheap-end win that does not exist. This is the
concrete vindication of the "dense anchors first" instruction that opened this handoff.

**WHY outlier cannot win — and the rule that explains every task in the campaign.** Compression beats
dense only where dense's own scaling curve is FLAT; where dense still converts compute into accuracy
efficiently, a cheaper method can at best sit on the curve.

| task | dense gain | over | soft-KV outcome |
|---|---|---|---|
| oolong | +0.087 | 7.8x FLOPs | dominates, 2.9x cheaper |
| nq | +0.071 | 8.1x | dominates, 2.4x |
| contradiction | +0.159 | 4.0x | dominates, 1.7x |
| **outlier** | **+0.500** | **32x** | **parity only** |

Outlier's curve is ~6x steeper than the tasks that win. **Recommendation: call outlier done at
parity.** The remaining levers (keep >1/2) only move it further right along a curve it already sits
on; they trade compression for accuracy at roughly the rate dense does, which is the definition of
"no Pareto gain". Predict this ratio before spending runs on a new task.

*(Qualified 2026-09-14: HEADER-REAL was never tried on outlier in TRAINING, and it is the
construction behind both winning tasks. Three arms are in flight -- see section 8. "Done at parity"
stands for the keep-ratio levers; it was never a verdict on the header rule.)*

**THREE of four tasks now beat dense with NO interpolation (2026-09-09, gen-3).** Each of these
dominates a dense point that was actually measured — at least as accurate AND strictly cheaper — so
none of them depends on the missing 4M/8M anchors:

**Oolong dominates at EVERY budget; nq at three of four.** Full gen-3 ladders (dominated dense point
in brackets):

| task | arm | 16M | 32M | 64M | 128M | FLOP ratio |
|---|---|---|---|---|---|---|
| oolong | ohdr33 | 0.615@171 [2.5x] | 0.654@312 [2.6x] | 0.655@638 [1.3x] | (eval running) | 0.38-0.41x |
| nq | kv33 | 0.792@149 | 0.849@286 [2.4x] | 0.878@556 [2.5x] | 0.887@1151 [1.2x] | 0.20-0.22x |
| nq | kv17 | 0.707@115 | 0.783@231 | 0.828@443 | 0.883@916 [1.5x] | 0.16-0.17x |
| contradiction | hdr33 | 0.652@229 | 0.810@437 [1.7x] | 0.855@854 | — | 0.28-0.30x |
| contradiction | hdr17 | 0.438@219 | 0.674@437 | 0.741@843 | — | 0.28-0.29x |
| dense oolong | | 0.603@419 | 0.644@811 | 0.661@1648 | 0.690@3270 | 1.00x |
| dense nq | | 0.844@683 | 0.865@1387 | 0.907@2777 | 0.915@5556 | 1.00x |
| dense contradiction | | 0.763@758 | 0.868@1539 | 0.922@3056 | (pool caps) | 1.00x |

nq compacts hardest (~0.21x) and its whole kv33 ladder sits left of dense's. Most dominated gaps are
within ~1 SE — phrase them as *parity at 1.2-2.6x less compute*. The one clear accuracy gain is
contradiction hdr33-32M at +0.047. The contradiction matched-FLOP headline also improved on gen-3:
**hdr33-64M is 0.855 @ 854 PF vs dense's interpolated 0.781, i.e. +0.075** (was +0.034 on gen-2).

⚠ **TRAP 6 BIT AGAIN, 2026-09-09 — a partial rung set was briefly read as a +0.104 oolong win.**
`ohdr33-u128M` showed mean 0.794, but its eval was still RUNNING with `n_rungs=2`: the mean covered
only 2k (0.876) and 8k (0.712), the two easiest rungs, with 16k/32k/64k ungraded. Those hard rungs
are what drag every other row down, so a partial ladder reads as a large gain. **Any query over
`results.csv` MUST filter `n_rungs == 5` (4 for outlier) — `mean_f1` being non-empty is not enough.**
`make_pareto_artifact.py` already enforces this, which is why the dashboard never showed the bad
point; the ad-hoc summary query did not.

**Check dominance first and the interpolated delta second.**

**CORRECTED 2026-09-09 — outlier is a LENGTH-GENERALISATION failure, not an at-chance bug.** An
earlier note in this file called outlier's soft arms "at chance at every rung"; that was read off the
16M runs alone and is wrong. Per-rung, keep 1/3 clearly learns the task at short context and loses it
as context grows, and it needs data to get there at all:

| run | 2k | 8k | 16k | 32k | 4-rung mean |
|---|---|---|---|---|---|
| dense 128M | 0.996 | 0.945 | 0.821 | 0.504 | 0.817 |
| kv33 128M | **0.890** | 0.163 | 0.030 | 0.018 | 0.275 |
| kv33 64M | **0.642** | 0.058 | 0.004 | 0.002 | 0.177 |
| kv33 32M | 0.024 | 0.006 | 0.005 | 0.007 | 0.011 |
| kv33 16M | 0.006 | 0.006 | 0.005 | 0.003 | 0.005 |
| kv17 (all budgets) | ≤0.008 | ≤0.032 | ≤0.006 | ≤0.007 | ≤0.011 |

Two different failures are in play and must not be pooled:
- **kv33 (keep 1/3)** reaches 0.890 at 2k against dense's 0.996, i.e. near parity, and its 2k score
  climbs 0.006 → 0.642 → 0.890 over 16M → 64M → 128M tokens. It then collapses from 8k onward. This
  is a real, quotable result: *the method works on outlier at short context and does not transfer to
  long context*.
- **kv17 (keep 1/6)** never clears 0.032 on any rung at any budget — degenerate, consistent with the
  keep-ratio collapse already documented for contradiction.

Training CE stays healthy throughout (0.05-0.09 vs dense's 0.16), so the long-context failure is a
train/eval distribution gap rather than underfitting: at training time most documents are pooled,
while at inference every document is real, and outlier is *defined* by comparing all of them. The
`--st-keep-mode gold_plus_random` shortcut (gold always real) remains the leading mechanism, and the
gold-blind probe below is still the decisive test.

**ROOT CAUSE (2026-09-09, corrected): ds64 forces GOLD REAL on nq and outlier, which the
2026-09-02 FLOP-scaling study had already measured as harmful on exactly those tasks.** An earlier
draft of this section blamed `--st-keep-frac` for disabling `--st-n-random-range`; that is true
mechanically but is NOT the explanation, because the prior study used a fixed keep fraction for both
its blind and its forced arms and the blind arms worked. The operative variable is the keep MODE.

`records/flop-scaling-report-2026-09-02.md`, keep 1/3, mean f1 (PF):

| task | gold-forced | gold-blind |
|---|---|---|
| nq @16M | 0.722 (172) | **0.832** (163) |
| outlier @160M | 0.202 (1992) | **0.369** (1753) |
| outlier @320M | 0.284 (4013) | **0.486** (3527) — vs dense 0.604 (3805) |
| contradiction @14M | **0.525** (166) | 0.268 (151); blind 1/6 = 0.053 |

That report states the rule outright: *"Forcing gold documents real leaks the answer on id-answer
tasks (nq 1/6: 0.603 forced vs 0.728 blind) but is required on contradiction"*, and *"[KV soft
tokens] fail where every document must be compared (outlier)"*. **ds64 nevertheless launched nq and
outlier with `--st-keep-mode gold_plus_random` (kv17/kv33), i.e. gold forced — against that finding.**
Oolong is gold-blind by construction (no gold subset) and is our best transferring task; contradiction
is correctly forced. So the ds64 arm table is right for two tasks and wrong for two.

**Subtlety that makes ds64's outlier look worse than the prior forced arms.** Those prior forced
nq/outlier arms were trained with the gold-index off-by-one ([[gold-sidecar-index-base-bug]], fixed
2026-09-08), so they kept the NEIGHBOUR document, not the gold one — they were only accidentally
"forced". ds64 has the fix, so its outlier runs are the first *genuinely* gold-forced ones, and they
are worse (ds64 kv33-128M scores 0.163/0.030/0.018 on 8k/16k/32k). That is what the shortcut story
predicts: correctly forcing the true outlier into the kept set is precisely what teaches
"the answer is one of the few real documents", a cue that is worthless at all-real inference.

**LOCAL PROBE (2026-09-11, `debug/ds64/probe_outlier_keepset.py`, CPU only) — the shortcut measured,
and why it bites harder with length.** Running the real `select_keep_docs` on outlier-shaped rows at
keep 1/3:

| rung | docs | kept | gold_plus_random | gold_subsample | gold-blind | distractors the cue hides |
|---|---|---|---|---|---|---|
| 2k | 15 | 6 | **1.000** (2.5x) | 1.000 | 0.340 | 9 |
| 8k | 55 | 19 | **1.000** (2.9x) | 1.000 | 0.340 | 36 |
| 16k | 110 | 37 | **1.000** (3.0x) | 1.000 | 0.340 | 73 |
| 32k | 220 | 74 | **1.000** (3.0x) | 1.000 | 0.340 | 146 |
| 64k | 448 | 150 | **1.000** (3.0x) | 1.000 | 0.340 | 298 |

(P = how often the answer document is among the REAL ones in training. At eval every document is
real, so the cue is worth 1x there.)

Three things fall out:
1. **`gold_plus_random` pins P(gold real) at 1.000 by construction** — the shortcut is not a
   statistical tendency, it is a guarantee the model can rely on every single step.
2. **Gold-blind is the only mode that removes it** (0.340 = the base rate). `gold_subsample` is
   *also* 1.000 here: it keeps `n_gold` randomly chosen gold docs, and with a single gold document
   subsampling 1 of 1 is a no-op. The mode's base-rate-preserving property only helps multi-gold
   tasks, so it is NOT an alternative for outlier. That leaves `kvgb` as the only available fix.
3. **The cue's ratio is constant (3.0x) but the absolute distractors it hides grows linearly with
   context** — 9 at 2k, 298 at 64k. That, not the ratio, is why the collapse deepens rung by rung,
   and it tracks the measured per-rung scores almost exactly (0.890 / 0.163 / 0.030 / 0.018).

**Fix: run outlier (and ideally nq) GOLD-BLIND.** Arm `kvgb` = `--st-gold-blind --st-keep-prob 0.3333`
is defined and dry-run clean in `launch_ds64.py` (ARM_MICRO 2). `kvrb`
(`--st-n-random-range 128,512`) remains worth one run — the prior report lists breadth-matched KV as
its own "not done / next" item — but gold-blind is the primary, with a measured 0.486-vs-0.284 prior.

    DS64_GEN=3 $PY debug/ds64/launch_ds64.py --tasks outlier --budgets 64M,128M --arms kvgb,kvrb launch

**Methodological lesson: judge an arm by its BEST rung, not its mean.** Averaging 0.890/0.163/0.030/
0.018 gives 0.275 and reads as "broken"; the mean hid a near-parity short-context result. The
dashboard now flags an arm suspect only if it clears no rung (kv17), and separately marks
"solves short context, collapses long" (kv33).

**Watch item RESOLVED: gen-3 is accuracy-neutral.** hdr33 gen-2 -> gen-3 by budget: 16M
0.660->0.652 (-0.008), 32M 0.773->0.810 (+0.037), 64M 0.839->0.855 (+0.016). Mean +0.015 against a
per-rung SE of ±0.021, and the sign is not consistent — the 32M pair was the outlier, not a trend.
Sorting changes speed, not the result, and gen-2/gen-3 numbers can be pooled.

**Oolong's first soft arm, in detail (the original no-interpolation result):**
`ohdr33-b128f3-u16M` scores **0.615 mean f1 at 171 PF** against dense's own 16M run at
**0.603 @ 419 PF** — at least as accurate as a point that was actually *measured*, for **0.41x** its
FLOPs (2.5x cheaper). The +0.012 accuracy edge sits inside the ±0.022 per-rung SE, so quote it as
*parity at 2.5x less compute*, not as a gain. This is a stronger class of evidence than the
contradiction headline: dominating a measured dense point requires no dense curve interpolation, so
it survives the missing 4M/8M anchors entirely. Rungs 2k/8k/16k/32k/64k = 0.86/0.64/0.56/0.51/0.50.
Oolong's realised compaction (0.41x) beats the ~0.6x floor its header-heavy lines predicted.
**When reading the rest of the soft arms, check dominance first and the interpolated delta second.**

**Other tasks: dense ladders complete, remaining soft arms still landing** (they were gated behind this
ablation, §3). Dense mean f1 (2k/8k/16k/32k/64k):

| task | 16M | 32M | 64M | 128M |
|---|---|---|---|---|
| nq | 0.844 | 0.865 | 0.907 | 0.915 |
| oolong | 0.603 | 0.644 | 0.661 | 0.690 |
| outlier | 0.322 | 0.452 | 0.606 | 0.659 |
| contradiction | 0.763 | 0.868 | 0.922 | (pool caps at 64M) |

Note outlier's 64k rung is ~0.0 at every budget (its 64k eval file is the v2 xlong one with random
fillers) — treat outlier@64k as broken, not as a result. **The outlier row above is the 5-rung mean
and is therefore deflated by ~0.15**; on the four working rungs (2k/8k/16k/32k) dense reads
0.400 / 0.564 / 0.754 / 0.817 at 16M / 32M / 64M / 128M. The dashboard now scores outlier on those
four rungs for every arm — averaging a rung that is ~0.0 for everything only compresses the
differences the campaign is trying to measure. Quote the 4-rung number for outlier, and say so.

---

## 2. Wall-clock: NOT yet answered, and it is the open risk

FLOPs are down 3x; **wall-clock is not**. Chronology of what was fixed and what remains:

1. **65536 microbenchmark (1 GPU, B=1, `debug/pooled_kv/bench_softtoken_throughput.py`, job
   01M226J7RBPPE8WS418AHER54H):** per-step time tracks compaction exactly — dense/flash 9.70 s,
   soft k=1/3 2.79 s, soft k=1/12 0.85 s, each equal to dense on a row of the compacted length.
   Backend irrelevant there; the `+log L` bias path is 2.7x slower (unused).
2. **On the real 8-GPU FSDP runs the soft arms were ~55–65 s/step** against dense ~10 s at 0.33x
   the FLOPs. Three causes found, two fixed:
   - *torch SDPA on padded multi-row micro-batches* is ~4x slower than flash (local test: micro 8,
     torch 234 s vs flash 71 s; equal at micro 1–2). → soft arms now run `--attn-backend flash_2`.
   - *per-document host syncs* in `compact_pooled_rows` (two `.item()` per pooled doc) stalled the
     FSDP collectives. → vectorised to one sync per row (commit 96f2fba11 lineage).
   - *padding waste inside a micro-batch*: compacted rows are padded to the longest member, so a
     56k example beside seven 2k ones wastes ~6x. Diagnostic: 8 rows/micro 55 s/step, 1 row/micro
     24 s/step, gold-blind (hook removed) 64 s/step → the keep hook is innocent.
     → **`microbatch_sort_pad_id`** added to `TransformerTrainModuleConfig`: each rank's batch is
     length-sorted before the micro-batch split, so micro-batches are length-homogeneous. Set
     automatically for `--variant softtoken`. Smoke-tested on 8 local GPUs; **its effect on Beaker
     step time is still unmeasured** — that is what the gen-3 runs will show.
3. Real fix if sorting is not enough: compact-then-**pack** the compacted rows with `cu_seqlens`
   (no padding at all). Not implemented.
4. **2026-09-09: sorting helps where there is padding to remove, and it is still not enough.**
   Two finished gen-2/gen-3 pairs, contradiction 16M, 8xH100 (`micro` = rows per micro-batch):

   | arm | budget | micro | gen-2 s/step | gen-3 s/step | change |
   |---|---|---|---|---|---|
   | dense | 16M | — | 9.7 | — | control |
   | hdr33 | 16M | 2 | 86.3 | 101.9 | 1.18x worse |
   | hdr17 | 16M | 4 | 82.7 | **52.5** | 1.58x better |
   | hdr17 | 32M | 4 | 65.2 | **41.7** | 1.56x better |

   The split is mechanistic, not noise: sorting can only recover padding that a micro-batch actually
   contains, so `micro=4` gains and `micro=2` has almost nothing to sort. Read the hdr33 regression
   as drift ([[gpu-benchmark-intra-process-drift]], up to 1.9x across jobs), not as sorting hurting —
   the hdr17 gain reproduces at two budgets (1.58x / 1.56x), which drift would not do.
   **But the FLOP win still does not convert:** even hdr17's improved 52.5 s/step is 5.4x dense's
   9.7 s/step while spending ~0.35x its FLOPs — ~15x worse FLOP utilisation. Compact-then-**pack**
   (item 3) remains the lever; sorting was a partial fix, not a dead end. Raising `ARM_MICRO` for the
   soft arms now has upside it did not have before sorting — worth one probe.

   *(An earlier note in this file, written from the hdr33 pair alone, said sorting "moved it the
   wrong way" and to stop spending runs on it. The hdr17 pair refutes that; this table supersedes it.)*

5. **The 900 s NCCL watchdog is now a hard failure mode.** `ds64-oolong-ohdr17-b128f3-u16M` and
   `-u32M` both died twice with
   `Watchdog caught collective operation timeout: WorkNCCL(..., OpType=_REDUCE_SCATTER_BASE, Timeout(ms)=900000)`
   — a single step exceeded the process group's 900 s timeout, so FSDP aborted with SIGABRT and no
   Python traceback. oolong is the worst case for padding: its `PadToLengthInstanceSource` expands an
   8.9M-token shard to **238M** padded tokens (27x). `ohdr33` (micro 2) survives on the same data;
   only `ohdr17` (micro 4) times out. Fix is `ARM_MICRO["ohdr17"] = 2` (or 1) in
   `debug/ds64/launch_ds64.py`; a longer timeout would only hide a step that is already pathological.
   **APPLIED 2026-09-09** (`ARM_MICRO["ohdr17"] = 2`). No commit/push is needed for this one:
   `launch_train` shells out to `launch_ds64.py`, so `--micro-batch-instances` is built locally and a
   plain edit takes effect on the next launch. Micro-batch is gradient-accumulation granularity only
   — rows/step stays 128 — so comparability with the other arms is untouched.

   The three already-FAILED ohdr17 arms (u16M/u32M/u128M; u64M was still running on micro 4 and will
   probably join them) do NOT come back on their own: the orchestrator launches only names absent
   from `state["runs"]` and never retries a `FAILED` one. Recover with
   **`debug/ds64/reset_failed_arms.py`**, which drops FAILED entries after a timestamped backup and
   clears the latched `done` flag (trap 7):

       pkill -f orchestrate_ds64.py
       python debug/ds64/reset_failed_arms.py --arms ohdr17     # --dry-run to preview first
       DS64_GEN=3 setsid nohup python debug/ds64/orchestrate_ds64.py >> debug/ds64/orchestrator_ds64.log 2>&1 &

6. **Sorting changes speed, not the result.** gen-3 vs gen-2 on hdr33-u16M, rung by rung:
   0.919/0.852/0.719/0.534/0.276 -> 0.920/0.851/0.713/0.517/0.261, mean 0.660 -> 0.652 against a
   per-rung SE of +/-0.021. So gen-2 accuracy stays valid and the two generations can be pooled;
   only the wall-clock columns differ.

**So: quote FLOPs today, do not quote wall-clock.** `results/ds64/results.csv` carries
`gpu_hours` per run, but every `-b128f2-` number in it is from the pre-sorting recipe and is
wall-clock-invalid (accuracy is fine). Only `-b128f3-` runs are wall-clock-faithful.

---

## 3. What is running right now

- **Orchestrator** (login node, restarted 08:51): `debug/ds64/orchestrate_ds64.py`, state
  `debug/ds64/orchestrator_ds64_state.json`, log `debug/ds64/orchestrator_ds64.log`. It launches
  training, then the 5-rung eval per finished run, retries once/thrice, harvests every 90 min.
  Restart with:
  `DS64_GEN=3 setsid nohup python debug/ds64/orchestrate_ds64.py >> debug/ds64/orchestrator_ds64.log 2>&1 &`
- **4 gen-4b outlier `xh2` runs** (`xh2k50`/`xh2warm17` x 16M/32M), launched 2026-09-14 19:22 PDT,
  4 GPUs each on jupiter+saturn -- **section 9**. They ask whether xhdr's collapse is an
  optimization shortcut or a hard leak; `xh2warm17` warm-starts from the finished `kvgb50-16M`
  checkpoint, so its budget and FLOPs are INCREMENTS (add 16M / 190.3 PF).
- **9 gen-4 outlier header-real runs** (`xhdr33`/`xhdr17`/`xhdr00` x 16M/32M/64M), launched
  2026-09-14 14:26 PDT -- **DEAD and all cancelled** (section 8's verdict,
  `debug/ds64/xhdr_collapse_diagnosis.md`); their 9 state entries are latched FAILED so a restart
  cannot resurrect them (trap 8). Job ids in section 8.
- **29 gen-3 runs relaunching** (`-b128f3-` names): contradiction hdr33/hdr17, oolong ohdr33/ohdr17,
  nq kv33/kv17, outlier kv33/kv17 at every budget. These are the wall-clock-faithful,
  keep-ratio-sane arms; the whole first launch of them died on an unpushed commit (§4, trap 1) and
  was reset at 08:51.
- Generations in run names: no tag = dense; `-b128-` torch backend (cancelled); `-b128f-` flash,
  per-doc syncs (cancelled); `-b128f2-` flash + vectorised syncs (**accuracy valid**, wall-clock
  not); `-b128f3-` + length-sorted micro-batches (**both valid**). Two very early `hdr03-u16M` /
  `runs03-u16M` runs carry no generation tag and used 16 rows/step — reference only.

---

## 4. Traps hit (all cost real time; do not repeat)

1. **An unpushed commit kills every Beaker job** with `fatal: remote error: upload-pack: not our
   ref <sha>`. It killed all 30 gen-3 launches and several evals. `git push` BEFORE any launch,
   and if a wave of jobs fails at once, check this first.
2. **Beaker data builds:** use `/opt/conda/bin/python` with `pip install -e . ./ctc` and gantry
   `--install false`. Gantry's uv venv has neither `pip` nor torch; the image's conda python lacks
   the repo. `ctc-data` must come from `origin/prasann/ctc_public:ctc/` (pip cannot clone
   github.com/PrasannS/ctc from a job). Run it as `python -m ctc.data.cli`.
3. **Batching arithmetic:** rows/step must be divisible by micro-batch × 8 GPUs, so micro ∈
   {1,2,4,8,16}. micro 6 dies with `global batch size ... must be divisible by`.
4. **Rows/step must be sized in TOKENS, not rows.** The short-heavy mix averages ~4.5k
   tokens/example, so the first soft launch at 16 rows/step gave the soft arm 7x dense's optimizer
   steps at 0.14x its tokens/step. 128 rows/step ≈ dense's 524k tokens/step.
5. **contradiction's pool caps at ~18k distinct 2k examples** (`ctc-data` refuses near-duplicates),
   so its grid stops at 64M and its 2k pool is built with `POOL_2K=15000`.
6. The collector used to cache eval logs mid-run and froze partial rung sets; it now caches only
   finished logs. If a run shows fewer than 5 rungs, delete `results/ds64/logs/eval_*.log`.
7. The orchestrator latches a `done` flag; after a full pass it exits immediately with `ALL_DONE`
   on restart. Clear `done` in the state JSON before restarting it for new work.

---

## 5. Next steps, in priority order

1. **Dense anchors at 4M and 8M tokens** — **now a one-liner; no data build needed.** BUILT and
   dry-run clean 2026-09-09: `launch_ds64.py` maps any budget below `SHARD_FLOOR` (16M) onto the 16M
   shard plus `--max-tokens <budget>`, which rides through the existing `--extra-args` passthrough.
   That is a *real* budget, not a truncation: the trainer sets `max_duration` from it so warmup and
   linear decay are relative to the short budget (`--max-steps` would hard-stop mid-schedule and
   silently understate the anchor, flattering every soft arm). Legitimate because
   `compose_uniform_arms.py` emits nested prefixes, so a 4M slice carries the same short-heavy mix.

       PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
       $PY debug/ds64/launch_ds64.py --tasks contradiction,oolong,nq,outlier --budgets 4M,8M --arms dense launch

   ~8 runs at ~8 and ~15 steps each (dense is 8x65536 = 524k tokens/step), so roughly 2 GPU-h total.
   To have the orchestrator launch AND eval them automatically instead, add `"4M", "8M"` to the front
   of `BUDGETS` (and to `TASK_BUDGETS["contradiction"]`) and restart it — one restart can also carry
   the `ohdr17` reset. Without these anchors every matched-FLOP claim below 758 PF is extrapolation —
   though note the oolong dominance result (§1) needs no interpolation and stands regardless.
2. **Measure gen-3 wall-clock**: compare a `-b128f3-` run's seconds/step with its `-b128f2-` twin
   and with dense (`debug/flop_scaling/collect_walltime.py --states debug/ds64/orchestrator_ds64_state.json`,
   or the `train_seconds`/`gpu_hours` columns of `results/ds64/results.csv`). If sorting did not
   close the gap, implement compact-then-pack with `cu_seqlens` (§2.3).
3. **Read out the other three tasks' soft arms** as the gen-3 evals land, and build the same
   FLOP-matched table. Oolong is the arm most likely to win (its 2026-09-02 result was the one
   matched-compute win, and its headers are half the line so its compaction floor is ~0.6).
4. **Plots**: performance vs tokens / vs FLOPs / vs GPU-hours per task.
   `debug/flop_scaling/make_axes_plots.py` is the template; nothing equivalent exists for ds64 yet.
5. **27B**: `DS64_SCALE=27b` on titan (B200), base `q35-27b-base-markerfix`, evals need ≥192 GB
   GPUs. Only worth starting once the 4B wall-clock story is settled.
6. Optional: a `hdr50`/`hdr66` arm — accuracy is monotone in keep and 1/3 is the cheapest keep that
   works, but nobody has checked whether 1/2 buys back the remaining gap to dense at lower cost
   than more tokens does.

---

## 6. Where everything lives

- **Code**: `debug/ds64/` — `build_ds64_data_beaker.sh` (per-rung `ctc-data` pools 2k…56k,
  short-heavy shares 30/20/15/13/12/10, nested arms, Qwen3.5 marker shards → weka
  `ds64/shards/<task>_u<B>`), `compose_uniform_arms.py`, `launch_ds64.py` (arm definitions,
  `ARM_EXTRA` / `ARM_MICRO` / `TASK_ARMS` / `TASK_BUDGETS`), `orchestrate_ds64.py`,
  `collect_ds64.py`, `harvest_ds64.sh`, `soft_arms.json` (gate for non-contradiction soft arms),
  `data_build_jobs.tsv`, `LAUNCH_LEDGER.tsv`.
- **Results**: `results/ds64/results.csv` (per-run per-rung f1, FLOP meter, wall-clock),
  `results/ds64/logs/` (cached Beaker logs), `results/ds64/pareto.html` + the artifact above.
- **Where the FLOP number comes from — the one trap to know.** Use `flop_meter/actual_pflops`, which
  a finished run prints in its wandb summary block and which charges the *compacted* sequence. Do NOT
  use the trainer's `throughput/total petaflops`: it charges every token the DENSE per-token cost
  (see the docstring in `train/callbacks/flop_meter.py`), so on a soft-token arm it reads ~10x the
  real compute and inverts the whole result — 11,053 PF instead of 277 PF on hdr33-16M.
- **Method code**: `mark_doc_headers_free` (`nn/attention/chunked_mask.py`),
  `enable_pooled_soft_tokens(header_stop_id=…, detach_soft_gdn=…)` (`nn/transformer/model.py`),
  `neighbour_runs` (`nn/attention/pooled_doc_kv.py`), `microbatch_sort_pad_id`
  (`train/train_module/transformer/{config,train_module}.py`), trainer flags
  `--st-header-stop-id/-count`, `--st-neighbour-runs`, `--st-no-detach-soft-gdn`, `--torch-profile`.
- **Method background**: `records/pooled-doc-kv-attention.md` (the eval-side probe, header-real
  parity, the GDN write detach), `records/soft-kv-slot-probe-handoff.md`,
  `records/flop-scaling-report-2026-09-02.md` (the previous campaign this supersedes for KV arms).

## 7. Method state (what "the soft token" is now)

Mean input embedding through an identity projector at the pooled document's centre position; slot
K/V detached; **GDN write channels also detached** (`q`/`k`/`v` pre-conv, `beta`, `g` — commit
96f2fba11) so a slot contributes zero gradient to any parameter, verified by
`src/test/nn/pooled_soft_token_gdn_test.py`; no logit bias (every bias variant lost); GDN still
*sees* the slot in the forward. Data-side: document headers stay real
(`--st-header-stop-id 25 --st-header-stop-count 1` contradiction / `3` oolong), keep set = gold +
a fixed fraction of random documents. This differs from the pre-2026-09-08 KV arms in the header
rule, the gold-index fix, and the GDN detach — old `kv17`/`kv33` numbers are not comparable.

---

## 8. gen-4: outlier HEADER-REAL arms (2026-09-14)

Prasann: *"do you ever try to keep a few header tokens per document un-detached (like OOLONG)? Maybe
that could solve the issue?"* Header-real is where contradiction (`hdr33`) and oolong (`ohdr33`) got
their matched-FLOP wins, and it had **never been tried on outlier in training** — outlier's arms were
only ever gold-forced (`kv17`/`kv33`) or whole-body gold-blind (`kvgb`/`kvgb50`/`kvgbmix`). §1 calls
outlier done at parity; these three arms are the one untried lever, and they are cheap.

**⚠ THE STOP ID FOR OUTLIER IS 5491 (`']:'`), NOT 25 (`':'`).** Decoded from the real rendering path
(`segment_prompt_to_chunks`, `chunk_by="document"`, qwen3_5 markers) with
`debug/ds64/probe_outlier_header.py`. Outlier (`ctc.tasks.outlier.sources.wiki100w`, titles off)
renders each document as `Document [N]: <100-word passage>`, and the wrap makes chunks contiguous so
the separator and the label are inside the chunk:

    <|box_start|> | '\n\n' | 'Document' | ' [' | '1' | ']:' | ' to' | ' finish' | ' ""' | 'Die' | ...
    <|box_start|> | '\n\n' | 'Document' | ' [' | '2' | ']:' | ' her' | ' debut' | ' in' | ' ""' | ...

Qwen3.5 fuses the bracket and the colon into **one** token `']:'` = 5491, so **token 25 never occurs
in an outlier header** — unlike contradiction (`Claim 269:` → `…, 24, 25, …`) and oolong
(`Date: … || User: …` → `1851, 25, …`), which do contain the standalone `':'`.

This matters because the failure is **silent**: `mark_doc_headers_free` keeps freeing tokens while
`n_before < stop_count`, bounded only by `cap` (32). With `--st-header-stop-id 25` every outlier
document would keep its **first 32 tokens** real — ~25% of a 130-token passage — which is a quarter
of the compaction given away, and the arm would still "work". Measured on one 110-document row:
stop id 25 frees 3467 tokens (≈ 32/doc, i.e. the cap), stop id 5491 frees 662 (≈ 6/doc, i.e. exactly
`\n\nDocument [N]:`).

**Arms** (`launch_ds64.py`, `ARM_EXTRA` / `ARM_MICRO` = 2 each, registered in `soft_arms.json`):

| arm | flags | keeps real |
|---|---|---|
| `xhdr33` | `--st-gold-blind --st-keep-prob 0.3333 --st-header-stop-id 5491 --st-header-stop-count 1` | every header + 1/3 of bodies |
| `xhdr17` | `… --st-keep-prob 0.1667 …` | every header + 1/6 of bodies |
| `xhdr00` | `… --st-keep-prob 0.0 …` | every header, **no** body |

`xhdr00` is the question asked directly: *is a real document id enough to name the odd one out?*
`keep_prob 0.0` is valid (`resolve_keep_docs` returns `u < keep_prob` with `u >= 0`, so nothing is
kept); forward+backward smoke-tested on CPU at 0.0 / 1/6 / 1/3.

**Budgets 16M / 32M / 64M only — enforced in code.** `TASK_BUDGETS["outlier"]` carries 4M/8M for the
*dense* anchors, but a soft arm at those budgets silently becomes a **1-step run** (§1's trap), which
is how `kvgb`/`kvgb50`/`kvgbmix` acquired their byte-identical 4M/8M rows. `launch_ds64.ARM_BUDGETS`
now narrows a budget grid per arm and `budgets_for(task, arm)` is what both the launcher and the
orchestrator iterate, so the mistake is no longer available.

**How to collect** (unchanged): the orchestrator evals each run on outlier's 5 rungs as it finishes,
then `debug/ds64/collect_ds64.py` writes `results/ds64/results.csv` and
`debug/ds64/make_pareto_artifact.py` redraws the dashboard. **Quote outlier on the 4 working rungs**
(2k/8k/16k/32k) — its 64k rung is the broken v2 xlong file and reads ~0.0 for everything (§1) — and
filter `n_rungs == 5` before reading any mean (trap 6).

Compare against dense outlier, 4-rung mean: `16M 0.400@780 PF | 32M 0.564@1582 | 64M 0.754@3168`,
and against the best existing soft arm `kvgb50` (`16M 0.285@190 | 32M 0.350@374 | 64M 0.388@761`).

**Launched 2026-09-14 14:26-14:30 PDT**, 4 GPUs each, `ai2/jupiter-cirrascale-2,ai2/saturn-cirrascale`,
priority urgent, workspace `ai2/flex2` (unallocated), by the orchestrator
(`DS64_GEN=3 DS64_NGPU=4 … setsid nohup python debug/ds64/orchestrate_ds64.py >> debug/ds64/orchestrator_ds64.log 2>&1 &`).
Also in `debug/ds64/LAUNCH_LEDGER.tsv`.

| run | Beaker |
|---|---|
| `ds64-outlier-xhdr33-b128f3-u16M` | `01M2GX1597NRXP0C32YEGSHVKY` |
| `ds64-outlier-xhdr33-b128f3-u32M` | `01M2GX22M7MPZKF9JKZN8FBH2H` |
| `ds64-outlier-xhdr33-b128f3-u64M` | `01M2GX2YZY4SJCYQP37Q6ND0R5` |
| `ds64-outlier-xhdr17-b128f3-u16M` | `01M2GX3RR0MNMHM3WN0427KK6Y` |
| `ds64-outlier-xhdr17-b128f3-u32M` | `01M2GX4J4B6XXRGZY5B3RN4V0H` |
| `ds64-outlier-xhdr17-b128f3-u64M` | `01M2GX5N3X0DVE8Z4H9FWGW2GM` |
| `ds64-outlier-xhdr00-b128f3-u16M` | `01M2GX6END3RK01AVNF96MVDYN` |
| `ds64-outlier-xhdr00-b128f3-u32M` | `01M2GX7CXFJVNSW8ZP7NJR5BHF` |
| `ds64-outlier-xhdr00-b128f3-u64M` | `01M2GX8924Y1BARWD0K70RQD9C` |

wandb group: https://wandb.ai/prasanns-allen-institute-for-ai/memory-networks/groups/ds64-q35-4b

Restarting the orchestrator for this wave needed `done` **and** `finishing` cleared in
`orchestrator_ds64_state.json` (trap 7) — done by hand, **not** with `reset_failed_arms.py`: the
state's 52 FAILED entries are the gen-1/gen-2 contradiction arms (`-b128-` / `-b128f-`), and dropping
them would relaunch all 52. The orchestrator never retries a FAILED run, so they stay dead on a
restart, which is what we want. `DS64_GEN=3` is also mandatory on any restart: without it every soft
arm's `run_name` loses the `3` and the orchestrator relaunches the whole gen-3 grid under gen-2 names.

**VERDICT 2026-09-14 (`debug/ds64/xhdr_collapse_diagnosis.md`) — DEAD, kill the rest. Not a bug.**
Every xhdr cell sits *exactly* on the uniform-guess floor `k/n` (3 gold of n docs, eval_size=500/rung):
2k 0.224 / 8k 0.053 / 16k 0.027 / 32k 0.014 vs measured 0.23-0.27 / 0.042-0.052 / 0.031-0.035 / 0.011-0.014 —
and `keep_prob` (0, 1/6, 1/3) and budget (16M/32M/64M) change **nothing** (CE plateaus at 0.48-0.50 for
keep 0 and keep 1/3 alike; kvgb50 reaches 0.35). Cause: header-real makes *every* id 1..N real text
including for fully-pooled docs, so gold-blind rows get a **copyable but unjustifiable** target and the
model converges to "sample 3 ids from the visible list"; without the header a pooled doc's id is absent,
so copying is impossible and supervision stays content-grounded (kvgb 0.684 / kvgb50 0.871 at 2k).
Contradiction `hdr*` escapes via gold-forcing (`gold_plus_random` always keeps gold real); oolong `ohdr*`
escapes because its answer is a phrase, not a line id. Header path itself verified clean on CPU
(`debug/ds64/repro_xhdr_compaction.py`: header span = exactly `\n\nDocument [N]:`, 5.6 tok/doc, 0 header
tokens labelled, labels 17→17, positions monotonic). **Never pair `--st-header-stop-id` with
`--st-gold-blind` when the answer is drawn from the header** (also blocks nq/rerank). Jobs to cancel are
listed in the diagnosis; drop xhdr* from `soft_arms.json` before restarting the orchestrator.

---

## 9. gen-4b: outlier `xh2` arms — is the header collapse a SHORTCUT or a LEAK? (2026-09-14)

§8's verdict ("header-real + gold-blind is dead on outlier") is consistent with every number we
have, but it does **not** separate two stories, and the difference decides whether header-real is
usable anywhere the answer is an id:

* **(a) HARD LEAK.** Once every `Document [N]:` id is real text, "sample 3 ids from the visible
  list" *is* the loss minimum. No keep ratio, no extra compute, and no better initialisation can
  beat it. The arm family is finished.
* **(b) OPTIMIZATION SHORTCUT.** The guess policy is merely *reachable in a few steps* — every
  xhdr CE curve is flat from step 8 — so SGD lands in it before it ever explores the
  content-grounded basin. The pooled mean-embedding slot should still carry **topic**, which is all
  outlier needs, so the content solution may exist and simply never be found. Header-free `kvgb50`
  had no such basin available (a pooled doc's id is absent, copying is impossible), was *forced*
  onto content, and reached dense parity.

Prasann's read, and mine: **(b)**. Two arms separate them. Both keep the xhdr flag set otherwise
byte-identical (gold-blind, `--st-header-stop-id 5491 --st-header-stop-count 1`, `ARM_MICRO` 2,
`ARM_BUDGETS` pinned to 16M/32M = 28/56 steps, well clear of the one-step floor):

| arm | flags | what a null / positive result means |
|---|---|---|
| `xh2k50` | `--st-gold-blind --st-keep-prob 0.5 --st-header-stop-id 5491 --st-header-stop-count 1` | Does the header **always** collapse, or only at low keep? Its header-free twin `kvgb50` (same keep 1/2, no header) is the parity arm, so a collapse here isolates the **header** and a descent isolates the **keep ratio**. Under (a) it must still sit on `k/n`. |
| `xh2warm17` | same flags at keep 1/6, **warm-started** from `ds64-outlier-kvgb50-b128f3-u16M/model_and_optim` | A model that *already reads document content* meets real headers for the first time. Under (a) it abandons content and falls back to guessing; under (b) it keeps descending. Keep 1/6 is the harshest xhdr cell, so a descent there is decisive. |

**Read the CE curve, not just f1.** The signature to watch is the plateau: xhdr00/xhdr33 both sat at
CE ≈ 0.48–0.50 from step 8 while `kvgb50` descended to 0.35. Any xh2 run that breaks below ~0.45 has
escaped the guess policy; any that plateaus at 0.48 has not, whatever its f1 says. Per-rung f1 floors
to beat are the `k/n` values in `debug/ds64/xhdr_collapse_diagnosis.md` §1
(2k 0.224 / 8k 0.053 / 16k 0.027 / 32k 0.014), and the target is `kvgb50`
(2k 0.871 / 8k 0.188 / 16k 0.066).

**Warm-start mechanics and FLOP accounting.** `launch_ds64.ARM_BASE` (new) points an arm's
`--base-checkpoint` at a finished run's model-only export instead of the raw base;
`train_ctc_suite.py` writes `<save_folder>/model_and_optim` at the end of every run, and
`_tolerant_base_load` restored **535 keys, 0 re-initialised** — so `pooled_projector.*` is warm too,
not just the backbone. Nothing downstream breaks: the FLOP meter charges only this run, and
`collect_ds64.py` takes task/arm/budget from the orchestrator state rather than parsing the run
name. **But the CSV's `budget` and `flops_meter` are then the INCREMENT, not the total.** For any
matched-FLOP quote add the warm start's own cost by hand:

    total tokens = 16M + row budget      (u16M row = 32M total, u32M row = 48M total)
    total PF     = 190.3 PF (kvgb50-b128f3-u16M) + this run's flop_meter/actual_pflops

Its 16M rung also re-reads the shard `kvgb50` already trained on (the arms are nested prefixes), so
that row is a second epoch over the same data, not new data. That is fine for the question being
asked — it is about the basin, not about data scaling — but it is not a data-scaling point.

**Launched 2026-09-14 19:22–19:24 PDT**, 4 GPUs each, `ai2/jupiter-cirrascale-2,ai2/saturn-cirrascale`,
priority **urgent**, workspace `ai2/flex2` (unallocated), by the orchestrator restarted as
`DS64_GEN=3 DS64_NGPU=4 DS64_CLUSTER="ai2/jupiter-cirrascale-2,ai2/saturn-cirrascale" setsid nohup
python debug/ds64/orchestrate_ds64.py >> debug/ds64/orchestrator_ds64.log 2>&1 &`. All four confirmed
loading the right flags (`header_stop_id=5491 gold_blind=True keep_prob=0.5 / 0.1667`) and
**28 / 56 steps**.

| run | Beaker | steps |
|---|---|---|
| `ds64-outlier-xh2k50-b128f3-u16M` | `01M2HDZAJQ990WFT042FXQW9AH` | 28 |
| `ds64-outlier-xh2k50-b128f3-u32M` | `01M2HE05Y07TE7RX9V2W53RY2W` | 56 |
| `ds64-outlier-xh2warm17-b128f3-u16M` | `01M2HE127ZVQ306PMPX1SC3MDQ` | 28 (+16M warm) |
| `ds64-outlier-xh2warm17-b128f3-u32M` | `01M2HE207EZVAC1S6EG63WBNZP` | 56 (+16M warm) |

wandb group: https://wandb.ai/prasanns-allen-institute-for-ai/memory-networks/groups/ds64-q35-4b

### The third arm does NOT exist: `--st-mix-*` is a silent no-op on every gold-blind arm

A `xh2mix17` (header + keep 1/6 with a fraction of rows trained DENSE) was planned and is **not
launched**, because the knob it needs is unreachable from a gold-blind arm. `--st-mix-start-p /
--st-mix-end-p / --st-mix-anneal-frac` are parsed unconditionally but are only ever *passed*
to `make_fingerprint_keep_docs_fn`, inside
`if opts.variant == "softtoken" and not opts.st_gold_blind:` (train_ctc_suite.py:1292-1319). The
curriculum itself lives in that closure (`pooled_doc_kv.py:539-566`). With `--st-gold-blind` no hook
is installed at all, `resolve_keep_docs` falls back to the seeded `u < keep_prob` draw
(`pooled_doc_kv.py:126`), and the mix flags are accepted and ignored — no warning. Implementing it
was explicitly out of scope, so it was not implemented.

⚠ **This retro-explains a result in §1: "the compression-mixing curriculum does nothing
(kvgbmix 0.284 vs kvgb 0.287 at 32M — identical)". `kvgbmix` is `--st-gold-blind` + the mix flags,
so the curriculum never ran.** Confirmed in the job log
(`01M2C7ABE93B059RBKAECYEAYM`): the run prints the usual softtoken config line but **no**
`[ctc-suite] softtoken: gold keep hook on N module(s)` line and no `[pooled-kv] … p_full=` lines —
the hook that carries the curriculum was never attached. `kvgbmix` is therefore a byte-identical
re-run of `kvgb`, which is exactly why their scores match. **The compression-mixing curriculum is
UNTESTED on a gold-blind task; do not cite kvgbmix as evidence against it.** Testing it needs either
a gold-blind keep_docs_fn with the mix built in, or the mix moved into `resolve_keep_docs`.

`xh2k17-64M` was also skipped deliberately: the 16M/32M points decide the question (the xhdr CE
plateau is visible by step 8), and 64M buys nothing they do not already show.

### Traps hit while setting this up

8. **Cancelling a Beaker job does not stop the orchestrator from relaunching it.** §8's nine xhdr
   jobs were killed by hand after the diagnosis, and dropping `xhdr*` from `soft_arms.json` only
   stops section **A2** (first launch) — sections **B** and **C** relaunch from `state["runs"]` /
   `state["evals"]` on exit code, and a cancelled job exits 1 (train) / 143 (eval). On restart the
   orchestrator would have resurrected 2 xhdr trainings and all 7 xhdr evals. Fixed before the
   restart by marking those 9 entries `FAILED` with retries maxed in
   `orchestrator_ds64_state.json` (backup `…_state.json.bak-20260914-192107`); the 52 old FAILED
   contradiction arms were left untouched, as §8 requires. **Anything you cancel by hand must be
   latched FAILED in the state, not merely removed from `soft_arms.json`.**
9. `done` / `finishing` were already `false` this time, so no hand-clearing was needed — but check
   both before every restart (trap 7), and keep `DS64_GEN=3` on the command line.

---

## 10. fast2k: a <1 h screening loop at 2k (2026-09-14)

`debug/ds64_fast2k/` — **read its README first**. Screens a soft-token recipe on **2k rows only**
in ~20–40 min and well under 1 GPU-hour per arm, instead of the 4–20 GPU-hours a ladder arm costs.

Rationale: in the linear-cost regime compaction saves the same *fraction* of FLOPs at any length,
so accuracy-vs-FLOPs at a matched budget is already well posed at 2k. Everything is held fixed
against ds64 — same task, same 2k rung pool, same `q35-4b-base-markerfix`, same `--st-*` flag
strings, same 2k eval file — except: 2k-only data, seq-len 4096, **every arm unpacked at 32
rows/step** (so dense and soft see identical rows for identical steps and the budget is matched by
construction; the FLOP meter charges non-pad tokens, so padding costs wall-clock, never FLOPs), and
budgets of 2M/4M/8M nominal tokens = 977/1953/3906 examples.

⚠ **fast2k cannot see length generalisation, which is where ds64's outlier arms actually died**
(`kv33`: 0.890 at 2k, 0.018 at 32k — §1). A fast2k win is a licence to spend a ladder, never a
substitute for one, and a fast2k number must not be quoted as a campaign result. A fast2k *loss*
is decisive.

Two signals, both cheap:
- **`ce_floor` = `ln C(n,k) / answer_tokens`** — for outlier's 2k rung (n=14, k=3, 15.1 answer
  tokens) that is **0.390**, with a matching uniform-guess f1 of **3/14 = 0.214**. `at_floor_step10`
  fires ~3 min into a run and is the exact xhdr-collapse signature (§8, `xhdr_collapse_diagnosis.md`).
- **`matched_flop_delta`** — f1 minus the dense f1 log-FLOP-interpolated to the arm's measured
  `flop_meter/actual_pflops`, with an `extrapolated` flag (hence three dense anchors per sweep).

Train/eval disjointness is checked at build time: **0 shared examples and 0 shared documents**
between the fast2k shards and `outlier_lengthmix/eval_rungs/outlier/rung_2048.jsonl`.

Data build (tokenize-only gantry job, no GPU, ~90 s; slices the EXISTING ds64 2k pool):
`TASK=outlier bash debug/ds64_fast2k/build_fast2k_data_beaker.sh` → weka `ds64/fast2k/{arms,shards}`.
Shards: `outlier_f2M` / `_f4M` / `_f8M`, `max_example_len` 2949, p50 2190.
wandb group: https://wandb.ai/prasanns-allen-institute-for-ai/memory-networks/groups/f2k-q35-4b
