# Cross-task data-scaling laws: does sparse ever overtake dense?

Extends the outlier / qdmatch_nq / nq length-mix campaign
(`debug/outlier_lengthmix_scaling/`) across the CTC suite. Per task: dense and sparse-landmark
Qwen3.5-4B trained on short-heavy length mixes at three token budgets, scored on the shipped
per-rung eval ladders, fit with the Hill law f(B) = fmax·B^g/(B^g+K^g), and read for a
sparse-vs-dense crossover.

Everything runs on Beaker (no mooney, no local slurm): CPU jobs build the data straight onto
weka, GPU jobs train and eval from it.

## Roster

Balanced across the suite's CTC classes, weighted toward high-CTC. `class` is
suite_table.json's: N = O_T(N) = low, NM / N² = high.

| task | class | rungs | status |
|---|---|---|---|
| outlier (scale-K) | high | 2k-32k | DONE (prior campaign) |
| qdmatch_nq | high | 8k-32k | DONE (prior campaign) |
| nq | low | 2k-8k done, 16k/32k in flight | prior campaign |
| contradiction (realistic) | high | 2k/8k/16k/32k | building |
| xabsence (EXACT) | high | 4k/8k/16k/32k | building |
| oolong | low | 2k/8k/16k/32k | building |
| grouping | high | 2k/8k/16k/32k | arms built, trains queued |
| reorder | high | 2k/4k/8k/16k | arms built, trains queued |
| textgroups | high | 2k/4k/8k/16k | arms built, trains queued |
| absence | low | 5.6k/10.9k/26.8k | arms built, trains queued |
| ~~rerank~~ | low | — | DROPPED: needs a pyserini index + a GPU cross-encoder pass, and data without CE scores is ungradeable |
| ~~qdmatch_hpqa~~ | high | — | DROPPED: dense is flat .999->.981 across 2k->32k (bridge entity appears verbatim in its gold doc), and train/eval share one 4,000-unit pool |

## Measured rung medians (Qwen3.5-0.8B-Base tokenizer, --query-position after)

Every mix share is computed from these, never from the rung label.

| task | knob | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|---|
| oolong | len band | 2042 | — | 7846 | 16342 | 31728 |
| contradiction | n docs | measured | — | measured | measured | 32811 |
| xabsence (EXACT) | P pairs | — | ~4.5k | ~8.7k | 15625 | 30989 |
| reorder | n chunks | measured | measured | 8812 | 17842 | no rung |
| textgroups | n docs | 1762 | (4k) 4361 | 9302 | 23429 | past window |
| grouping | docs/example | 2002 | — | 8272 | 16570 | 32696 |
| absence | n sents | (n90) 5604 | (n180) 11186 | (n360) 26811 | — | — |

## Arms built

| task | budgets (tokens) | instances |
|---|---|---|
| oolong | 20.3M / 40.6M / 81.3M | 6,711 / 13,422 / 26,844 |
| contradiction | 14.0M / 28.0M / 56.0M | 3,994 / 7,988 / 15,976 |
| reorder | 15M / 30M / 50M | 5,835 / 11,670 / 19,451 |
| xabsence | 20.0M / 40.1M / 80.2M | 3,570 / 7,141 / 14,280 |
| textgroups | 20.0M / 40.0M / 80.2M | 8,357 / 16,717 / 33,435 |
| absence | 20M / 40M / 80M | 2,832 / 5,666 / 11,330 |
| grouping | 20.0M / 40.1M / 80.2M | 6,835 / 13,670 / 27,340 |

All 21 tokenized arms pass the pre-flight audit (`audit_arms.py`): correct task string, Qwen3.5
eos 248044, zero skipped examples, and every max_example_len inside the 65536 training window.

reorder's 60M budget is not buildable: its 2k pool tops out at 18,973 examples inside the
20,000-book window the eval split forces (books 20,001+ are eval), against 19,300 needed.

## A free repeatability check

The reorder-50M eval was preempted and re-ran itself on the same checkpoint against the same rung
files. Two independent decodes of one checkpoint: 2k tau .686 vs .687, 4k .457 vs .453. So
decode-side nondeterminism contributes about +-.004 here -- an order of magnitude below the
+-.022 binomial floor at eval_size 500. When two of our numbers differ, it is sampling of the eval
set, not the decode, and quoting three decimals is over-precision either way.

## Scheduling note (2026-09-01)

**Root cause of the 9-hour stall: another user's stuck job, not our queue.** Every placement
message named the same head-of-queue blocker, `registry-mirror` (author shashankg, urgent, no GPU
request), pending from 06:29 UTC. Under jupiter's strict-priority policy everything behind it
waits. It was cancelled around 16:10 and our arms began placing within minutes. Diagnose this with
`beaker job get <id>` on the ID the launcher prints in "waiting for at least 1 job higher in the
queue" -- it is not necessarily ours.

Secondary observation, still true: jupiter reported free slots throughout, because that
capacity is FRAGMENTED -- a single-node 8-GPU request needs one node with 8 free GPUs, and 54 nodes
were cordoned. **ai2/saturn
(A100-80GB, eager scheduling) places the identical 8-GPU config in seconds once it has room** --
migrating an arm there is a stop-on-jupiter + relaunch with `--cluster ai2/saturn`, no config
change. Five dense arms moved and started immediately.

## Results

### oolong, sparse-landmark -- COMPLETE

Metric is oolong's `score`. eval_size 600 at 2k, 500 elsewhere.

| budget | 2k | 8k | 16k | 32k |
|---|---|---|---|---|
| 20M | .812 | .535 | .509 | .463 |
| 40M | .823 | .564 | .532 | .503 |
| 80M | .864 | .631 | .589 | .535 |

Monotone everywhere and still climbing at 80M (+.04 to +.07 per doubling). The length profile is a
THIRD distinct sparse shape: a step down from 2k to 8k, then near-flat -- 8k/16k/32k span only
.072-.096 at a given budget, where nq sparse fell .728 -> .248 over the same span. So sparse
degradation is not one phenomenon; it is task-specific.

### oolong, dense vs sparse -- FIRST COMPLETE 3x4 GRID, BOTH VARIANTS

Dense:

| budget | 2k | 8k | 16k | 32k |
|---|---|---|---|---|
| 20M | .865 | .639 | .576 | .564 |
| 40M | .886 | .672 | .643 | .607 |
| 80M | .913 | .692 | .668 | .617 |

Dense leads sparse at all 12 cells, by .053-.111. **Dense at the SMALLEST budget beats sparse at
the LARGEST at every rung except 2k**: 8k .639 vs .631, 16k .576 vs .589 (the one exception),
32k .564 vs .535. So sparse is roughly a 4x-data deficit behind dense on oolong, and the deficit
does not shrink with budget.

VERDICT (oolong): no crossover in the measured range. `fit_crossover.py` reports one at 301M (8k)
and 460M (16k), but both are 4-6x beyond the largest measured budget and both are driven by
sparse's fitted ceiling running into its 1.05 bound while dense's bends to .72/.68 -- i.e. the
extrapolation says "the curve still rising will pass the curve that is saturating", which three
points cannot support. Treat as NOT a prediction of crossover; it is a prediction that sparse has
not yet shown its ceiling.

### contradiction, dense -- COMPLETE (3 budgets)

| budget | 2k | 8k | 16k | 32k |
|---|---|---|---|---|
| 14M | .909 | .836 | .751 | .592 |
| 28M | .967 | .932 | .873 | pending |
| 56M | .982 | .971 | .941 | .881 |

The middle budget lands where a smooth curve wants it (2k .967 between .909 and .982), so the
steep 14M->56M rise is a real curve rather than two points that happen to be far apart.

eval_size 500, v3 realistic-mode gold. A 4x budget buys +.073/+.135/+.190/**+.289** -- the gain
grows monotonically with length. This is the strongest evidence in the campaign that long-context
failure on a genuinely O_T(N^2) task can be bought off with data: .881 at 32k from 56M tokens,
where outlier needs an extrapolated ~1.26B for .8 at the same rung. The 28M arm is still training.

### contradiction, sparse -- collapse

The 14M arm scores **.001 @2k, .000 @8k, .000 @16k** (eval_size 500) where its dense twin scores
.909/.836/.751. The dense arm
runs the same eval path against the same bundle, so this is the model, not the harness:
contradiction joins qdmatch as a task landmark compression cannot do at all.

### Sparse verdicts so far (one row per task)

| task | sparse profile | evidence |
|---|---|---|
| oolong | step down, then flat | .864/.631/.589/.535 @80M -- trails dense in all 12 cells |
| nq | compounding decay | .912/.728/.608/.248 @48M vs dense .973/.933/.910/.873 |
| outlier | constant ~0.9x factor, then plateau | needs ~1.5x tokens at 16k; 32k ceiling .57 |
| xabsence | near-floor and FLAT in budget | .217/.145/.061/.026 @20M vs .221/.140/.057/.032 @80M -- 4x data buys nothing |
| qdmatch_nq | dead | floor at every tested scale |
| contradiction | dead | .001/.000/.000/.000 vs dense .909/.836/.751/.592 |
| reorder | dead at every budget AND every rung | 15M ladder complete: .005/-.004/.000/.000 at 2k/4k/8k/16k; dense .686 at 2k |
| absence | LATE TAKEOFF, not dead | .017/.005/.001 @20M -> .224/.094/.025 @40M -> .357/.304/.250 @80M |
| grouping | at the degenerate floor | pairwise_f1 .419 @2k @20M -- BELOW the .44 that a
  single-cluster prediction scores, so it is not "44% right", it is chance; dense .794 |

⚠ CORRECTION (same day): absence sparse was first recorded as "dead" from its 20M arm alone
(.017/.005/.001). The 40M and 80M arms show a steep late takeoff -- .224/.094/.025 then
.357/.304/.250 -- so it is the outlier pattern (nothing happens, then the task switches on), not a
floor. A single small-budget arm cannot distinguish "cannot learn this" from "has not started
learning yet"; only the ladder can. Its 80M scores are still far under dense's .983/.983/.993.

**Six of nine sparse arms are at or near their task's floor**, and no task has shown sparse
leading dense at any rung or any budget. Read grouping's number against its floor, not against
zero: `pairwise_f1` gives ~0.44 for a degenerate all-one-cluster answer (ARI 0), so sparse
grouping's .419 is chance, not partial credit -- the same reason the metric is never quoted without
ARI. Contradiction sparse is confirmed dead at a second budget (28M: .001/.000/.000), so it is not
a data-starvation artifact of the smallest arm.

### reorder + grouping, dense (eval_size 500)

reorder (kendall tau): 2k .359/.610/.686, 4k .124/.356/.457, 8k .001/.017/.050 across 15/30/50M.
16k is 0.000 at BOTH 15M and 30M -- the one rung where more data changes nothing at all. The 8k rung IS scaling -- 50x from the smallest budget -- but from .001, so a 3.3x budget moves it
to .050 and it would take orders of magnitude more to matter.
Both short rungs scale cleanly; **8k is a cliff** -- .050 at the largest budget, against .686 at 2k
for the same checkpoint. Reorder is a short-context task at these budgets no matter the data.

grouping (pairwise_f1, floor .44): 80M gives .794 / .569 / .417 at 2k/8k/16k. The 16k cell is AT
the degenerate floor, so grouping's usable range under this recipe ends between 8k and 16k. Sparse sits at .419/.187/.090 (2k/8k/16k at 40M) against dense .794/.569/.417: the 2k cell is
already at grouping's .44 degenerate floor and every longer rung is well below it, i.e. sparse is
not producing even a degenerate clustering by 8k.

### textgroups 16k: measured once, deliberately not re-run

The first six textgroups evals fired while the weka rung bundle was still partly synced, so all six
skipped the 16k rung (caught by `audit_evals.sh`). The dense 80M arm was re-fired against the
complete bundle and returns **.006 @16k** (eval_size 500). The other five were NOT re-run: both
variants are already at the floor by 8k (dense .015, sparse .002), so five more jobs would buy five
more numbers indistinguishable from zero. `audit_evals.sh` will keep flagging them as short --
that is the correct behaviour for the tool, and this note is the reason they stay that way.

### Dense, first rungs across the four new tasks (eval_size 500 each)

| task | metric | 2k/4k rung | budget scaling |
|---|---|---|---|
| xabsence | f1 | **1.000 / .997** @20M | 1.000 / 1.000 @80M -- SOLVED at every budget |
| absence | f1 | .983 / .983 @40M | .997 / .993 @80M -- effectively solved |
| reorder | kendall_tau | .359 @2k @15M | **.686** @2k @50M -- steep, still climbing |
| textgroups | f1 | .040 / .016 @20M | .096 / .043 @80M -- near FLOOR even dense |

Two tasks are already saturated for dense at these budgets (xabsence, absence), so their scaling
laws are uninformative -- and note absence's labels understate length ~3x, so its "4k" rung is a
measured 14.5k. The informative dense curves are reorder (a 3.3x budget nearly doubles tau) and
textgroups, which is the campaign's one genuinely unsolved task for BOTH variants: dense at 80M
scores .096 at 2k, where sparse-vs-dense is a comparison between two floors.

The contrast that matters for sparse: on xabsence dense scores 1.000 at 4k where sparse scores
.217, and on reorder dense reaches tau .686 where sparse sits at chance (.009). These are not
tasks where sparse needs more data; they are tasks landmark compression cannot represent.

## Results (older)

(f1 per rung, eval_size stated per row; filled in as evals land)

## nq dense short-heavy mix ladder (eval_size 600/rung, --query-position after)

| budget | 2k | 8k | 16k | 32k |
|---|---|---|---|---|
| 16M | .977 | .917 | **.857** | **.760** |
| 32M | .977 | .930 | **.885** | **.837** |
| 48M | .973 | .933 | **.910** | **.873** |

Hill fits over the three budgets (fmax runs into its 1.0 bound at both rungs, so the .95 budgets
are soft): 16k g=.45 K=0.3M -> B(.90)=41M, B(.95)=216M; 32k g=.70 K=3.1M -> B(.90)=71M,
B(.95)=205M. Per-doubling increments are +.028/+.043 at 16k and +.077/+.062 at 32k.

The 16k/32k rungs are the informative ones -- 2k and 8k were already saturated, and the 16k column
moves +.028 then +.025 while 2k moves -.004. Pure-length anchor: nqD32k_4000 (4,000 examples at
n=200 docs, ~31.4k tokens) scores **.888 @32k**, slightly above the 48M mix, which is what you would
expect from spending every token at the rung being scored. A second pure-length arm, nqD64k_2000
(2,000 examples at ~62.8k tokens), scores **.875 @32k** -- the same place, from twice the length and
half the examples.

### Sparse vs dense on nq, at a matched 48M tokens (eval_size 600/rung)

| variant | 2k | 8k | 16k | 32k |
|---|---|---|---|---|
| dense | .973 | .933 | .910 | .873 |
| sparse-landmark | .912 | .728 | .608 | .248 |
| gap | -.061 | -.205 | -.302 | -.625 |

The deficit compounds with length rather than settling at a constant ratio: sparse holds 94% of
dense at 2k and 28% at 32k. This is the qdmatch profile (collapse at length), not the outlier
profile (converges to ~0.9x and plateaus). Sparse's 16M and 32M budgets are still queued, so the
crossover question -- whether MORE tokens buy sparse its way back -- is not answerable for nq yet.
The 64k transfer rung on the deep pure-length arm gives dense .838, so dense degrades gently over
the 32k->64k step that sparse cannot reach at all.

**Sparse arms cannot be evaluated on L40S.** The landmark prefill kernel asks triton for 104KB of
shared memory against sm_89's 99KB limit, so every sparse-landmark checkpoint dies with
OutOfResources on neptune -- and `--no-sparse-decode` does NOT help, because the fast decode path is
not what allocates it. Sparse evals have to go to an A100 (164KB) or H100 (228KB) cluster, which is
the scheduling constraint on this wave: dense arms can use the idle L40S capacity, sparse arms
compete for the jammed clusters.

## qdmatch_nq 64k transfer (eval_size 600)

Neither arm was trained at 64k; both are 32k pure-length arms scored on the 64k rung.
q32k_16000 (16,000 examples, ~487M tokens) -> **.706**, q32k_32000 (32,000 examples, ~975M) ->
.713. Doubling the corpus buys +.007 at 64k, the same flat tail the 32k ladder shows.

## outlier p32k_4000 (pure-length 32k arm, 4,000 examples)

Its 16k rung -- a DOWNWARD transfer, since every training example sits at ~32k -- scores **.119**
(eval_size 600), against .453 for the 64M short-heavy mix at the same rung. Training entirely at
one length does not buy the shorter rung; the short-heavy recipe exists for exactly this reason.
Its 32k rung scores **.302**, completing the three-point pure-length ladder at n=220 docs
(~32.8k tokens/example): 2,000 ex / 66M tok -> .209, 4,000 / 131M -> .302, 8,000 / 262M -> .472.
Per-doubling increments are +.093 then +.170 -- ACCELERATING, so this task is still in the takeoff
part of its curve rather than the saturating tail. The Hill refit (fmax 1.05, g .88, K 339M,
rmse .011) puts B(.5)=304M, B(.7)=742M and B(.8)=1.26B tokens; everything past 262M is
extrapolation from three points inside a rising regime, so treat the .8 figure as an order of
magnitude, not a number.

Set against the short-heavy mixes at the same rung (.107 at 64M, .266 at 160M, .352 at 320M), the
pure-length arms are AHEAD per token at 32k -- .302 at 131M vs ~.24 interpolated for the mix -- the
mirror image of the 16k transfer above, where the mix wins by a factor of four.

## Prior-campaign anchors

- qdmatch_nq dense short-heavy mixes @8k/16k/32k, eval_size 600: 64M .949/.883/.676,
  160M .956/.901/.709, 320M .968/.919/.745. Sparse 320M: .569/.283/.076.
- qdmatch_nq pure-length 32k ladder (2k/8k/16k/32k examples): .735/.793/.834/.812, asymptote ≈.83-.84.
- nq dense mixes @2k/8k: 16M .977/.917, 32M .977/.930, 48M .973/.933. Sparse 48M: .912/.728.
- outlier: no sparse token-crossover through 640M at any rung; sparse needs ~1.5-1.7x tokens at
  matched score, and ~1.16x cheaper GPU-hours at <=32k does not close that gap.
