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
| grouping | high | 2k-32k | registry pending |
| reorder | high | 2k-16k | registry pending |
| textgroups | high | 2k-32k | registry pending |
| rerank | low | 2k-16k | registry pending |
| absence | low | 2k-16k | registry pending |
| qdmatch_hpqa | high | 2k-32k | registry pending |

## Measured rung medians (Qwen3.5-0.8B-Base tokenizer, --query-position after)

Every mix share is computed from these, never from the rung label.

| task | knob | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|---|
| oolong | len band | 2042 | — | 7846 | 16342 | 31728 |
| contradiction | n docs | measured | — | measured | measured | 32811 |
| xabsence (EXACT) | P pairs | — | ~4.5k | ~8.7k | 15625 | 30989 |
| reorder | n chunks | measured | measured | 8812 | 17842 | no rung |

## Arms built

| task | budgets (tokens) | instances |
|---|---|---|
| oolong | 20.3M / 40.6M / 81.3M | 6,711 / 13,422 / 26,844 |
| contradiction | 14.0M / 28.0M / 56.0M | 3,994 / 7,988 / 15,976 |
| reorder | 15M / 30M / 50M | 5,835 / 11,670 / — |

reorder's 60M budget is not buildable: its 2k pool tops out at 18,973 examples inside the
20,000-book window the eval split forces (books 20,001+ are eval), against 19,300 needed.

## Results

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
expect from spending every token at the rung being scored.

## Prior-campaign anchors

- qdmatch_nq dense short-heavy mixes @8k/16k/32k, eval_size 600: 64M .949/.883/.676,
  160M .956/.901/.709, 320M .968/.919/.745. Sparse 320M: .569/.283/.076.
- qdmatch_nq pure-length 32k ladder (2k/8k/16k/32k examples): .735/.793/.834/.812, asymptote ≈.83-.84.
- nq dense mixes @2k/8k: 16M .977/.917, 32M .977/.930, 48M .973/.933. Sparse 48M: .912/.728.
- outlier: no sparse token-crossover through 640M at any rung; sparse needs ~1.5-1.7x tokens at
  matched score, and ~1.16x cheaper GPU-hours at <=32k does not close that gap.
