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

## Results

(f1 per rung, eval_size stated per row; filled in as evals land)

## Prior-campaign anchors

- qdmatch_nq dense short-heavy mixes @8k/16k/32k, eval_size 600: 64M .949/.883/.676,
  160M .956/.901/.709, 320M .968/.919/.745. Sparse 320M: .569/.283/.076.
- qdmatch_nq pure-length 32k ladder (2k/8k/16k/32k examples): .735/.793/.834/.812, asymptote ≈.83-.84.
- nq dense mixes @2k/8k: 16M .977/.917, 32M .977/.930, 48M .973/.933. Sparse 48M: .912/.728.
- outlier: no sparse token-crossover through 640M at any rung; sparse needs ~1.5-1.7x tokens at
  matched score, and ~1.16x cheaper GPU-hours at <=32k does not close that gap.
