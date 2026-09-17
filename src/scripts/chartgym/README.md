# ChartGym

A synthetic chart corpus with exact, code-derived ground truth, plus the olmo-eval task that
measures it. Built for Molmo2 stage-2.

## What it is for, and what it is not for

It trains **visual capabilities**, not a benchmark's question templates. The distinction is
enforced mechanically, not by care:

- `tests/test_chartgym.py::test_no_charxiv_phrasing` asserts every generated question has
  token-Jaccard < 0.6 and shares no 6-gram with any of CharXiv's 19 descriptive templates.
  It fails CI if someone borrows the benchmark's wording.
- `test_holdout_is_total` asserts the **held-out primitive** — panel layout — never appears
  in training emission, in any family and in any word (`row`, `column`, `grid`, `subplot`,
  `panel`). Multi-panel figures are still rendered and still questioned; panels are
  referenced *by title*, never by grid position.

The held-out primitive is what makes an improvement on CharXiv templates 18/19 readable as
capability transfer rather than as the benchmark's own question being fitted. It was chosen
by a rule fixed before the numbers were seen (referentially isolated, ≥2 CharXiv templates,
pooled accuracy ≤85%); colorbar was disqualified at 91.43%, legend at isolation, subplot
layout passed at 83.97%.

## Design rules that are load-bearing

1. **Nothing is automatic.** `set_xlim`/`set_xticks`/`set_xticklabels` are always explicit,
   so a matplotlib locator or formatter can never silently disagree with the spec. Without
   this, a version bump changes what is drawn while the spec still claims the old truth.
2. **`bbox_inches="tight"` is never used.** It re-crops *after* `RenderAudit` measured artist
   positions against `fig.bbox`, invalidating every pixel threshold and clipping edge tick
   labels the spec still counts. `layout="constrained"` instead.
3. **The audit guards, it never answers.** `RenderAudit` may be consulted for pixel-space
   guards (do two tick labels overlap, are two curves far enough apart to be distinguishable)
   but no question's answer is derived from it.
4. **Families decline rather than guess.** Tangential curve touches, near-threshold peak
   prominences and boundary correlations are dropped. A figure that fails
   `RenderAudit.validate` is discarded, not repaired — a spec/render disagreement means we do
   not know which one the image shows.
5. **`axes.unicode_minus = False`.** Matplotlib otherwise renders negatives with U+2212,
   which no model emits, so a verbatim gold of "-5" would be unmatchable.

## Inapplicable questions are a feature

`sample.py::NA_RATES` are **measurements**, not knobs: the gold "Not Applicable" rate per
CharXiv descriptive template on validation (25.0% of the benchmark overall). A generator that
always produced a titled, legended, numerically-ticked line chart would teach the model that
every question is answerable — the single most likely way this corpus makes the model *worse*,
since templates 1,2,3,8,9,10,11,12,13 all have an NA branch and the checkpoint already fails
them badly (template 11 NA recall 3.8–14.1%).

Inapplicability is therefore generated at the measured rate and answered **naturally** ("this
chart has no legend"), never as CharXiv's literal token. CharXiv's own prompt supplies that
format at eval time, so what has to transfer is the *detection*.

## Layout

```
chartgym/
  spec.py       FigureSpec & co -- the complete ground truth, no matplotlib import
  render.py     FigureSpec -> (png bytes, RenderAudit)
  sample.py     rng + difficulty -> FigureSpec, with NA_RATES
  families.py   question families by capability; pnl.* are the held-out primitive
scripts/
  generate.py    one shard: figures/ + specs.jsonl + qa.jsonl + reject_stats.json
  stage_train.py -> FineVision schema (texts list-of-struct + images); asserts no pnl.* leak
  stage_eval.py  -> flat one-row-per-question dataset for the olmo-eval `chartgym` task
tests/          11 CPU tests, ~85s
```

## Usage

```bash
export MPLCONFIGDIR=/tmp/mplconfig     # the cold font-cache build on weka takes minutes
python scripts/generate.py   --out raw/train-v1/shard-00000 --n 1000 --seed-base 1000000
python scripts/generate.py   --out raw/eval-v1/shard-00000  --n 200  --seed-base 900000 --eval-split
python scripts/stage_train.py --raw raw/train-v1 --out $MOLMO_EXPERIMENT_DATA_DIR/chartgym/train-v1
python scripts/stage_eval.py  --raw raw/eval-v1  --out $MOLMO_EXPERIMENT_DATA_DIR/chartgym/eval-v1 --n 4000
python -m pytest tests/ -q
```

Then train with `--chartgym_rate=0.25` (Molmo2-Stage2.py) and evaluate with
`-t chartgym -t chartgym:text_only`.

## Costs and properties, measured

- Render: **4.0 figures/s/core** at 2x2 mid-complexity (252 ms/figure, 134 KB PNG). 120k
  figures = 8.4 core-hours ≈ 10 min across 48 workers, ~16 GB.
- ~25% of figures are rejected by the audit, almost all for tick-label overlap. That is the
  guard working; it is also cheap, so it has not been tuned away.
- ~13-19% of emitted questions are inapplicable, matching the intended band.
- **16 questions on one chart cost one image encode.** Measured loss-mask sum scales ~sqrt(n):
  2.83 at n=1 to 16.0 at n=16, i.e. **5.66x**, not 16x. Packing on this tier is crop-bound
  (~31% token occupancy), so many-questions-per-chart is close to free.
- `chartgym:text_only` on a stage-2 checkpoint reads **8.80%** — the questions genuinely
  require the image.

## One caveat about the corpus's original motivation

The largest CharXiv target it was designed for, template 17 (total labelled ticks, 29.91%),
turned out to be an **elicitation** failure rather than a perception failure: asking the model
to enumerate the ticks before answering lifts it to 51.34% with no training at all
(McNemar chi2 = 17.82, p<0.001). Counting families should not be justified by that template.
What survives as a training target is applicability detection and dense label reading --
see `outputs/chartgym/stage0/RESULTS.md`.

## Where this lives, and why

The **code** is version-controlled here under `src/scripts/`, not in
`molmo-experimental-data/`, which is not a git repository — generator code left only there
would be exactly as durable as the uncommitted olmo-eval `key_mapping` fix that was silently
lost twice. It is not under `launch_scripts/` because that path is gitignored in this repo,
and `src/scripts/` is not packaged into the wheel (`pyproject.toml` includes only
`olmo_core*`), so the matplotlib dependency never reaches the training image.

The **generated data** stays under `$MOLMO_EXPERIMENT_DATA_DIR/chartgym/` and is not
committed: it is fully reproducible from `(run_name, seed)`, and the raw shards are large.
