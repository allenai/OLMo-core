# Raw data for Figure 1, the cross-task grid, and the CTC suite

Tidy CSVs, one row per plotted point, from two exporters:

```bash
cd paperdraft/figures
python3 export_figure_data.py       # fig1_*.csv, fig4_*.csv   (from the plotting scripts)
python3 export_ctc_suite_data.py    # ctc_*.csv                (from the harvested grade JSONs)
```

The first three sections below describe the figure CSVs. **The CTC suite CSVs are documented
at the bottom** — they come from a different source and follow slightly different rules.

The figure exporter **imports** `fig1_gap_vs_context.py` and `fig4_cross_task.py` rather than
re-typing their numbers, so the CSVs cannot drift from what the paper actually plots. If a
figure script changes, re-run the exporter.

## Which figures these are

| CSV | Script | PDF | Figure number in the current build |
|---|---|---|---|
| `fig1_gap_vs_context_data.csv` | `fig1_gap_vs_context.py` | `fig1_gap_vs_context.pdf` | **Figure 1** (`fig:motivatingtask`) |
| `fig4_cross_task_data.csv` | `fig4_cross_task.py` | `fig4_cross_task.pdf` | **Figure 5** (`fig:sec5-cross-task`) |

⚠ The second file is named `fig4_*` but compiles as **Figure 5**. Figure 4 in the current
PDF is `fig:masks`, the attention-mask schematic (`attention-masks-figure (6).pdf`), which
is a hand-drawn diagram with no underlying data. The cross-task grid is what's exported here.

## Columns common to both files

| Column | Meaning |
|---|---|
| `task` | Task name as it appears in the figure script |
| `ctc_class` | **`low`** or **`high`** corpus-traversal complexity — the orange/green split in Fig 1, the column split in the grid |
| `ctc_order` | Finer class: `O_T(N)` (= low) vs `O_T(NM)` / `O_T(N^2)+` (= high) |
| `metric` | Per-task metric from Table 1 (`table_suite.tex`) |
| `context_tokens` | Context length in tokens — the x-axis of Fig 1 |
| `n_docs` | Corpus size $N$ where the run was denominated in documents |
| `full_attention` | Score of the full-attention arm |
| `chunked_attention` | Score of the document-chunked + mask-mixing arm (best-case chunked) |
| `relative_gap` | `1 - chunked/full`, the y-axis of Fig 1. Blank where undefined (missing arm, or a zero full arm) |
| `eval_size` | Number of eval examples. `unknown (see README)` where neither figure script states it |

Both arms are Qwen3.5-4B unless noted below.

### `fig1_gap_vs_context_data.csv` (76 rows)

Extra columns:

- `series` — `background` (thin grey-alpha lines) vs `highlighted (star line)` (HotpotQA,
  Contradiction, drawn as bold star lines).
- `source` — `CTC suite (token ladder)`: the 2k/4k/8k/16k/32k budget rungs. `per_N runs`:
  the earlier per-$N$ sweeps, put on the token axis as $N \times$ per-doc tokens.
- `plotted_in_figure` / `drop_reason` — Figure 1 drops any rung whose **full** arm is under
  `FLOOR = 0.15`, because a ratio between two near-zero numbers is noise. Those rows are kept
  in the CSV with `plotted_in_figure=no` so the exclusion is visible rather than silent.
  8 of 76 rows are dropped this way: `mathmatch` entirely (3), `textgroups` down to a single
  point at 2k (4), and `reorder`'s 16k rung (1).

⚠ The `per_N` rows are **not the same runs as the suite rows**, and HotpotQA is not even the
same model (Qwen3.5-0.8B LoRA, vs 4B full fine-tune everywhere else). The figure script's
docstring explains why each was sourced that way — read it before re-plotting these together.

### `fig4_cross_task_data.csv` (77 rows)

Extra columns:

- `panel` — which of the 9 panels the row belongs to. `panel_y_label` is the axis label that
  panel actually prints, which is coarser than `metric` (the cross-document panel puts four
  different metrics on one axis).
- `x_axis` — `tokens` for the suite ladders, `documents` for the four kept per-$N$ panels.
- `context_tokens_exact` — `yes` for token-ladder rows. `no (N x per-doc tokens)` means the
  context length is **derived**: $N \times$ the per-doc token estimate from Table 5
  (contradiction ~45 tok/doc, grouping ~200, outlier-wiki / reorder ~130). Use `n_docs` if
  you want the number the run was actually parameterized by.
- `variant` — outlier's `scale-k` vs `fixed-k` ablation (solid vs dashed in the figure), and
  the single `chunked, NO mask mixing` X point on reorder.
- `notes` — currently only marks that X point, which has no matching full-attention run.

Reorder is Qwen3.5-9B; every other panel is 4B.

Blank `full_attention` / `chunked_attention` means **that run does not exist**, not zero.

## Caveats worth passing on

- **eval_size below 500.** OBLIQ is 126 per rung (SE ±0.045) and contradiction is 488 (the
  entire held-out file). Anything quoted off those two needs its error bar inline.
- **`unknown` eval sizes.** The per-$N$ ladders aren't covered by the figure scripts'
  "500 per rung" statement, so they're exported as unknown rather than assumed. Appendix
  Table 6 flags outlier-wiki at 56–100 examples per rung, and its own Table 5 says 400 for
  the same rows — that disagreement is noted in the appendix and is still unresolved.
- **`absence` is classified `low` / `O_T(N)` here**, following Prasann's 2026-08-03
  correction that both figure scripts implement. Table 1 (`table_suite.tex`) still lists it
  under the $O_T(N^2)$ block and is stale on this point. `xabsence` is the genuinely
  $O_T(N^2)$ one.
- **Metric naming.** The panel labels in the grid call NIAH / MS MARCO / OBLIQ "gold-ID F1"
  where Table 1 calls them set-F1. The `metric` column follows Table 1; `panel_y_label`
  preserves what the figure prints.
- **Sign convention.** A negative `relative_gap` means chunked *beat* full at that rung
  (real, e.g. NQ and GovReport). Don't clip it to zero.
- Each figure script's module docstring lists every task that was **excluded** and why
  (broken checkpoints, floor-level scores, voided ladders). Those reasons are not in the
  CSVs — read the docstrings before adding a task back.

---

# Raw data for the CTC suite (the "CTC Suite Grid" artifact, frozen 2026-08-13)

Three tidy CSVs, regenerated with:

```bash
cd paperdraft/figures && python3 export_ctc_suite_data.py
```

| CSV | Rows | What it is |
|---|---|---|
| `ctc_suite_grid_data.csv` | 110 | The 22-task 2k–32k ladder at 4B, both arms — one row per (task, rung) |
| `ctc_scale_data.csv` | 72 | The same ladders at 0.8B / 2B / 4B for the five retrained tasks |
| `ctc_length_generalization_data.csv` | 45 | 64k / 128k past the 32k training ceiling, plus the outlier fix-k n sweep |

Unlike the figure CSVs, these are **not** exported from a plotting script; the dependency runs
the other way, and `make_fig.py` / `make_fig2.py` read them (see the bottom of this file).
`export_ctc_suite_data.py` reads the same harvested grade JSONs that
`debug/ctc_final_suite/render_artifact.py` renders the artifact page from
(`suite_table.json`, `scale_table.json`, `debug/ctc_modelscale/lengthgen_results/**`), so the
CSVs and the page cannot disagree. All 220 grid cells and both aux tables were diffed against
the rendered page at export time; re-run that check if you change the exporter.

## Columns

Same names and meanings as the figure CSVs above (`task`, `ctc_class`, `ctc_order`, `metric`,
`context_tokens`, `n_docs`, `full_attention`, `chunked_attention`, `relative_gap`, `eval_size`,
`plotted_in_figure`, `drop_reason`), plus:

| Column | Meaning |
|---|---|
| `series` | Grid: the row's coverage status (`ok` / `partial` / `evals-owed` / `superseded`). Scale: the model (`Qwen3.5-0.8B` …). Length-gen: the task |
| `task_label` | Human name as printed on the artifact page; `task` is the machine key |
| `metric_key` | The grader's own metric name, for tracing a row back to its grade JSON |
| `absolute_gap` | `full - chunked`. This is the quantity the artifact page's "Gap" column shows; `relative_gap` (`1 - chunked/full`) is the Fig 1 quantity. Both are provided |
| `full_se` / `chunked_se` | Binomial SE `sqrt(p(1-p)/eval_size)` — an **upper** bound for a [0,1] per-example metric, since the graders don't persist per-example scores. Run-to-run seed variation is on top and is not measured |
| `caveats` | Things that qualify a number without invalidating it, `\|`-separated (several caveats contain their own semicolons) |
| `context_tokens_measured_p50` / `measured_over_label` | Measured p50 tokens for the rung, where an audit exists, and the ratio to the label |
| `ladder`, `task_dir`, `rung_file` | Provenance: which staged ladder and which rung file the cell was graded on |
| `full_parse_rate` / `chunked_parse_rate` | Scale CSV only. Below 1.0 means the grader could not parse the answer back out — see the stop-token bug below |
| `gold_semantics` | Length-gen only. Whether the long rungs may be built by injection at all — see below |

`drop_reason` is set **only** on rows with `plotted_in_figure=no`; anything that qualifies a
kept number lives in `caveats`. Blank `full_attention` / `chunked_attention` means **that run
does not exist**, not zero.

## What is dropped, and why

`plotted_in_figure=no` on 14 grid rows, 14 scale rows and 14 length-gen rows:

- **`qdmatch fiqa` (5 rows)** — training finished on Beaker 2026-08-13 with exit 0; the
  weka-to-S3 relay, HF export and eval are still owed. The only row with no number in either arm.
- **`xabsence` (5 rows)** — superseded. Training CE sat on the 0.78 flatline, so those numbers
  never came from a model that learned the task. Do not quote them.
- **Three never-built rungs** — `absence` 32k, `reorder` 32k, `outlier fix-k` 32k.
- **`niah contra` dense 2k** — 0.164 on disk vs 0.988 in the July table; the on-disk number
  predates the pipeline fix, so the 2k pair is not comparable. The 64k/128k niah rungs are
  dropped for a related reason: two independent builds disagree with each other (0.736 vs
  0.002 at the same length), so that ladder needs a rebuild before any of it is quoted.
- **Injection-unsafe long rungs (10 rows: outlier, outlier amazon, cycle, textgroups,
  absence)** — the 64k/128k rungs are built by injecting filler documents. That is sound when
  gold is defined by a relation to something specific (`query_match`, `pairwise`) and unsound
  when gold is defined by absence or by a structure over the whole corpus (`absence`,
  `structural`), because an injected document then *satisfies* the gold condition without
  being labelled. These were built and graded before the rule existed and all scored at the
  floor — that is the labels failing, not the model. They need natively-generated rungs. The
  rule is `gold_semantics` in `debug/ctc_modelscale/expand_ctc_rung.py`.
- **Missing scale arms (9 rows)** — `fiqa` was never run at 2B. `contradiction` and `reorder`
  have no 0.8B chunked arm: both OOM at 0.8B × chunked-mix × seq_len 40960 on saturn and
  jupiter while the same config trains fine at 2B/4B and at 0.8B × 33792. Four hypotheses were
  tested and refuted. **Do not read a gap from that absence.**

## Caveats worth passing on

- **The stop-token bug.** `stop_token_ids` ships only `<|endoftext|>` (248044) but our SFT
  targets end the turn with `<|im_end|>` (248046), so vLLM never stops and the model repeats
  itself to the token cap. Tasks graded with `stop="newline"` get an accidental text-level
  cut; the `stop="eos"` ones do not. It is not uniformly harmless — adding the stop token moved
  outlier fix-k from 0.3191 to 0.3913 at n=220, while moving reorder at 2B not at all (a model
  already at chance loses nothing to rambling). Affected cells carry the caveat and are
  **lower bounds, not measurements**, until the re-eval lands. In the scale CSV the bug is
  directly visible as `parse_rate` below 1.0.
- **`absence` is classified `low` / `O_T(N)`**, per Prasann's 2026-08-03 correction, the same
  as in the figure CSVs. `suite_table.json` still classes it `N2` following the stale Table 1;
  the exporter overrides it and says so in the code. `xabsence` is the genuinely `O_T(N^2)` one.
- **eval_size below 500.** obliq is 126 per rung, scifact 300, and absence's 16k rung 148.
  Each of those rows carries an inline `caveats` entry with its SE. The obliq cells are also
  carried from the 2026-07-27 table and have no grade JSON on disk.
- **Rung labels are not token counts on every ladder.** `absence` runs 3.0–3.6× its label,
  `qdmatch hpqa` 0.74–0.82×, `strmatch` 0.60×, `outlier amazon` 1.10–1.26×. Use
  `context_tokens_measured_p50` where it is populated; a label-only x-axis mislabels the axis
  rather than the metric. Four of those ladders are queued for a token-accurate rebuild.
- **`n_docs` is mostly blank on the token ladders.** It is filled only where a rung's corpus
  size is stated unambiguously (contradiction; the length-gen manifest; the fix-k n sweep).
  `debug/ctc_modelscale/hardneg_audit.json` has per-rung document counts for more tasks, but it
  carries three ladder variants per task (orig / rebuilt / ext32k) with no field saying which
  one was graded — joining it would risk a wrong N next to a right score.
- **oolong's 64k rung has eval_size 668, not 500.** The artifact page's length-generalization
  table says "eval_size 500 everywhere"; the grade JSON says 668. The CSV follows the JSON.
- **The outlier fix-k rows in the length-gen CSV are a DOCUMENT axis, not a token axis**
  (`context_tokens` is blank, `n_docs` is the x). They are there because they replaced fix-k's
  32k rung: the apparent cliff at 32k was rung spacing (the ladder steps n=111 → 220 in one
  jump), and filling the gap shows a smooth monotone decline that continues past the
  `n ~ U[14,220]` training support out to n=440. Dense arm only.
- **Two different contradiction rows.** `niah_contra` is NIAH-contradiction, a retrieval task.
  `contra_real` is the PubMed multi-claim task, scored on the IID `realistic`-mode ladder that
  matches its training generator — that choice is the whole result, and mixing it with the
  older `both`-mode ladder in one figure is the mistake to avoid. See
  `records/contradiction-train-eval-non-iid.md`.
- **Sign convention**, as in the figure CSVs: a negative gap means chunked *beat* full at that
  rung. Real (nq, hpqa at some rungs). Don't clip it to zero.
- These CSVs do **not** pre-filter on Fig 1's `FLOOR = 0.15` rule (drop a rung whose full arm is
  under 0.15, because a ratio between two near-zero numbers is noise). Apply it downstream if
  you are plotting `relative_gap`; `strmatch`, `textgroups` and `reorder` have rungs that need it.

## Figures built from these CSVs

`make_fig.py` and `make_fig2.py` (in the parent directory) are Prasann's plotting scripts,
repointed from the hand-made `/home/claude/ctc_data.csv` onto `ctc_suite_grid_data.csv` — same
columns, so the plotting is otherwise unchanged.

```bash
python3 paperdraft/figures/make_fig.py            # ctc_figure.{png,pdf}   aggregate + 6 task panels
python3 paperdraft/figures/make_fig2.py           # ctc_figure2.{png,pdf}  chained mean + penalty bars
python3 paperdraft/figures/make_fig_lengthgen.py  # ctc_lengthgen_figure.{png,pdf}
python3 paperdraft/figures/make_fig_scale.py      # ctc_scale_figure.{png,pdf}
```

Three things they do that the originals did not, each marked ⚠ in the source:

1. **They filter to `plotted_in_figure == "yes"`.** Without it the superseded xabsence row and the
   never-built rungs would be averaged in as measurements.
2. **`make_fig.py`'s "cross-document absence" panel shows `absence`, not `xabsence`.** xabsence is
   superseded, so that panel's meaning flipped from a high-CTC collapse to a low-CTC pair holding
   flat. `strmatch` is the natural stand-in if a high-CTC collapse is wanted back in that slot —
   one line in the `panels` list.
3. **`make_fig2.py`'s penalty bars apply `FLOOR = 0.10`.** `relative_gap` is a ratio, so once the
   full arm is near zero it is noise: grouping at 32k is full 0.0066 vs chunked 0.0095, a
   relative_gap of −0.44 that enters the mean as a spurious 44% chunked *win*. That cell plus
   reorder dropping out at 32k made the high-CTC bar read −61% at ≥32k after −70% at 16–32k, i.e.
   the penalty appearing to improve at the deepest bucket. With the floor the series is monotonic:
   −32, −37, −48, −63, −72. `fig1_gap_vs_context.py` uses 0.15 for the same purpose; this uses
   0.10 because 0.15 also swallowed qdmatch_fiqa@32k, a full arm of 0.1244 ± 0.0148 (8.4 SE above
   zero) — a real measurement, and the newest high-CTC task, dropped from the deepest bucket
   exactly where its gap is widest. 0.10 keeps it and still excludes every genuinely-noise cell.
   The floor bites only on high-CTC rows; the low-CTC bars are identical at any threshold.

### `make_fig_lengthgen.py` → `ctc_lengthgen_figure.{png,pdf}`

Reads `ctc_length_generalization_data.csv`. Qwen3.5-4B evaluated on 64k and 128k rungs of tasks it
was only ever trained on up to 32k. Four panels: (a) absolute dense score, (b) the same lines
divided by their own 32k score so absolute difficulty drops out, (c) qdmatch (NQ) — the only task
with a chunked arm run past the ceiling, (d) outlier (wiki, fixed $M$) on a **document-count**
axis, not a token axis.

The headline is panel (b): at 128k the five low-CTC ladder tasks keep 35–78% of their 32k score,
while the two high-CTC ones keep 3–6%.

Two things to know before quoting it:

- **Only 7 of the 13 tasks in the CSV have a slope to draw.** Everything else is 32k-only after the
  exclusions. The figure builds its own "excluded long rungs" footnote from the `drop_reason`
  column, so the note can never drift from the data — read it rather than assuming the 6 missing
  tasks were simply not run. Eight of the 14 excluded rows are the `gold_semantics=structural`
  retraction (injected filler silently satisfies the gold condition), two are the `absence`
  version of the same bug, and four are NIAH-contra, where two independent builds disagree
  0.736 vs 0.002 at 64k.
- **Panel (d) is on a different x-axis** and is labelled as such on the figure. It is the fixed-$M$
  outlier arm, dense only, sweeping 111→440 documents; score falls 0.98→0.06 across that range.

### `make_fig_scale.py` → `ctc_scale_figure.{png,pdf}`

Reads `ctc_scale_data.csv`. Qwen3.5 at 0.8B / 2B / 4B on identical shards and identical ladders, so
parameter count is the only thing varying within a panel. Colour = model scale (a purple ramp, kept
distinct from the blue/red CTC-class colours the other figures use); solid/dotted = full/chunked as
everywhere else. Panels (a)–(e) are one task each, (f) summarises `relative_gap` at a fixed rung.

- **Only complete full+chunked pairs are drawn**, and every missing pair is named in-panel with its
  reason rather than left as a silent absence. This matters: the `drop_reason` for the 0.8B chunked
  runs says explicitly *not* to read the missing arm as a gap — 0.8B × chunked-mix × seq_len 40960
  OOMs on both saturn and jupiter while 2B/4B and 0.8B@33792 train fine, and the cause is still
  unknown after four refuted hypotheses. The fiqa 2B pair was simply never launched.
- **Panel (f) applies the same `FLOOR = 0.10`** as `make_fig2.py`, for the same reason. It drops
  0.8B qdmatch (full arm 0.0000 at 32k) and 2B reorder (0.0041 at 8k) — cases where the model
  cannot do the task at all, so `1 - chunked/full` is a ratio of noise.
- **reorder is summarised at 8k, not 32k.** Its ladder stops at 16k and its full arm is already
  under the floor there (0.0473 at 4B). The panel's tick label carries the rung, so no group is
  silently compared at a different context length than it looks.
- The gap does **not** move in one direction with scale: contradiction goes 35%→26% from 2B to 4B
  while qdmatch goes 73%→85%. What is stable is that it stays large on high-CTC tasks at every
  scale measured, and stays at ~0 on HotpotQA at every scale.
