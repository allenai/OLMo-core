# Paper figures — label index

Every figure has a **stable label**. Refer to a figure by its label ("re-render CTC-SCALE-ROW with
the reorder panel back in") and any session can find the script, the data and the outputs from the
table below without guessing.

The label lives in three places, so it survives a file being renamed or a PDF being emailed around:

1. `FIGURE_LABEL` at the top of the script.
2. This table.
3. The PDF/PNG metadata — `pdfinfo ctc_scale_row.pdf | grep Title` prints the label and a
   one-line description of what the figure shows.

## The CTC suite figures

| Label | Script | Outputs (`.pdf` + `.png`) | Data | What it shows |
|---|---|---|---|---|
| **CTC-GRID** | `make_fig.py` | `ctc_figure` | `data/ctc_suite_grid_data.csv` | Low-vs-high-CTC chained-mean aggregate over the 2k–32k ladder, plus 6 high-CTC task panels (outlier, qdmatch_hpqa, reorder, contradiction, strmatch, textgroups). |
| **CTC-PENALTY** | `make_fig2.py` | `ctc_figure2` | `data/ctc_suite_grid_data.csv` | Same chained mean, plus per-context-bucket penalty bars. Low CTC −1.5 … −2.9%; high CTC −32 … −72%. |
| **CTC-LENGTHGEN** | `make_fig_lengthgen.py` | `ctc_lengthgen_figure` | `data/ctc_length_generalization_data.csv` | Four panels past the 32k training ceiling: (a) absolute dense score at 32k/64k/128k, (b) retention normalised to 32k, (c) qdmatch (NQ) — the only task with a chunked arm past the ceiling, (d) fixed-$M$ outlier on a document-count axis. |
| **CTC-RETENTION** | `make_fig_lengthgen_retention.py` | `ctc_lengthgen_retention` | `data/ctc_length_generalization_data.csv` | CTC-LENGTHGEN panel (b) alone, resized as a standalone paper figure. At 128k, low CTC keeps 35–78% and high CTC keeps 3–6%. |
| **CTC-SCALE** | `make_fig_scale.py` | `ctc_scale_figure` | `data/ctc_scale_data.csv` | Qwen3.5 0.8B/2B/4B on 5 tasks (HotpotQA, FiQA, contradiction, qdmatch NQ, reorder) plus a panel (f) summarising the relative gap at a fixed rung. |
| **CTC-SCALE-ROW** | `make_fig_scale_row.py` | `ctc_scale_row` | `data/ctc_scale_data.csv` | CTC-SCALE with reorder and the (f) summary dropped, laid out as a 1×4 row: two low-CTC then two high-CTC tasks. |

Re-render any of them with, e.g.:

```bash
python3 paperdraft/figures/make_fig_scale_row.py     # prints "[CTC-SCALE-ROW] wrote ..."
```

All six read only the CSVs in `data/`. Those are built by `export_ctc_suite_data.py` from
`debug/ctc_final_suite/suite_table.json` + `debug/ctc_modelscale/`, never from the rendered
artifact HTML. Rebuild the whole chain with:

```bash
python3 debug/ctc_final_suite/harvest_grades.py    # /net -> harvested_grades.json  (needs the nodes)
python3 debug/ctc_final_suite/build_suite_table.py # -> suite_table.json
python3 paperdraft/figures/export_ctc_suite_data.py # -> paperdraft/figures/data/*.csv
```

`data/README.md` documents the CSV columns, what is dropped from each figure and why, and the
`FLOOR = 0.10` rule that keeps near-zero dense arms out of any ratio.

## Older figures in this directory

These predate the CTC suite and are listed so the labels do not collide. Their scripts have not been
given a `FIGURE_LABEL`; the label here is for reference only.

| Label | Script | Outputs | Notes |
|---|---|---|---|
| **GAP-CONTEXT** | `fig1_gap_vs_context.py` | `fig1_gap_vs_context{,_labeled}` | Uses `FLOOR = 0.15`. Its data file `data/fig1_gap_vs_context_data.csv` is **0 bytes** — zeroed 2026-08-08, same day `iclr2026_conference.tex` was emptied. It will not re-render until that is restored. |
| **CROSS-TASK** | `fig4_cross_task.py` | `fig4_cross_task` | Reads `data/fig4_cross_task_data.csv`, built by `export_figure_data.py`. |

The loose `.pdf` files with no script (`attention-masks-figure*.pdf`, `Figure 1 — Print.pdf`,
`Figure_1 v2.pdf`, `Figure1v4.pdf`, `section4_*.pdf`, `section5_*.pdf`, `section6_*.pdf`,
`gap_2k_20k.pdf`) were made outside this repo and have no regeneration path here.

⚠ `Figure 1 — Print.pdf` has a space and a non-ASCII em-dash in its filename. `\includegraphics`
will not resolve that without bracing or `\detokenize`, and a failed `\includegraphics` aborts the
LaTeX compile — so everything after it in the document silently fails to render. Rename it before
using it in a paste-able section.
