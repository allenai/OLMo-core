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
| **CTC-GRID** | `make_fig.py` | `ctc_figure` | `data/ctc_suite_grid_data.csv` | 2×3 grid of six cells: the **Average** panel (low-vs-high-CTC chained mean over the 2k–32k ladder) takes the first **two** cells at double width, and four high-CTC task panels fill the rest — Outlier Detection, Reordering, Contradiction Detection, Text Grouping. Restructured 2026-08-14: `strmatch` and `qdmatch_hpqa` were dropped as panels to free the two cells (both remain in the Average lines and in the suite table — a figure-space decision, not a retraction). Conference format, matching CTC-PENALTY: Times-style serif, fonttype 42, **the same BLUE/ORANGE as `make_fig2.py`**, legend on top, no grey text anywhere — the chained-mean method note, per-series task counts and each panel's CTC class + metric are printed to stdout for the LaTeX caption instead of drawn. |
| **CTC-PENALTY** | `make_fig2.py` | `ctc_figure2` | `data/ctc_suite_grid_data.csv` | Same chained mean, plus per-context-bucket penalty bars. Low CTC −1.2 … −2.7%; high CTC −27 … −66% (re-rendered 2026-08-14: was −32 … −72% before the qdmatch_hpqa 2k override and before xabsence rejoined as a 10th high-CTC task). |
| **CTC-LENGTHGEN** | `make_fig_lengthgen.py` | `ctc_lengthgen_figure` | `data/ctc_length_generalization_data.csv` | Four panels past the 32k training ceiling: (a) absolute dense score at 32k/64k/128k, (b) retention normalised to 32k, (c) qdmatch (NQ) — the only task with a chunked arm past the ceiling, (d) fixed-$M$ outlier on a document-count axis. |
| **CTC-RETENTION** | `make_fig_lengthgen_retention.py` | `ctc_lengthgen_retention` | `data/ctc_length_generalization_data.csv` | CTC-LENGTHGEN panel (b) alone, resized as a standalone paper figure. At 128k, low CTC keeps 35–78% and high CTC keeps 3–6%. |
| **CTC-SCALE** | `make_fig_scale.py` | `ctc_scale_figure` | `data/ctc_scale_data.csv` | Qwen3.5 0.8B/2B/4B on 5 tasks (HotpotQA, FiQA, contradiction, qdmatch NQ, reorder) plus a panel (f) summarising the relative gap at a fixed rung. |
| **CTC-SCALE-ROW** | `make_fig_scale_row.py` | `ctc_scale_row` | `data/ctc_scale_data.csv` | CTC-SCALE with reorder and the (f) summary dropped, laid out as a 1×4 row: two low-CTC then two high-CTC tasks. Re-rendered 2026-08-15 when the 0.8B contradiction chunked-mix arm finally landed (0.798/0.755/0.687/0.609/0.520), completing that panel at all three scales. **The caption claim tightened as a result**: contradiction's relative gap *narrows* monotonically with scale (0.40 → 0.35 → 0.26 at 32k) while qdmatch NQ's *widens* (0.73 → 0.85), so "scaling does not close the gap" is right but "scaling does not help" would be contradicted by the contradiction panel. |
| **CTC-FAMILY** | `make_fig_family.py` | `ctc_family_figure` | imports `visualizations/make_family_figure.py` | Two panels, four model families (Qwen3.5-4B, OLMo-3-7B, Olmo-Hybrid-7B, Llama-3.2-3B): contradiction ($O_T(N^2)$) and HotpotQA ($O_T(N)$), solid = full / dashed = chunked. The pairing is the argument — the same models lose 0.005–0.089 on retrieval and 0.21–0.34 on pair-finding at 16k, so the chunking penalty is priced by the task, not the family. **Numbers are imported, not retyped**, from the artifact's data module, so the paper figure and the HTML ledger cannot drift. ⚠ Two series are deliberately absent and annotated in-panel: Olmo-Hybrid chunked on contradiction (final CE 0.958 vs OLMo-3's 0.171 on identical task/data/steps = optimization failure; its own qdmatch chunked arm converged at CE 0.156) and Olmo-Hybrid on HotpotQA (never trained on it — its retrieval-family run was `qdmatch_hpqa`, a different class). ⚠ Contradiction sits on a 2.5k bottom rung and HotpotQA on 2k; each panel draws its own ladder. Importing the module regenerates `visualizations/ctc_family_figure.html` as a side effect. |
| **TABLE-MASKMIX** | `make_table_maskmix.py` | `table_maskmix.{tex,pdf,png}` + `data/ctc_maskmix_data.csv` | `suite_table.json` + `debug/ctc_purechunk/harvest/` | LaTeX table: mask-mixing gain = (chunked-mix) − (pure-chunked), mean over the 2k–32k ladder and at the deepest rung, for 5 tasks. **This is the control the suite never had** — every one of the 31 chunked runs in `ctc_modelscale/LAUNCH_LEDGER.tsv` is `chunked-mix`, so the grid's "chunked" column silently includes the curriculum. Mixing is worth ~0 on $O_T(N)$ retrieval (+0.011) and +0.505/+0.791 on outlier/contradiction. ⚠ Two markers guard two different ways a gain can read wrong: † pure-chunked CE never descended (contradiction 1.175→1.071 flat) — a failure to fit, *not* evidence about what the mask can represent; ‡ floor-limited, the **full-attention** arm is itself under 0.15 at that rung so no arm has headroom (reorder's +0.000 at 16k is +0.473 at 2k). CE provenance: `debug/ctc_purechunk/purechunk_ce_curves.json`. |
| **CTC-QDMATCH** | `make_fig_qdmatch.py` | `ctc_qdmatch_figure` | `data/ctc_suite_grid_data.csv` | The same three corpora asked an $O_T(N)$ question (retrieval) and an $O_T(N^2)$ one (qdmatch): NQ, HotpotQA, FiQA panels + a retention (32k/2k) summary. The N² row loses more retention under chunked attention in **3/3** corpora; under full attention in **2/3** — qdmatch HotpotQA sits at ceiling and is labelled in panel (b) as the exception. |
| **CTC-QDMATCH-COMPACT** | `make_fig_qdmatch_compact.py` | `ctc_qdmatch_compact` | `data/ctc_suite_grid_data.csv` | Wrapfigure-sized (3.4×2.7in), **chunked attention only**, **$O(N^2)$ to $O(N)$ performance ratio on the same corpus** vs context (linear axis, 1.0 = parity), four corpora × **both arms** (solid = full, dashed = chunked). Chunked declines everywhere (NQ 0.82→0.12, HotpotQA 0.65→0.35, FiQA 0.67→0.06, contradiction 0.88→0.72); full sits at parity except FiQA (1.10→0.80, 1.00→1.04, 0.85→0.40, 1.00→0.98) — so the separation between solid and dashed *is* the chunked-attention penalty. Ratios slightly >1 are metric incommensurability, not the N² task being easier: read the slope, not the offset. Palette avoids blue/red/orange — those carry class/scale meaning in CTC-GRID, CTC-PENALTY and CTC-SCALE. Ratio rather than the two raw scores because the rows use different metrics and would need eight lines. Chunked-only: under full attention HotpotQA sits at ~1.0 throughout, so that arm lives in CTC-QDMATCH. ⚠ `niah_contra`'s **dense** 2k cell is the known-bad 0.164 — unused here (chunked 2k is 0.984), but resolve it before adding the dense arm. Earlier bar / scatter / retention variants are in git history. |

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
