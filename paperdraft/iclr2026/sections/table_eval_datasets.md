# Overview of evaluation datasets — CTC suite (HELMET Table 3 style)

Markdown mirror of `table_eval_datasets.tex` (`\label{tab:eval-datasets}`), 22 tasks.

Synced 2026-08-13 to the frozen 22-task suite (the CTC-suite grid artifact, backed by
`debug/ctc_final_suite/suite_table.json`). This table and `table_suite.tex` now agree row for row
and class for class; the old "reconcile before submission" deviations are resolved.

The `.tex` is prasann's self-contained rewrite: fixed-width `p`-columns sized as fractions of
`\linewidth` rather than hard cm values, `\providecommand` fallbacks for `\ctc`/`\cname`, and
`\ensuremath` inside `\ctc` so `\ctc{N^2}` needs no surrounding `$...$`. It compiles in a fresh
project given `\usepackage{booktabs,multirow,array}`. Keep it self-contained — that was the point.
If the paper goes two-column, switch `table` → `table*`: at ~1.2 pages tall in a single column the
float would never fit, and LaTeX defers an unfittable float silently rather than erroring.

Set-F1, pair-F1 and group-F1 are computed over document IDs, so every metric but partial credit and
Kendall-τ is scored against an exact ID set; grouping is additionally scored with ARI.

Three classes were corrected against the artifact, all at the source so they stop propagating:

- **Outlier (Amazon) is O_T(N)**, confirmed by prasann 2026-08-13 — its attribute set (star rating,
  product category) is small and fixed, so there is no category discovery, the same property that
  makes Oolong O_T(N).
- **Absence is O_T(N)**, confirmed by prasann 2026-08-13 — checking whether item *i* survived into
  the second copy is one operation per item, not an all-pairs search. This was the last class still
  disagreeing across sources: the figure scripts and the exported CSVs already had it as O_T(N) per
  the 2026-08-03 correction while both tables had it as O_T(N²). `xabsence` — two corpora of
  paraphrase twins, where finding an orphan genuinely does need all pairs — stays O_T(N²).
- **Textgroups is O_T(N³)**, though the artifact shows O_T(NM): `suite_table.json` has only three
  class values (`N`, `NM`, `N2`) and cannot express O_T(N³), so textgroups was bucketed rather than
  classified. Its oracle operation is a triple. Widening that schema is the outstanding fix.

The first two are fixed in `debug/ctc_final_suite/build_suite_table.py`, so the artifact will agree
on its next render and `export_ctc_suite_data.py` no longer carries a `CLASS_OVERRIDE` for either.
Re-running the exporter after the absence fix produced byte-identical CSVs, so no figure moved.

Still open: §3 prose has NOT been updated (still says "24 tasks", still describes qdmatch as
NQ/HPQA/OBLIQ, still references NarrativeQA, GovReport, cycle, groups-of-4). And the artifact grades
contradiction and strmatch with `set_f1` where these tables say pair-F1 / EM and pair-F1 — possibly
a real mismatch rather than the documented naming convention.

## O_T(N) — one pass over the corpus

| Category | Dataset | Metrics | Description |
|---|---|---|---|
| **Retrieval** | NQ | Gold-ID F1 | Factoid QA over 100-word Wikipedia passages |
| | HotpotQA (bridge) | Gold-ID F1 | Two-hop QA; two gold passages per question |
| | NIAH-contra | Set-F1 | Find the one claim contradicting the query claim |
| | BEIR SciFact | Set-F1 | Retrieve the abstract bearing on a science claim |
| | BEIR FiQA | Set-F1 | Financial opinion QA, median two golds per query |
| | MS MARCO | Set-F1 | Web-passage retrieval with mined hard negatives |
| **Passage re-ranking** | MS MARCO rerank | MRR@10 | Rank the full passage pool by relevance to a query |
| **Subjective retrieval** | OBLIQ | Set-F1 | Long, subjective queries (e.g. "similar in style") |
| **Label aggregation** | Oolong | Partial credit | Classify every item, then answer a distributional query |
| **Absence detection** | Absence | Set-F1 | Corpus vs. a copy with deletions; name what is missing |
| **Categorization, closed M** | Outlier (Amazon) | Set-F1 | Name the reviews carrying the minority attribute; the attribute set is small and fixed |
| | Outlier (wiki, fixed M) | Set-F1 | As below, but the article count stays fixed as N grows |

## O_T(NM) — each document against each of M categories

| Category | Dataset | Metrics | Description |
|---|---|---|---|
| **Categorization** | Outlier (wiki, scale-k) | Set-F1 | Name the chunks from the least-represented article |
| | Grouping (OpenAlex) | Pairwise-F1 | Partition N abstracts into k groups; axis unstated |

## O_T(N²) — all-pairs search

| Category | Dataset | Metrics | Description |
|---|---|---|---|
| **All-pairs semantic search** | Contradiction | Pair-F1 / EM | Find all K contradicting claim pairs among N claims |
| | xabsence | Set-F1 | Two corpora of paraphrase twins; name the orphans |
| **Lexical matching** | strmatch | Pair-F1 | Find every string pair sharing a ≥k-word run |
| **Query–document matching** | qdmatch (NQ) | Pair-F1 | Match M queries to N documents under one numbering |
| | qdmatch (HPQA) | Pair-F1 | …with two gold documents per query |
| | qdmatch (FiQA) | Pair-F1 | …with financial-opinion queries |
| **Ordering** | Reorder | Kendall-τ | Recover the original order of shuffled book segments |

## O_T(N³) and beyond — groups, not pairs

| Category | Dataset | Metrics | Description |
|---|---|---|---|
| **Higher-order groups** | Textgroups | Group-F1 | Find every three passages whose feature counts sum to T |
