# paperdraft — "Time to Pay Attention" (CTC suite paper)

LaTeX source for the CTC-suite paper (NeurIPS/ICLR style), unpacked from
`V2_TimeToPayAttention.zip`. Same layout as `corpus-reasoning/paperdraft/`.

## Build

```bash
cd paperdraft
make pdf      # -> iclr2026/iclr2026_conference.pdf
make watch    # latexmk -pvc, rebuilds on save
make clean    # remove build artifacts (keeps the source + figures)
```

`make pdf` runs `latexmk -pdf` (pdflatex + bibtex, repeated to convergence).
**Build from this directory**: the `.tex` files reference `iclr2026/sections/...`
and `figures/...` relative to `paperdraft/`, and the Makefile sets
`TEXINPUTS="iclr2026:"` so `\usepackage{neurips_2026}`, `math_commands.tex`,
`natbib.sty`, `fancyhdr.sty` and the `.bst` are found inside `iclr2026/`.

## Editing rules

- **Never add citations.** Prasann is the only one who adds citations, always by hand. Do not write
  `\citep{...}`/`\cite{...}` for a source that is not already in `iclr2026_conference.bib`, and do not
  append new `.bib` entries. Citation metadata written from
  memory is plausibly wrong (venue, year, a dropped author) and that error ends up in the published
  bibliography. Name the source in prose instead --- "two BEIR tasks", "following AbsenceBench
  (arXiv:2506.11440)" --- and leave the citation for the authors. Reusing a key that already exists in
  the .bib is fine.
- Mark machine-drafted passages with `\autonote{what changed}` (orange margin bubble) so each edit can
  be reviewed and signed off; `\autonotesoff` hides them all. `\autonote` is a `todonotes` float, so
  it cannot go inside a `figure`/`table` environment --- LaTeX drops it with "Float(s) lost".

## Layout

```
paperdraft/
├── Makefile                     # build pipeline
├── figures/                     # figure PDFs (tracked; referenced as figures/<name>.pdf)
└── iclr2026/
    ├── iclr2026_conference.tex  # MAIN file — preamble, author edit macros, \input of sections
    ├── iclr2026_conference.bib  # bibliography
    ├── sections/                # 1_introduction, 3_corpus_reasoning, 4_contradiction_case_study,
    │                            # 6_learning, 7_related_work, 8_conclusion, appendix_*
    ├── checklist.tex            # conference checklist appendix
    ├── neurips_2026.{sty,tex}   # style + the style's own template/instructions doc
    ├── natbib.sty, fancyhdr.sty, iclr2026_conference.bst, math_commands.tex
    └── iclr2026_conference.pdf  # build output (overwritten by `make pdf`)
```

Notes:
- `neurips_2026.tex` is the style package's template/instructions document, **not** the
  paper. The main file is `iclr2026_conference.tex`.
- Sections `2_related_work` and `5_scaling_task_computation` are currently commented out
  in the main file.
- The preamble defines per-author edit macros (`\prasann{...}`, `\prasannnote{...}`, and
  matching colors for the other authors) plus `\ctc{}` / `\cmc{}` complexity shorthands.
- Build artifacts (`.aux`, `.bbl`, `.log`, ...) are gitignored via `paperdraft/.gitignore`;
  the figure PDFs and the built paper PDF are not.
