# Vendored: CTC task package

This directory is a verbatim copy of the `ctc/` package (Corpus Task Complexity: task generators,
seed pools, prompt formatting, graders) from:

- **repo:** `PrasannS/corpustaskcomplexity` (`git@github.com:PrasannS/corpustaskcomplexity.git`)
- **branch:** `prasann/ctc`
- **commit:** `40a5c60d143b427b8c2cd276fa5009b809be6be3`

Only git-tracked files were copied (`git archive 40a5c60d1 ctc`). Edit it upstream first and
re-vendor; do not let the two copies drift.

## Install

`ctc` is its own package (its own `pyproject.toml`, source under `ctc/src/ctc`), separate from
`olmo_core`. Install it editable from the repo root:

```bash
pip install -e ctc/
```

This provides the `ctc-data` CLI used by `recipes/data/ctc_sft/` (e.g. `ctc-data build --pool auto`).

## Formatting

`ctc/src/ctc/format/prompts.py` is excluded from black/isort in both the root `pyproject.toml` and
`ctc/pyproject.toml`: its string literals ARE the training data, so reformatting would change the
tokenized prompts.
