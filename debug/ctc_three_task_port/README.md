# ctc_three_task_port

Calibration + build evidence for the last three generators (`reorder`, `qdmatch`,
`grouping_labeled`; commit `2fcc0fb57`).

- `measure_ladders.py` — tokenizer-measured tokens-per-document over each generator's own output;
  the fits are the `ctc.data.ladders` rows and their derivations are quoted there.
- `diagnose_gutenberg_runs.py` — why some reorder draws fail (contiguous-passage supply).
- `build/`, `calib/`, `tok_*/` — build outputs and the tokenizer copy (gitignored where large).
