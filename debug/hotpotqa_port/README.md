# hotpotqa_port

Calibration + contract evidence for the `hotpotqa` generator port (commit `510a18be6`).

- `measure_ladder.py` — the tokenizer-measured 17/36/72/144/288 ladder row (real pool, real
  generator, real prompt path), replacing BUILD_MATRIX row 2's 0.64–0.69× undershoot.
- `verify_shipped_rungs.py` — checks the ported loader's claims (ladder row, gold index base,
  distractor provenance) against the actual shipped pre-migration files, where the dataset card
  cannot answer.
- `download.log` — corpus fetch evidence.
