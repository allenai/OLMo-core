# ctc_strmatch_redundancy_port

Evidence for the `strmatch` + `redundancy` generator port (commit `510a18be6`).

- `probe_shipped_strmatch.py` — the measurement that showed shipped strmatch was solvable by
  shared-word count alone (200/200, chance 0.004), which drove the scattered-decoy fix.
- `compare_shortcut.py` — old vs fixed construction under the same probe (1.000 → 0.010 at n=38).
- `calibrate_strmatch.py` / `calibrate_ladders.py` — the tokenizer-measured rung rows now in
  `ctc.data.ladders` (shipped strmatch rungs were ~0.56× their labels).
- `harvest_redundancy_pairs.py` + `redundancy_pairs.jsonl` — the pair pool the redundancy
  generator builds from; `build_*.log`, `samples/` — build evidence.
