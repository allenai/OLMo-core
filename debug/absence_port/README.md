# absence_port

Calibration + shortcut evidence for the `absence` / `xabsence` generator port.

- `measure_ladders.py` — tokens-per-document for both ladders, measured by rendering real prompts
  through the task spec with a Qwen3 tokenizer (the method that replaced the ~3.4×-overshooting
  BUILD_MATRIX estimates in `ctc.data.ladders`).
- `probe_xabsence_sim.py` — how solvable xabsence is by lexical overlap and what closes it; runs
  the same `unmatched_by_lexical_overlap` probe the build audit runs.
- `check_long_runs.py` — whether the Gutenberg pool holds *unbroken* prose runs long enough for
  the long absence rungs (a run breaks at every heading, not at book end).
