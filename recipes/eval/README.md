# recipes/eval — eval-side tooling for the canonical pipeline

The canonical CTC-suite eval runs from olmo-eval (`scripts/ctc_suite/launch_ctc_suite.py`); see
`recipes/README.md` §3 and `records/ctc-fast-suite-eval.md`. This folder holds what this repo
contributes. Eval outputs live on weka (`.../_olmoeval_ctc/`) and in results-hub, not here.

| Path | What |
|---|---|
| `setA_audit/` | Train/eval IID audit of a built set vs the eval's own rows (`audit_train_eval_iid.py`), doc-marker token parity (`check_doc_marker_parity.py`), forward parity (`forward_parity*.{py,sh}`), token counts. `olmo_eval_roster.json` is the frozen olmo-eval roster (`dump_olmo_eval_roster.py`) that the audit, `measure_eval_ladder.py` and `legacy_native/ctc_eval/olmo_eval_backend.py` read. |
| `ctc_eval_speed/` | `answer_tokens.json` (measured answer lengths per row, read by `olmo_eval_backend.py`'s planner) + `measure_answer_tokens.py`, `profile_landmark_setup.py`. |
| `make_yarn_copy.py` | Config-only YaRN serving copy of a step dir (symlinked weights) for rungs past the trained context. |
| `legacy_native/` | **Legacy** in-repo native eval path; see its README. |
