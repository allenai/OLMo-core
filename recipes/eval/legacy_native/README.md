# recipes/eval/legacy_native — the in-repo native eval path (LEGACY)

Superseded for the CTC suite by olmo-eval's `launch_ctc_suite.py` (`recipes/README.md` §3). Kept
because it still scores the older singletask / xlong eval bundles (v2/v3 ladders,
`records/v3-eval-howto.md`) and is what the `run-evals` skill's multirung path drives.

| File | What |
|---|---|
| `ctc_eval/` | Python package `ctc_eval` (ex-`src/scripts/ctc_eval/`): `eval/eval_lc_native*.py` drivers, graders, prompts, `olmo_eval_backend.py` planner, `launch_ctc_evals.py`. Import with `PYTHONPATH=recipes/eval/legacy_native:src`. |
| `run_q4b_beaker_multirung_eval.py` | Beaker launcher: one job per (run, task) → `run_beaker_multirung_eval.sh` (8-way torchrun on node). |
| `run_q4b_stl_multirung_eval.sbatch` | Local slurm twin; submit from the repo root (it derives `REPO` from `SLURM_SUBMIT_DIR`). |

Ledger entries go to `records/eval_launches/` (gitignored); results to results-hub.
