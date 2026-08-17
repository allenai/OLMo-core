# local_eval_smoke

The first end-to-end local run of `ctc-eval` on a real checkpoint (Berkeley cluster, node-local
bundle + checkpoint), kept as the reference invocation: node-local `HOME`/`TMPDIR`/`HF_HOME`,
`PYTHONPATH` pinned to this checkout, `--dry-run` first, then the graded run. See
`records/running-ctc-evals.md` ("Running it without Beaker") for the current instructions.
