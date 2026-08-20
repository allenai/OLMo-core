# records/ — standalone writeups

Experiment diagnoses, port records, and setup notes that outlive one working session. New writeups
of this kind go here; the repo root keeps only the standard project files. Each record leads with
its status and verification date — trust the newest dated statement over anything below it.

| record | what it is |
|---|---|
| [`reference-runs.md`](reference-runs.md) | The main experiments as node-local commands with internal paths: the 22-task per-task protocol, the model-scale sweep, the 5-task mix — plus the holes the config recovery surfaced (unaudited Qwen3-4B markers, unattested CPT long-text run). Public twin: `REPRODUCING.md` on `prasann/ctc_public`. |
| [`ctc-data-generation.md`](ctc-data-generation.md) | The paper's data appendix: every task's construction, gold-uniqueness guarantee, and anti-shortcut controls. |
| [`data-generator-port.md`](data-generator-port.md) | The port record for `ctc.data`: what crossed over, the 19-trap index, and the defect ledger. **Read §3–4 before any rebuild.** Port complete as of 2026-08-17. |
| [`running-ctc-evals.md`](running-ctc-evals.md) | How anyone with a checkpoint gets its numbers: the Beaker one-liner, the bundles (`v2` / `v2_clean` / `fast`), ultra-long rungs, and the flags that must match training. |
| [`training-launcher-curation-plan.md`](training-launcher-curation-plan.md) | The audit that collapsed 161 pre-migration launchers into the keep set behind `src/scripts/ctc/train/`. |
| [`vendored-prompt-builder-audit.md`](vendored-prompt-builder-audit.md) | Duplication audit of `ctc.format` against the vendored prompt builder; its `grouping_labeled` finding is fixed. |
| [`oolong-preamble-trap-investigation.md`](oolong-preamble-trap-investigation.md) | Verdict on port trap 3: the code-side mismatch was the item-regex bug already fixed; the pre-2026-07-26 shard on disk is what still needs rebuilding. |
| [`256k-dense-vs-complm-short-context.md`](256k-dense-vs-complm-short-context.md) | Open question: why dense is weak at *short* context in the 256k runs; two hypotheses ruled out, discriminating experiment specced. |

Related but living elsewhere:

- **How to build data** — `ctc/src/ctc/data/README.md` (user-facing; includes rungs beyond 32k).
- **How to train** — `src/scripts/ctc/README.md`.
- **The olmo-eval integration** — repo `olmo-eval`, branch `prasann/ctc-suite`
  (`src/olmo_eval/evals/tasks/ctc_suite/`), scoring the public HF dataset
  `PrasannSinghal/ctc-suite-eval`.
- **Evidence for one-off validations** — `debug/<topic>/`, each with its own README.
