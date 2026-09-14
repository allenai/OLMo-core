# Recover the six preempted SFT-only EMO-off evaluations

Based on the exact original worker/controller pin89e7dcb7. Model weights,
inference kernels, sampling, prompts, scorers and qualification tolerances are
unchanged. Only Math500/Alpaca tasks canceled by SYSTEM_PREEMPTION are eligible.

- Controller keeps original successful tasks and original immutable submission
  intents. A separate `-r2-preemption` experiment is submitted once per preempted
  task and recorded with `replaces` in the existing collector status.
- Replacements use urgent priority, six-hour minimum runtime and Beaker-native
  auto-resume. Runtime/qualification failures and manual cancellations do not
  cause retries. The original 12-hour task timeout is unchanged.
- The wrapper locks its original output directory and verifies the complete
  saved recipe, model conversion and inference qualification before reuse.
- Completed raw responses are replayed into the same native scoring pipeline;
  native ID, question, reference, model and sampling must match. Only missing
  requests reach the original inference path. In-flight unsaved generations are
  regenerated. Restarting stochastic inference need not reproduce the exact
  missing responses of an uninterrupted run.
- Existing response records must compare exactly and are never overwritten.
  Newly completed response records are published atomically. A completed suite
  is a no-op on restart only after checking its receipt and result hashes.
- No checkpoints, original recipes or saved generations are deleted. Scope is
  this campaign only; other running SFT campaigns are unchanged.

The resume hook is loaded in spawned evaluator workers. It does not modify
vLLM, model kernels, task definitions or scoring algorithms. Unit tests exercise
cached/missing dispatch, duplicate-record equality, idempotence, and rejection
of mismatched model/sampling/question/response. CPU-only validation builds all
six original recipes in the actual eval runtime and stops before inference.
