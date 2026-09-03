## Summary (written 2026-09-02 18:10; grid 92/93 runs and 87/89 evals done, three outlier-320M tail items pending)

**Question.** At matched *training* FLOPs, does either method beat dense SFT on Qwen3.5-4B short-heavy 2k–32k mixes?

**Answer per task** (f1 = mean over rungs; PF = training PFLOPs at real example lengths; dense anchors at 4–10M tokens added today so every comparison is inside the measured range):

| task | best method arm | matched-compute verdict | FLOPs to the task target, method ÷ dense (Hill fit) |
|---|---|---|---|
| oolong | KV soft tokens, keep 1/6 (gold-blind) | **wins below ~500 PF**: 0.637 @134 PF vs dense 0.624 @243 PF; 0.652 @272 vs dense 0.661 @486; 0.665 @564 vs dense 0.702 @972 | 1.4× at f1 0.68 (fit is flat; direct comparison says ≈0.55–1.2×) |
| contradiction | KV soft tokens, keep gold + 1/3 | **parity to slight loss**: 0.762 @338 PF vs dense 0.772 @337; 0.861 @695 vs dense 0.924 @674; gold-blind collapses (gold pair must be real) | 1.21× at f1 0.85 (FFN stage 1: 1.28×, two-sided: 1.34×) |
| nq | KV soft tokens, gold-blind 1/3 | **dense wins by ~0.03 everywhere**: 0.832 @163 PF vs dense-8M 0.860 @190; 0.878 @509 vs dense-16M 0.878 @379 | 2.5× at f1 0.88 (FFN stage 1: 2.35×) |
| outlier | none | every KV arm dead (≤0.30 at 8k vs dense 0.51–0.89); FFN routing 0.28 vs 0.43 at 64M | — |

**Routed FFN** never reaches compute-optimality here: FFN is 57% of training FLOPs at these lengths, so the saving caps at ~0.70×, and every arm gives up more than that (losses concentrate on the 32k rung). Routing all layers collapses on 30–150-step runs. The two-sided budget lands exactly on target and is the recipe to keep; the L12+ ladder at 0.01 (stage 1) is the better of the two at ≥48M budgets. Deployed inference cost is the one thing FFN routing buys that KV does not.

**KV soft tokens** are task-shaped: they win where the answer is an aggregate over many documents (oolong), tie where two specific documents must survive but usually do (contradiction with gold forced real), trail slightly where one passage must be read verbatim (nq), and fail where every document must be compared (outlier). Forcing gold documents real leaks the answer on id-answer tasks (nq 1/6: 0.603 forced vs 0.728 blind) but is required on contradiction (blind 1/6: 0.053).

**What changed today** (all in §10 of the plan record): a random-router init on every meta-built FFN run (fixed, all FFN numbers re-run), Qwen3 token ids hardcoded in the marker-aware evaluator, a dead rung-file override, an unset-variable death in the eval script, `beaker experiment cancel` being a no-op, marker-data builds that reported success without input, Hub rate limits killing eval starts (tokenizers now staged on weka). Every number above is post-fix.

**Not done / next.** Breadth-matched KV (absolute 128–512 documents, the Qwen3-4B v25 recipe) on 3.5 would test whether contradiction and nq reach parity the way the Qwen3-4B runs did; the all-layer FFN ceiling should be re-measured on the fixed init (v12 rerun, layer-0 exempt from the budget); importance-permuting FFN units before nesting is free and untried.
