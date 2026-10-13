---
experiment: gdn_int32_boundary
title: Debug — r256k Qwen3.5 evals crash past 262,144 tokens (Triton illegal memory access); GDN int32 hypothesis + chunked-prefill workaround
status: done
updated: 2026-10-12
branch: prasann/landmark
launcher: debug/gdn_int32_boundary/run_beaker.sh
records:
  - records/ctc-fast-suite-eval.md
results:
  - debug/gdn_int32_boundary/probe.py
  - debug/gdn_int32_boundary/chunk4_diag.py
wandb_group:
next_step: None required — r256k runs with chunked prefill (131,072-token chunks) + YaRN x2 by default; optional: find the real cause of the one-shot >262,144-token crash.
---

# gdn_int32_boundary

## Question
Every YaRN r256k eval (chat + doc-marker prompts run to ~272k tokens) died with a Triton "illegal memory
access" on 7 different nodes (noted 2026-09-29). Is it an int32 overflow in the GDN chunk kernel — whose
per-chunk state holds ceil(T/64)·H·K·V = ceil(T/64)·32·128·128 values, exactly 2^31 at T = 262,144 — and
does splitting the prefill into chunks avoid it?

## Decisions
- 2026-09-29 (agent): port Amanda's `GenerationConfig.prefill_chunk_size` (from 186cce0af on prasann/ctc-setA-sft) onto the eval branch, extended to the fused-QKV GatedDeltaNet path (commit 287df8eff, branch prasann/landmark-chunked-prefill); add `probe.py` testing one-shot T = 262,144−64 / 262,144 / +64 / 272,000 vs a two-chunk 272,000 call, each case in its own process.
- 2026-10-02 (agent): chunked-prefill equivalence test failed only at chunk_size=4 (5 and 16 passed) → `chunk_matrix.py` (chunk 3–7,16 × FLA length-autotune freeze on/off) and `run_chunk4_beaker.sh` (port vs Amanda's original branch).
- 2026-10-03 (agent): `chunk4_diag.py` showed chunk 4 splits from one-shot at generated step 7 where the one-shot top-2 logit margin is 0.0039, with every logit within 0.0039 before that (same deltas as passing sizes) → bf16 greedy near-tie, not a state bug. Test now accepts a split only below `NEAR_TIE_MARGIN` with logits matching up to it (b8341d734).
- 2026-10-12 (agent/Prasann): prasann/landmark-chunked-prefill merged into prasann/landmark (d28e458b2); olmo-eval launcher (8e63b917) runs r256k cells with `--yarn-rungs r256k` by default: YaRN x2, max_model_len 524,288, prefill chunks 131,072. `--fit-window` (shave 0–3% distractor docs; not oolong) is the alternative and agreed within noise on setA.

## Current best numbers
Debug investigation — no task metrics. Outcomes:

| arm | task / rung | metric | eval_size | ±SE | cot | source |
|---|---|---|---|---|---|---|
| chunked prefill chunk=4 vs one-shot (bf16) | toy hybrid GDN, 12-tok prompt | first divergence at gen step 7, top-2 margin 0.0039 = near-tie | — | — | — | commit b8341d734 message (diag output itself not in repo) |
| GDN chunk kernel, one-shot T>262,144 | probe.py | int32-overflow hypothesis REFUTED (crash is not in the GDN chunk kernel) | — | — | — | memory:qwen35-4b-chat-template-eval-mismatch (probe log not in repo) |

## Next step
None queued. The workaround (chunked prefill) is the default path. If the root cause matters (e.g. for training
or one-shot prefill past 262,144), the crash location is still unidentified — only "Triton illegal access, not the
GDN chunk kernel" is known. GPU validation of chunked prefill (GDN left-pad + fused QKV) is owed on the new
`prasann/flopdrop_clean` branch (tracked on the flopdrop_clean page).

## Open issues / traps
- One-shot forwards over 262,144 tokens still crash; anything that bypasses `prefill_chunk_size` (old launchers, runs before olmo-eval 8e63b917 without explicit flags) has either crashed or LEFT-TRUNCATED r256k rows (43–100% of rows lost the instruction → fake 256k collapse). Treat pre-2026-10-12 r256k numbers as suspect.
- Even with YaRN, some 256k rows of hpqa / outlier / qdmatch_nq / rerank exceed 262,144 tokens and are left-truncated (records/ctc-fast-suite-eval.md).
- `contra_fever` r256k p90 is 264,294 tokens — past Qwen3.5's 262,144 position ceiling; needs YaRN.
- Separate, easily confused cause: node jupiter-cs-aus-116 gave CUDA illegal-memory-access on EVERY long cell 2026-09-28 — steer with `--hostname`.
- `probe.py`'s docstring still states the int32 hypothesis as if open; it was not updated with the refutation.
- Chunked-prefill equivalence at bf16 can split on near-ties; do not read a late greedy divergence as a state bug without checking the logit margin.

## Verified vs unverified
- Verified 2026-10-12 from git: scripts and their purpose (all 7 files read), commit chain 287df8eff (09-29) → dd0828613 → 4bed64628 / 33fba4d1c / 89f11ed21 (10-02) → acf33ef5e / b8341d734 (10-03) → merge d28e458b2 into landmark (10-12); b8341d734's near-tie numbers (from its message); records/ctc-fast-suite-eval.md r256k default text.
- UNVERIFIED — confirm with Prasann: the probe's actual per-case results (which T values crashed, whether the two-chunk call passed). The only statement of the outcome is memory `qwen35-4b-chat-template-eval-mismatch`: "One-shot >262,144-token forwards crash (Triton illegal access; NOT the GDN chunk kernel -- probe refuted that)". No Beaker log or result file for `gdn-int32-boundary-probe`, `chunked-prefill-matrix` or `chunked-prefill-chunk4(-diag)` is in the repo, and no ledger row records their job ids.
- The "7 nodes" count comes only from the 287df8eff commit message / probe docstring.
