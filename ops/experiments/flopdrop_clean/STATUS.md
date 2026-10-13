---
experiment: flopdrop_clean
title: INFRA — clean squashed port of landmark + ctc-setA-sft onto upstream 59a339f53 (recipes/ + research/ layout); v3.0.0 upgrade last
status: active
updated: 2026-10-12
branch: prasann/flopdrop_clean
launcher: recipes/README.md
records:
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/records/clean-repo-target-structure.md
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/records/clean-repo-migration-plan.md
results:
wandb_group:
next_step: Land the env_flags commit (b11ff1ff8, tmp/flopdrop-envflags) on prasann/flopdrop_clean, then path pass + central recipe flags, owed GPU validation, and only then the v3.0.0 upgrade — never push without asking.
---

# flopdrop_clean

## Question
Not an experiment: produce one clean, current branch for the FLOP-reduction research (token dropping,
FFN dropping, KV route / block skip, landmark/chunked attention) sitting on the canonical setA
train → olmo-eval pipeline, without the history/debug clutter of `prasann/landmark`.

## Decisions
- 2026-08-08 (Prasann): earlier clean-repo effort — long-lived branch `prasann/ctc` on upstream 064b172e5, `ctc/` package split at tokenization (records/clean-repo-*.md). Outcome: `prasann/ctc_public` (2026-08-20) and the public repo PrasannS/corpustaskcomplexity, squashed (2026-09-19). That line is finished; `prasann/ctc` is 0/0 with origin, last commit 40a5c60d1 (2026-10-01).
- 2026-10-12 (Prasann): fresh port with SQUASHED history, not merge-then-clean. Integration = real merge of origin/prasann/ctc-setA-sft into origin/prasann/landmark (`tmp/flopdrop-integration` @ fd49fc8a8), then a keep-list copied onto upstream 59a339f53 (the base every env/image uses).
- 2026-10-12 (Prasann): upgrade to origin/main v3.0.0 LAST (needs torch>=2.10 + fla 0.5.2 vs our torch 2.9.1 / fla 0.4.1; 16 core conflicts).
- 2026-10-12 (Prasann): layout `recipes/{data,train/setA,train/research_trainer,eval,env}`, `research/{token_routing,ffn_compute,kv_route,landmark,common}`, `ctc/` vendored (prasann/ctc @ 40a5c60d1); no results dirs, logs or .pt in git; learned-router .pt stay on local storage.
- 2026-10-12 (Prasann): carry ALL repo skills in `.claude/skills/` (run-evals, pull-evals, storage-audit, where-to-run, training-smoke-test) and rewrite their paths after the move.
- 2026-10-12 (agent): untracked orchestrators left behind unless recent/part of latest recipes (dropped hybridish, ctc_purechunk, ctc_vllm_validation/general; kept flexcompute_40x/_train, ds64 extras, pooled_kv/layer_probe).

## Current best numbers
Infra project — no metrics. Tests are the gate:

| arm | task / rung | metric | eval_size | ±SE | cot | source |
|---|---|---|---|---|---|---|
| integration merge (CPU) | pytest | 1143 pass; 11 pre-existing failures on landmark (kv_route_test ×3, landmark_compressive_ragged_decode ×8); gloo Ulysses CP hangs on CPU | — | — | — | memory:flopdrop-clean-branch (2026-10-12) |
| GPU triage on mooney | pytest | no merge regressions; inherited failures listed in memory | — | — | — | memory:flopdrop-clean-branch (2026-10-12) |

## Branch state (2026-10-12 21:57, from git)
`prasann/flopdrop_clean` @ 94abf522f, 14 commits on 59a339f53, local only (no `origin/prasann/flopdrop*`):
library (3d9b5a5c1), vendored ctc (f06393726), recipes/ (b74d7d633), research/ (e44a38915), records+skills+docs
(a14d01dfa), landmark test fix (633919f00), canonical prompt module / single eval harness / dedupe parity
(2e8393a5c, d8a6c78aa, cf7053b2d), block-skip compaction + KV-route + sparse-loss lm_head that were uncommitted on
landmark (686ea1488, = c5b495d39), converter move (2e496fecb), nested_ffn_moe docstrings (3c4e02983), variants
registry (3e9494f74, = 66f7f2f69 from tmp/flopdrop-variants), block-skip parity check (94abf522f).
Worktrees: `flopdrop_port` (the branch, clean), `flopdrop_lib` (tmp/flopdrop-envflags, untracked `.gpu_triage.sh`),
`flopdrop_variants` (tmp/flopdrop-variants), `flopdrop_clean` (detached at fd49fc8a8 = integration merge).

## Next step
1. Bring `b11ff1ff8 env_flags: one documented home for olmo_core runtime env toggles` (tmp/flopdrop-envflags) onto the branch — it is NOT an ancestor of prasann/flopdrop_clean yet.
2. Path pass (REPO_ROOT / env vars) and central recipe flags (`--tasks/--window/--variant`); dedupe step appears done (2e8393a5c/d8a6c78aa/cf7053b2d).
3. Owed GPU validation: GDN left-pad + chunked prefill (fused QKV), landmark chunked prefill + sparse decode bf16, multi-landmark generation, Ulysses CP on NCCL (cp4 needs 4 GPUs).
4. v3.0.0 upgrade. Push only when Prasann says so.

## Open issues / traps
- Committed `prasann/landmark` was INCONSISTENT: block_skip.py / kv_route.py / lm_head.py edits (Sep 8–14) + 3 tests existed only uncommitted in the MAIN checkout; captured as c5b495d39 → 686ea1488. Any later diff of landmark vs flopdrop_clean will show these as "new".
- `~/.claude/skills/*` symlink into the MAIN checkout's `.claude/skills` — they keep working off landmark until incorporation; skill bodies need path rewrites for the new layout.
- Public-release hygiene (from the 2026-09-19 release): before any push, check `git log --name-only --diff-filter=A -- records debug .claude` for internal files reachable in history — memory:ctc-public-repo-release.
- `convert_unified_to_document_landmark.py` needed a hand merge (setA summary emit + landmark gold-index metadata); `fix_marker_embeddings.py` taken from setA (norm-aware version) — re-check marker repair on any base built from this branch.
- Validation gate for ported modules is numeric parity (<1e-6), not import success — memory:clean-repo-migration.

## Verified vs unverified
- Verified 2026-10-12 via read-only git: commit list/count/timestamps on prasann/flopdrop_clean, env_flags not an ancestor, variants commit cherry-picked, no remote flopdrop branch, worktree states, research/ + recipes/ directory layout in flopdrop_port, prasann/ctc 0/0 with origin.
- Read only in memory (not re-run): test counts (1143 pass / 11 pre-existing failures), GPU triage outcome, v3.0.0 dependency requirements and 16-conflict count.
- Memory `flopdrop-clean-branch` says branch @ 633919f00 "6 commits"; git now shows 94abf522f / 14 commits (later the same evening) — memory is stale on state, not on decisions.
- Memory `clean-repo-migration` (2026-08-20) says the only gap is "5 unpushed commits on prasann/ctc"; git shows prasann/ctc 0/0 with origin — resolved.
- UNVERIFIED — confirm with Prasann: whether `flopdrop_clean` (detached at the integration merge) and `flopdrop_lib`/`flopdrop_variants` worktrees are still needed once env_flags lands; and whether the vendored `ctc/` should track prasann/ctc or corpustaskcomplexity main.
