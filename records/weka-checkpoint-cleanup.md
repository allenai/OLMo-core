# Weka checkpoint cleanup — 21.4 TB → 7.2 TB (2026-07-28)

Audit and cleanup of `/weka/oe-training-default/ai2-llm/checkpoints/prasanns/`, which had grown to
**21.37 TB** while the shared `oe-training-default` bucket sat at **98% full** (45 T free of 1.8 P).

**Result: 14.26 TB reclaimed, 21.37 TB → 7.2 TB, bucket free space 45 T → 59 T.** No final
checkpoint was lost — verified afterwards, see §5.

## 1. Why it was so large: the weights were stored twice

Every completed `ctc_suite` run held its weights in two places:

| Path | Size @4B | Contents | Who reads it |
|---|---|---|---|
| `<run>/stepN/` | ~54 GB each | model + optimizer + train state | nothing, after the run ends |
| `<run>/model_and_optim/` | ~18 GB | **model only**, post-fit save | the eval path |

74 runs carried up to 4 `stepN/` dirs apiece. The cause is documented in our own code —
`src/scripts/train/memexpress/ctc_suite/train_ctc_suite.py:834-840` warns that omitting
`--save-interval` silently activates the framework default (`save_interval=250`,
`max_checkpoints=3`), writing *"step*/ dirs that nothing reads, since eval loads the model-only
`model_and_optim`"*. The 2026-07-19/20 fan-out predates that fix; runs launched after it are clean.

The model-only save is written post-fit by `train_ctc_suite.py:884-888`, which passes only
`train_module.model` to `save_model_and_optim_state()` — hence no optimizer state.

## 2. What was deleted

| Phase | Scope | Rule | Freed |
|---|---|---|---|
| 1 | `ctc_suite/ckpts/` — 74 runs | delete **all** `stepN/`; the model-only save is the surviving artifact | **11.22 TB** (210 dirs) |
| 2 | 49 top-level `q4b-*`/`q06b-*` run dirs | these have **no** model-only save, so keep the highest `stepN/` and delete earlier ones | **3.04 TB** (64 dirs) |

48 further top-level runs had exactly one `stepN` and were not touched. 38 `ctc_suite` dirs were
dead/failed launches holding ~0 bytes and were skipped (they have no weights at all — do not
mistake them for damage from this cleanup).

**Not done:** stripping optimizer state from the 97 surviving top-level finals (~3.1 TB more). It
needs a load-and-resave GPU job per run and is the only irreversible step. The
`outlier_datascale_sweep` runs still have owed evals, so they should not be stripped yet.

## 3. Tooling — `debug/weka_cleanup/`

Deliberately **two stages, with no pattern expansion in the delete step** (see that directory's
README for usage). The planner discovers targets and writes a manifest of exact absolute paths;
the applier reads those literal paths and re-derives every safety property from the filesystem, so
a corrupted or hand-edited manifest still cannot cause an unintended delete. Manifests are archived
to `s3://ai2-llm/checkpoints/prasanns/_inventory/manifest_<phase>.txt`, so what was removed stays
auditable after the fact.

Weka is not mounted at Berkeley, so all of it runs as 0-GPU gantry jobs.

## 4. Two traps worth remembering

**The freshness guard tripped on its own writes.** The in-flight check (`skip runs modified in the
last 90 min`) was evaluated inline per path — but deleting a `stepN` updates its *parent's* mtime,
so after the first delete in a run every remaining path was refused. The first Phase 1 apply
deleted 74 of 210 paths and refused 136. Fixed with a pre-pass that computes each parent's verdict
once, before any deletion. It failed in the safe direction, but it silently halves the work.
Resuming an interrupted apply then needs a lowered `FRESH_MIN`, because the tooling itself has just
touched those dirs.

**The "weights survive" invariant differs per family.** `modelonly` mode requires a non-empty
`model_and_optim/.metadata`; `keepfinal` mode requires that a strictly higher-numbered `stepN`
survives on disk *and is not itself in the manifest*. Applying the first check to the second family
refuses everything (those runs have no `model_and_optim/` at all). The mode must match how the
manifest was planned.

## 5. Verification performed

- **0** top-level runs left with more than one `stepN`; **97 loadable finals, 0 broken** (each has
  a non-empty `model_and_optim/.metadata`).
- For Phase 1, the 74 runs deleted from were cross-checked against the 38 that lack a final model:
  **zero overlap** — i.e. nothing we touched lost its weights.
- For Phase 2, the manifest's highest step per run was checked against the pre-cleanup inventory:
  **0 runs** would have had their final checkpoint deleted.
- Keep-list intact: `ctc_suite/bases` 35 G, `ctc_suite/shards` 44 G, `xlong5_2k256k_qwen35` 23 G,
  `_eval_bundle_eval500_v2` 3.6 G, `cr_suite_data` 22 G.

## 6. Related change

`src/corpus_reasoning/train/export_olmo_to_hf.py` — `latest_step_dir()` used to `SystemExit` when
no `step<N>/` existed. It now falls back to the run dir itself when that holds a
`model_and_optim/`, which is required for exporting any `ctc_suite` run after Phase 1. The main CTC
eval path (`debug/ctc_vllm_validation/sweep_task_vllm.sh`, `pipeline_4b.sh`, `sweep_contra_vllm.sh`)
already passed `--ckpt` explicitly and never used the resolver.

## 7. Do not delete

`ctc_suite/bases/*` and `ctc_olmo3/bases/*-fixmark` (marker-audited bases every run derives from —
re-deriving them is the failure mode in `n100-chunked-marker-position-bug.md`);
`xlong5_2k256k_qwen35/*` (weka is the **only** copy — the local one was `rm -rf`'d);
`ctc_suite/shards/contra_mix_{qwen3,qwen35}_10k_128k` (feeds the open 128k progressive-extension
eval); `_eval_bundle*`, `cr_suite_data`, tokenizer dirs (shared eval infra);
`ctc_suite/ckpts/ctc-olmo3-7b-hpqaret-*` (retrain still owed). These are encoded in the tooling's
default `KEEP_REGEX`.
