# olmo-ddp port — conflict resolutions (for review)

Running log of places where core and `origin/olmo-ddp` conflicted and I took **olmo-ddp's** version,
dropping/changing a core-side cleanup or guard. Grouped by PR. Review whether any core cleanup is
worth re-applying on top.

## PR-A: EP/MoE compute + kernels

Took olmo-ddp wholesale for the EP compute layer. Notable overrides / deviations:

- **comm.py**: took olmo-ddp's version → **dropped core's `_check_input_grads`** grad-validation
  guard (core-only cleanup; olmo-ddp's autograd `backward`s return bare grad tuples).
- **ep_config.py**: took olmo-ddp's version → **lifted the `validate()` rejections** I'd added
  (#61) for `rowwise_wave` / `deepep_v2` / `schedule='tbo'`. These now validate OK but are NOT yet
  dispatched: block dispatch is PR-C, train wiring PR-D, and the wave/deepep modules are deferred to
  PR-B. **Interim gap**: a config selecting those validates but won't run end-to-end until the later
  PRs land. (Whole stack is reviewed together before merge to main, so no mid-stack breakage.)
- **_nvtx_colors.py**: olmo-ddp *removed* this module; **kept core's copy** because core's
  PR-B/PR-C files (routed_experts/router/shared_experts/model) still import it. Reconcile when those
  land.
- **Deferred to PR-B**: experimental backends `ep_no_sync_rowwise_wave.py`, `ep_deepep_v2.py` and
  their exclusive kernels (`grouped_mm_row_offset`, `swiglu`) — they import `ExpertActivation` from
  the experts refactor (routed_experts.py), which lands in PR-B.
- **Skipped**: `ep_no_sync_tma_ibgda.py` — olmo-ddp abandoned/removed it ("Remove abandoned TMA
  IBGDA rowwise backend"); no core refs.
- **mypy**: added `# type: ignore` at 12 sites in olmo-ddp code that trip core's stricter mypy
  (fp8.py callable-assignment ×4 on the grouped_mm wgrad variants; union-attr ×8 for
  Optional-lease/config access that's non-None by construction). Behavior unchanged.

### Round 2 — re-applied core cleanups + Codex fixes (per "keep olmo-ddp features + core cleanup")
- **_check_input_grads**: re-added core's grad-tuple guard to comm.py and wrapped all 7 autograd
  `backward`s (had been dropped when taking olmo-ddp's comm.py). It caught a real olmo-ddp bug:
  `_CombineVDevAutograd.backward` returned 10 gradients for a 9-input forward — fixed to 9.
- **nvtx optional (core convention)**: the pulled EP modules + symm_mem loader used top-level
  `import nvtx` (breaks base installs without the `profiling` extra); switched all to
  `from olmo_core._nvtx import nvtx` (real nvtx or no-op), keeping olmo-ddp's `@nvtx.annotate` usage.
- **_nvtx_colors.py**: kept core's centralized module (olmo-ddp removed it); the pulled EP files
  don't annotate with colors, so no change needed there.
- **nn/output_discard_checkpoint.py**: reverted to core's version (I'd overwritten it with
  olmo-ddp's, breaking core's test + benchmark_odc.py which use `_SHARED_STORAGE_LOADER`). This
  also resolves the Codex P2 about the olmo-ddp fallback's storage-identity bug.
- **ep_config.validate()**: re-added the loud rejections for not-yet-wired paths
  (`rowwise_wave`, `deepep_v2`, `schedule='tbo'`) — lift each in the PR that wires it.
- **CMakeLists.txt**: updated to the split `olmo_symm_mem_*` sources (was still referencing the
  deleted monolithic files → cmake build would fail).

## PR-B: experts (routed/router/shared)

- Scoped down from "experts + model variants": **model variants** (nemotron/qwen/gpt_oss) deferred —
  need MoERouterGatingFunction.topk_softmax [added], LayerNormType.nemotron_rms, RoPEConfig.rotary_dim,
  YaRNRoPEScalingConfig.truncate, and OLMoDDPModelConfig block-config typing. **Experimental backends**
  (deepep_v2/rowwise_wave) deferred to #53 (need PR-C block dispatch; DeepEP external-API).
- **shared_experts.py**: re-applied core's `reset_parameters()` (trunc_normal init on build) — olmo-ddp's
  version left w_up_gate/w_down as `torch.empty`, so standalone-built modules produced NaN. (core cleanup)
- **router**: added `topk_softmax` to MoERouterGatingFunction (nn/moe/router.py); dropped core's
  `test_router_use_quant_scores_*` test (the `use_quant_scores` router option was removed upstream); fixed
  dead-branch `get_top_k().indices` -> `[1]`, removed duplicate `record_routing_batch_size` field.
- **Tests**: followed olmo-ddp's flatten (v2 tests -> src/test/nn/moe/*_v2_test.py); kept + adapted core's
  router/shared_experts unit tests to the new API; took olmo-ddp's routed_experts_v2_test / fp8 bench /
  router_output_discard_checkpoint_test. `SimpleNamespace` fakes -> `# type: ignore[arg-type]`.

### Round 2 — CI routing + Codex fixes
- **CI routing**: flattening `routed_experts_v2_test` from `moe/v2/` -> `moe/` moved it off the torch-2.10
  "Test MoE kernels" job onto the plain "Test MoE (GPU)" job, whose image lacks torch-native
  `F.grouped_mm` + TE. It was also mis-marked `@requires_grouped_gemm` (the grouped_gemm *lib*, present
  there) so it ran and failed. Re-marked `@requires_torch_grouped_mm` (skips on the plain image) and added
  the test to the 2.10 job's pytest paths in `.github/workflows/main.yml`.
- **routed_experts.py (core cleanup, Codex P1)**: re-applied a `reset_parameters()` (trunc_normal std=0.02
  on w_up_gate/w_down, zeros on b_up_gate/b_down) — olmo-ddp left the params as `torch.empty`, so
  standalone build + `bias=True` biases (which `init_moe_v2` never touches) started from arbitrary memory.
- **nn/hf/config.py (Codex P2)**: the olmo3moe HF exporter guarded on `router.use_quant_scores`, which the
  ported router no longer defines (olmo-ddp removed the option) -> `AttributeError` on export. Dropped the
  dead reference (kept the `bias_gamma` guard).

### Round 3 — Codex (richer router modifiers)
- **router.py (Codex P2)**: the ported `topk_softmax` gating path selects top-k directly from all logits,
  bypassing the `n_group`/`topk_group` group masking (which only runs in `get_top_k`, the softmax/sigmoid
  path). So grouped-routing configs were silently ignored under `topk_softmax`. Reject that combination
  loudly in the router constructor (`OLMoConfigurationError`) rather than route outside the allowed group.
- **nn/hf/config.py (Codex P2)**: my round-2 guard only rejected `bias_gamma`, but the richer router adds
  `score_correction_bias`, `topk_softmax` gating, and grouped routing — none representable by the HF
  Olmo3Moe router (softmax/sigmoid only, no score-bias/group path). Expanded the export guard to reject
  all of these modifiers so a bad export fails loudly instead of diverging / crashing on the HF forward.

### Round 4 — CI env skip + more Codex guards
- **CI (env, not a source bug)**: `test_routed_experts_forward_row_offset_writes_canonical_window` JIT-builds
  the CUTLASS-backed `grouped_mm_row_offset` CUDA ext, but the torch-2.10 RMA image ships no CUTLASS headers
  (`OLMO_CUTLASS_ROOT` unset) so the build fails. Added `cutlass_headers_available()` (cheap file-existence
  probe) + a `requires_grouped_mm_row_offset` test marker and applied it. **Coverage note**: `forward_row_offset`
  is therefore *skipped* in CI until an image with CUTLASS headers is available (tracks #58); the other
  routed-experts GPU tests run on the 2.10 job.
- **router.py scores-only (Codex P2)**: `topk_softmax` on the `scores_only` path (shared-expert mixing) returned
  the dense softmax over all experts, ignoring `top_k`. Reject that combination (`NotImplementedError`) rather
  than mix in non-selected experts. Also removed a dead `HACK` debug comment block in `forward`.
- **nn/hf/config.py (Codex P2)**: also reject biased router / biased routed experts / non-SwiGLU expert
  activation — the HF olmo3moe linears are bias-free and the converter assumes contiguous SwiGLU up/gate,
  so those configs would drop learned state or export a layout HF can't represent.

### Round 5 — Codex (bias init, grouped top_k, more HF guards, TBO down bias)
- **transformer/init.py (Codex P2)**: `init_moe_v2` initialized only weights, so on the meta/`to_empty`
  DDP path the newly-supported biases (router `bias`, expert `b_up_gate`/`b_down`) started from arbitrary
  memory (the constructor `reset_parameters` runs before `to_empty` wipes them). Zero-init each bias in
  `init_moe_v2` when present.
- **router.py get_top_k (Codex P2)**: grouped routing with `top_k > topk_group * experts_per_group` let
  `torch.topk` return masked (`-inf`) experts whose weights are then gathered from the *unmasked* scores,
  dispatching out-of-group. Reject that config at build time.
- **nn/hf/config.py (Codex P2)**: also reject the weight-rescaling modifiers `expert_weight_scale`,
  `original_top_k`, `restore_weight_scale` — core multiplies the selected expert weights in the router
  forward but the HF Olmo3Moe router has no equivalent, so exports would run at a different scale.
- **routed_experts.py (Codex P2)**: on the rowwise TBO no-sync path the down output aliases the caller's
  symmetric combine buffer (`out=down_proj_out`), but the `b_down` bias add allocated a new tensor, so the
  combine kernel read the un-biased buffer and dropped the learned down bias. Add the bias in-place when
  `down_proj_out` is provided (out-of-place elsewhere to stay autograd-safe).

### Round 6 — Codex (expert bias on rowwise paths)
- **routed_experts.py (Codex P2)**: expert biases are only partially wired for the rowwise EP paths — the
  up_gate bias `repeat_interleave(output_size=capacity)` crashes when capacity padding exceeds the valid
  routed rows, and the down bias aliases a per-expert capacity-slotted combine buffer. Rather than expand
  biases correctly across the padded/aliased layout (per-expert slots, not a contiguous prefix), reject
  routed-expert biases on the rowwise path (`down_proj_out`/`row_weights`), generalizing the existing
  `row_weights require bias-free routed experts` guard. Superseded the round-5 in-place down-bias tweak
  (now unreachable on that path) and removed the now-redundant row_weights+b_down check. Bias-free experts
  (the default) and biased experts on the non-rowwise paths are unaffected.

### Round 7 — Codex (HF export guard tuning)
- **nn/hf/config.py (Codex P2, regression fix)**: my round-5 guard over-rejected `original_top_k` and
  `restore_weight_scale` — the HF exporter DOES support them (`_get_olmo3moe_config` maps them to
  `original_num_experts_per_tok` / `restore_weight_scale`, and `Olmo3MoeRouter.forward` applies both).
  Dropped them from the unsupported set; kept `expert_weight_scale` (HF has no equivalent multiply).
- **nn/hf/config.py (Codex P2)**: reject a non-default `sigmoid_stability_epsilon` under sigmoid gating —
  the HF router hard-codes `1e-7`, so any other value routes differently after export.
- **nn/hf/config.py (Codex P2)**: reject non-SwiGLU *shared* experts (the guard already covered routed
  experts) — the HF converter reshapes `shared_experts.w_up_gate` as SwiGLU up/gate.

### Round 8 — Codex (more unsupported-combo guards) + flaky integration test
- **CI (flaky, not a source bug)**: `test_train_small_model_gpu` failed with a NCCL ALLREDUCE
  collective timeout / `ProcessExitedException` — a distributed-training integration test unrelated to
  the round-7 `hf/config.py` export-guard change. Re-ran the job.
- **router.py (Codex P2)**: `random_expert_assignment=True` + `gating_function='topk_softmax'` — the
  randomization replaces `scores`, but the topk_softmax branch selects from raw `logits`, so routing
  stayed deterministic. Reject the combination in the router constructor.
- **fp8.py (Codex P2)**: the rowwise-FP8 shared-expert helpers reshape the projection as SwiGLU
  unconditionally, so `SharedExperts(activation=relu2)` + `rowwise_fp8` would misfire. Guard both
  helpers (`_require_swiglu_shared_experts`) to reject non-SwiGLU shared experts on the FP8 path.

### Round 9 — Codex (partial grouped-routing config)
- **router.py (Codex P2)**: a partially-specified grouped-routing config (only `n_group` or only
  `topk_group` set) silently fell back to global top-k — every group-masking/validation branch checks
  both knobs, so a half-set config ran with the wrong policy and no error. Require both-or-neither in the
  router constructor.

## PR-C: OLMoDDP block + model (wholesale)

Took `origin/olmo-ddp` wholesale for `nn/ddp/block.py` + `nn/ddp/model.py`, then reconciled:
- **te/cpu_offload.py**: pulled forward (user decision). Only referenced by commented-out (disabled)
  offload code in model.py, so currently unwired — the real wiring lands with PR-F/G. Dropped the
  unused `cpu_offload_simple_varlen.py`.
- **Experimental backends**: lazy-imported `ep_no_sync_rowwise_wave` / `ep_deepep_v2` at their dispatch
  sites (deferred to #53; `ExpertParallelConfig` rejects the paths meanwhile).
- **nvtx**: `import nvtx` -> `from olmo_core._nvtx import nvtx` (model.py); block.py's was dead, dropped.
- **Cruft removed**: dead selective-checkpoint machinery (`policy_fn`/`should_save_ops`), duplicate
  `apply_ep` stub (referenced non-existent `apply_epdp`; the multi-arg impl shadowed it at runtime),
  scattered mid-file imports, unused locals.
- **Norm-split (kept core design)**: olmo-ddp put the attention/feed-forward norm split at the *config*
  level; core keeps a single `layer_norm` config field and splits in the constructor ([[moe-v2-block-config-single-layer-norm]]).
  Restored core's `build()` (maps `layer_norm` -> both norm kwargs) and `num_params`/`num_active_params`.
- **Guarded CUDA events (core cleanup)**: olmo-ddp created `torch.cuda.Event()` unconditionally (crashes
  constructing on CPU) with `...1`-suffixed names; restored core's guarded init (`None` unless
  `torch.cuda.is_available()`, via `install_cuda_events()`) and renamed `...1` -> `..._tbo1` to match the
  names the already-merged PR-A EP modules use.
- **FLOPs (core cleanup)**: olmo-ddp's block flops called a config-level `AttentionConfig.flops_per_seq`
  that core lacks; routed attention through core's `build().num_flops_per_token`. Re-added the block-level
  `num_flops_per_token` override core had (the base method is abstract).
### transformer/block.py + model.py reconcile — take core's version (no change)
The shared `nn/transformer/block.py` + `model.py` are intentionally left unchanged: olmo-ddp's diff to
them is entirely things core rejects or defers, and the OLMoDDP block/model (in `nn/ddp/`) are
self-contained (they don't reference any of these shared additions — verified: 0 OLMoDDP refs in the
block.py diff; the overlap logic lives in `nn/ddp/block.py`'s own `get_dense_stream`). Breakdown:
- **shared-block norm split (#687)**: olmo-ddp splits `layer_norm` -> `attention_norm`/`feed_forward_norm`
  on the shared `TransformerBlock`/`PeriNorm`/`MoE*` blocks. Core deliberately keeps single `layer_norm`
  ([[moe-v2-block-config-single-layer-norm]]); taking it would also cascade into `transformer/config.py`.
  **Rejected.**
- **`tie_word_embeddings` removal**: olmo-ddp drops weight tying; core keeps it (used by other models).
  **Kept core.**
- **v1 `MoEHybrid` dense/sparse overlap**: benefits the old v1 MoE blocks, not the OLMoDDP path. **Skipped.**
- **CP precursor**: added `Transformer.prepare_cp_sequence_inputs` (pulled forward) — it uses core's
  existing `_cp_load_balancer.batch_shard`, so it's a clean self-contained addition. Full CP train wiring
  is PR-D.

### Deferred
- `nn/transformer/flops.py` (only imported by `transformer/config.py`; main fn is a dead deprecated-raise)
  -> lands with the config reconcile.
- `v2_context_parallel_test.py` -> PR-D (needs the MoE train module).

## PR-D: DDP train module + pipeline stack (IN PROGRESS)

Wholesale-ported olmo-ddp's DDP/pipeline mechanics; imports clean (nvtx reconciled). Shared-infra
reconciliation still pending.

**Ported wholesale (+ nvtx reconciled to `from olmo_core._nvtx import nvtx`):**
- `ddp_train_module.py` (OLMoDDPTrainModule: CP integration, P2P/RMA executor wiring, per-layer grad-norm
  diagnostics, reduce-scatter grads).
- `pipeline/p2p_executor.py` (NEW — PipelineP2PExecutor / TorchDist / NCCLRMA + `build_p2p_executor`).
- `pipeline/p2p_transport.py` (RMA slot-depth, ACK channel / `nccl_rma_ack`, single-node guard).
- `pipeline/pipeline_schedule.py` (1F1B-V via executor factory, CP token kwargs, env-var schedule plot).
- `pipeline/pipeline_stage.py`, `pipeline/helpers.py`, `pipeline/gpu_activation_offload.py`.
- `moe_train_module.py` (NEW — thin facade: `MoEV2TransformerTrainModule = OLMoDDPTrainModule`).

**Remaining reconciliation (preserve core features; add olmo-ddp DDP/CP/RMA additions):**
- `config.py`: add `reset_optimizer_states_on_resume` + `MoEV2TransformerTrainModuleConfig` alias + match
  `build()` to olmo-ddp's `OLMoDDPTrainModule.__init__` contract; KEEP core's `determinism_check`,
  `save_schedule_plot`/`schedule_plot_dir`, docstrings, schedule validation, tied-embedding+PP check, `@beta`.
- `pipeline_train_module.py`: KEEP core's `eval_only` support (olmo-ddp dropped it); take the `loss_fn`
  plumbing + drop of schedule-plot params.
- `common.py`: KEEP core's legacy-DDP-on-TransformerTrainModule path + `prepare_experts_for_ddp` (olmo-ddp
  disables it); take the `determinism_check`-forwarding reconcile.
- `distributed/parallel/pipeline_parallel.py`: add `PipelineP2PBackend.nccl_rma_ack`; KEEP core's
  TYPE_CHECKING/lazy-matplotlib import hygiene + save_plot/plot_dir params.
- `transformer/__init__.py` + `train_module/__init__.py`: export the MoEV2 aliases.
- Tests: `context_parallel_test.py` (standalone CP prims), `v2_context_parallel_test.py` (needs train
  module), `pipeline_rma_*` smokes/tests, `pipeline_schedule_test.py`.

### PR-D reconciliation (mypy + FLOPs + conventions)
- **FLOPs (per moe_rewrite_part2.md decision)**: olmo-ddp's DDP train module computed throughput FLOPs
  from `model.config.num_flops_per_token` (the config-level FLOPs system that was DROPPED as
  not-upstreamable — duplicates main's `lm_head` accounting). Restored moe-v2-core's model-level path:
  capture `self._full_model_num_flops_per_token = model.num_flops_per_token` before the PP split (#680
  capture-before-split) and query that; core's `Transformer.num_flops_per_token` = Σ block + lm_head, with
  the v2 block implementing the MoE formula (PR-C). Did **not** port the config-level `num_flops_per_token`
  / block-config `flops_per_token` / `flops.py`.
- **config.py**: added `reset_optimizer_states_on_resume` / `reduce_scatter_grads` fields +
  `MoEV2TransformerTrainModuleConfig` alias; kept core's fields (determinism_check, save_schedule_plot, @beta).
- **mypy**: `helpers.normalize_model_output_as_tuple` return `tuple[Any]` -> `tuple[Any, ...]` (olmo-ddp
  wrongly narrowed it); `send_stream_context` -> `AbstractContextManager[None]`; `isinstance(x, Callable)`
  -> `callable(x)`; `_resolve_model_checkpoint_key` param `Sequence[str]` -> `Collection[str]`; annotations
  for kwargs/args_split/model_parts/ddp_model_parts/rowwise_slots; `num_flops_per_token` kept
  `@lru_cache  # type: ignore[override]`; a few `type: ignore` on module-attr / bare `return None`.
- **conventions**: removed dead `debug_mem_*` memory snapshots, a dead `total_ops`, a dead `reload_event`
  + commented-wait cruft, an autocast-less `ExitStack` wrapper (-> plain `yield`); `progress_callback`
  lambda -> `def`; de-duplicated `DataParallelType` import + moved a mid-file checkpoint-metadata import
  to the top; nvtx -> optional `_nvtx` across the 5 wholesale files.

### PR-D review round 1 (head ceada120d, rebased onto merged PR-C #764)
Four comments from `final_review.md`:
- **P0 — checkpoints broke through the standard API**: `OLMoDDPTrainModule` raises from
  `state_dict()`/`state_dict_to_load()`/`load_state_dict()`, but `Checkpointer` still called them, so every
  trainer save/resume/eval-load failed. Forward-ported olmo-ddp's `Checkpointer` dispatch: `save()`/`load()`
  route to `save_state_dict_direct()`/`load_state_dict_direct()` when present; `save_async()` rejects the
  direct path (no async support). This `checkpoint.py` change lands with PR-D (coupled — PR-D is dead without it).
- **P1 — resume reset flag never read**: forward-ported olmo-ddp's `Trainer.load_checkpoint()` /
  `maybe_load_checkpoint()` `reset_optimizer_states_on_load` param + the derivation
  `if ...is None and load_trainer_state is True: = getattr(train_module, "reset_optimizer_states_on_resume")`,
  threaded through `Checkpointer.load()` -> `load_state_dict_direct()`. Non-resume `reset_optimizer_states_on_load`
  preserved.
- **P1 — MultiGroupDDP per-microbatch reduce**: MultiGroupDDP accumulates grads in-place into flat bucket
  views (reduced once per accumulation window), so `only_allreduce_last_microbatch=False` (all-reduce every
  backward with a single post-loop `finalize_grad_reduce()`) leaves grads unreduced/corrupted. Per-microbatch
  finalize would double-reduce earlier microbatches, so chose the review's second option: reject the flag at
  `__init__` and always `no_sync()` non-final microbatches.
- **P2 — double rowwise-FP8 cache rebuild**: `OLMoDDPOptimizer.step()` ->
  `_copy_main_params_to_model_params()` already rebuilds every rowwise-FP8 cache; the post-step
  `model.refresh_rowwise_fp8_cache()` repeated it. Kept the block **commented out** (per user) with the reason.
- **Out-of-band blocker fix**: `__init__` guarded `isinstance(model.config, OLMoDDPModelConfig)`, but the model
  build never attaches `.config` — this blocked *all* construction. The reviewer's sandbox couldn't run the
  distributed construction tests, so it went unnoticed. Removed the vestigial guard (model type already asserted;
  FLOP accounting uses `model.num_flops_per_token` directly) + dropped the now-unused import.
- **tests**: resume-flag config roundtrip (CPU), per-microbatch all-reduce rejection (gloo, CPU), GPU resume test
  asserting moments reset vs restored. GPU resume test not executed locally (no GPU).

### PR-D review round 2 (head 29cab1b57)
Three new comments:
- **P1 — resume reset flag on `load_trainer_state=None`**: round 1 only derived the flag when
  `load_trainer_state is True`, but the default `None` is the normal auto-resume path, so a default
  resume never applied the reset. Changed the guard to `load_trainer_state is not False`.
- **P1 — model buffers omitted from direct checkpoints**: `save_state_dict_direct()` saved only
  `optim.state_dict()`, dropping persistent buffers like the router's aux-loss-free `score_bias`
  (updated outside the optimizer), so resumed/eval routing restarted from fresh bias. Added
  `_persistent_model_buffer_state_dict()` (buffers keyed by wrapper-stripped name so wrapped-training
  checkpoints reload into the unwrapped eval model), merged into the save dict, and restored via a
  separate load pass on both training and eval paths (independent of optimizer-state handling). GPU
  round-trip test mutates `score_bias` before save.
- **P2 — unsupported direct-checkpoint options**: `state_dict_save_opts` / `state_dict_load_opts` /
  `load_key_mapping` were stored but never read by the direct path. Reject non-default (non-None)
  values at `__init__` rather than silently ignoring them.
- **conventions**: `make checks` equivalent (isort/black/ruff/mypy) clean on all touched files; config
  field style is section-comments (matches the class); test inline imports hoisted to module top.

### PR-D review round 3 (head e0af8f9aa)
- **P1 — resume reset wrongly applied to weights-only loads**: round 2 derived the resume flag in
  `Trainer.load_checkpoint()` for any load that wasn't `load_trainer_state=False`, before checking
  whether trainer state existed. So loading a standalone model/optimizer checkpoint (default
  `load_trainer_state=None`, no trainer state) with `reset_optimizer_states_on_resume=True` discarded
  its moments. Moved the default derivation into `Checkpointer.load()` after the trainer-state probe:
  defaults from the resume flag only when trainer state was found (real resume), still honoring an
  explicit `reset_optimizer_states_on_load` override. Added a CPU parametrized test
  (`test_checkpointer_resume_reset_only_applies_on_actual_resume`) covering both default-None cases via
  a recording fake direct train module + stubbed metadata probe.
- **P2 (bare `# TODO review`)**: left in place per user instruction (do not action).

### PR-D style/type CI fix (head 7e5de38f7)
- isort import order in `pipeline/p2p_executor.py` (missed earlier — only per-file isort was run, not
  `make style-check`); recording fake in `checkpoint_test.py` cast to `TrainModule` for mypy.
- Process note: run the full `make checks` (style/lint/type over the whole repo) before every commit,
  not targeted per-file mypy/ruff.

### PR-D Codex review round 1 (#765, head e5f6601d1)
Four comments, all in `ddp_train_module.py` / its collaborators:
- **P1 — `num_microbatches` leaked into the PP model kwargs**: `run_pipeline()` passed
  `num_microbatches` to `PipelineSchedule.step()`, which funnelled it through `**kwargs` into the
  microbatch input splitter, so `CustomSchedule._split_inputs()` rejected the unknown key and every PP
  step raised. Fixed in `distributed/parallel/pipeline_parallel.py`: `step()` now takes
  `num_microbatches` explicitly and applies it via `reset_n_microbatches` for the step (restoring
  after), mirroring the existing `forward_only` override — so it is consumed, not forwarded, and the
  reduced-microbatch dry run actually takes effect.
- **P2 — disabled Float8**: `__init__` rejected any `Float8Config`; now only rejects when
  `float8_config.enabled` (matches `parallelize_and_init_model`, which gates FP8 on `.enabled`).
- **P2 — `-1` EP-degree shorthand**: normalize `ep.degree < 0` to `folded_dp_cp_world_size` before the
  `<= 1` validation and use the normalized `ep_degree` for the EP mesh reshape (matches the
  `build_expert_parallel_mesh` convention).
- **P2 — pre-sharded CP inputs (the CP-consumer gap from the plan)**: `Transformer._prepare_inputs`
  now consumes `cp_already_sharded` / `cp_original_seq_len` — skips re-sharding input_ids/labels while
  still sharding RoPE buffers from the original seq len — preventing double-shard under PP + CP. Ported
  from olmo-ddp's `_prepare_inputs`. (final_rewrite.md PR-D item 9.)

### PR-C Codex round (rebased head 23af3c26d)
Four regressions from the wholesale block/model port; first three restored moe-v2-core behavior:
- **tie_word_embeddings (P1)**: re-tie after `to_empty()` in `init_weights` (tied model was left with an
  independent LM head after materialization).
- **shared-FP8 CPU fallback (P2)**: guard `use_rowwise_fp8` with `device.type == "cuda"` so CPU
  smoke/eval/materialize falls back to the dense shared path instead of asserting.
- **checkpoint_attn shared-only (P2)**: route through `_checkpointed_res_norm_attn` (olmo-ddp called
  `_res_norm_attn` directly, dropping attention activation checkpointing → OOM risk).
- **TBO 1d vs rowwise (P1) — DECISION: adopt olmo-ddp's rowwise-only TBO, retire 1d.** olmo-ddp replaced
  moe-v2-core's 1d TBO (`ep_sync_tbo` / `ep_no_sync_tbo_1d`, landed in PR-A #33) with a
  `rowwise_nvshmem`-only `forward_tbo`. Kept olmo-ddp's rowwise TBO; removed the superseded 1d-TBO modules
  (`ep_sync_tbo.py`, `ep_no_sync_tbo_1d.py`) and their tests (`tbo_test.py`, `ep_no_sync_tbo_1d_test.py`).
  `tbo_state.py` (`SyncedTboPendingContext`, only consumed by the removed `ep_sync_tbo`) is now orphaned
  but kept (a module-importable smoke test references it) — flag for later cleanup.

### PR-C Codex round 2 (head d5a59437)
Three comments; two were re-posts of already-resolved items, one new:
- **tie_word_embeddings re-tie (P1)** — already fixed (re-tie present at the end of `init_weights`); stale re-post.
- **"restore non-rowwise TBO" (P1)** — moot/intentional: this IS the retire-1d-TBO decision above; the model
  correctly rejects sync_1d/no_sync_1d TBO now, and the 1d-TBO tests it referenced were removed.
- **world_mesh required kwarg (P2, NEW — fixed)**: `OLMoDDPModel.init_weights` took `world_mesh` as a
  required keyword, so the non-parallel path `init_weights(device=...)` raised `TypeError`. `world_mesh` is
  only used under PP/EP, so defaulted it to `None` + assert-guarded the PP/EP branches. Also moved the
  method's docstring above its local import (it was a no-op string after the import, not a real docstring).

### PR-C Codex round 3 (head 78d0960)
- **tie re-tie (P1)** / **non-rowwise TBO (P1)** — stale/intentional again (fix present; 1d-TBO retired by decision).
- **te/cpu_offload disabled path (P2, NEW — fixed)**: `get_cpu_offload_context(enabled=False)` constructed
  `CpuOffloadHandler` (whose `__init__` allocates `torch.cuda.Stream()`) before the no-op return, so the
  documented disabled path raised on CPU-only processes. Moved the handler construction inside the
  `enabled` branch so `enabled=False` returns `(nullcontext(), None)` with no CUDA allocation.

### PR-C final review (head c33ce513a)
Three comments from `final_review.md`:
- **P1 — CP kwargs missing in routed TBO loop**: `_prepare_inputs()` put per-block CP RoPE shards in
  `per_block_kwargs`, but the routed `combined_forward_rowwise_nvshmem_tbo` loop only passed
  `all_block_kwargs`, so under TBO + CP the routed blocks missed their `pos_sin`/`pos_cos`/`freqs_cis`
  shards. Fixed: merge `per_block_kwargs[int(block_key)]` into the call, matching the dense/non-TBO paths.
- **P2 — unused cpu-offload module**: reverses the earlier "pull cpu_offload forward into PR-C" decision.
  The te/cpu_offload.py module (~385 lines, private autograd hooks, no tests) was never wired up — the
  only integration in `OLMoDDPModel` was commented out and explicitly disabled. Removed the module
  (`te/cpu_offload.py`, `te/__init__.py`) and the dead commented offload hooks in model.py. Defer to a
  later dedicated PR (Phase F/G) when the feature is actually wired and testable.
- **P2 — restored config docstrings**: re-added the class docstring + per-attribute docstrings on
  `OLMoDDPTransformerBlockConfig` (already done in the prior reconcile round; confirmed present).

### PR-C final review round 2 (head 5c1ce671a)
First-pass comments confirmed addressed. One new P1:
- **P1 — surviving sync-TBO test**: `ep_sync_tbo_test.py` still built the TBO model with
  `ExpertParallelPath.sync_1d`, but `_check_tbo_requirements()` now rejects every path except
  `rowwise_nvshmem`, so the test would fail at the guard on a GPU runner. Deleted it — the supported
  rowwise-NVSHMEM TBO surface is covered by `ep_no_sync_tbo_rowwise_test.py`, and the other legacy TBO
  tests were already removed with the 1d-TBO retirement. (Chose delete over migrate for consistency with
  the rest of the 1d-TBO retirement.)

### PR-C final review round 3 (head 61d03c665)
One new P2:
- **P2 — orphaned sync-TBO state module**: `tbo_state.py` held only `SyncedTboPendingContext`, which
  nothing constructs/consumes now that `_tbo_last_step()` is narrowed to `_NoSyncRowwiseTboPendingContext`
  (defined in `ep_no_sync_tbo_rowwise.py`). Its sole reference was a smoke assertion in
  `block_no_ep_test.py` kept alive only to preserve the dead type. Deleted the module and the assertion
  (+ its `tbo_state` import and docstring mention), completing the synchronous-TBO retirement.

## PR-E: general train infra (#766)

Train-side deltas ported (speed_monitor device table + pipeline/OLMoDDP FLOP coverage;
monkey_patcher torch.compiler.disable; wandb/comet priority=3; console_logger MoE patterns;
trainer `_metric_value_to_tensor` float64 coercion + test; `SubCmd.eval_checkpoints`;
`eval_only` threaded through `TransformerTrainModuleConfig.build` with a dense-path guard +
`TODO(dense-train-module-eval-only)`).

### Verification: `_join_bookkeeping_ops` vs olmo-ddp's `_drain_bookkeeping_ops`
Confirmed core's mechanism fully covers olmo-ddp's `_drain_bookkeeping_ops` — **no change needed**.
olmo-ddp's `_drain` did `_log_metrics()` + a `while` loop of `future.result()` until the queue
emptied. Core decomposed this into explicit `_log_metrics()` + `_join_bookkeeping_ops()` (single
`concurrent.futures.wait`) at the post-`fit()` site and in `_shutdown`. Coverage:
- `_log_metrics()` called explicitly before each `_join` (trainer.py:884, 966).
- The snapshot `wait()` runs after `_log_metrics()` enqueues metric ops, so they're included.
- The drain-loop existed for ops-that-enqueue-ops; bookkeeping ops here are leaf ops submitted from
  the main thread (`_check_if_canceled`, `reduce_metrics`, checkpoint saves) and never re-submit.
  As a backstop `_shutdown(gracefully=True)` follows `_join` with `pool.shutdown(wait=True)` on both
  pools (waits for everything, incl. anything enqueued during the join); all fit/eval teardown routes
  through `_shutdown()`.
- Bonus: core invokes the completion cb *inside* `wrapped_op` (before FINISHED), so `_join` guarantees
  the cb ran — fixes an olmo-ddp race (cb via `add_done_callback` ran after FINISHED → stale state_dict).

### D15: zero-token path filtering in SourceMixtureDataset (#767)
Ported olmo-ddp's zero-token filtering behind an **opt-in flag** for backwards compatibility:
`SourceMixtureDataset.filter_zero_token_paths` (default `False`) gates a shared
`selected_path_tokens()` helper that `to_index()` and `to_paths()` both route through (so their
`(path, idx)` numbering stays aligned for the downstream `_path_offset_index` lookups in
`NumpyFSLDatasetMixture`).
- **Default off** preserves core's historical path indexing for direct/serialized `SourceMixtureDataset`
  use. The one real consumer — `NumpyDatasetConfig.build()`'s mixture branch → `NumpyFSLDatasetMixture`
  — sets the flag `True`, matching olmo-ddp (zero-token paths produce no instances, so the concatenated
  instance space / resume state is unchanged; this just keeps the index tight and stops no-op paths
  reaching the prep loop).
- (Briefly made it unconditional, then reverted to the opt-in flag per request.)

## PR-F1/F2: Qwen-MoE + GPT-OSS variants (#769)

Config-builder variants for the fused-MoE-v2 stack. Both keep their builders internal to their
module (`olmo_core.nn.moe.v2.{qwen,gpt_oss}`) rather than re-exporting from the package namespace,
since neither has an HF checkpoint converter yet (config-mapping only; slugged TODOs +
`.. warning::`). API adaptations vs olmo-ddp: RoPE `rotary_dim`→`partial_rotary_factor`, dropped
`AttentionConfig.d_attn`, collapsed `attention_norm`/`feed_forward_norm` to the single block
`layer_norm` field, dropped `YaRNRoPEScalingConfig.truncate` (absent in core).

### Attention sinks re-introduced (core had removed them)
GPT-OSS requires per-head learnable attention-sink logits, which core had dropped. Restored them as
a self-contained addition rather than the full olmo-ddp attention surface:
- `AttentionConfig.attention_sinks` (default `False`); counted in `num_params` (`+n_heads`); build
  guard rejects it for non-default attention.
- `Attention.sinks` learnable parameter (`normal_`-initialized) + a guard in `Attention.sdpa` that
  rejects sinks on any non-torch backend.
- Torch SDPA backend applies softmax manually with an extra sink column when sinks are present;
  other backends accept the `sinks` argument for interface uniformity but never receive a non-None
  value. Verified numerically on CPU: a large-negative sink logit collapses the sink softmax to the
  plain SDPA result (~2e-9).
- NOT ported from olmo-ddp: `d_attn` (core uses `head_dim` throughout) and the flash/TE sink paths
  (torch-backend only, matching olmo-ddp's own restriction).

## PR-F3: Nemotron-3 Nano variant (#769)

Hybrid Mamba2 / attention / fused-MoE architecture ported wholesale (~940 lines): Mamba2 SSM
sequence mixer, a Nemotron fused-MoE sequence mixer, a hybrid `NemotronBlock`
(`TransformerBlockBase`) selected per-layer via core `TransformerConfig` `block`-dict +
`block_pattern`, and config builders returning a core `TransformerConfig`. Builders stay internal to
the module (`olmo_core.nn.moe.v2.nemotron`); no HF checkpoint converter yet (warning + follows the
`convert_nemotron3_nano_hf_to_olmo.py` script, deferred with the scripts in Phase D).

Core addition: `LayerNormType.nemotron_rms` + `NemotronRMSNorm` in `layer_norm.py` (fp32 variance,
cast-back before affine weight — matches HF Nemotron-H), mirroring the existing `qwen_rms` addition.

API adaptations vs olmo-ddp:
- Dropped `AttentionConfig.d_attn` (core uses `head_dim`).
- `NemotronBlockConfig` neutralizes core's base fields (single `layer_norm` + `name`, not the
  olmo-ddp `attention_norm`/`feed_forward_norm` split) via `init=False`, and overrides `build()` so
  the base's `name`-based block dispatch is never exercised.
- Fixed a latent olmo-ddp bug: `NemotronMamba2Mixer.init_weights` `del`-ed `block_idx`/`num_blocks`
  then referenced them in the llama/llama_depth branches (only reachable under non-default init).
- TP/PP are `NotImplementedError` for the Mamba2/MoE mixers (as in olmo-ddp). CPU build + forward
  verified end-to-end across all three block kinds; SSM numerical parity is GPU/FLA-gated.

### PR-F3 review follow-ups (fifth pass)
- **dt_bias init (P1):** olmo-ddp discarded `time_step_max`/`time_step_floor` and init'd `dt_bias`
  to ones (`softplus(1)=1.31`, ~13x the configured max). Fixed to the Nemotron-H/Mamba2 init:
  log-uniform `dt` in `[time_step_min, time_step_max]`, clamp by `time_step_floor`, store
  inverse-softplus. Regression test asserts effective timesteps land in range.
- **Mamba2 relocated (P2):** moved `NemotronMamba2Config`/`NemotronMamba2Mixer` (+ `NemotronRMSNormGated`
  and the chunk-scan helpers) into the reusable SSM layer `nn/attention/recurrent.py`, registered as
  `@SequenceMixerConfig.register("nemotron_mamba2")` mirroring `GatedDeltaNet`, and re-exported from
  `nn/attention/__init__.py`. `nemotron.py` now only assembles the variant. Also added
  `LayerNormType.nemotron_rms` in the earlier commit. Mamba2 unit tests live in `recurrent_test.py`.

### PR-F3 review follow-up (sixth pass)
- **dt_bias generator (P1):** `reset_parameters()` sampled `dt_bias` via `torch.rand()` on the global
  RNG, breaking the model-init seed contract (identical seeded generators diverged under different
  ambient RNG). Threaded the `init_weights` generator through `reset_parameters(generator=...)` into
  the `torch.rand` call (device matches the param per the framework's `_apply_init` contract), so
  `dt_bias` is tied to `init_seed`. Regression test varies the global seed and asserts an identical
  supplied generator yields identical `dt_bias`.

## PR-G: Experimental EP backends + attention FP8 (one PR)

Three olmo-ddp features that core had scaffolded but left rejected/absent:

### DeepEP V2 + rowwise-wave EP backends
Core's `nn/ddp/block.py` already dispatched `ExpertParallelPath.{deepep_v2,rowwise_wave}` to
lazily-imported transport modules, and `ep_config` already carried all their fields — only the
modules and the `validate()` rejection were missing. Dropped in `ep_deepep_v2.py` (647L) and
`ep_no_sync_rowwise_wave.py` (1385L) verbatim; lifted the path rejection (kept the `tbo` schedule
rejection). Both are GPU-only (DeepEP also needs the optional `deep_ep` pkg + NCCL≥2.30.4; wave needs
the compiled `symm_mem_vdev2d` ext, same as the existing rowwise path). Adaptations: typed the
untyped DeepEP handle/buffer as `Any` (olmo-ddp used `object`, which fails core mypy), and cast the
`sync_tail_drop_allowed_splits_single_a2a` union return (mirroring the existing rowwise call).

### MXFP8Linear + typed prequantizer hook
Restored the injectable `FP8WeightStore.prequantizer` core had hardcoded. Instead of olmo-ddp's
`Callable[..., Any]`, typed it: the cache/RHS type widens to a `PrequantizedRHS` union
(`ScaledGroupedMMPrequantizedRHS | ScaledMMPrequantizedRHS`); the two concrete consumers (routed
experts → grouped-mm; MXFP8Linear → scaled-mm) narrow back with `cast`. Added
`kernels/mxfp8_linear.py` (deps `mxfp8_utils`/`mxfp8_tensor` already in core) + `nn/mxfp8_linear.py`
(`@beta`). GPU parity via `mxfp8_linear_test.py` (skips w/o CUDA); hook wiring via CPU
`fp8_weight_test.py`.

### FusedAttentionV2
Packed-QKV attention variant with optional MXFP8 projections. Extracted a `_prepare_qkv` hook from
`Attention.forward` (behavior-preserving; verified numerically identical) so FusedAttentionV2
overrides only the packed projection and reuses `forward`. Added `AttentionType.fused_v2`, the
mxfp8/recompute config fields (routed to fused_v2 in `build()`, rejected elsewhere), `@beta`. Dropped
the olmo-ddp `d_attn` param (core uses `head_dim`). `use_recompute_qkv_prep` /
`mxfp8_save_qkv_for_backward` are honored by the shared `Attention.forward` (olmo-ddp wires them in
the base attention, not the FusedAttentionV2 subclass): recompute wraps `_prepare_qkv` in an
`OutputDiscardCheckpoint`; MXFP8-save uses saved-tensor hooks. Both live on the base `Attention`
(honored by default + fused_v2; rejected for fused/normalized which override forward). rowwise-wave sibling backends (wave-mega, TMA-IBGDA) remain not ported (olmo-ddp
deleted them).

### PR-G review round 2 (MXFP8-attention hardening)
- **Optimizer bypass (P1):** MXFP8Linear freezes its weight + routes wgrad to an FP8WeightStore that
  only OLMoDDPOptimizer discovers. Added `_assert_no_unowned_fp8_weight_stores` in the general
  `TransformerTrainModule` build — raises for optimizer-enabled fp8 stores (points users to
  OLMoDDPTrainModule). Sibling class, so the OLMoDDP path is unaffected.
- **Saved-QKV match (P1):** `_MXFP8SavedQKVHooks` matched by `id(q/k/v)`, but the torch backend
  transposes/repeats before SDPA so autograd saved derived tensors → pack always no-oped. Switched
  to storage-pointer matching (transpose views match; GQA repeat-copies stay unmatched, never
  mis-packed). CPU test covers the predicate.
- **Autograd anchor:** MXFP8 kernel's custom Function only tracked `mat_a`; when it doesn't require
  grad (first layer / after frozen embedding), the output had no grad_fn and the wgrad routing was
  skipped. Added a differentiable anchor when a wgrad sink is set and mat_a doesn't require grad.
- **GPU marker (P2):** MXFP8 test suite uses `pytestmark = list(GPU_MARKS)` (gpu mark + skip)
  instead of an inline `torch.cuda.is_available()` skip.
- GPU-only, still unverified here: MXFP8 GEMM parity and the fused-optimizer end-to-end weight-update
  step (belongs to the OLMoDDP optimizer GPU suite).

## Material runtime-diffs port (slow-batch warn, GDN FLOPs, logger test, torch-stream, attention FLOPs)

Ported the genuinely-missing material runtime diffs identified by the verified audit. Each was
checked against current core before porting (the raw tree diff over-reports gaps).

Ported:
- Slow batch-load warning `OLMO_BATCH_LOAD_WARN_SECONDS` (`5a4afb68e`, speed_monitor.py) + CPU
  valid/invalid-threshold tests.
- GDN training-FLOPs x3 factor (`39659e83b`, recurrent.py) + analytical test (`@requires_fla`).
- Logger callback-order regression test only (`2c8815e3a`); the comet/wandb init-ordering impl was
  already present in core.
- Torch stream custom-op patch `patch_torch_stream_custom_ops()` (`c4c5319a6`, utils.py) wired into
  `prepare_cli_environment()` + test. Guarded to no-op without `torch._dynamo.variables.streams`, so
  inert until torch 2.11; landed now so it's ready. NOTE: the setup_logging/`OLMO_RICH_LOGGING`
  refactor bundled in the same upstream commit was already present in core — nothing to port there.
- Causal + sliding-window attention FLOP accounting (`329e0a203`). Adapted rather than copied: core's
  FLOP API is `num_flops_per_token` (per-token), not olmo's `flops_per_seq(d_model, seqlen)` (that API
  survives only on the unused `OLMoDDPTransformerBlockConfig.flops_per_seq`). Added
  `_causal_attention_positions()` and applied exact causal / windowed counting in `Attention` and
  `FusedAttention`. This corrects a ~2x overcount of full-attention compute FLOPs (core never applied
  the causal /2), so reported model TFLOPs / MFU drop on the attention term vs older logs; a historical
  note is left at each call site. The live MFU path (`train_module -> model.num_flops_per_token` over
  built blocks) is per-layer correct because each layer's real `window_size` is read.

Intentionally NOT ported from `329e0a203`:
- The 422-line `fa4_intra_doc_attention_bench.py` (benchmark; scripts workstream).
- The NVTX annotation relocation in olmo `nn/ddp/model.py` (orthogonal profiling labeling, not FLOPs).
- `layer_idx`/`n_layers` threading into `OLMoDDPTransformerBlockConfig.flops_per_seq`: that config-level
  estimator has no callers in core and routes through the now-fixed `num_flops_per_token` anyway.

Held for evidence / separate workstreams (per audit): none from this batch beyond the above; NCCL-RMA
ACK transport was found already present in core (only its smoke/transport tests are olmo-only).

## TE CPU activation-offload prototype

Ported olmo-ddp's canonical CPU activation-offload module to `nn/moe/v2/te/cpu_offload.py`
(vendored NVIDIA code, torch-only despite the `te` dir name) plus a package `__init__.py` and an
import/GPU-disabled-path smoke test. Marked experimental in the docstrings.

- Not wired into `nn/ddp/model.py`: olmo itself keeps the offload path disabled (`self.cpu_offload =
  False`, `get_cpu_offload_context(...)` commented out — "not useful due to low PCIe bandwidth"), and
  core had already dropped the import. Left unwired rather than reintroducing dead/commented wiring.
- Did NOT port `cpu_offload_simple_varlen.py`: it is an earlier, unreferenced near-duplicate of the
  canonical module (which is itself the "variable tensors per group" variant). Available to add later
  if a reason emerges.

## Reduce-scatter for OLMo DDP gradients (e3b5d69d9)

Ported the opt-in normal-parameter reduce-scatter path for `MultiGroupDistributedDataParallel`
(new `use_reduce_scatter` flag, `configure_reduce_scatter_params`, per-bucket RS packing +
`reduce_scatter_tensor`, `_olmo_ddp_reduced_grad_shard` intake in `OLMoDDPOptimizer`), wired
through `dp_config.use_reduce_scatter` on the train module.

Conflict resolutions (kept core's cleanups, took olmo-ddp's functional additions):

- `moe_optimizer.py`: did NOT reintroduce olmo-ddp's `@overload def step` stubs — core had already
  removed them. Dropped an unused `dbg_mem_before_cp1` debug line. Flipped `_use_reduce_scatter_grads`
  default to `False` (legacy optimizer-owned reducer stays off; MultiGroupDDP owns AR and RS).
- `ddp_train_module.py`: removed the old `reduce_scatter_grads` constructor arg and its
  `NotImplementedError` guard, replaced by the config-driven `configure_reduce_scatter_params` path.
- `config.py`: removed the now-dead `reduce_scatter_grads` field from `OLMoDDPTrainModuleConfig`
  (the train module no longer accepts it; `build()` would otherwise pass an unknown kwarg). Kept
  core's Sphinx-cross-ref docstrings and folded in the reduce-scatter semantics.
- Tests: the source modified `multigroup_distributed_test.py`, which core had renamed to
  `distributed_test.py`. Folded the new reduce-scatter models/helpers/tests into `distributed_test.py`
  and added `start_method="spawn"` to every new `run_distributed_test` call (per the spawn policy).
  Adapted `config_test.py` to core's build path (`as_dict(...)` instead of olmo-ddp's `_build_kwargs()`).
  3 CPU-backend RS tests pass locally; optimizer-step / EP-parity tests are multi-GPU-gated.
