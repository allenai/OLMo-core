# OLMoE ladder v2 experiment queue

This is the concrete post-migration architecture plan. The scientific and
operational rules in `EXPERIMENT_RULES.md` govern every item below. All isolated
tests train from scratch at Cx1/Cx2/Cx4/Cx8 and compare against the 275M-active
wide v1 integration model.

| Order | Experiment | Change from parent recipe | State |
|---:|---|---|---|
| 1 | GDN hybrid | On wide, replace sliding-attention layers with GatedDeltaNet; keep geometry, global-attention placement, RoPE, initialization, and `expand_v=1` fixed. | 275M sweep, 480M/810M scale cells, and 1.2B Cx1/Cx2/Cx4 are complete; only 1.2B Cx8 remains in progress. |
| 2 | Aligned geometry, mixer ratio, and GDN value width | Use the corresponding dense ladder width, depth, four-GDN/one-global pattern, and `expand_v=2` while retaining MoE, the dense-first-FFN design, our GQA ratio, RoPE, and initialization. | 275M four-LR Cx1/2/4/8 sweep complete. Larger RoPE configs are audited but intentionally deferred while NoPE is tested directly. |
| 3 | NoPE | On the aligned-geometry recipe, remove RoPE only from global-attention layers and train from initialization. | 275M sweep complete. Larger wave is 10/12 complete; 1.2B Cx4 is running, while 1.2B Cx8 has now reproduced the same non-finite-grad failure on three attempts. |
| 4 | Full-attention gating, then exact head geometry | First add only the dense ladder's elementwise full-precision gate to the 275M NoPE model while retaining 8-Q/4-KV GQA. If useful, separately test the dense 8-Q/8-KV attention shape. | 275M gated sweep complete. Larger wave is 8/12 complete; 810M Cx8 and 1.2B Cx2/Cx4/Cx8 are running, with the requeued Cx2 past its prior failure point. Exact KV-head geometry remains deferred. |
| 5 | RoPE × attention-gating interaction | Starting from the gated NoPE geometry model, restore RoPE only on full-attention layers 4 and 9. Keep GQA, gating, geometry, initialization, data, and optimization fixed. | Cx4 MB16 smoke passed; 16-cell 275M LR sweep submitted urgent unallocated on 2026-07-20. |
| 6 | Initialization | On the promoted control, change only initialization standard deviation from 0.01 to 0.02. | Optional/planned. |
| 7 | Remove QK norm | On the promoted control, remove per-head RMSNorm from Q and K in global-attention layers only. Keep GDN's internal output norm and every other norm unchanged. | Deferred isolated architecture ablation. This intentionally departs from the dense ladder, which uses QK norm. |
| 8 | FP8 training | Hold the promoted architecture fixed and compare an explicitly selected OLMo-core FP8 training mode against the BF16 control, recording loss, stability, memory, tokens/s, and TFLOPs/GPU. | Deferred precision/systems ablation; choose and smoke the exact FP8 mode after the architecture is stable. |
| 9 | LatentMoE | Add the coworker's LatentMoE implementation to the promoted hybrid recipe and test it as an isolated MoE intervention before composition. | Deferred until the coworker's implementation is available and verified in this branch. |
| 10 | Combined 275M pilot | Combine only interventions whose isolated evidence is neutral-to-positive. | Blocked on isolated results. |
| 11 | Promote combined recipe | Run the full pretraining ladder, then midtraining, then 8K-to-65K long-context adaptation. | Blocked on the combined pilot. |

## Dense-hybrid alignment target

Apart from the intentional dense-versus-MoE FFN difference, full alignment with
the coworker dense hybrid requires all of the following changes from the current
wide-derived `expand_v=1` hybrid:

- adopt the dense ladder's per-rung width, depth, head, and attention geometry,
  resolving its 450M/1.4B rung labels against our 480M/1.2B rungs;
- use the exact four-GDN/one-global repeating pattern (20% global attention),
  including GDN as the sequence mixer in our dense-first-FFN block when needed
  to preserve the ratio;
- change GDN from `expand_v=1` to the dense recipe's `expand_v=2`;
- match the dense rung's exact query/KV-head geometry and add its elementwise
  attention gate. This is size-dependent: dense 275M/450M use 8 Q / 8 KV
  heads (MHA), while dense 810M and above use 2:1 GQA. Our current 275M/480M
  models use 8 Q / 4 KV heads;
- remove RoPE from global-attention layers (NoPE); and
- change initialization standard deviation from `0.01` to `0.02`.

Peri-norm placement, RMSNorm and QK-norm types/epsilons, 128-dimensional heads,
SiLU, embedding scaling/normalization, bias settings, and the remaining GDN
dynamics already agree and do not need interventions. Re-audit active parameters
and FLOPs after composing the alignment changes. `expand_v=2` is now folded
into the geometry experiment; exact KV-head/gate matching, NoPE, and the new
initialization remain separately testable.

For geometry-changing experiments, record both token-matched quality and
active-parameter/FLOP efficiency. Do not promote a result until each relevant
Cx has a finished, bracketed LR sweep under the rules in
`EXPERIMENT_RULES.md`.

The original migration-era statement of this plan remains in the repository
root at `JACOBM_MIGRATION_PLAN.md`; this file is the live v2 queue.

The larger NoPE geometry family has completed checkpoint-free unallocated
smokes on Holmes B300s. Exact parameter counts, measured TFLOPs/GPU, memory,
the all-Cx ETA matrix, and the selected 192-GPU layout are recorded in
`GEOMETRY_MATCHED_SCALE.md`. Before launching full runs, wait for the 275M NoPE
LR decision, insert the four transferred LRs, and explicitly decide whether
the production wave should remain unallocated or return to the allocated
queue.

## Training-schedule transition

The v1 pretraining ladder used a 10%-of-training linear warmup followed by
cosine decay to 0.1x peak LR. Midtraining and long-context adaptation instead
used a fixed 2,000-step linear warmup followed by constant LR. That discrepancy
was accidental, but the remaining v1 long-context continuations (including
810M and 1.2B integration-wide) must retain the fixed-2,000-step schedule so the
wave remains internally comparable.

Before promoting the v2 recipe, run controlled scheduler experiments that
change only the schedule. Compare the v1 incumbent against the pretraining
schedule separately for midtraining and long-context adaptation, starting with
a smaller representative model before applying the result across the ladder.
Do not silently mix the new schedule into the remaining v1 runs.
