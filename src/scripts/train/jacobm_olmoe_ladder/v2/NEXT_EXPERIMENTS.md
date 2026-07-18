# OLMoE ladder v2 experiment queue

This is the concrete post-migration architecture plan. The scientific and
operational rules in `EXPERIMENT_RULES.md` govern every item below. All isolated
tests train from scratch at Cx1/Cx2/Cx4/Cx8 and compare against the 275M-active
wide v1 integration model.

| Order | Experiment | Change from parent recipe | State |
|---:|---|---|---|
| 1 | GDN hybrid | On wide, replace sliding-attention layers with GatedDeltaNet; keep geometry, global-attention placement, RoPE, initialization, and `expand_v=1` fixed. | In progress; finish and bracket the current LR sweeps. |
| 2 | Aligned geometry, mixer ratio, and GDN value width | Use the corresponding dense ladder width, depth, four-GDN/one-global pattern, and `expand_v=2` while retaining MoE, the dense-first-FFN design, our GQA ratio, RoPE, and initialization. | 275M inherited-LR sweep in progress. Active-matched 480M/810M/1.2B configs designed and locally audited; no larger-size runs launched. |
| 3 | NoPE | On the 275M aligned-geometry recipe, remove RoPE only from global-attention layers and train from initialization. | Independent variant implemented; all 2/4/8-GPU Cx1/Cx8 scaling smokes passed. Production sweep uses 4/4/4/8 GPUs for Cx1/2/4/8 and is ready to submit unallocated. |
| 4 | Initialization | On the wide control, change only initialization standard deviation from 0.01 to 0.02. | Optional/planned. |
| 5 | Combined 275M pilot | Combine only interventions whose isolated evidence is neutral-to-positive. | Blocked on isolated results. |
| 6 | Promote combined recipe | Run the full pretraining ladder, then midtraining, then 8K-to-65K long-context adaptation. | Blocked on the combined pilot. |

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

The larger geometry family will return to allocated runs. Before launching it,
smoke-test the largest plausible per-rank microbatch on Holmes B300s for each
size and choose EP from measured throughput. Do not copy the temporary
unallocated scheduling exception from the 275M sweep.

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
