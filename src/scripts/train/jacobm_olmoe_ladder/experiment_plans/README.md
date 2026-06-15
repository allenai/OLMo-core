> Planning note: these runbooks are useful implementation references, but some tables predate the current `480m` naming and canonical Cx2 `b384k` policy. Confirm launch settings against `../SETTINGS_AUDIT.md` and queue priority against `../LAUNCH_QUEUE.md`.

# Experiment Plans

This folder contains detailed runbooks for post-baseline MoE A0 experiments.
Each plan should be specific enough that a fresh session can implement, launch,
monitor, summarize, and evaluate the experiment without relying on oral context.

Current plans:

- `REMAINING_V0_ABLATION_PLAN.md`: succinct plan for the three independent V0
  ablations to finish before the first integrated run.
- `CONCRETE_RUNBOOK_SUMMARY.md`: short-form index of the concrete first-wave
  experiment runbooks, shared LR-transfer policy, systems defaults, and launch
  intent.
- `expert_granularity.md`: fixed active/total routed-capacity sweep over
  24E/top2, 48E/top4, and 96E/top8.
- `shared_expert_dense_schedule.md`: shared-expert size/presence and dense/MoE
  schedule interactions.
- `width_depth_geometry.md`: deeper/narrower vs shallower/wider backbone shapes
  at controlled active/total parameter budgets.
- `total_sparsity_fixed_active.md`: total routed expert capacity sweep at fixed
  active routed compute.
- `swa_full_attention_ratio.md`: sliding-window versus full-attention layer
  ratio sweep, separate from larger hybrid-attention changes.
- `qwen3_like.md`: Qwen3-30B-A3B-inspired no-shared, all-sparse, fine-grained
  top-8 MoE geometry variants.

Plan template:

- Goal and hypothesis.
- Exact architecture variants.
- Required code changes.
- Launch settings and LR transfer protocol.
- Monitoring and failure criteria.
- Analysis and promotion criteria.
- Documentation/plotting expectations.
- Open TODOs.
