# Experiment Plans

This folder contains detailed runbooks for post-baseline MoE A0 experiments.
Each plan should be specific enough that a fresh session can implement, launch,
monitor, summarize, and evaluate the experiment without relying on oral context.

Current plans:

- `expert_granularity.md`: fixed active/total routed-capacity sweep over
  24E/top2, 48E/top4, and 96E/top8.
- `shared_expert_dense_schedule.md`: shared-expert size/presence and dense/MoE
  schedule interactions.
- `width_depth_geometry.md`: deeper/narrower vs shallower/wider backbone shapes
  at controlled active/total parameter budgets.

Plan template:

- Goal and hypothesis.
- Exact architecture variants.
- Required code changes.
- Launch settings and LR transfer protocol.
- Monitoring and failure criteria.
- Analysis and promotion criteria.
- Documentation/plotting expectations.
- Open TODOs.
