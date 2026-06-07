# MoE A0 Ablation And Research Plan

This document is the forward-looking plan for experiments after the v0 baseline
ladder is settled. It separates near-term ablations that should help define a
strong standard MoE shape from weirder research ideas that may become more
interesting papers or longer-term systems work.

The baseline lineage for comparison is the current MoE A0 family:

- 48 experts per MoE layer
- top-4 routing
- `moe_hidden_size = d_model`
- one shared expert
- `shared_mlp_hidden_size = d_model / 2`
- one dense prefix layer
- GQA with `n_kv_heads = n_heads // 2`
- mostly sliding-window attention with periodic full attention

For now, LR selection remains based on final-window training CE loss only. Eval
metrics and validation losses are observational until we explicitly change that
policy.

## Goals

The near-term goal is to turn the current good yolo-ish architecture into a
well-understood baseline family. The longer-term goal is to use that baseline to
ask more research-flavored questions about sparse models as modular, expandable,
and adaptive compute systems.

The planning principle:

- First, find a good standard MoE shape under controlled active/total parameter
  budgets.
- Then, test stranger mechanisms against that standard shape.
- Use the ladder to avoid overfitting to a single data scale.
- Promote only promising variants from 275M to 810M/1.2B.

## Baseline Ablation Roadmap

### 1. Expert Geometry

This is the first ablation family to plan concretely.

Question:

Can we improve the MoE block by changing expert count, expert hidden size, and
`top_k` while keeping active and total parameter counts approximately fixed?

Why this comes first:

- It directly tests whether the inherited 48E top-4 point is actually special.
- It is the most MoE-specific knob.
- It should be easier to interpret than width/depth changes because the dense
  backbone can remain mostly unchanged.

Candidate variants:

| Variant | Direction | Intuition |
| --- | --- | --- |
| More smaller experts, higher `top_k` | More routing diversity at same active compute | Tokens can combine more specialized small functions. |
| Fewer larger experts, lower `top_k` | More capacity per active expert | Each active expert may be more expressive and easier to train. |
| Same `top_k`, changed expert count/hidden | Isolate pool size vs per-expert size | Tests whether total expert pool cardinality matters independent of active fan-in. |
| Shared expert size sweep | `0`, `d_model/4`, `d_model/2`, maybe `d_model` | Turns shared expert from binary choice into an active-capacity allocation question. |

The first expert-size sweep should use a clean granularity family:

```text
num_experts / top_k = 12
top_k * moe_hidden_size = 4 * d_model
num_experts * moe_hidden_size = 48 * d_model
```

This exactly preserves routed active hidden units and routed total expert hidden
units. Shared expert size, dense prefix, attention, width, depth, and batch rules
stay fixed. Total parameter counts will differ slightly because the router has
one output per expert, but the expert FFN budget is matched.

The three core variants are:

| Variant name | Experts | `top_k` | `moe_hidden_size` rule | Motivation |
| --- | ---: | ---: | ---: | --- |
| `coarse_24e_top2` | 24 | 2 | `2 * d_model` | Phi-style coarse top-2 endpoint. |
| `baseline_48e_top4` | 48 | 4 | `1 * d_model` | Current MoE A0 baseline. |
| `fine_96e_top8` | 96 | 8 | `d_model / 2` | DeepSeek/Qwen-style fine-grained endpoint. |

Exact 275M configs:

| Variant | `d_model` | `d_attn` | Layers | Heads | KV heads | Experts | `top_k` | `moe_hidden_size` | Shared experts | Shared hidden | Dense prefix | Dense MLP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `coarse_24e_top2` | 768 | 1024 | 12 | 8 | 4 | 24 | 2 | 1536 | 1 | 384 | 1 | 3456 |
| `baseline_48e_top4` | 768 | 1024 | 12 | 8 | 4 | 48 | 4 | 768 | 1 | 384 | 1 | 3456 |
| `fine_96e_top8` | 768 | 1024 | 12 | 8 | 4 | 96 | 8 | 384 | 1 | 384 | 1 | 3456 |

Exact 810M configs:

| Variant | `d_model` | `d_attn` | Layers | Heads | KV heads | Experts | `top_k` | `moe_hidden_size` | Shared experts | Shared hidden | Dense prefix | Dense MLP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `coarse_24e_top2` | 1280 | 1536 | 20 | 12 | 6 | 24 | 2 | 2560 | 1 | 640 | 1 | 5760 |
| `baseline_48e_top4` | 1280 | 1536 | 20 | 12 | 6 | 48 | 4 | 1280 | 1 | 640 | 1 | 5760 |
| `fine_96e_top8` | 1280 | 1536 | 20 | 12 | 6 | 96 | 8 | 640 | 1 | 640 | 1 | 5760 |

Exact 1.2B configs:

| Variant | `d_model` | `d_attn` | Layers | Heads | KV heads | Experts | `top_k` | `moe_hidden_size` | Shared experts | Shared hidden | Dense prefix | Dense MLP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `coarse_24e_top2` | 1536 | 2048 | 22 | 16 | 8 | 24 | 2 | 3072 | 1 | 768 | 1 | 6912 |
| `baseline_48e_top4` | 1536 | 2048 | 22 | 16 | 8 | 48 | 4 | 1536 | 1 | 768 | 1 | 6912 |
| `fine_96e_top8` | 1536 | 2048 | 22 | 16 | 8 | 96 | 8 | 768 | 1 | 768 | 1 | 6912 |

Code-style config fragments:

```python
EXPERT_GEOMETRY_VARIANTS = {
    "coarse_24e_top2": dict(num_experts=24, top_k=2, moe_hidden_mult=2.0),
    "baseline_48e_top4": dict(num_experts=48, top_k=4, moe_hidden_mult=1.0),
    "fine_96e_top8": dict(num_experts=96, top_k=8, moe_hidden_mult=0.5),
}
```

Important controls:

- Keep active params approximately fixed.
- Keep total params approximately fixed for the first wave.
- Keep layer count, width, attention schedule, dense prefix count, and batch
  rules fixed.
- Re-tune LR coarsely for each serious variant; do not assume the baseline LR
  transfers perfectly.

First concrete experiment shape, to decide next:

- Start at 275M.
- Run Cx1 and Cx4 for each variant first.
- Use a four-point LR sweep centered from the baseline LR rule, widened if the
  structural change is large.
- Promote only variants that beat baseline at both Cx1 and Cx4, or show a clear
  scaling reason to keep them.

Open design questions:

- Which exact expert geometries hit the active/total budgets cleanly?
- Should active params include embeddings/head for matching, or use active
  non-embedding params for architecture matching?
- Should shared expert be counted as part of the expert-geometry sweep or saved
  for the dense/shared schedule sweep?

### 2. Dense/MoE Layer Schedule

Question:

How much dense computation should the model have, and where should it live?

Candidate variants:

- Current: one dense prefix layer, then MoE layers.
- More dense prefix layers: 2, 4, perhaps scaled by model depth.
- Alternating dense/MoE layers.
- Periodic dense layers, e.g. one dense layer every 3-4 layers.
- MoE every layer but no shared expert.

Why it matters:

- Dense layers may stabilize early representations before sparse routing.
- Alternating dense/MoE layers may give tokens a shared mixing path throughout
  the network.
- This interacts strongly with shared experts.

Suggested design:

- Treat shared expert on/off as a small factorial with dense schedule.
- Keep active and total params approximately fixed by adjusting MoE hidden sizes
  or dense MLP widths.

### 3. Shared Expert Ablations

Question:

Is the shared expert helping because it is a generally useful dense path, or is
it merely compensating for too few dense layers?

Candidate variants:

- No shared expert.
- Current shared expert, `shared_mlp_hidden_size = d_model / 2`.
- Smaller shared expert, `d_model / 4`.
- Larger shared expert, `d_model`.

Best tested alongside dense-layer schedule:

- One dense prefix + shared expert.
- One dense prefix + no shared expert.
- More dense prefix + no shared expert.
- Alternating dense/MoE + shared/no shared.

### 4. Width/Depth At Fixed Active And Total Params

Question:

Does this MoE family prefer width or depth when active and total parameter counts
are fixed?

Candidate variants:

- Deeper/narrower.
- Shallower/wider.
- Current shape.

Why this is second-wave rather than first-wave:

- It changes attention, residual depth, MoE routing frequency, optimization, and
  systems behavior at the same time.
- It is still important, but expert geometry should probably be understood first.

### 5. Attention Schedule

Question:

How much full attention is needed relative to sliding-window attention?

Candidate variants:

- Current sliding/full attention ratio.
- Fewer full-attention layers.
- Full attention only at regular sparse intervals.
- More local-heavy variant for efficiency.

Why it may be later:

- It may matter more for downstream/long-context behavior than early
  pretraining-loss ladder decisions.
- We should avoid changing attention and MoE shape simultaneously in the first
  standardization wave.

### 6. More Total Params At Fixed Active Params

Question:

How sparse should the model be? Can we increase total capacity dramatically while
holding active compute fixed?

Candidate variants:

- Current active/total ratio.
- 2x total expert capacity at same active params.
- 4x total expert capacity at same active params.
- Eventually, very sparse ratios where only a small fraction of total parameters
  are active per token.

Why this is important:

- It tests whether inactive capacity improves loss once routing has enough
  options.
- It moves us toward modern high-sparsity MoE regimes.

Why it should wait:

- We should first know whether our fixed-total expert geometry is sane.

### 7. Hybrid Attention Instead Of SWA

Question:

Should the eventual baseline use a hybrid attention scheme rather than the
current SWA/full-attention mix?

This is worth doing, but likely after the MoE block shape is less uncertain.

## Research Ideas

These ideas are less immediately utilitarian than the baseline ablations, but
they are likely more interesting academically. A useful shared framing:

Can sparse models become adaptive computation systems, rather than just cheaper
dense models?

### 1. Progressive / Growing MoE

Related paper:

- [EMO: Frustratingly Easy Progressive Training of Extendable MoE](https://arxiv.org/abs/2605.13247)

Core idea:

Start with fewer experts and grow the expert pool over training. For example:

- `12E -> 24E -> 48E`
- `24E -> 48E -> 96E`
- grow at fixed token milestones or based on loss/usage diagnostics

Questions:

- Can we match the final fixed-expert baseline while spending less memory and
  communication early?
- Can growth make continual learning cheaper by adding new capacity over time?
- Should new experts be random, copied, split from existing experts, or
  initialized from high-traffic experts?

First runnable version:

- Train a 275M Cx4 run with staged expert growth.
- Compare to a fixed final-capacity baseline.
- Keep average active compute comparable.
- Measure loss, wall-clock, expert usage, and stability around growth events.

Hard parts:

- Optimizer state for new experts.
- Router rebalancing after expansion.
- Checkpoint compatibility.
- Whether to briefly warm up LR or router loss after growth.

### 2. Conditional Compute / Variable `top_k`

Core idea:

Different tokens or sequences use different numbers of experts. Easy tokens
spend less compute; hard or uncertain tokens spend more.

Possible policies:

- Router entropy controls `top_k`.
- Token loss proxy controls `top_k`.
- Sequence/document type controls `top_k`.
- Learned compute budget with an auxiliary compute penalty.

First runnable version:

- Fixed average compute target.
- Let tokens choose among `top_k in {2, 4, 8}` by a simple entropy rule.
- Compare against fixed top-4 with matched average active params.

Metrics:

- Loss at matched average FLOPs.
- Distribution of `top_k` by token type, position, domain, and loss.
- Whether high-compute tokens actually correspond to harder examples.

### 3. Universal Expert Pools

Related paper:

- [Mixture of Universal Experts: Scaling Virtual Width via Depth-Width Transformation](https://arxiv.org/abs/2603.04971)

Core idea:

Reuse a layer-agnostic expert pool across multiple layers, creating a form of
virtual width by sharing expert capacity across depth.

First runnable version:

- Current per-layer experts vs grouped universal pools.
- Example: layers 0-3 share one expert pool, layers 4-7 share another, etc.
- Keep active and total params approximately fixed.

Questions:

- Does cross-layer sharing improve parameter efficiency?
- Does it hurt specialization because the same experts see different depth
  distributions?
- Do we need depth-aware load balancing or routing state?

Longer-term version:

- A model made mostly from a universal expert pool.
- Tokens can recurse through the pool a variable number of times.
- Expert computation becomes an iterative workspace.

### 4. Expert Lifecycle / Birth-Death

Core idea:

Experts should not be static. The model can split, merge, retire, freeze, or
grow experts based on usage and specialization.

Possible mechanisms:

- Split high-traffic or high-loss experts.
- Merge experts with similar routing/use patterns.
- Retire dead experts.
- Freeze mature experts and allocate new experts for new data.
- Reserve new capacity for new domains in continual learning.

Why this is exciting:

- It combines progressive growth with data-driven modularity.
- It gives MoEs a natural continual-learning story.
- It may create models that are easier to prune, specialize, or update.

First runnable version:

- Offline lifecycle analysis on completed runs:
  - expert load distribution
  - router entropy
  - specialization by token/domain
  - redundancy between experts
- Then a simple online split rule for overused experts.

### 5. Document/Token Expert Pools And Emergent Modularity

Related paper:

- [EMO: Pretraining Mixture of Experts for Emergent Modularity](https://arxiv.org/abs/2605.06663)

Core idea:

Encourage tokens from the same document, domain, or sequence to use a coherent
subset of experts. This makes the model more modular and may allow useful expert
subsets to be retained, pruned, or composed.

First runnable version:

- Assign each document a candidate expert subset.
- Route tokens within that subset.
- Compare normal training loss to baseline.
- Evaluate subset survivability: how much performance remains if only a
  document/domain's preferred expert subset is kept?

Why it matters:

- It creates a bridge between MoE training and modular deployment.
- It may help continual learning by localizing new knowledge to new expert
  subsets.
- It pairs naturally with growth and expert dropout.

### 6. Expert Path Dropout / Routing Robustness

Core idea:

Randomly perturb or drop routed experts during training so tokens cannot depend
on a brittle expert path.

This may be especially useful with multiple data epochs or continual learning:

- It discourages permanent route lock-in.
- It makes backup expert paths stronger.
- It may reduce forgetting when old paths are masked or stale.
- It pairs well with document expert pools and expert growth.

First runnable version:

- During training, occasionally mask one selected routed expert and force the
  router to choose a backup.
- Keep average active compute matched.
- Evaluate robustness to expert pruning or route perturbation.

### 7. Adaptive Expert Depth / Early-Exit MoE

Core idea:

Some tokens may not need all layers or all expert passes. Attach intermediate
prediction or halting heads and let tokens decide whether to continue spending
expert compute.

Related older ideas:

- Early-exit transformers.
- Deep supervision with auxiliary layer-wise LM heads.
- Adaptive computation time / pondering.
- Self-speculative decoding with useful intermediate predictions.

MoE-specific version:

- Every few layers, predict whether a token is "done."
- Easy tokens skip later MoE computation.
- Hard tokens continue, use more experts, or recurse through a universal expert
  pool.

First runnable version:

- Add auxiliary LM heads at a few intermediate layers.
- Train with a small auxiliary loss.
- Analyze which layers produce usable predictions before adding learned halting.

### 8. Router As Learned Curriculum

Core idea:

Use router uncertainty and expert usage to decide where the model should spend
more training compute or replay.

Possible signals:

- High router entropy.
- High token loss.
- Unstable expert assignment across epochs.
- Rare-domain expert concentration.

This links conditional compute to data selection and continual learning.

## Suggested Priority

### Standardization Wave

1. Expert geometry at fixed active and total params.
2. Dense/MoE schedule plus shared expert interaction.
3. Width/depth at fixed active and total params.
4. Total sparsity increase at fixed active params.
5. Attention schedule and hybrid attention.

### Research Wave

1. Document expert pools / emergent modularity.
2. Expert path dropout and pruning robustness.
3. Progressive/growing MoE.
4. Universal expert pools.
5. Expert lifecycle / birth-death.
6. Adaptive expert depth and recurrent universal experts.
7. Router-as-curriculum.

## Next Concrete Planning Task

The next document update should specify the first expert-geometry variants.

For each candidate, record:

- target model size, likely 275M first;
- exact architecture settings;
- active parameter estimate;
- total parameter estimate;
- expected active/total ratio;
- first LR sweep grid;
- which Cx rungs to run first;
- promotion criteria to 810M.

The first wave should be small enough to finish and interpret. A good target is
three or four expert-geometry variants plus the baseline, with Cx1/Cx4 first.
