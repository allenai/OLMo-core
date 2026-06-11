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

The ideas currently split into two rough families:

- Adaptive compute:
  - conditional compute / variable `top_k`;
  - expert path dropout / routing robustness;
  - adaptive expert depth / early-exit MoE;
  - router as learned curriculum.
- Adaptive modularity and capacity:
  - progressive / growing MoE;
  - expert lifecycle / birth-death;
  - document/token expert pools and emergent modularity;
  - universal expert pools.

The current discussion focus is the adaptive-compute family, especially adaptive
numbers of experts per token or sequence and expert dropout for robustness.

### 1. Conditional Compute / Variable Expert Count

Core idea:

Different tokens or sequences use different numbers of experts. Easy tokens
spend less compute; hard or uncertain tokens spend more. This could either save
compute at similar quality or improve quality at matched average compute by
allocating extra experts to the tokens that most need them.

Related work:

- [AdaMoE: Token-Adaptive Routing with Null Experts](https://arxiv.org/abs/2406.13233)
- [Mixture-of-Experts with Expert Choice Routing](https://arxiv.org/abs/2202.09368)
- [Route Experts by Sequence, not by Token](https://arxiv.org/abs/2511.06494)
- [DynMoE: Towards Resource-Efficient Mixture of Experts for Multitask Learning](https://arxiv.org/abs/2405.14297)
- [DynaMoE: Dynamic Token-wise Expert Activation](https://arxiv.org/abs/2603.01697)

Related-work takeaways:

- Null experts are a clean way to express "up to `k` real experts" while
  preserving a fixed router top-k interface. If a selected expert is null and
  returns zero, the token effectively uses fewer real experts.
- Expert Choice Routing lets experts choose tokens rather than tokens choosing
  experts, which naturally creates variable token fan-in and strong load
  control. For autoregressive LM pretraining, the main caveat is avoiding
  sequence/batch-level future-token leakage.
- Sequence-level top-k budgets are especially relevant. A sequence can receive a
  fixed total number of token-expert assignments, e.g. `T * K`, while individual
  tokens receive fewer or more experts. This directly tests "work harder on
  harder tokens" at fixed average compute.

Implementation notes for this codebase:

- The current MoE v2 path is fixed-shape around `top_k`: the router returns
  `[B, S, K]`, permute duplicates exactly `tokens * K`, and combine expects
  `[tokens, K]` weights.
- Null experts are therefore mostly a modeling interface unless dispatch can
  skip null slots. Dispatching null slots like real experts preserves shape but
  does not save compute.
- Actual compute savings require filtering null or pruned slots before expert
  dispatch, then combining a variable number of routes per token or using a
  compacted fixed-shape representation.
- Starting from a higher `k_max`, such as the `top_k = 8` small-expert geometry,
  is attractive because the marginal later experts may be lower-value and easier
  to replace with null/skip decisions.

Null-expert semantics to test:

- Renormalized null experts:
  - route among real plus null experts;
  - remove null slots before expert compute;
  - renormalize selected real expert weights;
  - interpretation: fewer mixture components at roughly matched MoE output
    scale.
- Non-renormalized null experts:
  - keep null probability mass as zero output;
  - interpretation: fewer experts and possibly a smaller MoE update.

Why tokens would choose null experts:

- Without an incentive, null selection may be weak. If null mass is renormalized
  away and there is no compute penalty, choosing a null only helps if the
  marginal real expert would hurt loss.
- More reliable incentives include:
  - a compute penalty on real expert count;
  - a target average real expert count;
  - a learned or scheduled null bias;
  - pruning selected experts whose normalized weights are below a threshold;
  - allowing null mass to reduce MoE output scale.

Candidate policies:

- Router entropy controls real `top_k`.
- Token loss proxy controls real `top_k`.
- Sequence-level budget chooses `T * K_avg` assignments across a sequence.
- Prune selected experts whose normalized weights fall below a threshold.
- Learned compute budget with an auxiliary compute penalty.
- Null experts with an average-real-expert target.

First runnable versions:

- Diagnostic fixed-shape version:
  - route top `k_max`, e.g. 8;
  - compute all selected experts;
  - log how many experts would survive threshold/null rules;
  - use this to estimate potential compute savings without changing kernels.
- Simple adaptive-compute version:
  - route top `k_max`;
  - prune slots below a normalized-weight threshold;
  - renormalize surviving weights;
  - compact dispatch so null/pruned slots are not computed;
  - compare to fixed top-`k_max` and fixed top-4 at matched average compute.
- Null-expert version:
  - add `N_null` router outputs with no corresponding expert MLP;
  - train with a target average real expert count;
  - test renormalized vs non-renormalized null semantics.

Metrics:

- Loss at matched average FLOPs and matched total tokens.
- Average real experts per token and per layer.
- Distribution of real expert count by token type, position, domain, document,
  and token loss.
- Whether high-compute tokens actually correspond to harder examples.
- Potential versus realized wall-clock savings.
- Inference behavior under the same pruning/null policy used during training.

Open questions:

- Should inference use the same threshold/null policy, or should it use a fixed
  budget chosen to hit a latency target?
- Does pruning low-weight experts during training make the model robust to using
  fewer experts at inference, or does it mostly train around the specific
  threshold?
- Are sequence-level budgets easier to make efficient than per-token ragged
  budgets?

### 2. Expert Path Dropout / Routing Robustness

Core idea:

Randomly perturb or drop routed experts during training so tokens cannot depend
on a brittle expert path. This is a robustness and regularization idea rather
than primarily a compute-savings idea.

Why it may matter:

- It discourages permanent route lock-in.
- It makes backup expert paths stronger.
- It may reduce forgetting when old paths are masked or stale.
- It may reduce overfitting in multi-epoch settings by preventing repeated
  examples from always training through the exact same expert path.
- It pairs well with document expert pools, expert growth, pruning, and
  continual learning.

Related work:

- [Gating Dropout: Communication-Efficient Regularization for Sparsely Activated Transformers](https://arxiv.org/abs/2205.14336)
- [Taming Sparsely Activated Transformer with Stochastic Experts](https://arxiv.org/abs/2110.04260)
- [ST-MoE: Designing Stable and Transferable Sparse Expert Models](https://arxiv.org/abs/2202.08906)
- [Sparse MoE as the New Dropout: Scaling Dense and Self-Slimmable Transformers](https://arxiv.org/abs/2303.01610)

Related-work takeaways:

- Routing perturbations can act as regularization, but the evidence is mixed.
- ST-MoE is an important caution: some dropout/noise choices improve stability
  or fine-tuning generalization but hurt pretraining quality.
- Stochastic expert methods suggest that learned routing is not always
  necessary for strong sparse models, but they are more radical than the first
  experiment needed here.

Dropout variants:

- Chosen-path dropout with replacement:
  - compute top `k + r` candidates;
  - normally use top `k`;
  - with probability `p`, drop one selected expert and replace it with the
    next-best backup;
  - renormalize weights;
  - active compute remains exactly `k`.
- Chosen-path dropout without replacement:
  - compute top `k`;
  - drop one selected expert;
  - use `k - 1` real experts;
  - useful later for compute reduction, but confounds robustness with lower
    active compute.
- Pool dropout with replacement:
  - mask a subset of the available expert pool before routing;
  - route top `k` among the remaining experts;
  - active compute remains exactly `k`;
  - tests global redundancy and pruning/fault tolerance.

Recommended first sequence:

1. Start with chosen-path dropout with replacement, because it directly tests
   backup-path robustness with constant per-token compute.
2. Compare dropping the lowest-weight selected expert versus dropping a uniform
   selected expert.
3. Add pool dropout with replacement if chosen-path dropout shows any useful
   robustness or overfitting signal.

Initial experiment matrix:

| Variant | Description | Compute confound? |
| --- | --- | --- |
| Baseline | Normal fixed top-`k` routing | No |
| Chosen dropout `p=0.05` | Drop one selected expert, replace with next-best | No |
| Chosen dropout `p=0.10` | Same, stronger perturbation | No |
| Pool dropout `p=0.05` | Mask available experts before routing, still choose top-`k` | No |

Metrics:

- Normal training and validation CE.
- Loss under evaluation-time route perturbation.
- Loss under expert pruning or expert masking.
- Expert load entropy, dead experts, and overused experts.
- Router entropy and route stability for a fixed probe set across checkpoints.
- Backup quality: loss delta when replacing the top, random, or lowest-weight
  chosen expert with the next-best candidate.

Multi-epoch overfitting test:

- Choose a fixed subset of the data, e.g. 50B-100B unique tokens.
- Train for multiple epochs over the same subset, e.g. `1x`, `2x`, `4x`, and
  possibly `8x`.
- Track both unique tokens and total tokens seen.
- Compare baseline, chosen-path dropout, and pool dropout.
- Evaluate on held-out data from the same mix and on the normal broad
  validation/eval suite.

The specific hypothesis is that repeated exposure may cause the router to
memorize stable token/domain-to-expert paths. If so, assignments on a fixed
probe set should become sharper and more stable across epochs, and route
perturbation should hurt more. Expert dropout should reduce that brittleness or
at least reduce the eval-time degradation under route perturbation.

### 3. Adaptive Expert Depth / Early-Exit MoE

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

### 4. Router As Learned Curriculum

Core idea:

Use router uncertainty and expert usage to decide where the model should spend
more training compute or replay.

Possible signals:

- High router entropy.
- High token loss.
- Unstable expert assignment across epochs.
- Rare-domain expert concentration.

This links conditional compute to data selection and continual learning.

### 5. Progressive / Growing MoE

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

### 6. Expert Lifecycle / Birth-Death

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

### 7. Document/Token Expert Pools And Emergent Modularity

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

### 8. Universal Expert Pools

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

## Suggested Planning Order

### Standardization Wave

1. Expert geometry at fixed active and total params.
2. Dense/MoE schedule plus shared expert interaction.
3. Width/depth at fixed active and total params.
4. Total sparsity increase at fixed active params.
5. Attention schedule and hybrid attention.

### Research Wave

This list is not currently prioritized. The research ideas should be ordered
after the standardization wave based on which baseline architectures perform
well and which implementation paths are cheapest to test first.

- Adaptive expert count / conditional compute.
- Expert path dropout and pruning robustness.
- Document expert pools / emergent modularity.
- Progressive/growing MoE.
- Universal expert pools.
- Expert lifecycle / birth-death.
- Adaptive expert depth and recurrent universal experts.
- Router-as-curriculum.

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
