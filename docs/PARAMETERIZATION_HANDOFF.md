# Handoff: implementing a new parameterization (init + per-layer optimizer settings)

Written 2026-10-17 against commit `9059a225c` (branch `tanushy/hero-12p4t-mt20-20261012`).
Line numbers refer to that commit; re-grep before editing. Statements marked **(verified)** were
confirmed by running code (see [Section 6](#6-verification-recipes)), not only by reading it.

Notation: `std` = `init_std`, `d` = `d_model`, `L` = number of layers (`n_layers`),
`l` = block index (0-based), `in` = the layer's input (fan-in) dimension.

---

## 0. Summary

There is **no muP-style machinery** in the codebase (no width multipliers, no per-layer LR
multipliers, no output/residual multipliers). A "parameterization" here is built from three
independent levers, each implemented in a different place:

1. **Initialization**: the `InitMethod` enum in `src/olmo_core/nn/transformer/init.py`, plus
   per-module `init_weights` methods that branch on it.
2. **Per-parameter-group optimizer hyperparameters**: `OptimGroupOverride` (fnmatch globs over
   parameter names, mapped to *absolute* option values such as `lr` and `weight_decay`).
3. **Forward-pass multipliers**: for the standard and hero models, only `embed_scale`, the
   attention `softmax_scale` (constructor argument only, not on `AttentionConfig`), and the
   learned per-head `scalable_softmax` scale. The nGPT variant (`TransformerType.normalized`)
   has its own learned scales, e.g. in `NormalizedLMHead`.

Gotchas, in rough order of how likely they are to bite (each detailed below):

1. In the DDP (hero) path, the optimizer sees parameter names prefixed with `module.`. The hero's
   `embeddings.weight` override therefore matches nothing, and embeddings train with
   `weight_decay=0.1` instead of the intended 0.0 **(verified)**. The only signal is a log warning.
2. Overlapping override patterns put one parameter in multiple groups. `build_groups` does not
   reject this **(verified)**, and `OLMoDDPOptimizer` has no duplicate check.
3. Override `lr` values are absolute. Hero scripts change `config.optim.lr` *after* the optimizer
   config is built, which does not update any group with a hard-coded `lr`.
4. Weight decay is applied as `p *= 1 - lr * weight_decay`, so scaling a group's LR also scales
   its effective decay.
5. `olmoe3_hero_mt.py` asserts that every logged per-group LR equals the single cosine schedule;
   any per-group LR multiplier will trip it.
6. Scheme coverage is inconsistent across modules (Section 1.5). In particular, KDA and
   GatedDeltaNet raise `NotImplementedError` for `fan_in`, and `normalized` does not touch MoE
   weights.
7. MoE-v2 expert weights are 3-D (or flattened 1-D for the router) with non-`nn.Linear` layouts.
   Fan-in cannot be read off a fixed dimension; existing code uses module attributes
   (`d_model`, `hidden_size`).

---

## 1. Initialization

### 1.1 Config surface

Fields on `TransformerConfig` (`src/olmo_core/nn/transformer/config.py:322`, fields at 339-347):

| Field | Default | Effect |
|---|---|---|
| `init_method` | `InitMethod.normal` | Selects the scheme. |
| `init_seed` | `0` | Seeds the `torch.Generator` passed to every scheme hook (and, offset, the EP generator). The earlier `reset_parameters` pass runs before this generator exists; its values are overwritten. |
| `init_std` | `0.02` | Base std passed to every hook as `std`. |
| `embedding_init_std` | `None` | Embedding std. **Only used by `normal`**; every other scheme ignores it. |
| `embed_scale` | `None` | Forward multiplier on embedding outputs. Also used by `fan_in` (embedding std = `1/embed_scale`). |
| `tie_word_embeddings` | `False` | If tied, the shared weight ends up with the embedding init. |

`OLMoDDPModelConfig` (`config.py:2022`) subclasses `TransformerConfig`; its `build()`
(`config.py:2053-2095`) forwards all of these to `OLMoDDPModel`.

`ExperimentConfig.init_seed` (`src/olmo_core/internal/experiment.py:104`, default 12536) is a
*different* seed (passed to `seed_all`). It is not the model's init generator seed.

Hero values (`src/examples/olmo_ddp/olmoe3_small_medium_models.py:263-266`):
`init_method=InitMethod.normal`, `init_seed=0`, `init_std=0.02`, `embed_scale=sqrt(d_model)`.

### 1.2 Call flow

**Generic `Transformer.init_weights`** (`src/olmo_core/nn/transformer/model.py:314-435`):

1. `self.to_empty(device=...)`.
2. Call `reset_parameters()` on every module that has one (norm gains become 1, `nn.Linear`
   gets PyTorch's default init, etc.).
3. Build `generator = torch.Generator(device).manual_seed(init_seed)`, offset by the PP rank and
   `model_part_idx` under pipeline parallelism.
4. `init_method.init_embeddings(...)`, then re-tie weights if tied.
5. For each block: `init_attention` (delegates to the mixer's own `init_weights`), then
   `init_feed_forward` if the block has `feed_forward`, `init_feed_forward_moe` if it has
   `feed_forward_moe`, and `init_moe_v2` if it is an `OLMoDDPTransformerBlock`.
6. `init_final_w_out` on `lm_head.w_out`, skipped when weights are tied.

**DDP `OLMoDDPModel.init_weights`** (`src/olmo_core/nn/ddp/model.py:883-990`) follows the same
pattern, with these differences:

- It casts the whole model to bf16 **before** `to_empty` (`ddp/model.py:908`), so init values are
  sampled into bf16 parameters.
- Seed offsets: `+ pp_rank + model_part_idx * 997` under PP. Under expert parallelism (EP), a
  separate `ep_generator` seeded with `seed + (1 + ep_mp_rank) * 653` is used **only** for routed
  expert weights.
- Every block gets `init_attention` + `init_moe_v2`.
- It always initializes `lm_head.w_out`, then re-ties if tied (net effect is the same as the
  generic path).
- In `OLMoDDPTrainModule`, EP is applied (`ddp_train_module.py:2716`) **before** weights are
  initialized (`init_model_weights` call at 2771), and DDP wrapping happens after
  (2787-2809). Routed experts are therefore already local shards when they are initialized.

Step 2 matters: any parameter no hook writes keeps its `reset_parameters` value. For the hero
model, every parameter is overwritten by either a scheme hook or a deliberate fixed rule; none
keeps PyTorch's default init and none is left uninitialized **(verified)**.

### 1.3 Hooks on `InitMethod` (`src/olmo_core/nn/transformer/init.py`)

| Lines | Name | Notes |
|---|---|---|
| 16-27 | `_apply_init(init_fun, x, ...)` | DTensor-safe: materializes the full tensor, applies `init_fun`, copies the local shard. Use it for new code. |
| 30-43 | `init_linear(m, std, generator)` | Truncated normal at +/-3 std; zeroes the bias. Works for `nn.Linear` and `nn.Conv1d`. |
| 46-77 | `class InitMethod(StrEnum)` | Members `normal`, `normalized`, `llama`, `llama_depth`, `fan_in`. |
| 79-105 | `init_embeddings` | `normal` uses truncated normal; all other schemes use a plain (untruncated) normal. |
| 107-122 | `init_final_w_out` | `normal`: `std`; every other scheme: `1/sqrt(d)`. |
| 124-141 | `init_attention` | Pure delegation to `m.init_weights(init_method=self, ...)`. |
| 143-180 | `init_feed_forward` | Dense FFN (`w1` gate, `w3` up, `w2` down). |
| 182-261 | `init_feed_forward_moe` | MoE v1 (`MoEBase`): router, latent projections, expert `w1/w2/w3`. |
| 263-380 | `init_moe_v2` | MoE v2 (`OLMoDDPTransformerBlock`): routers, latent projections, routed and shared experts. |

The abstract mixer hook is `SequenceMixer.init_weights` (`src/olmo_core/nn/attention/base.py:50`).

### 1.4 What each scheme does (hero model; verified)

| Parameters | `normal` (hero) | `normalized` | `llama` | `llama_depth` | `fan_in` |
|---|---|---|---|---|---|
| `embeddings` | std | 1/sqrt(d) | 1.0 | 1.0 | 1/`embed_scale` (1.0 if unset) |
| `lm_head.w_out` | std | 1/sqrt(d) | 1/sqrt(d) | 1/sqrt(d) | 1/sqrt(d) |
| KDA `w_q/w_k/w_v`, `f_proj_*`, `w_b`, `g_proj_*`, `q/k/v_conv1d` | std | 1/sqrt(d) | std | std | raises |
| KDA `w_out` | std | 1/sqrt(d * 2L) | std/sqrt(2L) | std/sqrt(2(l+1)) | raises |
| Attention `w_q/w_k/w_v/w_g` | std | 1/sqrt(d) | std | std | 1/sqrt(in) |
| Attention `w_out` | std | 1/sqrt(d * 2L) | std/sqrt(2L) | std/sqrt(2(l+1)) | 1/sqrt(in) |
| Router, `latent_down/up_proj`, routed `w_up_gate/w_down` | std | std | std/sqrt(2L) | std/sqrt(2(l+1)) | 1/sqrt(in) |
| Shared experts `w_up_gate/w_down` (including block 0's dense MLP) | std | std | std/sqrt(2L) | std/sqrt(2(l+1)) | std |

The `fan_in` column for KDA rows reflects the real behavior (`NotImplementedError`). The other
`fan_in` entries were verified by temporarily bypassing that error during the audit.

Things the table implies that are easy to miss:

- **MoE v2 depth scaling is global.** `init_moe_v2` reduces `std` once at the top
  (`init.py:284-287`), so under `llama`/`llama_depth` *every* MoE matrix is depth-scaled, not
  just output projections. Attention and KDA scale only `w_out`.
- **`normalized` never touches MoE weights.** Both `init_moe_v2` and `init_feed_forward_moe`
  lack a `normalized` branch, so MoE weights stay at the base `std`.
- **`fan_in` skips shared experts** (and `shared_experts_router`) in `init_moe_v2`; they keep `std`.
- **KDA convs are treated exactly like q/k/v**: never depth-scaled or fan-in scaled. Their true
  fan-in is `conv_size` (4), because they are depthwise (`groups=hidden_size`,
  `src/olmo_core/nn/convolution.py:47`).

Dense FFN (generic `Transformer`, verified on `olmo2_30M`):

| Parameter | `normal` | `normalized` | `llama` | `llama_depth` | `fan_in` |
|---|---|---|---|---|---|
| `w1` (gate) | std | 1/sqrt(d) | std | std | 1/sqrt(in) |
| `w3` (up) | std | 1/sqrt(d) | std/sqrt(2L) | std/sqrt(2(l+1)) | 1/sqrt(in) |
| `w2` (down) | std | 1/sqrt(d * 2L) | std/sqrt(2L) | std/sqrt(2(l+1)) | 1/sqrt(in) |

`w2`'s `llama`/`llama_depth` values come from reusing the `std` variable already reduced for
`w3` (`init.py:163-180`), not from an explicit `w2` branch. Preserve this if you refactor.

### 1.5 Scheme support per module

| Module | File:line | Notes |
|---|---|---|
| `Attention` | `nn/attention/__init__.py:1134` | All 5 schemes. Also sets `ssmax_scale = 1`; `sinks` (if enabled) ~ N(0, final `w_out` std). |
| `FusedAttention` | `nn/attention/__init__.py:1538` | All 5 (`w_qkv`, `w_out`). |
| `FusedAttentionV2` | `nn/attention/__init__.py:1894` | All 5 (`w_qkv`, `w_g`, `w_out`). |
| `KimiDeltaAttention` | `nn/attention/kda.py:245` | `fan_in` raises. `A_log`, `dt_bias`, `o_norm` are fixed (Section 1.6). |
| `GatedDeltaNet` | `nn/attention/recurrent.py:241` | `fan_in` raises. |
| `NemotronMamba2Mixer` | `nn/attention/recurrent.py:818` | No `fan_in` branch: silently uses base `std`. Under `normalized`, `in_proj` stays at `std` (no `1/sqrt(d)`) and `out_proj` gets `std/sqrt(2L)`. Ends by calling its own `reset_parameters`. |
| `NemotronMoE` | `nn/moe/v2/nemotron.py:289` | Discards `init_method`, `d_model`, `block_idx`, `num_blocks`; uses only the base `std`. |
| `NormalizedTransformer` | `nn/transformer/model.py:1207` | Calls the parent `init_weights`, then `normalize_matrices()`. |

### 1.6 Fixed regardless of scheme (verified)

- **Norm gains = 1**, via `LayerNorm.reset_parameters` (`nn/layer_norm.py:222-224`):
  `embedding_norm`, `attention_norm`, `attention_input_norm`, `feed_forward_norm`,
  `feed_forward_input_norm`, attention `q_norm`/`k_norm`, `lm_head.norm`. KDA's `o_norm` is FLA's
  `FusedRMSNormGated`, reset explicitly in `kda.py:279`.
- **KDA `A_log`** = log(Uniform(1, 16)) (`kda.py:276`).
- **KDA `dt_bias`** = 0, via a direct `.zero_()` (`kda.py:278`), not `nn.init`.
- **Attention `ssmax_scale`** = 1 (`nn/attention/__init__.py:1146-1147`).
- **All biases** that go through `init_linear` = 0. In the hero model this is only KDA's
  `g_proj_2.bias`.

### 1.7 Hero parameter inventory (small, verified)

Geometry (`olmoe3_small_medium_models.py:74-82`): d=1024, L=16, 8 heads, 4 KV heads,
head_dim=128, 512 routed experts, top-k 16, latent_dim=512. Block 0 is KDA plus a dense shared
MLP (hidden 8d). Blocks 7 and 15 use gated full attention with per-head QK-norm gains (set in
`olmoe3_small_integration.py:220-222`). All other blocks are KDA + MoE. Parameter counts:
794,233,472 active, 12,496,341,632 total (these match the hero's own assertion at
`olmoe3_small_hero.py:367`).

Names below are as returned by `model.named_parameters()`. The optimizer sees them with a
`module.` prefix (Section 2.6).

| Name (per block unless top-level) | Shape (small) | Fan-in | Notes |
|---|---|---|---|
| `embeddings.weight` | [100352, 1024] | n/a | |
| `embedding_norm.weight`, `lm_head.norm.weight` | [1024] | n/a | gain 1 |
| `lm_head.w_out.weight` | [100352, 1024] | d | |
| KDA `w_q`, `w_k` | [1024, 1024] | d | |
| KDA `w_v` | [2048, 1024] | d | `expand_v=2` |
| KDA `f_proj_1` / `f_proj_2` | [256, 1024] / [1024, 256] | d / 256 | low-rank decay projection |
| KDA `w_b` | [8, 1024] | d | |
| KDA `q/k/v_conv1d.weight` | [1024 or 2048, 1, 4] | 4 (depthwise) | |
| KDA `g_proj_1` / `g_proj_2` (+bias [2048]) | [256, 1024] / [2048, 256] | d / 256 | output gate |
| KDA `A_log` / `dt_bias` / `o_norm.weight` | [8] / [1024] / [256] | n/a | fixed |
| KDA `w_out` | [1024, 2048] | 2048 | |
| Attn `w_q`, `w_g` | [1024, 1024] | d | elementwise gate |
| Attn `w_k`, `w_v` | [512, 1024] | d | |
| Attn `w_out` | [1024, 1024] | 1024 | |
| Attn `q_norm` / `k_norm` | [8, 128] / [4, 128] | n/a | per-head gains, 2-D |
| Attn `ssmax_scale` | [8] | n/a | fixed |
| `latent_down_proj` / `latent_up_proj` | [512, 1024] / [1024, 512] | d / 512 | MoE blocks only |
| `routed_experts_router.weight` | [524288] = E * d | d | **flattened 1-D** (`moe/v2/router.py:265`) |
| `routed_experts.w_up_gate` | [E, 2H, D] = [512, 2048, 512] | D = latent (last dim) | `moe/v2/routed_experts.py:495` |
| `routed_experts.w_down` | [E, H, D] = [512, 1024, 512] | H (dim 1) | |
| `shared_experts.w_up_gate` | [D, 2H] = [1024, 2048]; block 0: [1024, 16384] | D (**dim 0**) | `moe/v2/shared_experts.py:102` |
| `shared_experts.w_down` | [1, H, D] = [1, 1024, 1024]; block 0: [1, 8192, 1024] | H (dim 1) | |
| 4 peri-norms per block | [1024] | n/a | gain 1 |

Consequences for a new scheme:

- **Fan-in is not a fixed dimension** across expert tensors (last dim, dim 1, or dim 0). Existing
  code uses module attributes (`routed_experts.d_model`, `.hidden_size`,
  `routed_experts_router.d_model`, `projection.in_features`). Do the same.
- **Shape-based classification breaks.** The router weight is 1-D, so any rule keyed on `ndim`
  (e.g. `MatrixAwareOptimConfig.categorize_parameters`) treats it as a vector. QK-norm gains are
  2-D, so an `ndim == 2` rule treats them as matrices.

### 1.8 Forward-pass multipliers that exist today

- **`embed_scale`**: applied as `h = embeddings(ids) * embed_scale`, then `embedding_norm`
  (`ddp/model.py:1045-1049` and `1355-1362`; generic model at `model.py:720-724`).
  Derived, not tested: because an RMSNorm with gain 1 immediately follows, `embed_scale` has
  essentially no effect on the hero's forward pass or embedding gradients, apart from interaction
  with the norm's `eps`.
- **Attention `softmax_scale`**: an `Attention.__init__` argument (`nn/attention/__init__.py:613`)
  that is **not** a field on `AttentionConfig` (`:324`). `None` means the backend default.
  `NormalizedAttention` passes `sqrt(head_dim)` (`:1259`). Making it configurable (e.g. a
  `1/head_dim` scale) requires adding a config field.
- **`scalable_softmax`**: a learned per-head scale (`ssmax_scale`), initialized to 1.
- **Not present**: logit multipliers on the LM head, residual-branch multipliers, per-block
  output multipliers. A logit multiplier would have to cover every path in `LMHead.forward`
  (`nn/lm_head.py:201`): the no-labels path (`:246`), `default` (`:253`), and `fused_linear`
  (`:267-283`, which passes `w_out.weight` straight to the fused kernel). `NormalizedLMHead`
  (`:429`, `logits = sz * self.w_out(x)` at `:491`) is the only existing precedent. The DDP model
  calls `self.lm_head(...)` (`ddp/model.py:1212`, `1442`), so it uses the same paths.

---

## 2. Per-layer optimizer hyperparameters

### 2.1 `OptimGroupOverride` and `build_groups`

`OptimGroupOverride(params: List[str], opts: Dict[str, Any])` (`src/olmo_core/optim/config.py:43-54`).

Semantics (base `OptimConfig.build_groups`, `config.py:120-154`; DDP version in
`optim/moe_optimizer.py:224-291`):

- Each pattern is matched with `fnmatch` against `named_parameters()` names, considering only
  `requires_grad=True` parameters. `fnmatch` is a full-string match, so `*` is needed for any
  prefix.
- Each override becomes one param group: `{"params" or "named_params": [...], **opts}`.
  Unmatched parameters form the default group. Empty groups are dropped.
- A pattern that matches nothing raises under `strict=True` and only logs a warning under
  `strict=False` (`config.py:104-116`; `moe_optimizer.py:208-220`). Patterns matching only frozen
  parameters log a different warning.
- **Overlaps are not deduplicated.** A parameter matching two overrides lands in both groups
  (`OLMoDDPOptimizerConfig.build_groups`, **verified**). Standard `torch.optim` optimizers raise
  on this when constructed. `OLMoDDPOptimizer` contains no duplicate check (by inspection); its
  per-parameter state is keyed by name, so the parameter would presumably be stepped twice
  (not tested).
- Any key missing from `opts` falls back to the optimizer's defaults (the config's `lr`, `betas`,
  `eps`, `weight_decay`). Unknown keys are carried along silently. For example, `use_muon` in
  `OLMoE3-dev-*.py` is never read by `OLMoDDPOptimizer`.
- Group order: the base `OptimConfig` puts overrides first and the default group last.
  `OLMoDDPOptimizerConfig` puts the default group first (`moe_optimizer.py:277`), and builds the
  dense (`pg="dp"`) partition before the expert-parallel (`pg="ep_dp"`) partition
  (`moe_optimizer.py:353-366`). Group indices in the `optim/LR (group i)` metrics follow this
  order.

### 2.2 How LR flows

- At build, each group's `lr` is set (from the group's opts, else the config) and
  `initial_lr` defaults to that value (`config.py:181-196`; `moe_optimizer.py:391-406`).
- Each step, the train module calls `scheduler.set_lr(group, trainer)` for **every** group
  (`ddp_train_module.py:2073-2076`; `train_module.py:574-575`). `Scheduler.set_lr`
  (`optim/scheduler.py:52-102`) computes `new_lr = get_lr(group["initial_lr"], step, max)`, so
  per-group LR ratios are preserved through warmup and decay.
- `BatchSizeSchedulerCallback` multiplies every group's `initial_lr` by the same factor
  (`train/callbacks/batch_size_scheduler.py:202-221`).
- Override `lr` values are **absolute**. Hero scripts set `config.optim.lr = LR` late
  (`olmoe3_small_integration.py:229`, `olmoe3_small_hero.py:283`, `olmoe3_hero_mt.py:181`).
  Per-group LRs must be computed after the final base LR is known, or expressed as multipliers
  resolved at `build()` time. Multipliers would be new code; nothing like that exists today.

### 2.3 Weight decay is coupled to LR

`foreach_adamw_step` does `p *= 1 - step_factor * (lr * weight_decay)`
(`src/olmo_core/optim/adamw.py:71`), and `OLMoDDPOptimizer._step_foreach` uses it with the
group's `lr`, `betas`, `eps`, `weight_decay` (`moe_optimizer.py:2093-2105`). If a scheme wants
LR-independent decay, set each group's `weight_decay` to compensate.

### 2.4 Other optimizer-level facts

- Gradient clipping uses one global `max_grad_norm` with a single clip coefficient
  (`moe_optimizer.py:1078`), not per group.
- `OLMoDDPOptimizer.step` always goes through `_step_foreach` (`moe_optimizer.py:1232`).
- Muon/Dion (`MatrixAwareOptimConfig`, `optim/config.py:230-399`) build their own groups
  (embed / matrix / vector / lm_head) and **reject** manual `group_overrides`. Their
  `categorize_parameters` asserts the model is a `Transformer` and has no >2-D parameters, so they
  cannot be used with the hero's 3-D expert tensors. Dion already sets `lm_head lr = lr / sqrt(d)`
  (`optim/dion.py:95-99`).

### 2.5 Checkpoint interaction

- **`OLMoDDPOptimizer`**: `state_dict()` (`moe_optimizer.py:2223-2276`) contains only
  per-parameter states keyed by `"{param_name}.{suffix}"`, plus skip-step histories. No group
  hyperparameters are saved, so on resume every group's settings come from the current config,
  and moving a parameter between groups does not affect its saved state. The post-load
  `reset_fixed_fields` hook is commented out (`moe_optimizer.py:437`) and is unnecessary for this
  reason.
- **Generic `OptimConfig` path**: group hyperparameters are part of the torch optimizer state. A
  post-load hook re-applies only `fixed_fields` (default: `initial_lr`) from the config
  (`config.py:220-225`); other group fields come from the checkpoint. Verify this before changing
  group settings mid-run on that path.

### 2.6 The hero's actual optimizer config (verified)

Config chain: `olmoe3_hero_mt.train_module_config` -> `olmoe3_small_hero.train_module_config`
-> `olmoe3_small_integration.train_module_config` ->
`olmoe3_small_medium_profile.build_train_module_config` (`:232-272`), which defines:

```python
OLMoDDPOptimizerConfig(
    lr=LEARNING_RATE,            # later overwritten by each wrapper above
    weight_decay=0.1,
    betas=(0.9, 0.95),
    group_overrides=[
        OptimGroupOverride(params=["embeddings.weight"], opts={"weight_decay": 0.0}),
        OptimGroupOverride(params=["*routed_experts.w_up_gate", "*routed_experts.w_down"], opts={}),
    ],
    compile=True, dtype=DType.float32, sigma_factor=6, max_grad_norm=1.0, use_distributed=True,
)
```

- The optimizer is built on the `MultiGroupDistributedDataParallel`-wrapped model
  (`ddp_train_module.py:292`, `342-346`, `strict=False`). The wrapper stores the model as
  `self.module` (`nn/parallel/distributed.py:133`) and does not override `named_parameters`, so
  names are `module.embeddings.weight`, `module.blocks.1.routed_experts.w_up_gate`, and so on.
- **Result:** `embeddings.weight` matches nothing. The log shows `optim group 0 override pattern
  'embeddings.weight' does not match any parameters`, and embeddings stay in the default group
  with `weight_decay=0.1`. Use `*embeddings.weight` (or `module.embeddings.weight`). This has not
  been fixed, because changing it alters the production hero runs.
- The `*routed_experts.*` patterns do match (leading `*`), creating a second group with
  default hyperparameters.
- The small hero runs with `ep_config=None` (asserted at `olmoe3_small_hero.py:372`), so all
  parameters are in the `dp` partition. Group 0 is the default group; group 1 is the routed
  experts (2 tensors x 15 MoE blocks).
- `OLMoDDPOptimizerConfig.build` logs every group's fields **and parameter names**
  (`moe_optimizer.py:412-427`). Use this log to confirm grouping on real runs.
- `olmoe3_hero_mt.py:166-171` (`log_metrics`) asserts that every `optim/LR (...)` metric equals
  `CosWithWarmup(warmup=WARMUP, alpha_f=0.0).get_lr(LR, step, END)`. Per-group LR multipliers
  require updating this audit.

---

## 3. Design decisions to settle with the user before implementing

1. **Which levers does the scheme use?** Init only; init + per-group LR/WD/eps; or also forward
   multipliers (which need new config fields and model code; Section 1.8).
2. **Which model paths?** DDP MoE-v2 hero model (`init_moe_v2`, KDA, `Attention`), generic
   `Transformer` (`init_feed_forward`, `init_feed_forward_moe`), or both.
3. **Fixed parameters**: keep norm gains, `A_log`, `dt_bias`, `ssmax_scale` and biases on their
   fixed rules, or make them scheme-dependent?
4. **KDA convs**: what should their rule be (true fan-in is 4)?
5. **Base-width reference**: muP-style schemes need a base shape (e.g. `base_d_model`). Decide
   where it lives, most likely a new field on `TransformerConfig`, and how optimizer groups learn
   it.
6. **Weight decay**: coupled to LR (current behavior) or LR-independent.
7. **Resumed runs**: init has no effect on runs that load a checkpoint (all current hero
   midtraining/decay runs). LR and WD changes take effect immediately on resume in the DDP path
   (Section 2.5).

## 4. Implementation checklist

**Initialization**

- [ ] Add a member to `InitMethod` and document it in the enum docstring
      (or agree with the user on a different structure).
- [ ] Add branches in every hook the target model reaches: `init_embeddings`,
      `init_final_w_out`, `init_feed_forward`, `init_feed_forward_moe`, `init_moe_v2`.
- [ ] Add branches in every sequence mixer the target model uses (Section 1.5). For the hero:
      `KimiDeltaAttention.init_weights` and `Attention.init_weights`. Raise
      `NotImplementedError` for unsupported mixers rather than silently falling through.
- [ ] Watch the `std`-variable reuse in `init_feed_forward` and the single top-of-function `std`
      reduction in `init_moe_v2`; both make branch order significant.
- [ ] Use `init_linear` / `_apply_init` so DTensor-sharded parameters stay consistent.
- [ ] Pass `generator` everywhere, and `ep_generator` for routed experts in `init_moe_v2`.
- [ ] Compute fan-in from module attributes, not tensor shapes (Section 1.7).

**Optimizer groups**

- [ ] Generate overrides from actual parameter names, with patterns that tolerate the
      `module.` prefix (leading `*`).
- [ ] Make patterns disjoint, or add an overlap check; nothing enforces this today.
- [ ] Compute group LRs from the *final* base LR, or add multiplier support resolved in `build()`.
- [ ] Decide on WD compensation (Section 2.3).
- [ ] Update `olmoe3_hero_mt.py`'s per-group LR assertion if the scheme is used there.

**Forward multipliers (only if needed)**

- [ ] Add config fields, plumb them through `TransformerConfig.build` and
      `OLMoDDPModelConfig.build` (`config.py:2074-2095`), and cover every `LMHead.forward` path.

## 5. Tests

- `src/test/nn/transformer/init_test.py`: pattern for checking per-parameter empirical std
  against an expected formula with `TransformerConfig.llama_like` on CPU and meta devices
  (e.g. `test_fan_in_init_dense`, `test_fan_in_init_moe`, `test_fan_in_init_latent_moe`), plus
  seed tests for PP model parts.
- `src/test/nn/ddp/block_test.py:146` (`test_latent_block_projection_initialization`): calls
  `InitMethod.fan_in.init_moe_v2(...)` directly on a single DDP block.
- `src/test/optim/moe_optimizer_test.py`: `build_groups` with a `group_overrides` glob.
- Follow AGENTS.md: `pytest.mark.parametrize` across schemes and devices, `@pytest.mark.gpu` for
  GPU-only tests.

## 6. Verification recipes

These scripts lived in `/tmp/init_audit/` and are not checked in; the techniques are
straightforward to recreate.

**Per-parameter init audit.** Build a scaled-down hero config with the real block types and layout:
patch `olmoe3_small_medium_models` with `NUM_EXPERTS=32`, `TOP_K=4`, a tiny `Geometry`
(d_model 256, 8 layers, 2 heads, 1 KV head) and `use_cute_kernel=False` (`kernel_fun` may be
missing; it does not affect init). Then:

1. Wrap `torch.nn.init.{trunc_normal_, normal_, ones_, zeros_, uniform_, kaiming_uniform_}`
   to record `(function, std, caller)` keyed by `tensor.data_ptr()`.
2. Wrap `model.to_empty` so that right after allocation it fills every parameter with NaN and
   clears the records.
3. Call `model.init_weights(device=cuda)` for each `InitMethod` and report each parameter's
   last writer, std argument and empirical std. A surviving NaN means uninitialized; a value with
   no record means it was written outside `nn.init` (only KDA `dt_bias`).
4. For `fan_in`, temporarily make `KimiDeltaAttention.init_weights` map `fan_in` to `normal`, so
   the other `fan_in` branches can be exercised.

**Optimizer-group audit.** In a single process, run
`dist.init_process_group("nccl", world_size=1, rank=0, ...)`, init the tiny model, cast it to
bf16, and wrap it with `MultiGroupDistributedDataParallel(model, init_sync=False)`. Then call
`OLMoDDPOptimizerConfig(...).build_groups([wrapped], strict=False)` with the candidate overrides
and inspect each group's `named_params` and options.

**Real shapes and parameter counts.** Build `build_model_config("small", ...)` with the
integration's `qk_norm_per_head_gains=True` tweak on `init_device="meta"` (no memory needed),
and compare `num_active_params` / `num_params` with `olmoe3_small_hero.py:367`.
