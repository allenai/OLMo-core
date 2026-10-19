# Handoff: NanoChat initialization (`InitMethod.nanochat`)

Written 2026-10-17 against commit `9059a225c` (branch `tanushy/hero-12p4t-mt20-20261012`) plus
the uncommitted changes listed in [Section 3](#3-files-changed). **Nothing has been committed.**
Line numbers refer to the working tree at the time of writing; re-grep before editing.

Read this together with `docs/PARAMETERIZATION_HANDOFF.md`, which describes the codebase's
parameterization machinery in depth. This doc only repeats what is needed to follow the change.

Notation: `d` = `d_model`, `fan_in` / `fan_out` = the number of input / output features of a
weight matrix, `s(fan_in, fan_out)` = the NanoChat std defined below.

---

## 0. Summary

`InitMethod.nanochat` is a new member of the existing `InitMethod` enum. It is **init only**: no
optimizer settings, no forward-pass multipliers, and no hero scripts were changed. Production
hero runs still use `InitMethod.normal`.

The rule, for every weight matrix:

```text
s(fan_in, fan_out) = (1 / sqrt(fan_in)) * min(1, sqrt(fan_out / fan_in))
```

This is the spectral-condition init from Yang, Simon & Bernstein, "A Spectral Condition for
Feature Learning" (arXiv:2310.17813), which nanochat's `_init_weights` cites. The Theta-constant
is 1, as in nanochat. In words: expanding or square matrices (fan_out >= fan_in) get the classic
`1/sqrt(fan_in)`; contracting matrices (fan_out < fan_in) get the smaller `sqrt(fan_out)/fan_in`.

To use it, set `init_method=InitMethod.nanochat` on the model config (a `TransformerConfig` or
`OLMoDDPModelConfig`). For the hero that would be in `build_model_config` in
`src/examples/olmo_ddp/olmoe3_small_medium_models.py`, or via a dotlist override such as
`--model.init_method=nanochat` if the launching script supports overrides.

---

## 1. What the scheme does

### 1.1 Rules

| Parameter kind | Rule |
|---|---|
| Any `nn.Linear` (attention, KDA, GDN, FFN, latent projections) | truncated normal (+/-3 std), std = `s(in_features, out_features)` |
| Fused linear (`w_qkv` in `FusedAttentionV2`) | each fused projection (Q, K, V row block) gets its own `s(in, rows_of_that_block)` |
| MoE routers (v1 and v2, routed and shared) | `s(router.d_model, router.num_experts)`; router bias = 0 |
| Routed / shared experts `w_up_gate` (fused gate + up) | `s(d_model, hidden_size)`: the fan-out of **each half**, not `2 * hidden_size` |
| Routed / shared experts `w_down` | `s(hidden_size, d_model)` |
| MoE v1 expert `w1`, `w3` / `w2` | `s(mlp.d_model, mlp.hidden_size)` / `s(mlp.hidden_size, mlp.d_model)` |
| Depthwise causal convs (KDA, GDN `q/k/v_conv1d`) | `1/sqrt(kernel_size)` (treated as square, fan_in = fan_out = kernel_size) |
| `embeddings.weight` | plain (untruncated) normal, std = 1.0 |
| `lm_head.w_out` | `s(d, vocab_size)`, which is `1/sqrt(d)` whenever vocab >= d |
| Attention `sinks` (if enabled) | normal with the same std as that layer's `w_out` (mirrors the existing coupling) |
| Norm gains, QK-norm gains, `ssmax_scale`, KDA/GDN `A_log` and `dt_bias`, biases | unchanged fixed rules (ones / log-uniform / zeros) |

Ignored inputs: `init_std` (the base std) and `embedding_init_std`, the same way `fan_in` ignores
them. `embed_scale` is also ignored for the embedding std.

No depth scaling and **no zero-init** (nanochat zero-inits the LM head and the residual output
projections; we deliberately do not; see [Section 4](#4-decisions-made-with-the-user)).

### 1.2 Resulting stds on the small hero (d = 1024)

Today's hero (`InitMethod.normal`) uses 0.02 for all of these.

| Parameter | fan_in -> fan_out | nanochat std |
|---|---|---|
| KDA `w_q`, `w_k`, `w_v` | 1024 -> 1024 / 2048 | 0.03125 |
| KDA `f_proj_1`, `g_proj_1` | 1024 -> 256 | 0.01562 |
| KDA `f_proj_2`, `g_proj_2` | 256 -> 1024 / 2048 | 0.06250 |
| KDA `w_b` | 1024 -> 8 | 0.00276 |
| KDA `w_out` | 2048 -> 1024 | 0.01563 |
| KDA `q/k/v_conv1d` | kernel 4 | 0.50000 |
| Attention `w_q`, `w_g` (elementwise), `w_out` | 1024 -> 1024 | 0.03125 |
| Attention `w_k`, `w_v` | 1024 -> 512 | 0.02210 |
| `routed_experts_router` | 1024 -> 512 experts | 0.02210 |
| `latent_down_proj` / `latent_up_proj` | 1024 -> 512 / 512 -> 1024 | 0.02210 / 0.04419 |
| `routed_experts.w_up_gate` / `w_down` | 512 -> 1024 / 1024 -> 512 | 0.04419 / 0.02210 |
| `shared_experts.w_up_gate` (all blocks) | 1024 -> 1024 or 8192 | 0.03125 |
| `shared_experts.w_down` (MoE blocks / block 0 dense MLP) | 1024 -> 1024 / 8192 -> 1024 | 0.03125 / 0.00391 |
| `embeddings` | n/a | 1.0 |
| `lm_head.w_out` | 1024 -> 100352 | 0.03125 |

These were confirmed on the real model; see [Section 6](#6-verification).

---

## 2. How initialization flows through the codebase

(Condensed from `PARAMETERIZATION_HANDOFF.md` Section 1.2.)

1. **Config.** `TransformerConfig.init_method` (`src/olmo_core/nn/transformer/config.py:339`)
   selects the scheme; `OLMoDDPModelConfig` subclasses it and forwards it in `build()`.
   `init_seed` seeds the init generator. Nothing about the config surface changed: adding an enum
   member is enough for YAML/JSON/dotlist configs to accept `"nanochat"`.
2. **Model driver.** Two `init_weights` drivers walk the model and call hooks on the enum:
   - Generic `Transformer.init_weights` (`src/olmo_core/nn/transformer/model.py:314`):
     `to_empty`, then `reset_parameters()` on every module, then `init_embeddings`, per block
     `init_attention` + `init_feed_forward` / `init_feed_forward_moe` / `init_moe_v2`, then
     `init_final_w_out` (skipped when embeddings are tied).
   - DDP/hero `OLMoDDPModel.init_weights` (`src/olmo_core/nn/ddp/model.py:883`): same idea, but
     it casts the model to **bf16 before `to_empty`** (values are sampled directly into bf16),
     calls `init_attention` + `init_moe_v2` for every block, uses a separate `ep_generator` for
     routed experts under expert parallelism, and always initializes `lm_head.w_out` before
     re-tying.
3. **Hooks** live on `InitMethod` in `src/olmo_core/nn/transformer/init.py`. `init_attention` is
   pure delegation to the sequence mixer's own `init_weights(init_method=..., ...)`, so every
   mixer class (`Attention`, `FusedAttentionV2`, KDA, GDN, Mamba2, NemotronMoE)
   has to branch on the scheme itself.
4. Any parameter no hook writes keeps its `reset_parameters()` value. That is how norm gains
   become 1, and it is also how a missed parameter silently keeps PyTorch's default init.

---

## 3. Files changed

`git status` also shows `src/examples/olmo_ddp/olmoe3_hero_mt.py` modified (a 3-line
reformatting) and `docs/PARAMETERIZATION_HANDOFF.md` untracked. **Both predate this work** and are
not part of it.

### 3.1 `src/olmo_core/nn/transformer/init.py` (the core change)

New module-level helpers:

| Line | Name | Purpose |
|---|---|---|
| 46 | `_init_trunc_normal(x, *, std, generator)` | Private shorthand for `_apply_init(nn.init.trunc_normal_, ...)` at +/-3 std. Used only by the new code. |
| 58 | `nanochat_std(fan_in, fan_out)` | The formula. Public, unit-tested. |
| 71 | `init_linear_nanochat(m, *, generator)` | `nn.Linear`: std from `in_features` / `out_features`. `nn.Conv1d`: must be depthwise (`groups == in_channels == out_channels`, else `NotImplementedError`), std = `1/sqrt(kernel_size)`. Delegates to `init_linear`, so the bias is zeroed and DTensors are handled. |
| 94 | `init_fused_linear_nanochat(m, out_features, *, generator)` | Splits the weight's rows into the given blocks and inits each with its own std. Raises `ValueError` if the sizes don't sum to `m.out_features`. Runs inside `_apply_init`, so it is DTensor-safe (the full tensor is materialized, then sharded). |

Enum and hooks:

| Line | Hook | Change |
|---|---|---|
| 156 | `InitMethod.nanochat` | New member with a docstring. |
| 177 | `init_embeddings` | Added to the `(llama, llama_depth)` tuple, which already did `normal_` with std 1.0. |
| 211 | `init_final_w_out` | New `elif`: `std = nanochat_std(m.in_features, m.out_features)`. |
| 244 | `init_feed_forward` | Early-return branch: `init_linear_nanochat` on `w1`, `w3`, `w2` (same order as the existing code). |
| 290 | `init_feed_forward_moe` (MoE v1) | Early-return branch: router, latent projections, expert `w1`/`w2`/`w3`, **and `shared_mlp`** (see Section 5.6). |
| 420 | `init_moe_v2` (DDP blocks) | Early-return branch placed **after** the `ep_generator` fallback: shared and routed routers (+ bias zeroing), latent projections, routed experts (with `ep_generator`), shared experts. |

Design choice: every nanochat branch is an **early return** that leaves the existing code for
other schemes untouched. The alternative (threading `elif nanochat` through each existing std
computation) would have interacted with the `std`-variable reuse in `init_feed_forward` and the
single top-of-function depth reduction in `init_moe_v2` (`PARAMETERIZATION_HANDOFF.md` Section
1.4). The cost is that the nanochat branch duplicates the "which tensors exist" traversal; see
Section 5.1.

### 3.2 `src/olmo_core/nn/attention/__init__.py`

| Line | Class | Change |
|---|---|---|
| 1154 | `Attention.init_weights` | Early-return branch after `ssmax_scale` is set to 1: `init_linear_nanochat` on `w_q`, `w_k`, `w_v`, `w_g` (if present), `w_out`; `sinks` (if present) get the `w_out` std. Inherited by `NormalizedAttention`. |
| 1941 | `FusedAttentionV2.init_weights` | Early-return branch: `w_qkv` split as `(q_dim, kv_dim, kv_dim)` (the order used by its forward, `qkv.split(...)`), then `w_g` and `w_out`. Under grouped-query attention, K and V get a smaller std than Q. |

### 3.3 `src/olmo_core/nn/attention/kda.py` (`KimiDeltaAttention.init_weights`, line 245)

The two existing loops (linears, then convs) were **merged into one loop with the same order**,
and each element branches between `init_linear_nanochat` and `init_linear`. The fixed-parameter
init (`A_log`, `dt_bias`, `o_norm`) stays exactly where it was. `w_out` gets a nanochat
early-return (line 289) just before the existing depth-scaling code. `fan_in` still raises.

### 3.4 `src/olmo_core/nn/attention/recurrent.py`

- `GatedDeltaNet.init_weights` (line 241): the same restructuring as KDA (merged loop at line
  276, `w_out` early-return at line 292). `A_log` / `dt_bias` unchanged. `fan_in` still raises.
- `NemotronMamba2Mixer.init_weights` (line 836): raises `NotImplementedError` for nanochat
  (line 848). Its `in_proj` fuses several projections of different widths and it ends by calling
  its own `reset_parameters`, so a correct implementation needs more thought.

### 3.5 `src/olmo_core/nn/moe/v2/nemotron.py`

`NemotronMoE.init_weights` (line 289) used to discard `init_method` entirely. It now raises
`NotImplementedError` for nanochat (line 299) instead of silently using the base std. Other
schemes are unchanged.

### 3.6 Tests

- `src/test/nn/transformer/init_test.py` (docstring updated to cover nanochat). Helpers
  `_assert_std` (line 261; empirical std within 10%) and `_assert_linear_nanochat` (line 268).
  Tests:
  - `test_nanochat_std`: square, expanding, contracting, narrow-output cases.
  - `test_nanochat_init_dense`: CPU and meta->CPU, d in {256, 512}, grouped-query attention
    (K/V contracting), vocab 128 < d (contracting LM head), and `init_std` /
    `embedding_init_std` set to 0.5 to prove they are ignored.
  - `test_nanochat_init_attention_gate`: headwise and elementwise `w_g`.
  - `test_nanochat_init_fused_attention`: v1, marked `@requires_flash_attn_2`, because v1's
    constructor rejects backends without packed-QKV support (the torch backend fails).
  - `test_nanochat_init_fused_attention_v2`: per-slice stds with grouped-query attention.
  - `test_nanochat_init_moe`: MoE v1 with and without latent projections, with `shared_mlp`.
  - `test_nanochat_init_kda` / `test_nanochat_init_gdn`: skipped without FLA. They check every
    linear, the convs (0.5), and that the fixed parameters are intact.
  - `test_nanochat_init_raises_for_unsupported_mixers`: Mamba2 and NemotronMoE, built on meta.
- `src/test/nn/ddp/block_test.py`: `test_block_nanochat_initialization` (line 193) calls
  `InitMethod.nanochat.init_moe_v2` on a DDP block with and without latent projections, with a
  shared expert.

### 3.7 `CHANGELOG.md`

New `### Added` entry under `## Unreleased`.

---

## 4. Decisions made with the user

The user approved the plan with these choices (recommended option chosen each time):

1. **Embedding std = 1.0** (nanochat-literal). Rejected: `1/embed_scale` like `fan_in`, and
   honoring an explicit `embedding_init_std`. The formula is meaningless for one-hot inputs. The
   hero normalizes embeddings immediately (`embedding_norm`), so this choice changes only
   optimizer dynamics, not the forward pass.
2. **No zero-init anywhere.** nanochat zero-inits the LM head and the attention/MLP output
   projections. The hero is **peri-norm** (`x + RMSNorm(f(x))`, `use_peri_norm=True`): with a
   zero output projection, RMSNorm(0) = 0 but its Jacobian there is about `1/sqrt(eps)` = 1000
   (eps = 1e-6). After one step the norm rescales the tiny output back to unit RMS, so zero-init
   buys nothing and makes the first step odd. Zero-init would be meaningful for pre-norm-only
   models; it could be added later as an option (Section 7).
3. **Depthwise convs: `1/sqrt(kernel_size)` = 0.5.** Rejected: keeping the base `init_std`
   (0.02) as every other scheme does. Each output channel sums `kernel_size` taps of a single
   channel, so the conv is treated as square with fan_in = kernel_size.
4. **Scope: init only.** Per-group learning rates (the other half of nanochat's
   parameterization) were offered as a follow-up and not requested.

Decisions I made without asking (all low-stakes; revisit if needed):

- Constant = 1 (nanochat's choice); `init_std` is ignored rather than used as a gain.
- Truncated normal at +/-3 std (codebase convention via `init_linear`) instead of nanochat's
  plain normal. This lowers the realized std by about 1.3%.
- Embeddings use a plain normal (as the `llama` / `fan_in` schemes already do).
- Fused matrices use the fan-out of each fused projection (per Q/K/V block, per gate/up half).
- Attention sinks keep the existing "same std as `w_out`" coupling.
- Mixers I did not implement raise rather than fall through to the base std.
- MoE v1 `shared_mlp` is initialized under nanochat (Section 5.6).

---

## 5. Peculiarities and things that may cause problems later

### 5.1 Adding a parameter to a module now needs two edits

Because nanochat branches are early returns that re-enumerate the tensors, a new weight added
to, e.g., `OLMoDDPTransformerBlock`, `Attention` or `MoEBase` must be added to **both** the
existing code path and the nanochat branch. If it is missed, it keeps its `reset_parameters()`
value with no error. The audit recipe in Section 6.3 catches this.

### 5.2 RNG order in KDA and GDN must be preserved

The merged loops in KDA and GDN were kept in exactly the old order so that existing schemes
produce **bit-identical** weights for a given seed (verified, Section 6.2). Do not move the
`A_log` / `dt_bias` init or reorder the projections, or `normal` / `llama` / ... inits change for
every seed. The same applies to the router order in `init_moe_v2` (shared router before routed).

### 5.3 Fan-in/fan-out come from module attributes, not shapes

The router weight is flattened 1-D (`[num_experts * d_model]`), routed `w_up_gate` is
`[E, 2H, D]` (fan-in in the last dim), shared `w_up_gate` is `[D, E*2H]` (fan-in in dim 0), and
`w_down` is `[E, H, D]`. The code uses `d_model`, `hidden_size`, `num_experts`, `in_features` /
`out_features`. Under latent MoE, `routed_experts.d_model` is the **latent** dim (512 for the
small hero), while the router's `d_model` is the model dim (1024). Shape-based rules
(`ndim == 2`, "last dim is fan-in") would get these wrong.

### 5.4 Tied embeddings give the LM head std 1.0

`_tie_weights` does `lm_head.w_out.weight = embeddings.weight`, so under tying the embedding's
std 1.0 wins. With a unit-RMS input from `lm_head.norm`, the initial logits then have std about
`sqrt(d)` (32 for d = 1024). The hero is untied. This already happens with `llama` /
`llama_depth`, so it is not new, but it matters if someone uses nanochat on a tied model.

### 5.5 Interaction with the optimizer (the biggest practical caveat)

- With one global AdamW LR, a parameter's relative update per step is roughly `lr / std`.
  Embeddings at std 1.0 with the hero's LR of 1.1e-3 move about 50x slower relative to their
  size than today's std 0.02. Relative to today, other parameters range from about 25x slower
  (the convs, std 0.02 -> 0.5) to about 7x faster (KDA `w_b`, std 0.02 -> 0.0028); see
  Section 1.2.
  nanochat pairs this init with per-group LRs (Muon for matrices, separate AdamW LRs for
  embeddings and the LM head); we did not.
- Weight decay is coupled to LR (`p *= 1 - lr * weight_decay`), so any future per-group LR also
  changes effective decay (`PARAMETERIZATION_HANDOFF.md` Section 2.3).
- The hero's `embeddings.weight` weight-decay override does not match under DDP (it needs a
  leading `*`), so embeddings currently train with weight_decay 0.1 (`PARAMETERIZATION_HANDOFF.md`
  Section 2.6). That interacts with a std-1.0 embedding too.
- `olmoe3_hero_mt.py` asserts every per-group LR equals one cosine schedule; per-group LRs would
  need that audit updated.

### 5.6 MoE v1 `shared_mlp` is never initialized by the other schemes

`init_feed_forward_moe` never touched `MoEBase.shared_mlp`, so under `normal`, `llama`, `fan_in`,
etc. it keeps PyTorch's default (Kaiming-uniform) init. This is a **pre-existing gap**. Under
nanochat it is now initialized via `init_feed_forward`; the other schemes were deliberately left
alone so their results don't change.

### 5.7 Consequences of applying the formula literally

- Narrow outputs get very small stds: KDA `w_b` (1024 -> 8 heads) gets 0.0028, so beta logits
  start near 0 and beta starts near 1.0 (doubled by `allow_neg_eigval`). A headwise attention gate
  (d -> n_heads) gets a similarly small std.
- Block 0's dense shared MLP `w_down` (8192 -> 1024) gets 0.0039.
- The MoE top-k combine (a weighted sum of k expert outputs) is not accounted for; each expert
  matrix is treated independently.
- Router logits start with std of about 0.71 (vs 0.64 under `normal`), so routing entropy at init
  is similar to today.
- Under peri-norm, the scale of each sublayer's output does not affect the forward pass. The
  init mainly matters through nonlinearity regimes (SwiGLU, conv SiLU, gates), router logits, and
  the optimizer's relative update size.

### 5.8 Not exercised by any test

- Expert parallelism (routed experts drawn from `ep_generator` on real shards), and DTensor /
  tensor-parallel sharding through `_apply_init`. The code follows the same patterns as the
  existing schemes but was only run single-GPU.
- A shared-experts router (more than one shared expert). It goes through the same router loop as
  the routed router, which is tested.
- `MXFP8Linear` projections in `FusedAttentionV2` (`MXFP8Linear` subclasses `nn.Linear`, and the
  existing `init_linear` already handles it).
- `NormalizedTransformer` + nanochat: it would run (parent init, then `normalize_matrices()`), but
  the combination is not meaningful.

### 5.9 Init does nothing for resumed runs

All current hero midtraining / decay runs load checkpoints, so they are unaffected. nanochat
only matters for runs that start from scratch.

### 5.10 Pre-existing check failures (not caused by this change)

- `isort --check` fails on `src/olmo_core/nn/attention/__init__.py` at `HEAD`: the
  `from torch.distributed.tensor import ...` line is 92 columns and the project's isort profile
  (`profile = "black"`) wraps at 88. Running `make style` will rewrap it; I reverted that
  unrelated hunk to keep the diff focused.
- `mypy` reports one error in `src/olmo_core/nn/moe/v2/routed_experts.py:940` (an untouched
  file).
- Ruff and black are clean on all changed files.

### 5.11 Environment notes

- `olmo_core` is not pip-installed in this environment; pytest finds it through its config.
  Standalone scripts need `PYTHONPATH=src`.
- `kernel_fun` is not installed, so building the hero config requires setting
  `use_cute_kernel=False` on the KDA mixer configs (it does not affect init).
- flash-attn 2.8.2 is installed (olmo-core warns that it supports <= 2.8.1, but the backend
  constructs fine); FLA 0.4.1 is installed, so KDA/GDN tests run.

---

## 6. Verification

### 6.1 Tests

```bash
pytest -v src/test/nn/transformer/init_test.py src/test/nn/ddp/block_test.py -k nanochat
```

20 nanochat tests pass and none are skipped on this machine (B300 GPU with FLA and flash-attn 2).
The full `init_test.py` + `block_test.py` run is 60 passed. The related existing suites
(`kda_test.py`, `recurrent_test.py`, `nemotron_test.py`, `attention_test.py`,
`src/test/nn/ddp/`) gave 351 passed, 1009 skipped (multi-GPU / kernel-specific), 40 xfailed, and
0 failed.

### 6.2 Existing schemes are bit-identical

On the real small hero, per-parameter integer fingerprints for `normal`, `llama`, `llama_depth`
and `normalized` were compared between the working tree and a `git archive HEAD` copy: all 415
parameters are bit-identical for every scheme, and `fan_in` raises for KDA in both trees. A
standalone `GatedDeltaNet` (not in the hero) was also bit-identical across the same four schemes.

### 6.3 Per-parameter audit on the real hero

The audit built the real small hero (`build_model_config("small")`, 512 experts, real vocab,
per-head QK-norm gains, `use_cute_kernel=False`) with `init_method=nanochat` on meta, then ran
`OLMoDDPModel.init_weights` on one B300. The parameter count was 12,496,341,632, matching
`olmoe3_small_hero.py`'s assertion. The audit:

1. Wrapped `torch.nn.init.{trunc_normal_, normal_, ones_, zeros_, uniform_, kaiming_uniform_}`
   to record the last writer and std argument per `tensor.data_ptr()`.
2. Wrapped `model.to_empty` to fill every parameter with NaN right after allocation.
3. Derived each parameter's expected std **from its name and shape only** (independent of the
   module-attribute logic in the implementation), and checked that (a) no NaN survived, (b) the
   last writer is a normal init, (c) the std argument matches exactly, and (d) the empirical std
   is within 5%. Fixed parameters were checked for exact ones / zeros / log-uniform range.

Result: all checks passed. Empirical stds sit about 1.8% below the target (truncation plus bf16
rounding). The script lived at `/tmp/nanochat_audit/audit.py`, which is **not checked in** and
will not survive the machine; the recipe above is enough to recreate it.

---

## 7. Possible follow-ups

- **Per-group learning rates** (spectral muP for Adam scales LR with `fan_out / fan_in`; nanochat
  uses Muon plus separate embedding / LM-head LRs). See `PARAMETERIZATION_HANDOFF.md` Section 2
  for the `OptimGroupOverride` machinery and its gotchas (`module.` prefix, absolute LRs,
  overlapping patterns, WD coupling).
- **Optional zero-init** of residual output projections and the LM head for pre-norm models,
  e.g. a flag on `TransformerConfig`. Not useful for the peri-norm hero (Section 4, decision 2).
- **nanochat for `NemotronMamba2Mixer` / `NemotronMoE`**, which currently raise.
- **Fix the MoE v1 `shared_mlp` gap** for the other schemes (Section 5.6). This changes their
  results, so it should be a deliberate, separate change.
- **Check the audit script in** (e.g. under `src/scripts/`) if per-parameter init audits will
  be needed again.
