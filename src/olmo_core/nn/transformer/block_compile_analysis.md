# Torch.compile Analysis: ReorderedNormTransformerBlock

## Executive Summary

This report analyzes the compilation behavior of `ReorderedNormTransformerBlock` under `torch.compile` with the inductor backend. The analysis reveals that torch.compile creates **4 major fused Triton kernels** for the forward pass, achieving significant operation fusion particularly around LayerNorm, RoPE, SwiGLU, and residual connections.

## Test Configuration

- **Model**: `ReorderedNormTransformerBlock`
- **Input Shape**: `[4, 2048, 1024]` (batch=4, seq_len=2048, d_model=1024)
- **Dtype**: `bfloat16`
- **Attention**: 16 heads, RoPE enabled
- **Feed-Forward**: 4x expansion (4096 hidden), SwiGLU activation
- **Compiler**: torch.compile with inductor backend, mode="max-autotune"

## Fused Kernel Overview

The forward pass is compiled into 4 main fused Triton kernels plus several extern kernel calls (matrix multiplications via cuBLAS/CuTLASS):

1. **Kernel 0**: RoPE application on Q and K projections
2. **Kernel 1**: Post-attention LayerNorm + residual connection
3. **Kernel 2**: SwiGLU activation in feed-forward
4. **Kernel 3**: Post-FFN LayerNorm + residual connection

Plus unfused operations:
- 7x Linear projections (handled by extern `mm`/`addmm` kernels)
- 1x Flash Attention (handled by `scaled_dot_product_flash_attention`)

## Detailed Kernel Analysis

### Kernel 0: RoPE Fusion for Attention
**Location**: `src/olmo_core/nn/transformer/block.py:294` (within attention call)

**Fused Operations**:
```python
triton_poi_fused__scaled_dot_product_flash_attention__to_copy_add_addmm_cat_mul_neg_transpose_unbind_unsqueeze_view_0
```

**Operation Sequence** (8.4M elements):
1. **Q/K Projection Bias Add**: `q = mm(x, W_q) + bias_q`
2. **View Reshaping**: Reshape to `[4, 2048, 16, 64]` (multi-head format)
3. **Type Conversion**: Convert from bf16 → fp32 for RoPE computation
4. **RoPE Position Encoding**:
   - Load sin/cos position encodings
   - Slice to sequence length
   - Broadcast to query/key dimensions
5. **Rotary Embedding Application**:
   - `q_cos = q * cos_pos`
   - Split q into two halves via `unbind`
   - Negate second half: `neg(q2)`
   - Concatenate: `cat([neg(q2), q1])`
   - Apply sin component: `q_sin = rotated_q * sin_pos`
   - Combine: `q_rope = q_cos + q_sin`
6. **Type Conversion**: Convert back fp32 → bf16
7. **Transpose**: Rearrange for attention `[B, H, S, D]`

**Applied to both Q and K in parallel** (dual output kernel).

**Performance Characteristics**:
- Pure pointwise operations (no reductions)
- 14 loads, 2 stores per element
- Optimal for GPU: high arithmetic intensity
- Eliminates intermediate materialization of 6+ tensors

---

### Kernel 1: Post-Attention Fusion
**Location**: `src/olmo_core/nn/transformer/block.py:294` (attention output processing)

**Fused Operations**:
```python
triton_per_fused__to_copy_add_addmm_native_layer_norm_view_1
```

**Operation Sequence** (8192 rows × 1024 reduction):
1. **Output Projection Bias**: `att_out = mm(attention, W_out) + bias_out`
2. **Type Conversion**: bf16 → fp32 for LayerNorm
3. **LayerNorm Statistics**:
   - Compute mean: `mean = sum(x) / 1024`
   - Compute variance: `var = sum((x - mean)^2) / 1024`
   - Compute inverse std: `rstd = 1 / sqrt(var + eps)`
4. **LayerNorm Normalization**:
   - Normalize: `x_norm = (x - mean) * rstd`
   - Scale: `x_scaled = x_norm * weight`
   - Shift: `x_final = x_scaled + bias`
5. **Type Conversion**: fp32 → bf16
6. **Residual Connection**: `h = input_x + normalized_attention_output`

**Performance Characteristics**:
- Persistent reduction kernel (reduces 1024 dimensions)
- 5 loads, 4 reductions, 1 store per row
- **Critical**: Fuses LayerNorm + residual into single kernel
- Avoids materializing attention output before normalization

**Key Insight**: In `ReorderedNormTransformerBlock`, LayerNorm is applied to the OUTPUT of attention (line 294), which allows torch.compile to fuse:
- attention output linear projection
- LayerNorm computation
- residual connection

This is more efficient than the standard transformer where LayerNorm on the input cannot be fused with the residual connection.

---

### Kernel 2: SwiGLU Fusion
**Location**: `src/olmo_core/nn/transformer/block.py:295` (feed-forward)

**Fused Operations**:
```python
triton_poi_fused_addmm_mul_silu_view_2
```

**Operation Sequence** (33.5M elements):
1. **W1 Projection Bias**: `gate = mm(h, W1) + bias_1`
2. **SiLU Activation**:
   - Convert to fp32
   - Compute: `silu = x * sigmoid(x)`
   - Convert back to bf16
3. **W3 Projection Bias**: `value = mm(h, W3) + bias_3`
4. **SwiGLU Gating**: `output = silu(gate) * value`

**Performance Characteristics**:
- Pure pointwise operations
- 4 loads, 1 store per element
- **Critical fusion**: Combines two linear outputs with activation
- Eliminates 3 intermediate tensor materializations

**Architecture Note**: SwiGLU (`F.silu(w1(x)) * w3(x)`) is perfectly suited for fusion. The gate and value paths are computed in parallel (separate matmuls), then combined in a single fused kernel.

---

### Kernel 3: Post-FFN Fusion
**Location**: `src/olmo_core/nn/transformer/block.py:295` (feed-forward output)

**Fused Operations**:
```python
triton_per_fused__to_copy_add_addmm_native_layer_norm_view_3
```

**Operation Sequence** (identical to Kernel 1):
1. **Output Projection Bias**: `ffn_out = mm(swiglu_out, W2) + bias_2`
2. **LayerNorm** (mean, var, normalize, scale, shift)
3. **Residual Connection**: `output = h + normalized_ffn_output`

**Performance Characteristics**:
- Same as Kernel 1: persistent reduction
- 5 loads, 4 reductions, 1 store per row
- Fuses FFN output projection + LayerNorm + residual

---

## Unfused Operations

### Matrix Multiplications (7 total)
These are **not** fused into Triton kernels but instead use highly optimized BLAS libraries:

1. `W_q` projection: `[8192, 1024] × [1024, 1024]` → uses cuBLAS `mm`
2. `W_k` projection: `[8192, 1024] × [1024, 1024]` → uses cuBLAS `mm`
3. `W_v` projection: `[8192, 1024] × [1024, 1024]` → uses cuBLAS `mm` (used directly in attention)
4. `W_out` projection: `[8192, 1024] × [1024, 1024]` → uses cuBLAS `mm`
5. `W1` projection: `[8192, 1024] × [1024, 4096]` → uses cuBLAS `mm`
6. `W3` projection: `[8192, 1024] × [1024, 4096]` → uses cuBLAS `mm`
7. `W2` projection: `[8192, 4096] × [4096, 1024]` → uses cuBLAS `mm`

**Rationale**: Large matrix multiplications are better handled by vendor-optimized BLAS libraries (cuBLAS, CuTLASS) than custom Triton kernels. Torch.compile correctly delegates these to extern kernels.

### Flash Attention
The attention computation itself uses PyTorch's native `scaled_dot_product_flash_attention`:
```python
att = torch._C._nn.scaled_dot_product_flash_attention(
    q, k, v,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=True,
    scale=None
)
```

This is a single fused kernel implementing the Flash Attention algorithm, which is already highly optimized.

---

## Key Optimization Opportunities

### 1. ✅ **Already Optimal: Post-Operation LayerNorm Fusion**

The `ReorderedNormTransformerBlock` architecture (applying LayerNorm to outputs rather than inputs) enables excellent fusion:

```python
# Standard Transformer (suboptimal):
h = x + dropout(attention(layer_norm(x)))
# LayerNorm on input → cannot fuse with residual connection
# Requires 2 separate kernels

# ReorderedNormTransformerBlock (optimal):
h = x + dropout(layer_norm(attention(x)))
# LayerNorm on output → fuses with output projection + residual
# Single kernel handles: projection + layernorm + residual
```

**Why this matters**:
- Reduces kernel launches from 3 to 1 for each sublayer
- Eliminates intermediate tensor materializations
- Better memory bandwidth utilization

### 2. ⚠️ **Potential: Pre-LayerNorm Input Reuse**

Currently, the input `x` is used directly for attention without LayerNorm. If we examine the current pattern:

```python
# Current: line 294
h = x + dropout(layer_norm(attention(x)))
```

The attention operates on raw `x`, then LayerNorm is applied to the output. This is good for fusion, but it means attention doesn't benefit from normalized inputs (which can improve training stability).

**Not a performance issue**: This is an architectural choice, not a compiler limitation.

### 3. ✅ **Already Optimal: SwiGLU Parallel Projection Pattern**

The SwiGLU implementation is ideal for compilation:

```python
# Current implementation (src/olmo_core/nn/feed_forward.py:128):
return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

The compiler recognizes that `w1(x)` and `w3(x)` can be computed in parallel (both use the same input), then fuses the SiLU and multiplication into a single pointwise kernel. This is optimal.

### 4. ⚠️ **Possible Optimization: RoPE Frequency Computation**

The RoPE kernel currently loads sin/cos tables from memory. For very long sequences or frequent kernel launches, pre-computing and caching these on-device is beneficial (already done via `BufferCache` in the codebase).

**Current status**: Already optimized via caching in `src/olmo_core/nn/rope.py`.

### 5. ⚠️ **Advanced: Fusing Linear Projections into Attention**

Theoretically, you could fuse Q/K/V projections with the RoPE kernel:

```
Current:   [mm(Q)] → [RoPE fusion kernel] → attention
Possible:  [QKV projection + RoPE fusion kernel] → attention
```

However:
- **Tradeoff**: Matrix multiplications are better in cuBLAS than Triton for these sizes
- **Complexity**: Would require custom Triton matmul implementation
- **Verdict**: Not recommended unless using smaller models where Triton matmul beats cuBLAS

---

## Comparison: Standard vs Reordered LayerNorm

### Standard TransformerBlock
```python
# forward (line 143-144):
h = self.attention_residual_stream(x, self.attention(self.attention_norm(x)))
return self.feed_forward_residual_stream(h, self.feed_forward(self.feed_forward_norm(h)))
```

**Fusion pattern**:
1. Kernel: LayerNorm (attention_norm)
2. Extern: Q/K/V projections
3. Kernel: RoPE
4. Extern: Flash Attention
5. Extern: Output projection
6. Kernel: Residual add (no LayerNorm)
7. Kernel: LayerNorm (feed_forward_norm)
8. Extern: W1/W3 projections
9. Kernel: SwiGLU
10. Extern: W2 projection
11. Kernel: Residual add (no LayerNorm)

**Total**: ~11 kernel launches

### ReorderedNormTransformerBlock
```python
# forward (line 294-295):
h = x + self.dropout(self.attention_norm(self.attention(x)))
return h + self.dropout(self.feed_forward_norm(self.feed_forward(h)))
```

**Fusion pattern**:
1. Extern: Q/K/V projections
2. Kernel: RoPE
3. Extern: Flash Attention
4. **Kernel: Output projection + LayerNorm + Residual** ✨
5. Extern: W1/W3 projections
6. Kernel: SwiGLU
7. Extern: W2 projection
8. **Kernel: Output projection + LayerNorm + Residual** ✨

**Total**: ~8 kernel launches

**Savings**: ~3 fewer kernel launches (27% reduction), better memory efficiency due to fewer intermediate materializations.

---

## Memory Access Pattern Analysis

### Kernel 0 (RoPE): Compute-Bound
- **Memory reads**: 14 loads/element (Q, K, biases, sin/cos tables, intermediate results)
- **Memory writes**: 2 stores/element (Q and K outputs)
- **Compute**: ~20 FLOPs/element (multiply, add, negate, etc.)
- **Arithmetic Intensity**: ~1.25 FLOPs/byte (good for modern GPUs)

### Kernel 1 & 3 (LayerNorm + Residual): Memory-Bound
- **Memory reads**: 5 loads/row × 1024 elements = 5120 loads
- **Memory writes**: 1 store/row × 1024 elements = 1024 stores
- **Compute**: 4 reductions + normalization = ~10 ops/element
- **Arithmetic Intensity**: ~0.2 FLOPs/byte (memory-bound, but reduction is unavoidable)
- **Critical**: Fusing residual connection saves a full pass over the data

### Kernel 2 (SwiGLU): Compute-Bound
- **Memory reads**: 4 loads/element
- **Memory writes**: 1 store/element
- **Compute**: sigmoid + 3 multiplies + add = ~8 FLOPs/element
- **Arithmetic Intensity**: ~1.6 FLOPs/byte (good balance)

---

## Recommendations

### For Maximum Performance

1. **✅ Keep using ReorderedNormTransformerBlock**: The post-normalization pattern is superior for compilation and memory efficiency.

2. **✅ Ensure RoPE caching is enabled**: The current implementation already does this via `BufferCache`. Verify it's being used in production.

3. **Consider for future work**:
   - **Grouped Query Attention (GQA)**: If `n_kv_heads < n_heads`, fusion opportunities remain similar
   - **Flash Attention 2/3**: Ensure using latest version for best attention kernel performance
   - **Custom attention patterns**: If implementing custom attention (e.g., sliding window), verify similar fusion patterns

4. **Don't over-optimize**:
   - Matrix multiplications should stay as extern kernels
   - Flash Attention should stay as is
   - Focus optimization efforts on operations that are NOT currently fused

### For Different Model Sizes

**Small models (d_model < 512)**:
- Consider fusing Q/K/V projections with RoPE (Triton matmul may be competitive)
- May benefit from more aggressive fusion

**Large models (d_model > 2048)**:
- Current fusion strategy is optimal
- Focus on tensor parallelism and model parallelism
- Ensure communication overlapping with computation

**Long sequences (seq_len > 8192)**:
- RoPE caching becomes critical
- Consider mixed-precision RoPE (keep tables in fp16/bf16)
- Flash Attention with sequence parallelism

---

## Appendix: Fusion Verification Commands

To reproduce this analysis:

```bash
# Run the analysis script
python analyze_block_compile.py

# Examine generated graphs
cat graph_1.txt  # Full FX graph

# Examine generated Triton kernels
ls torch_compile_debug/run_*/torchinductor/model__0_inference_0.0/
cat torch_compile_debug/run_*/torchinductor/model__0_inference_0.0/output_code.py
```

## Summary

The `ReorderedNormTransformerBlock` achieves excellent compilation efficiency under `torch.compile`:

- **4 fused Triton kernels** handle all elementwise and reduction operations
- **Post-normalization architecture** enables optimal LayerNorm fusion with residuals
- **SwiGLU pattern** is perfectly structured for fusion
- **RoPE application** is efficiently fused with projection outputs
- **Matrix multiplications** correctly delegated to optimized extern kernels

The primary optimization target for improving transformer block performance should be:
1. Improving attention mechanism efficiency (already addressed by Flash Attention)
2. Optimizing parallelism strategies (TP, PP, SP)
3. Memory management and activation checkpointing

The current operation fusion achieved by torch.compile is near-optimal for this architecture.