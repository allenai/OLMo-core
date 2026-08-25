# KDA — the math these kernels implement

Everything below is fla 0.5.2's own decomposition (`fla/ops/kda/{chunk_fwd.py,chunk_bwd.py,
chunk_intra.py,wy_fast.py}` and `fla/ops/common/chunk_delta_h.py`), stated precisely so each
port has one written contract. All logs are base-2 (fla multiplies the raw log-space gate by
`RCP_LN2 = 1/ln2` in the cumsum), all gates are ≤ 0 and decreasing down a chunk, and `BT` is
the chunk size (64), `BC = 16` the sub-chunk. Per (batch b, value-head hv); q/k live at head
`h = hv // (HV//H)`. `scale = K^-0.5`.

This package ports the forward scan+readout and four of the backward's seven stages out of
fla. Everything else is fla's, called stage by stage from `kda/chain.py` — which is the
authoritative table of who owns what; this file is the math each stage implements.

| where | module | replaces |
|---|---|---|
| fwd, Aqk zero-fill | `kda/_kernels/fwd_intra_triton.py` | a `masked_fill` over the whole tile |
| fwd, scan + o | `kda/_kernels/fwd_state.py` | `chunk_gated_delta_rule_fwd_h` + the o stage |
| bwd 2, re-scan (B1) | `kda/_kernels/bwd_scan.py` | `chunk_gated_delta_rule_fwd_h` |
| bwd 4, dhu (B2a) | `kda/_kernels/bwd_dhu.py` | `chunk_gated_delta_rule_bwd_dhu` |
| bwd 5, wy_dqkg (B2b) | `kda/_kernels/bwd_wy.py` | `chunk_kda_bwd_wy_dqkg_fused` |
| bwd 6, intra | `kda/_kernels/bwd_intra.py` (+`bwd_intra_triton.py`) | `chunk_kda_bwd_intra` |
| bwd 7, dg cumsum | — | folded into stage 6's epilogue |

The gate activation (`-exp(A_log)·softplus(g + dt_bias)`, when `use_gate_in_kernel`) is
fused into stage 1's cumsum via fla's `kda_gate_chunk_cumsum`, not run as eager fp32 torch
ops. At prod8192 the eager form is four passes over a [B,T,HV,K] fp32 tensor — ~2ms and
~2GiB of saved activations *per layer*, against 1.9ms for the entire fused stage.

---

# Forward

## Stage 1 — cumsum (fla, stays Triton)

    g2[t, d] = (1/ln2) * sum_{s <= t, s in chunk(t)} g[s, d]        fp32 [B, T, HV, K]

Chunk-local inclusive cumsum, per dimension. `G[c, d] = g2[last row of chunk c, d]` is the
per-chunk total decay — a K-vector, where a scalar-gate op (gdn) would have a scalar.

## Stage 2 — intra / WY (fla, stays Triton)

`chunk_kda_fwd_intra` produces, per chunk (row indices i, j within the chunk):

    Aqk[i, j] = scale * sum_d q[i,d] * k[j,d] * exp2(g2[i,d] - g2[j,d])     for i >= j, else 0
    Akk       = the solved lower-triangular WY inverse, beta folded in
    w[i, :]   = (Akk @ (beta * exp2(g2) * k))[i, :]                         bf16 [B, T, HV, K]
    u[i, :]   = (Akk @ (beta * v))[i, :]                                    bf16 [B, T, HV, V]
    kg[i, d]  = k[i,d] * exp2(G[d] - g2[i,d])                               bf16 [B, T, HV, K]
    qg        = q * exp2(g2)   (only materialized when disable_recompute; None on our path)

Two things to internalize:

- **`scale` ships inside Aqk** (the intra kernels multiply it in), while the `q @ h` readout
  term gets `scale` applied in the o stage. Don't apply it twice.
- **The per-dim gate sits INSIDE Aqk's dot product.** `exp2(g2_i - g2_j)` would be a scalar
  under a per-head gate, so it could be factored out of one MMA and applied afterwards in
  SIMT. Per-dim, that factorization needs `exp2(+g2_j)` on one operand — unbounded — so fla
  computes Aqk in sub-chunks (BC=16) where relative gates stay bounded. That is Triton work
  worth keeping: the CuTe kernel LOADS Aqk instead of computing it.

## Stage 3+4 — scan + o (`kda/_kernels/fwd_state.py`, fused)

Serial over chunks c, state h [K, V] fp32:

    v'_c = u_c - w_c @ h_c                                  (MMA "WH" + subtract)
    o_c  = scale * (q_c * exp2(g2_c)) @ h_c + Aqk_c @ v'_c  (MMAs "OH" + "OI")
    h_{c+1}[d, :] = exp2(G_c[d]) * h_c[d, :] + (kg_c^T @ v'_c)[d, :]    (MMA "DH" + decay)

after the last chunk, `ht = h` (fp32, TMA store). fla stores v' (`v_new`) and per-chunk h to
HBM between its two kernels; the fusion keeps both in smem/tmem — that traffic is the win.

Notes that cost time to rediscover: the state decay multiply indexes the tmem fragment's row
(Ld16x256b lanes span rows r and r+8), and OH's q operand must be pre-gated in SIMT
(`qg = q·exp2(g2)`, per-dim) from the q tile plus a g2 tile (fp32 [BT,K], 32KB/stage smem).

---

# Backward

Residuals from the forward: `(q, k, v, g2, beta, Aqk, Akk, h0)` — exactly what fla's own
autograd Function saves when `disable_recompute=False`. The forward must preserve that
rounding contract: Aqk/Akk stored bf16 by intra, g2 the fla cumsum, v_new/h recomputed by
the backward itself.

One deliberate deviation, and it is fla's own behaviour rather than ours: under
`use_gate_in_kernel`, **g2 is not saved**. The backward recomputes it from the raw (and
half-size) `g` with one kernel launch, which is what fla's recompute path does — saving it
would cost 1GiB per layer at prod8192 for something a single launch reproduces.

## The seven launches

1. **recompute** (`recompute_w_u_fwd`): w = Akk @ (β·exp2(g2)·k), u = Akk @ (β·v),
   qg = q·exp2(g2), kg = k·exp2(G−g2). All bf16 [B,T,HV,K/V].
2. **re-scan** (`chunk_gated_delta_rule_fwd_h`): the forward recurrence again, this time
   materializing every per-chunk state: h_c [K,V] checkpoints (bf16, [B,NT,HV,K,V]) and
   v'_c = u_c − w_c @ h_c (`v_new`). Serial over chunks.
3. **dAv** (`chunk_kda_bwd_dAv`): dAqk[r,s] = scale·Σ_v do[r,v]·v'[s,v] (lower tri, fp32
   [B,T,HV,BT]); dv[r,:] = Σ_{s≥r} Aqk[s,r]·do[s,:] (Aqk^T @ do, upper-masked).
4. **dhu** (`chunk_gated_delta_rule_bwd_dhu`): the reverse scan. State dh [K,V] fp32; fla
   folds the per-chunk decay on the K axis, updates dv (dv2 = dv + w-side term) per chunk,
   and emits dh checkpoints (bf16 [B,NT,HV,K,V]) + dh0. Serial, reverse.
5. **wy_dqkg** (`chunk_kda_bwd_wy_dqkg_fused`), per chunk, consuming h_c and dh_c:
   - dq[r,d] = scale·exp2(g2[r,d])·Σ_v do[r,v]·h_c[d,v]                    (h consumer)
   - dk[r,d] = exp2(G[d]−g2[r,d])·Σ_v v'[r,v]·dh_c[d,v] + WY corrections   (dh consumer)
   - dw[r,d] = −Σ_v dv[r,v]·h_c[d,v]; folded via Akk into dk/db/dAkk       (h consumer)
   - dgk[d] += exp2(G[d])·Σ_v h_c[d,v]·dh_c[d,v] (last row of chunk)       (both)
   - dv2 = β·(Akk^T @ dv); db += rowsum terms; dAkk (strict lower, fp32)
   - dg[r,d] = q·dq − k·dk + last-row dgk + kg·(Akk@dw)·β  (the inter-chunk part)
6. **intra** (`chunk_kda_bwd_intra`): the intra-chunk dAqk/dAkk consumers — **the dominant
   stage, half of fla's backward**; see below.
7. **dg cumsum**: dg = chunk-local REVERSE cumsum of dg (fp32).

Ours are 2, 4, 5 and 6, and 7 disappears into 6's epilogue. Stages 2/4/5 are the "B1/B2"
fusions listed as standing headroom in the first version of this file: B1 re-runs the
forward scan with h checkpointed via TMA store off the critical path, B2a keeps dh^T
tmem-resident through the reverse scan, and B2b is a full-K restructure of fla's fused
kernel — the same math over one K slab instead of NK, so v_new/do/dv are read once each
instead of four times.

GVA (HV > H): dq/dk are produced at HV and reduced to H after stage 6.

## Where the time and bytes go (prod8192 = B16 × T8192 × H16, K128/V256, b300)

Stage times measured 2026-08-18: 25.3ms total — intra 12.62, wy_dqkg 6.15, dhu 2.22,
fwd_h 1.88, recompute 1.18, dAv 0.77, dg cumsum 0.51. Inter-stage HBM: h 2.1GB + dh 2.1GB
+ v_new 1.1GB + w/u/qg/kg 2.7GB + dv 2×1.1GB + fp32 dq/dk/dg 3.2GB + dAqk/dAkk 1.1GB.

## Stage 6, bwd_intra — half the backward

### What it computes

Adds the intra-chunk gradient parts, given dAqk and dAkk (both fp32, strictly-masked). With
r, s chunk-local rows, all per (b, hv, chunk), and every gate factor per-dim d:

    dq[r,d]  += Σ_{s ≤ r} dAqk[r,s] · k[s,d] · exp2(g2[r,d] − g2[s,d])
    dwk[r,d]  = Σ_{s ≤ r} dAkk[r,s] · k[s,d] · exp2(g2[r,d] − g2[s,d])
    db[r]    += Σ_d dwk[r,d] · k[r,d]
    dk[s,d]  += Σ_{r ≥ s} (dAqk[r,s]·q[r,d] + dAkk[r,s]·β_r·k[r,d]) · exp2(g2[r,d] − g2[s,d])
                + β_s · dwk[s,d]
    dg[r,d]  += q[r,d]·dq_intra[r,d] + β_r·dwk[r,d]·k[r,d] − dkt[r,d]·k[r,d]
                (dkt = the Σ_{r ≥ s} term above, i.e. the column-side accumulation)

### The numerics law (do not relitigate)

Every exp2 argument above is `g2[r,d] − g2[s,d]` with r ≥ s — one-sided, ≤ 0, safe at any
gate magnitude. KDA's real init makes decay ~16 log2 units PER STEP (`exp(A_log) ∈ [1,16]`),
so any factorization through a reference row m — `exp2(g_r − g_m)·exp2(g_m − g_s)` — has one
factor of the pair unbounded unless m sits between r and s. On a diagonal block no single m
does, which is why fla's diagonals are scalar j-loops and why the SAFE_GATE midpoint trick
carries a bounded-gate contract this op does not satisfy. Cross-sub-chunk blocks ARE safe
(any boundary between the blocks separates all (r,s) pairs) — fla already exploits that with
MMAs there. Falsified before: two-sided midpoint/binary-tree diagonal forms — they overflow
(a production NaN, 2026-08-17) or don't win. `test_kimi_delta_attention_cute_extreme_decay`
in `src/test/nn/attention/kda_test.py` is the guard for exactly this class.

### Why fla's version costs 12.6ms

Grid (NK·NC, NT, B·HV) = (16, 128, 256) → **524k CTAs of [BC=16, BK=32] work**:

- **Stream multiplicity.** Each of the NK=4 K-slabs re-reads the same [BC,BC] dAqk/dAkk
  tiles (fp32 → 4× the 1.1GB), and each of the NC=4 sub-chunk owners re-reads its j-loop
  partners' q/k/g tiles. db is written as an NK-slab and reduced afterwards.
- **exp2 volume.** ~1.1e10 exp2 at prod8192 → ~2.5ms of pure SFU even at perfect
  utilization, and utilization is poor at [16,32] granularity. The *inherent* one-sided
  minimum is ~4× smaller (~2.3e9, ~0.5–0.6ms) IF each factor is computed once and shared
  between the dAqk/dAkk products AND between the row-side (dq) and column-side (dk/dg)
  passes. fla shares the first pair but recomputes across passes — they live in different
  CTAs.
- **Latency.** 16-iteration serial scalar loops of tiny loads with nothing to overlap them.

Floors: traffic ~7.5GB at one-read-one-write → ~1.1ms; exp2 ~0.6ms (overlappable). The
honest target is ~2.5–3ms; fla is 5× above it.

### What `kda/_kernels/bwd_intra.py` does instead (6.34ms SIMT; ~5.0ms with the MMA path)

One CTA per (chunk, b·hv), 512 threads as (32 d-lanes) × (16 row-lanes) — one warp spans a
full row. Each thread walks one row per 16-block with mirror pairing (even blocks r0+rlane,
odd blocks r0+15−rlane) so diagonal work is uniform per thread, and owns 4 **strided**
columns d = lane + 32c (consecutive-column ownership was 4-way bank-conflicted on every
scalar load — 27ms). q/k/g2/dq/dk/dg and the [64,64] dA tiles are read once, dq2/dk2/dg2
written once, db completed in-CTA: the 4× multiplicities are gone.

Cross-block pairs factor through the **s-block end** boundary e(s) = 16(j+1):
`exp2(g_r − g_s) = exp2(g_r − g_e)·exp2(g_e − g_s)`, both exponents ≤ 0 at any gate
magnitude (r ≥ e > s on a decreasing g) — the same safety class as fla's own `kg` operand.
Diagonal pairs (same 16-block) keep exactly one one-sided exp2 per (r,s,d): nothing inside a
diagonal block is ever factorized. All three prescale arrays (kb, qb, kbb) are built in one
phase up front, so the two sweeps fuse per block with no inter-sweep barrier.

Outputs land ~1e-7..2e-6 abs of a fp64 reference, where fla's tf32 `tl.dot`s land 3e-4..1.3e-3.

The shipped kernel additionally runs the **cross-sub-chunk** blocks on tcgen05 MMAs rather
than SIMT (the diagonals stay scalar j-loops — see the numerics law above, which is not
negotiable), and folds two things into its epilogue that fla spends separate work on: the
chunk-local reverse cumsum of dg (deleting stage 7 entirely) and the bf16 cast of dq/dk,
which makes the wrapper's casts no-ops. The dg fold is unconditional; the bf16 emit is
gated on `HV == H`, because the GVA reduction that follows must sum in fp32.

Constraints: K=128, BT=64, fixed-length, no `safe_gate`, and a grid of at least 1024 CTAs
(smaller grids underfill a 148-SM box and the per-call marshaling isn't amortized — T512
regressed to 0.90× without the gate). Off that box the wrapper falls back to
`bwd_intra_triton.py` (the same math restructured in Triton, 9.26ms, K ≤ 128), which falls
back to fla's kernel for varlen/`safe_gate`. **That floor is per stage, not per call**: a
shape can clear `is_supported` and still run this one stage on Triton, which is why the
test arms in `src/test/nn/attention/kda_test.py` are sized to clear it explicitly.

### Register-pressure notes for whoever edits the CuTe kernel

ptxas left alone targets 64 registers (dynamic smem hides the 1-CTA/SM cap from it) and
spills 1–2KB/thread. `min_blocks_per_mp=1` plus `--maxrregcount=128` (frozen as `_MAXREG`)
and *un*-unrolling the inner 16-iteration loops (`cutlass.range(unroll=4..8)`; full unroll
lets ptxas hoist whole load batches) gets `LOCAL_SIZE_BYTES=0`. The incoming-grad gmem reads
cost 3.25ms read-at-use; prefetching them a compute-section ahead is worth ~3ms. 1024
threads was falsified twice (0.45×, 64-reg spills), as was holding P1/P2 in smem (27.5ms).

## Standing headroom

The B1/B2 fusions listed here originally are now stages 2, 4 and 5. What is left, in the
order the ladder is working on it:

- **recompute_w_u** (stage 1, 2 × 1.17ms) and the forward cumsum (0.32ms) — both fla's,
  both counter-blocked on the profiling box.
- **wy_dqkg** at 4.99ms is now the largest single stage.
- Remaining intra headroom is ~4ms (latency-bound loads/prescale/epilogue at 16 warps/SM).

None of these are edited here. This directory is a vendored snapshot; kernel work happens
in the `kernel-fun-2` ladder, where a bench row can say whether an idea was worth keeping —
three of idea 004's did not survive that test.
- `recompute_w_u`/`dAv`/cumsum (2.4ms combined) are near their traffic floor — leave them.
