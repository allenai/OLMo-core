# What fla 0.5.2's chunked KDA *backward* computes

Read from `fla/ops/kda/{chunk_bwd.py,wy_fast.py,chunk_intra.py}` and
`fla/ops/common/chunk_delta_h.py` at the pinned 0.5.2 — the same reading 001's ALGORITHM.md
did for the forward; read that one first for `BT=64`, log2-space `g2`, the WY factors and the
shapes. gdn 003's ALGORITHM.md is the scalar-gate ancestor of this file; the diff tables below
are against it.

Notation, per `(b, hv)` and per chunk `c`: `g2[i,d]` the chunk-local inclusive cumsum of `g`
over ln2 (fp32 `[B,T,HV,K]` — a K-vector per row where gdn had a scalar), `G[d] = g2[BT-1,d]`
the per-chunk total decay, `Akk` the bf16 unit-lower-triangular WY inverse, `Aqk` the bf16
gate-weighted lower-triangular `q k^T` (scale folded in), `h_c` the forward state entering
chunk `c`, `v_new = v'` the post-`w@h` values. `scale = K^-0.5`.

Saved tensors are `(q, k, v, g2, beta, Aqk, Akk, h0)`; everything else is recomputed.

## The six launches

fla's gdn backward was seven launches; kda restructures the tail: gdn's `dqkwg` and
`wy_repr_bwd` merge into ONE fused kernel (stage 5), and a new `intra` backward (stage 6)
consumes `dAqk`/`dAkk` — the per-dim gate sits inside those dot products, which forces the
same BC=16 sub-chunk decomposition the forward's intra stage needed.

| # | stage | kernel | grid | what |
|---|---|---|---|---|
| 1 | `recompute_w_u_fwd` (kda) | `recompute_w_u_fwd_kda_kernel` | NT × B·HV | w, u **and qg, kg** |
| 2 | `fwd_h` | common `..._fwd_kernel_h_blockdim64`, USE_GK | V/BV × B·HV, serial NT | h checkpoints + v_new |
| 3 | `chunk_kda_bwd_dAv` | `chunk_kda_bwd_kernel_dAv` | NT × B·HV | dAqk + intra dv |
| 4 | `bwd_dhu` | common `..._bwd_kernel_dhu_blockdim64`, USE_GK | V/BV × B·HV, serial NT rev | dh checkpoints, dh0, dv2 |
| 5 | `wy_dqkg_fused` | `chunk_kda_bwd_kernel_wy_dqkg_fused` | NT × B·HV | dq, dk, dw-folded, dg, dv, db, dAkk |
| 6 | `bwd_intra` | `chunk_kda_bwd_kernel_intra` | NK·NC × NT × B·HV | dAqk/dAkk → dq, dk, dg, db |
| 7 | reverse `chunk_local_cumsum(dg)` + GVA head-sum of dq/dk | | | |

## The math, stage by stage

**1. `recompute_w_u_fwd`** — from the saved `Akk`, all bf16 out:
```
u  = Akk @ (v · beta)                          [B,T,HV,V]
w  = Akk @ (k · beta · exp2(g2))               [B,T,HV,K]
qg = q · exp2(g2)                              [B,T,HV,K]   (kda only — gdn had no qg/kg here)
kg = k · exp2(G - g2)                          [B,T,HV,K]
```
`qg`/`kg` are the pre-scaled scan operands: they are why the per-dim gate largely vanishes
from stages 2 and 4, exactly as it vanished from 001's fused forward.

**2. `fwd_h`** — the forward scan re-run (USE_GK path), `output_final_state=False`:
```
store h checkpoint h_c (bf16)
v'  = u - w @ h_c                 -> v_new (bf16, saved BEFORE any gating — none is applied)
h  += kg^T @ v'   after   h[d,:] *= exp2(G[d])
```
vs gdn's USE_G path: **no** `v' *= exp2(G - g2_i)` rescale before the state dot (kg carries
it), and the decay is a K-vector on the state's rows, not a scalar.

**3. `chunk_kda_bwd_dAv`** — chunk-parallel, the analog of gdn's `dv_local`:
```
dv   = Aqk^T @ do                              (Aqk loaded transposed, masked i <= j)
dAqk = tril(do @ v_new^T) · scale              (fp32 [B,T,HV,BT])
```
Note it also emits `dAqk` — gdn rebuilt the equivalent inside `dqkwg` (its `ds`); kda cannot
(the per-dim gate factor `exp2(g2_i - g2_j)` is unbounded outside sub-chunks), so the raw
product is handed to stage 6.

**4. `bwd_dhu`** — reverse scan (USE_GK path), `dh` starts at `dht`, chunks `NT-1 … 0`:
```
store dh checkpoint dh_c (bf16, PRE-update — the gradient flowing out of chunk c)
dv2 = kg @ dh_c + dv                           (completes stage 3's dv; NO gate factor)
dh  = exp2(G[d]) · dh  +  scale · qg^T @ do  -  w^T @ dv2
after chunk 0: dh0 = dh (fp32)
```
vs gdn: `q·exp2(g2)` and the `exp2(G-g2)` on the k-side both live in the pre-scaled operands;
the decay is per-dim.

**5. `chunk_kda_bwd_wy_dqkg_fused`** — chunk-parallel, K-tiled outer loop with a V-tile
reduction reading `h_c`/`dh_c` back from HBM. One kernel doing gdn's stages 5 AND 6, minus
what moved to stage 6. Per k-tile:
```
per v-tile:  dq  += do @ h_c          dk += v_new @ dh_c       dw += dv @ h_c
             dgk += sum_v(h_c · dh_c)                          (K-vector now, was scalar)
  (i_k==0):  dA  += dv @ v^T          dvb = Akk @ dv
             dv2 = dvb · beta         db += sum(dvb · v)       (the WY du-backward, inlined)
then:        dq  = dq · exp2(g2) · scale
             dk  = dk · exp2(G - g2)
             dgk = dgk · exp2(G) + sum_i(k · dk)
             kgw = k · exp2(g2)                                 (fla's b_kg — NOT stage-1 kg!)
             dw  = -dw;   dA += dw @ kgw^T;   dkgb = Akk @ dw;  db += sum(dkgb · kgw)
             dg  = q·dq - k·dk + [i==last]·dgk + kgw·dkgb·beta  (per-dim rows)
             dk += dkgb · exp2(g2) · beta
finally:     dAkk = -tril_strict(Akk^T @ (tril_strict(dA) · beta_j) @ Akk^T)
```
The `Akk^T (…) Akk^T` triangular sandwich is gdn's `wy_bwd` dA path; the `exp2(g2_i - g2_j)`
factor that gdn applied to dA is *deferred* — it lands inside stage 6's sub-chunk loops.

**6. `chunk_kda_bwd_intra`** — sub-chunk (BC=16) decomposed, grid `(NK·NC, NT, B·HV)`:
```
dq += (dAqk @ (k·exp2(gn - g2_j))) · exp2(g2_i - gn)            (lower blocks + diag loop)
dkt = (dAqk^T @ (q·exp2(g2_j - gn)) + dAkk^T @ (k·beta·exp2(g2_j - gn))) · exp2(gn - g2_i)
dk += dAkk-analog of dq's path, · beta;    db += sum(that · k)
dg += q·dq_contrib - k·dkt + …                                   (row/col asymmetric)
```
This is the backward of the forward intra's bounded-relative-gate trick; it is as awkward as
the forward version and stays fla for the same reason (001 kept the forward intra).

**7.** `dg` (fp32 `[B,T,HV,K]`) gets a **reverse** chunk-local cumsum; GVA reduces dq/dk from
HV to H heads by summing groups.

## Cost at prod8192, and where the ms are

Not yet measured stage-by-stage (dbg_perf.py's backward table produces it — run on the
B300 before porting anything beyond the scans). What is known from the bench: the whole
backward is `35.18 - 7.56 ≈ 27.6ms` against gdn's 14.3ms at the identical shape — the
per-dim gate roughly doubles fla's backward. The scans (stages 2+4) carry gdn's costs plus
a full fp32 `[BT,K]` g2 read per chunk-step that our port deletes (kg/qg carry it); stage 5
reads both 2.15GB checkpoint tensors back like gdn's dqkwg, plus the fp32 g2 stream.

## Rounding contract (match it or the tolerances will not hold)

- `Aqk`, `Akk` are read as the saved **bf16**; `w/u/qg/kg/v_new` and both checkpoint tensors
  are bf16 with fp32 accumulation. `dh0` is fp32.
- `beta`, `g2`, `dAqk`, `dq/dk/dg/db` intermediates are **fp32** throughout.
- `dv` is written three times: stage 3 (intra part), stage 4 (`dv2`, + state part), stage 5
  (the WY `du`-backward overwrite). The returned `dv` is stage 5's.
- One deliberate divergence in the cute dhu port: fla computes `dot(qg_bf16, do_bf16)` in fp32
  and multiplies `scale` afterwards; the port folds `scale` into `do` and rounds `do·scale`
  to bf16 (the UPD MMA's A operand). Same order of error as gdn 003's `dog` note; measured
  there at dh 3.8e-5, well inside the stage tolerances.

## Diff vs gdn 003's kernels

### `kernel_fwdh.py` (gdn USE_G → kda USE_GK)

| gdn 003 | kda 002 |
|---|---|
| `k` `[B,T,H,K]`, mn-major B of UPD | **`kg`** `[B,T,HV,K]` — fla pre-scales `k·exp2(G-g2)`; per-VALUE-head, `h_idx` gone |
| `v~ = v'·exp2(G-g2_i)` built in SIMT from a `[BT]` g2 row | **deleted** — UPD's A operand is `v'` itself; the vta and v_new stores hold the same value |
| `g2` staged as `[BT]` scalars/stage | **`gd`** staged as `[K]` fp32 chunk-decay vectors (g2's last row, sliced host-side, 001's `gdc` pattern) |
| `h ← exp2(G)·h + UPD`, one scalar | `h[d,:] ← exp2(gd[d])·h[d,:] + UPD` — `coordUPD`'s `kk` indexes a broadcast view of sGd, per-element exp2 (001's state-decay pattern, same fragment geometry) |

### `kernel_dhu.py` (gdn USE_G → kda USE_GK)

| gdn 003 | kda 002 |
|---|---|
| `q`,`k` `[B,T,H,K]` | **`qg`,`kg`** `[B,T,HV,K]` pre-scaled, per-value-head |
| `dog = do·exp2(g2_i)·scale` in SIMT | `dos = do·scale` — the gate lives in qg; see the rounding note above |
| `dv2 = DV·exp2(G-g2_i) + dv` | `dv2 = DV + dv` — kg carries the factor |
| `dh ← exp2(G)·dh + …`, scalar | `dh[d,:] ← exp2(gd[d])·dh[d,:] + …` per-dim |
| `g2` `[BT]` staging, row-broadcast views | **`gd`** `[K]` staging, one broadcast view for the decay only |

Net effect in both scans: LESS in-kernel work than gdn's (two exp2-multiply passes deleted,
one added), identical pipeline structure, identical MMA shapes — only operand sources and the
decay geometry change. Keep gdn 006's two TMA-store deferrals exactly as they are; they were
worth 0.5ms/stage.

### Stage 5 (NOT ported yet — the sketch for when the profile justifies it)

`kernel_dqkwg.py` + `kernel_wy_bwd.py` in this folder are gdn's kernels, kept verbatim as the
raw material. The kda fused kernel is their union with three structural changes: (a) `ds` is
gone (arrives as `dAqk` from stage 3 / leaves as `dAkk` for stage 6 — no in-kernel segsum
mask, which was dqkwg's SIMT hot spot); (b) all the row scalings widen to per-dim `[BT,BK]`
exp2 factors built from a g2 tile; (c) wy_bwd's `dA · exp2(g2_i-g2_j)` and the `dA@k`-family
second-pass products move out (to stage 6). Port only after dbg_perf shows stages 5+6
dominating the post-scan backward, and consider whether fusing stage 3 into it (both are
NT × B·HV readers of do) pays first.
