# Matched KDA/GDN2 numerical audit

The 2026-07-25
[one-B300 Beaker job](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYBZX8MHJ611ZSJD43SYS9HZ)
compared each optimized kernel with its own sequential PyTorch recurrence.
KDA used the production FLA `0.4.1` package; GDN2 used the isolated FLA
`0.5.2` commit `cbb0a72`. The full per-tensor JSON and 40-row report are in
the job result dataset `01KYBZX8MSG9PSF60QBG91XM8M`.

Every case used BF16, H=2, K=128, identical common random tensors and upstream
gradients, and the same initial-state scale. The matrix crossed V=128/256,
T=64/256 (one/four chunks), and output-only/state-only losses. GDN2 also
crossed negative eigenvalues and retained-intermediate versus recomputed
backward.

## Worst gradient relative-L2 error

| Mixer | T | V | Loss | Negative eigvals | Worst relative-L2 | Tensor |
|---|---:|---:|---|---:|---:|---|
| KDA | 64 | 128 | output | no | 0.44% | `dt_bias` |
| KDA | 64 | 128 | state | no | 1.01% | `A_log` |
| KDA | 64 | 256 | output | no | 0.39% | K |
| KDA | 64 | 256 | state | no | 0.56% | scalar gate |
| KDA | 256 | 128 | output | no | 1.76% | `A_log` |
| KDA | 256 | 128 | state | no | 0.41% | K |
| KDA | 256 | 256 | output | no | 0.67% | `A_log` |
| KDA | 256 | 256 | state | no | 0.41% | scalar gate |
| GDN2 | 64 | 128 | output | no / yes | 0.51% / 0.54% | erase gate |
| GDN2 | 64 | 128 | state | no / yes | 0.88% / 0.54% | `A_log` / erase gate |
| GDN2 | 64 | 256 | output | no / yes | 0.53% / 0.50% | erase gate |
| GDN2 | 64 | 256 | state | no / yes | 0.92% / 0.84% | erase gate |
| GDN2 | 256 | 128 | output | no / yes | 1.13% / 1.33% | `A_log` |
| GDN2 | 256 | 128 | state | no / yes | 0.44% / 0.58% | K / erase gate |
| GDN2 | 256 | 256 | output | no / yes | 1.93% / 3.80% | `A_log` |
| GDN2 | 256 | 256 | state | no / yes | 0.53% / 0.47% | erase / write gate |

All tensor comparisons passed FLA's elementwise combined absolute/relative
tolerances with zero violating elements. All worst-gradient cosine
similarities exceeded `0.99984`. Retained-intermediate and recomputed GDN2
backward produced identical metrics in every case.

## Interpretation

The earlier 1.56e-2 maximum `dV` difference was dominated by gradient scale
and an unmatched combined output/final-state loss. Under matched output losses,
KDA and GDN2 Q/K/V and decay gradients are generally in the same 0.3--0.5%
relative-L2 range. There is no broad GDN2 backward disagreement.

The meaningful exception is the widest four-chunk output-loss case. GDN2's
two-element `A_log` gradient reaches 1.93% relative-L2 without negative
eigenvalues and 3.80% with them, versus 0.67% for KDA at the same T/V shape.
For the negative-eigenvalue GDN2 case, `A_log` has max absolute error
`2.88e-3`, cosine similarity `0.999939`, and zero tolerance violations; every
other common gradient remains between 0.35% and 0.52% relative-L2. This is a
localized accumulation effect worth stress-testing, not evidence of a general
incorrect backward kernel.
