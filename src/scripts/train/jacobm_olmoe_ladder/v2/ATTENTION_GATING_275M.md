# 275M NoPE attention-gating intervention

This intervention starts from `geometry_nope` and changes only the two global
attention blocks at layers 4 and 9. Each block gains the dense ladder's
elementwise, full-precision sigmoid attention gate. The existing 8-query /
4-KV-head GQA shape, MoE widths, NoPE, `expand_v=2`, and `init_std=0.01` remain
fixed so this is a clean gating test.

## Gate definition

The selected gate is:

```python
GateConfig(
    granularity=GateGranularity.elementwise,
    full_precision=True,
)
```

For each token and gated attention layer, a bias-free `640 -> 1024` projection
produces one gate value per query-head element. A sigmoid maps those values to
`[0, 1]`; the result multiplies the attention output before the output
projection. The sigmoid function and placement are fixed by the implementation.

There are only two exposed gating choices:

- `granularity`: `headwise` or `elementwise`; use `elementwise` to match the
  dense ladder.
- `full_precision`: whether sigmoid and gating are computed in FP32; use
  `true` to match the dense ladder.

There is no separate gate learning rate, bias, temperature, activation, dropout,
or loss coefficient. Gate weights use the model's normal initialization. Under
this intervention that is still `init_std=0.01`; the dense ladder uses its
default `0.02`, which remains a separate initialization experiment.

## Architecture comparison

| Field | NoPE + gated MoE candidate | Dense 275M hybrid | Alignment |
|---|---:|---:|---|
| Model dimension / layers | 640 / 10 | 640 / 10 | exact |
| GDN / global layers | 8 / 2 at layers 4, 9 | 8 / 2 at layers 4, 9 | exact |
| GDN Q/V heads, head dim, `expand_v` | 8 / 8 / 128 / 2 | 8 / 8 / 128 / 2 | exact |
| Global query heads / head dim | 8 / 128 | 8 / 128 | exact |
| Global KV heads | 4 | 8 | intentionally unmatched |
| Global position encoding | NoPE | NoPE | exact |
| Attention gate | elementwise, FP32 | elementwise, FP32 | exact |
| QK norm | per-head RMSNorm, eps `1e-6` | per-head RMSNorm, eps `1e-6` | exact |
| Block/norm layout | peri-norm RMSNorm | peri-norm RMSNorm | exact |
| Embedding scale/norm | `sqrt(640)` / RMSNorm | `sqrt(640)` / RMSNorm | exact |
| Initialization standard deviation | `0.01` | `0.02` | still different |
| FFN | dense-first layer 0, then MoE | dense in every layer | intentional MoE difference |

Ignoring the intentional dense-versus-MoE FFN difference and the explicitly
held 4-versus-8 KV-head ratio, initialization is the only remaining
architectural/training-recipe mismatch at 275M. There is no residual width,
depth, mixer-ratio, norm-type, norm-placement, head-dimension, positional
encoding, or GDN-value-width difference.

The current MoE config uses the Transformer Engine attention backend inherited
from the converted v1 config, whereas the dense Holmes launcher selects
FlashAttention 4. This is an execution-kernel difference, not an architectural
one; the gate is applied after scaled dot-product attention in either case. We
retain Transformer Engine for the isolated intervention and can benchmark a
backend change separately if desired.

The gate adds `640 * 1024 = 655,360` parameters to each of two global attention
layers:

| Profile | Active params | Active non-embedding | Total params |
|---|---:|---:|---:|
| NoPE, ungated | 290,782,080 | 226,556,800 | 3,136,314,240 |
| NoPE, elementwise gated | 292,092,800 | 227,867,520 | 3,137,624,960 |
| Delta | +1,310,720 | +1,310,720 | +1,310,720 |

This is a 0.45% active-parameter increase over the ungated NoPE control. We do
not shrink the experts to compensate, because doing so would confound the
gating intervention.

## Prepared launch path

- Capacity smokes:
  [`launchers/pretraining/manifests/275m_geometry_gdn_ev2_nope_gated_smokes.yaml`](launchers/pretraining/manifests/275m_geometry_gdn_ev2_nope_gated_smokes.yaml)
- Full sweep:
  [`launchers/pretraining/manifests/275m_geometry_gdn_ev2_nope_gated.yaml`](launchers/pretraining/manifests/275m_geometry_gdn_ev2_nope_gated.yaml)
- Beaker wrapper:
  `src/scripts/train/jacobm_olmoe3_geometry_275m_nope_gated_beaker.sh`

The checkpoint-free smokes reuse the maximum successful NoPE microbatches:
Cx1 MB8 on four GPUs, Cx2 MB12 on four, Cx4 MB16 on four, and Cx8 MB12 on
eight. The full manifest retains the same four LRs (`4e-4`, `8e-4`, `1.6e-3`,
`3.2e-3`) for every Cx and has an 80-GPU peak if fully concurrent.

The urgent unallocated capacity smoke
[01KXSZKW55FZKJSD9CPFW4WZ82](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXSZKW55FZKJSD9CPFW4WZ82)
passed all four cells on 2026-07-18. Each reached step 11, exited 0, and wrote
no checkpoint:

| Cx | GPUs | Rank MB | Median final-5 TFLOPs/GPU | Active / reserved memory |
|---:|---:|---:|---:|---:|
| 1 | 4 | 8 | 340.7 | 108.0 / 108.8 GiB |
| 2 | 4 | 12 | 368.9 | 148.8 / 149.8 GiB |
| 4 | 4 | 16 | 394.3 | 189.5 / 191.1 GiB |
| 8 | 8 | 12 | 376.0 | 144.4 / 145.3 GiB |

The exact 16-point LR sweep was then submitted as urgent unallocated work:
[01KXT07N6AGD1S0REJA3TH897G](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXT07N6AGD1S0REJA3TH897G).

## Sweep results

All 16 tasks exited 0. Every Cx is bracketed and supports the formal quadratic
curve check; the observed-best point, rather than the fitted prediction, is
used in the summary:

| Cx | Observed-best LR | Final-250M CE | Delta vs ungated NoPE | Delta vs RoPE geometry |
|---:|---:|---:|---:|---:|
| 1 | `8e-4` | `2.711104` | `-0.001701` | `+0.003223` |
| 2 | `1.6e-3` | `2.580768` | `-0.004411` | `+0.001805` |
| 4 | `8e-4` | `2.476065` | `-0.001719` | `+0.001434` |
| 8 | `8e-4` | `2.390397` | `-0.001556` | `+0.000540` |

Elementwise gating is therefore a small, consistent improvement over the
ungated NoPE control at every data multiple. It nearly closes the NoPE-to-RoPE
gap, especially at Cx8, but does not beat the otherwise-identical RoPE
geometry model.

This result promoted the gated architecture to the full 480M/810M/1.2B
production wave using the same transferred wide-integration LRs and the same
192-GPU peak layout as the larger ungated NoPE wave. The exact launch settings
and Beaker work links are in
[`GEOMETRY_MATCHED_SCALE.md`](GEOMETRY_MATCHED_SCALE.md).

## RoPE interaction control

The larger gated-NoPE family showed regressions at several scale cells, so a
275M interaction control restores RoPE while retaining the gate. The new
`geometry_rope_gated` profile is identical to `geometry_nope_gated` except for
the RoPE object on global-attention layers 4 and 9. It retains 8 Q / 4 KV
heads, expert hidden size 664, `expand_v=2`, elementwise full-precision
gating, QK RMSNorm, and `init_std=0.01`.

RoPE uses theta `500000`, full-precision rotary application, and no scaling.
Because RoPE adds no weights, the gated RoPE and gated NoPE profiles have the
same 292,092,800 active and 3,137,624,960 total parameters. A strict
construction test confirmed exact config equality after removing RoPE from
the new profile.

- Smoke manifest:
  [`launchers/pretraining/manifests/275m_geometry_gdn_ev2_rope_gated_smoke.yaml`](launchers/pretraining/manifests/275m_geometry_gdn_ev2_rope_gated_smoke.yaml)
- Sweep manifest:
  [`launchers/pretraining/manifests/275m_geometry_gdn_ev2_rope_gated.yaml`](launchers/pretraining/manifests/275m_geometry_gdn_ev2_rope_gated.yaml)
- Smoke work:
  [01KY0G559WGWP50DE05B8DJQGY](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY0G559WGWP50DE05B8DJQGY)
- Sweep work:
  [01KY0GVX8SM5998GFMGAKR3AQ6](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY0GVX8SM5998GFMGAKR3AQ6)

The Cx4 MB16 checkpoint-free smoke completed 11 steps and exited 0. The full
four-LR Cx1/Cx2/Cx4/Cx8 sweep was then submitted urgent and unallocated on 80
Holmes B300s at peak concurrency. It deliberately reuses every training and
systems setting from the gated-NoPE sweep.

All four curves finished with complete final-250M-token windows. Observed best
LRs at Cx1/2/4/8 are `1.6e-3`, `1.6e-3`, `8e-4`, and `1.6e-3`; their CEs are
`2.691980`, `2.573449`, `2.470110`, and `2.386206`. The quadratic visual-fit
minima are approximately `1.4e-3`, `1.4e-3`, `1.1e-3`, and `1.2e-3`.

At observed best, gated RoPE beats ungated RoPE geometry by `0.015901`,
`0.005514`, `0.004521`, and `0.003651` CE at Cx1/2/4/8. It also beats ungated
NoPE by `0.020825`, `0.011731`, `0.007674`, and `0.005747`, and gated NoPE by
`0.019124`, `0.007319`, `0.005955`, and `0.004191`. Relative to the first
hybrid, it is better at Cx1/Cx4/Cx8 by `0.002661`/`0.008055`/`0.010174`, and
worse at Cx2 by `0.003461`.

The larger 1.2B gated models have an audited 6.1155% active-parameter delta
from the corresponding wide-integration reference. A shared 6% sanity guard
initially rejected those runs before step 1; the production entrypoint now
permits 6.2% only when attention gating is enabled, while preserving the
stricter 6% limit for all ungated variants. All four 1.2B cells were
re-submitted after that fix; the work links are recorded in
[`GEOMETRY_MATCHED_SCALE.md`](GEOMETRY_MATCHED_SCALE.md).

On 2026-07-21, this result promoted gated RoPE to a complete
480M/810M/1.2B transferred-LR wave. Its 12 urgent allocated jobs use the
compact 124-GPU layout and were submitted Cx1, Cx2, Cx4, then Cx8, with model
sizes ordered 480M, 810M, then 1.2B inside each group. Exact settings and work
links are recorded in
[`GEOMETRY_MATCHED_SCALE.md`](GEOMETRY_MATCHED_SCALE.md#gated-rope-transferred-lr-wave).

- U-plot:
  [`plots/pretraining/geometry_gdn_ev2_nope_gated/275m_uplot.png`](plots/pretraining/geometry_gdn_ev2_nope_gated/275m_uplot.png)
- Observed-best summary:
  [`plots/pretraining/geometry_gdn_ev2_nope_gated/summary_observed_best.png`](plots/pretraining/geometry_gdn_ev2_nope_gated/summary_observed_best.png)
- Exact results:
  [`results/pretraining/geometry_gdn_ev2_nope_gated/results.md`](results/pretraining/geometry_gdn_ev2_nope_gated/results.md)
- Gated-RoPE U-plot:
  [`plots/pretraining/geometry_gdn_ev2_rope_gated/275m_uplot.png`](plots/pretraining/geometry_gdn_ev2_rope_gated/275m_uplot.png)
- Gated-RoPE observed-best summary:
  [`plots/pretraining/geometry_gdn_ev2_rope_gated/summary_observed_best.png`](plots/pretraining/geometry_gdn_ev2_rope_gated/summary_observed_best.png)
- Gated-RoPE exact results:
  [`results/pretraining/geometry_gdn_ev2_rope_gated/results.md`](results/pretraining/geometry_gdn_ev2_rope_gated/results.md)
