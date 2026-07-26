# Kimi Delta Attention

## Dependency and kernel choice

KDA training uses the FLA `0.4.1` package already present in the standard
pre-GDN2 image. This is the version pinned by OLMo-core's `fla` dependency
extra and is deliberately separate from the temporary FLA `0.5.2` overlay
used only by GDN2 jobs.

FLA `0.4.1` contains Moonshot's released Triton KDA forward and backward
kernels. The KDA Beaker wrapper asserts `fla.__version__ == "0.4.1"` and the
presence of `fla.ops.kda.chunk_kda` before training. It installs nothing and
therefore leaves GDN1 and GDN2 environments unchanged.

MoonshotAI/FlashKDA is a trusted official implementation, but it currently
exposes only a forward kernel and requires `torch.inference_mode()`. It cannot
train the model or run an LR sweep. It may be evaluated separately for
inference and RULER after the training architecture is selected. The unrelated
`Itssshikhar/Flash-Flash-KDA` H100 fork is not used.

## Audited 275M candidate

`geometry_275m_kda_ev1_noneg_nope_gated` changes only the eight recurrent
mixers in the existing geometry-matched gated-NoPE model. It retains 640 model
width, ten layers, global attention at layers 4 and 9, the dense-first FFN,
all MoE dimensions, attention gating, GQA, NoPE, and initialization.

The KDA layers match FLA `0.4.1`'s released Kimi configuration:

- eight 128-dimensional recurrent heads;
- `expand_v=1`;
- nonnegative eigenvalues;
- per-key-channel decay and one scalar delta gate per head;
- four-token unbiased short convolutions; and
- the canonical low-rank decay and sigmoid output-gate projections.

| Variant | Active params | Active non-embedding | Total params |
|---|---:|---:|---:|
| Canonical KDA (`expand_v=1`, nonnegative) | 274,470,720 | 210,245,440 | 3,120,002,880 |
| GDN1-settings KDA (`expand_v=2`, negative eigenvalues) | 290,503,488 | 226,278,208 | 3,136,035,648 |

The candidate's default Cx1 token budget is 4,204,908,800 tokens under the
usual `20 * active_non_embedding_params * Cx` rule.

## Qualification and sweep

The qualification wrapper first compares the Triton kernel's output and all
input/parameter gradients with FLA's sequential Torch KDA recurrence. It then
checks that packed document boundaries match independent per-document calls.
Only after those checks pass does it run the compiled model for 50 optimizer
steps at 8K, MB16, and a 2 Mi-token optimizer batch on one B300. It writes no
checkpoint and runs no evals.

The qualification passed on 2026-07-25 in
[Beaker experiment `01KYBX6WX46F9B3HV3W59G368R`](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYBX6WX46F9B3HV3W59G368R).
The maximum output absolute difference against the sequential reference was
`4.88e-4`; gradient differences were at most `1.70e-3`, and packed-document
output matched independent documents exactly. The subsequent 50-step training
smoke completed with zero skipped steps. Its steady-state actual averages were
404.7 TFLOPs/GPU and 290,450 tokens/s on one B300; active and reserved memory
were 214.6 GiB and 226.4 GiB respectively. The W&B run is
[`3s14s676`](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/3s14s676).

The later
[matched KDA/GDN2 numerical audit](results/diagnostics/matched_kda_gdn2_numerics.md)
tested KDA at the same H=2, K=128, V=128/256, T=64/256, initial states, and
separate output/state losses used for GDN2. KDA's worst gradient relative-L2
error ranged from 0.39% to 1.76%, with zero tolerance violations.

The four-LR Cx1/Cx2/Cx4/Cx8 sweep was submitted on 2026-07-25 in four
Cx-first works: [Cx1](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYC1F615GDDN5Z9W48W072CQ),
[Cx2](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYC1F92YYNMSPH3SAQD5NXDJ),
[Cx4](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYC1FBW5T3GC4XHEK0SHJG5J),
and [Cx8](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYC1FER04GND05GA5F6EFH1N).
It uses `4e-4`, `8e-4`, `1.6e-3`, and `3.2e-3` at every Cx, EP1,
ordinary 10%-of-token warmup and cosine decay, rolling ephemeral checkpoints,
no in-loop evals, urgent priority, and unallocated Holmes scheduling. The
fully concurrent sweep is 16 tasks / 80 GPUs.

The paired plotting entry point is `plot_canonical_gdn2_kda.py`. It writes one
KDA U-plot, one canonical-GDN2 U-plot, and one shared observed-best plot against
wide integration, matching gated-NoPE GDN1, and original `expand_v=2` GDN2.
Only bracketed curves enter the shared plot, and labels remain observed-best
LRs rather than predicted minima.

Status at 2026-07-25 16:44 UTC: all 16 KDA jobs finished successfully without
a failed attempt. Every Cx curve is bracketed and its observed best is
`1.6e-3`. The strict final-250M-token CEs at Cx1/2/4/8 are `2.717057`,
`2.587990`, `2.486896`, and `2.405198`. The generated U-plot and shared
comparison live under `plots/pretraining/canonical_gdn2_kda/`; the complete
machine-readable and Markdown summaries are under
`results/pretraining/canonical_gdn2_kda/`.

The matched 480M stability transfer uses the same four wide-transfer LRs and
resource layouts as canonical GDN2. At the 2026-07-26 22:05 UTC refresh, all
four cells had finished cleanly without a failed attempt. Their strict
final-250M CEs at Cx1/2/4/8 are `2.517826`, `2.412884`, `2.323228`, and
`2.237558`. All four KDA transfer points are included in
`gdn2_fixed_lr_scale_comparison.png`.

## GDN1-settings KDA transfer

The separately named `geometry_*_kda_ev2_neg_nope_gated` family tests KDA
with the recurrent settings used by matching GDN1: `expand_v=2` and negative
eigenvalues enabled. It otherwise retains the same gated-NoPE geometry and
FLA 0.4.1 KDA kernel as canonical KDA.

The fixed-LR transfer covers 275M and 480M at Cx1/2/4/8. The 275M LRs are the
observed-best matching-GDN1 values (`8e-4`, `1.6e-3`, `8e-4`, `8e-4`); 480M
uses the standard transferred wide LRs (`1.2e-3`, `9e-4`, `8e-4`, `8e-4`).
The two sizes request 20 and 40 GPUs respectively, use accumulation one, and
write distinct W&B/checkpoint identities containing `kda-ev2-neg`.

`plot_canonical_gdn2_kda.py` registers these identities separately and writes
`kda_ev2_neg_fixed_lr_scale_comparison.png` plus its own JSON/Markdown result
ledger. These fixed-LR points do not enter the canonical 275M LR-sweep U-plots
or observed-optimal summary.
