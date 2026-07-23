# Migration to `akshitab/moe-v2-core`

## Provenance

- Source experiment branch: `jacobm/olmo-ddp` at `d7198425831a0f45eb483acce68296891bd86660`.
- Upstream migration base: `akshitab/moe-v2-core` at
  `f5376c18424e3f7329fa6e39312c63b84c5f845a`.
- Migration branch: `jacobm/moe-v2-core`.

This branch starts from upstream and imports Jacob's experiment layer. It does
not merge or copy the old branch's `src/olmo_core` implementation.

## Config boundary

`v2/moe_v2_core_adapter.py` is the only compatibility layer for recorded
`olmo-ddp` configs. It strictly translates:

1. identical `attention_norm` and `feed_forward_norm` configs to `layer_norm`;
2. `d_attn / n_heads` to `head_dim`;
3. flat legacy expert-parallel controls to `ExpertParallelConfig`;
4. legacy dense first layers to equivalent shared-only DDP blocks;
5. YaRN `truncate=true` to the upstream representation; and
6. Muon-only optimizer controls only after requiring `use_muon=false`.

Checkpoint tensor names and values are not rewritten.

## Gates

- [x] Import tracked v1/v2 configs, launchers, results, plots, and docs.
- [x] Copy the ignored results cache so result refreshes do not redownload all W&B data.
- [x] Update active Beaker wrappers and manifests to this worktree.
- [x] Match the old branch's three model-summary outputs byte-for-byte.
- [x] Build all 20 active family/size/profile combinations with unchanged parameter counts.
- [x] Dry-run 275M EP1 and 1.2B EP8 pretraining configs.
- [x] Dry-run weight-only midtraining and 64K long-context continuation configs.
- [x] Rerun the strict tensor/logit/block/router checkpoint gate on the pinned upstream tip.
  Beaker experiment `01KY87GSERYVG92Q8D8Q8PMTJA` passed exactly on
  2026-07-23. The report records `bitwise_equal=true` for all 216 checkpoint
  tensors and 69 fixed-input output tensors, including full logits.
- [x] Rerun exact checkpoint gates for the current 275M geometry model, a 1.2B
  EP8 model, and a completed 275M long-context model. All three tasks in Beaker
  experiment `01KY87ZKBT2A54A15AQ47ESGKZ` passed exactly on 2026-07-23.
- [x] Run representative pretraining, EP8, midtraining, long-context, and eval
  smokes. The result-bearing experiments are `01KY88T80JBGCHSF86MM7VM0QD`
  (275M pretraining, 1.2B EP8, checkpoint-backed eval),
  `01KY8AFCNMG03JD6XPRC9BS94K` (weight-only midtraining), and
  `01KY89W2PBDS597FH1KHWM52A3` (64K long context). Every result-bearing task
  exited 0. The earlier aggregate smoke exposed harness-only source-mixture,
  credential, and glob issues; it did not expose a model/checkpoint mismatch.
- [ ] Run the 275M throughput matrix only after the functional gates pass.
- [ ] Make this branch canonical and mark `jacobm/olmo-ddp` read-only.

The first throughput stage is prepared in
`v2/launchers/pretraining/manifests/275m_rope_gated_large_batch_capacity.yaml`.
It holds the 275M geometry GDN + gated-RoPE architecture fixed, disables
checkpoints and evals, and runs up to 50 steps. The MB16 controls use 2 Mi- and
4 Mi-token optimizer batches; the 2 Mi MB32 cell is the only larger legal
microbatch divisor and determines whether the previous B300 capacity ceiling
changed on this branch. All three cells need only one GPU on one Holmes node.

After that capacity gate, compare 1/2/4/8-GPU EP1 and EP2/4/8 using an untouched
upstream `ExpertParallelConfig()` on a single eight-GPU node. Do not select an
EP path or tune rowwise block counts: this gate is specifically intended to
exercise the codebase defaults. Separately compare DDP all-reduce with the
upstream reduce-scatter option. Hold model, sequence length, global batch,
optimizer, compile mode, and selected microbatch fixed within each comparison,
and report steady-state TFLOPs/GPU, TPS/GPU, aggregate TPS, step time, peak
memory, and skipped updates.

### Matched SWA throughput control

Repeat the complete capacity and parallelism study with a one-variable
sequence-mixer control named `geometry_275m_swa_rope_gated`. Keep `d_model=640`,
10 layers, the dense-first FFN at layer 0, every MoE width/router setting, 8 Q / 4
KV heads, head dimension 128, RoPE, initialization, norms, optimizer settings,
and elementwise gating on the two global-attention layers fixed. Replace only
the eight GDN mixers at layers 0--3 and 5--8 with 2,048-token sliding-window
attention; retain global attention at layers 4 and 9. The replacement SWA
layers should use the integration-wide attention template without adding an
extra attention gate, so the comparison does not introduce a second gating
intervention.

Run this in the same stages as the GDN study:

1. Validate the resolved mixer pattern and parameter counts, then run one
   checkpoint-free functional smoke.
2. Repeat the one-GPU 2 Mi MB16, 4 Mi MB16, and 2 Mi MB32 capacity gate.
3. Do not repeat the MB17--MB20 fine boundary sweep for SWA or future variants;
   the GDN sweep already answered the one-GPU capacity-envelope question.
4. Select MB16 for the paired throughput comparison and repeat the full 2 Mi /
   4 Mi, 1/2/4/8-GPU EP1, upstream-default EP2/4/8, and DDP reduce-scatter
   matrix with otherwise identical Beaker settings.

Use raw TPS/GPU, aggregate TPS, and optimizer-step time as the primary GDN/SWA
comparison because the two mixers have different FLOP definitions and active
parameter counts. Record TFLOPs/GPU and MFU within each family, plus active and
reserved memory, accumulation, skipped updates, node, and exit status. Run all
cells urgent and unallocated on Holmes with checkpoints and evals disabled.

The initial allocated capacity submission `01KY8BWKJ790QVXFSE6ZEYVAK6` was
canceled before scheduling. Its unallocated replacement was submitted at
urgent priority in the MoE workspace as Beaker work
`01KY8CBCG3BZZW6CR3616NQY57` (`minRuntime=0`, `autoResume=true`). The prepared
follow-on manifest uses the same unallocated settings and contains 16 tasks /
80 maximum concurrent GPUs across the 2 Mi- and 4 Mi-token batches.

The one-GPU capacity gate established MB16 as the largest tested production
microbatch: the 2 Mi- and 4 Mi-token controls completed 50 steps at about 453.6
and 445.1 TFLOPs/GPU with 221.5 GiB active memory, while MB32 OOMed in the
compiled dry run at 267.6 / 267.7 GiB. A finer MB17--MB20 boundary sweep was
completed as unallocated urgent work `01KY8DZZFWDJAFX754AP373DMP`. Each cell
used a per-run optimizer batch equal to exactly 16 microbatches, and disabled
checkpoints and evals. All four cells completed 50 steps without a skipped
update:

| MB | Optimizer tokens | Final-10 median TFLOPs/GPU | Final-10 median TPS/GPU | Final-10 median MFU | Stable step time | Final actual-average TPS/GPU | Active / reserved GiB | W&B |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 17 | 2,228,224 | 456.0 | 300,359 | 20.27% | 7.42s | 300,622 | 229.1 / 232.9 | [8673vgnj](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/8673vgnj) |
| 18 | 2,359,296 | 450.9 | 297,009 | 20.04% | 7.94s | 296,608 | 239.4 / 243.9 | [ob1u80gq](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ob1u80gq) |
| 19 | 2,490,368 | 446.2 | 293,901 | 19.83% | 8.47s | 294,154 | 249.8 / 253.9 | [5ckd8rrz](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/5ckd8rrz) |
| 20 | 2,621,440 | 446.5 | 294,129 | 19.85% | 8.91s | 272,252 | 260.2 / 264.6 | [b22cuic5](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/b22cuic5) |

MB20 is the measured fit boundary but has only about 3.1 GiB of reserved-memory
headroom and experienced several wall-clock stalls before recovering to about
446.5 TFLOPs/GPU over the final ten samples. MB17 is the best fine-sweep
efficiency point, but these deliberately non-production batches measure the
capacity curve only. Use MB16 for the follow-on paired 2 Mi / 4 Mi parallelism
matrix.

The GDN MB16 parallelism matrix was submitted as urgent unallocated work
`01KY8GWR68YYNQ15Q46F4D998V`: 16 tasks / 80 maximum concurrent GPUs, split
evenly between the 2 Mi- and 4 Mi-token batches.

The audited SWA control resolves to 265,665,280 active parameters, 201,440,000
active non-embedding parameters, and 3,111,197,440 total parameters. Its mixer
pattern is 2,048-token SWA at layers 0--3 and 5--8, with gated global attention
at layers 4 and 9. This is 26,427,520 fewer active parameters than the matched
GDN model; retain both TFLOPs/GPU and MFU in the report, but use raw TPS and
step time for the direct wall-clock comparison.

## Functional gate results

| Gate | GPUs / EP | Work | Result |
|---|---|---|---|
| 275M geometry RoPE + gated pretraining | 1 / EP1 | `01KY88T80JBGCHSF86MM7VM0QD` | 5 optimizer steps; exit 0 |
| 1.2B first-hybrid pretraining | 8 / EP8 `sync_1d` | `01KY88T80JBGCHSF86MM7VM0QD` | 5 optimizer steps; exit 0 |
| Checkpoint-backed validation | 1 / EP1 | `01KY88T80JBGCHSF86MM7VM0QD` | load + eval step; exit 0 |
| 275M weight-only midtraining | 1 / EP1 | `01KY8AFCNMG03JD6XPRC9BS94K` | fresh optimizer, 5 steps / 1,310,720 tokens; exit 0 |
| 275M 64K long context | 8 / EP1 | `01KY89W2PBDS597FH1KHWM52A3` | 2 steps / 4,194,304 tokens; exit 0 |

The continuation smokes use explicit bounded data globs so the migration gate
tests model loading, optimizer reset, compile, and training instead of spending
the GPU allocation scanning or packing the full production corpus. Production
MT and LC defaults are unchanged unless the explicit smoke override is set.

## Checkpoint policy

Existing OLMo DDP checkpoint directories remain the source of model weights.
Pretraining-to-midtraining and midtraining-to-long-context transitions load
weights only and start a fresh optimizer. Cross-branch optimizer-state resume
is not part of this migration.

The strict verifier must require exact key/shape/value mapping and
`torch.equal` for fixed-input logits, block outputs, and router tensors. A
tolerance-based comparison is not an acceptable migration gate.
