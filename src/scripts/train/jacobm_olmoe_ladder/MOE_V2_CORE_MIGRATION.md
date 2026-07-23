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

The capacity experiment was submitted at urgent priority in the MoE workspace
as Beaker work `01KY8BWKJ790QVXFSE6ZEYVAK6`. The prepared follow-on manifest
contains 16 tasks / 80 maximum concurrent GPUs across the 2 Mi- and 4 Mi-token
batches; render it with the capacity-winning MB16 or MB32 before submission.

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
