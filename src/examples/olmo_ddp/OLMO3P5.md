# OLMo 3.5 hybrid latent MoE: reusable production recipe

This draft extracts the reusable architecture, accepted optimizations, export
support and checkpoint notifications from the production integration branch.
It does not launch jobs, change live runs, or include campaign watchers/IDs.

## Model family

| Name | Active parameters | Total parameters | Layers | Model / latent width | Q / KV heads | KDA / FA |
|---|---:|---:|---:|---:|---:|---:|
| Small | 794,233,472 | 12,496,341,632 | 16 | 1024 / 512 | 8 / 4 | 14 / 2 |
| Medium | 2,387,533,440 | 42,759,806,592 | 24 | 1536 / 768 | 16 / 8 | 21 / 3 |
| Large | 3,780,515,200 | 72,237,847,936 | 40 | 1536 / 768 | 16 / 8 | 35 / 5 |

Counts include embeddings and the untied output head, and incorporate independent
per-head Q/K normalization gains. They are asserted by the config builder.

Common settings: 512 routed experts, top-16, full-width softmax router,
L1-normalized selected weights with top-K scale restoration, one shared expert;
latent width exactly half model width. Block zero has a dense 8x-width FFN.
Subsequent experts have FFN hidden width equal to model width. KDA uses head
dimension 128, expand_v=2, negative eigenvalues, conv4; every eighth block uses
gated NoPE full attention with scalable softmax. Peri-RMSNorm, epsilon 1e-6;
KDA gated norm epsilon 1e-5. Untied embedding/head, embedding scale sqrt(d_model),
normal initialization std .02, seed 12536; vocabulary padded to 100352.

EMO is explicit via `build_model(..., use_emo=True/False)`. Its training document
pool is 16–512 and inference pool is 512. The two hero PT arms differ in EMO only.
Use EMO **off** for the intended downstream MT/LC/SFT recipe; those stage
launchers and datasets are not introduced in this draft.

## Training and topology

| Preset | GPUs | PP / EP | Rank microbatch (8192-token sequences) | Accumulation | LR |
|---|---:|---:|---:|---:|---:|
| SMALL | 64 | 1 / 1 | 4 | 8 | 1.1e-3 |
| MEDIUM_64 | 64 | 1 / 8 | 2 | 16 | 9.2e-4 (CBS seed only) |
| MEDIUM_128 | 128 | 1 / 8 | 2 | 8 | 9.2e-4 (CBS seed only) |

Global batch is 16,777,216 tokens. BF16 compute, FP32 master weights/Adam
state/gradient accumulation and reductions. Distributed compiled AdamW:
betas .9/.95, epsilon 1e-8, weight decay .1 (zero on embeddings), gradient
clip 1, skip-step sigma 6. Compiled model, fused-v2 MoE, no activation
checkpointing, FP8, TP, CP or TBO in these baseline presets. EP uses rowwise
NVSHMEM with capacity factor 1.25 and shared output buffers disabled.
Medium metrics collection interval is 5; small retains its original interval 1.
Large topology/batch/LR and a medium 14T LR remain unqualified: do not treat the
medium CBS LR or the large geometry as approved hero training settings.

The WSD **stable trunk** is `ConstantWithWarmup(warmup=2000)`; a hard stop does
not trigger decay. At 16Mi tokens/step, 14T is 834466 steps. Decays are distinct
full-state continuations with their own save root and WSD scheduler on the
original step clock, not an implicit switch in the stable trunk.

Checkpointing is synchronous, step zero on fresh training only, every 100 steps
through 18000, every 250 through 60000, then every 500. Both
`remove=never` and `max_checkpoints=None` are explicit: current main's default
retention must not delete unuploaded checkpoints. The optional notifier publishes
atomic ready events; all upload verification/retention lives in the independent
[uploader](https://github.com/allenai/olmo-checkpoint-uploader).

## Using the builders

Add `src/examples/olmo_ddp` to your import path. This imports configuration only:

```python
from olmo3p5_recipe import SMALL, build_model, build_train_module, build_trainer

model = build_model(SMALL, use_emo=True)
train_module = build_train_module(SMALL)
trainer = build_trainer(
    SMALL, save_folder="/checkpoints/my-run", work_dir="/work/my-run",
    stop_step=179000,  # optional ~3T stopping point; full horizon remains 14T
    inbox_dir="/checkpoints/uploader/inbox",
    run_id="my-run", lineage_id="my-run",
)
```

Use `build_data(SMALL, manifest=..., data_root=..., work_dir=...)` for the dataset
and loader configs. The manifest must be the unchanged ordered
`ladders/mainline/workloads/mixes/Dolma3p5-14t.txt` shipped in scaling-ladders.
After replacing `{TOKENIZER}` with `allenai/dolma2-tokenizer` and stripping the
trailing newline (absent in the historical file), its SHA256 must be
`992ea0c56506fe0e03140f7094b5f52022885f5accfafd390c50a8bf193b0c1b`.
Set data_root to a local mirror's `ai2-llm` directory, or `s3://ai2-llm`.
This preserves the Dolma2 tokenizer, repetition filters and data seed 928543231.
No mount, credentials, WANDB project, or scheduler allocation is hard-coded.
Attach these configs to the normal olmo-core experiment runner, including its
ConfigSaver and desired evaluation/logging callbacks. For resume, preserve
optimizer/trainer/data state and use the same lineage; for branching, use a
new save root and lineage with an explicit source checkpoint.

Before model construction, explicitly call `enable_optimizations(SMALL)` to opt
into the accepted bundle. The returned environment switches should be logged.
These are historically named `OLMO_PROFILE_*`; library defaults stay off.
The helper also selects inverse-scatter EMO masks and kernel-fun MIN_CTAS=128.
This is process-wide setup: use separate processes for A/B comparisons.

## Mainline compatibility and required requalification

The source production environment was Torch 2.11/CUDA 13 on B300s. Current main
uses newer dependency pins and `use_experimental_kernels` (not the old
`use_cute_kernel` config name). Legacy checkpoint export normalizes that key.
We preserve main's current dependencies rather than downgrade the whole repo.

The custom top-16 selection kernel reproduces **Torch 2.11's** CUDA tie ordering;
it is deliberately disabled on other Torch versions. Native top-k is the
fallback, not an unverified claim of bitwise equivalence across Torch releases.
Rounded weight-gradient accumulation is opt-in and explicitly requires
QuACK 0.5.0 and SM100. Its epilogue preserves the reference BF16 rounding before
FP32 addition. Revalidate dependency and hardware constraints before rollout.
The CTA setting still uses a guarded private kernel-fun interface.

Historical small-model 100B A/B: about 76,848 -> 100,645 TPS/GPU (64 B300s);
final-500-step CE 1.968182 -> 1.967852, with 8/11 held-out losses improved.
These are historical qualification results, **not measurements of this rebase**.
Medium early-step gradient-norm differences remain a scientific qualification
item; this draft does not assert equivalence from aggregate loss alone.

Before merge/deployment: run B300 fast-path parity (including EP8), checkpoint
save/resume, 65K EMO/recomputation tests, and a matched reference/optimized
training comparison on current dependencies. For inference, exact tensor export
and CPU modeling tests are not a substitute for end-to-end GPU generation parity.
The paired scaling-ladders plugin documents its separate fast-runtime limits.

## Review map and provenance

Logical commits separate attention architecture, EMO, expert/communication
optimizations, HF export, checkpoint notifications and recipes.
Source production tree: `d031ab975c0dd11e986b3f017e0aac2e3608b521`.
Accepted bundle baseline: `107dfa3ff42f4ee4984d5c445d587ceb7db5e4f4`.
Recomputation/long-context fix: `cdb2cb663542cfad268a1be050e87cb90beda148`.
Deferred communication experiments, FP64 inference monkeypatches and campaign
automation are intentionally excluded. KDA kernels remain in kernel-fun;
this PR consumes them rather than vendoring another copy.
