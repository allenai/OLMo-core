# OLMoE Ladder Eval Plan

This document tracks the near-term plan for evaluating ladder checkpoints with the
new `olmo-eval` checkout in `/weka/oe-adapt-default/jacobm/olmoe3/olmo-eval`.

## 2026-07-14 Long-Context Backend Decision

- Use converted HF checkpoints with vLLM for RULER.
- The full `ruler_all__65536` suite completed on the final 275M baseline
  long-context checkpoint with one Jupiter H100: 1,300 examples in 651.1
  scoring seconds (2.0 examples/second), with aggregate recall `0.1753`.
- Beaker experiment: `01KXHATN799NGFT5ADBAK343MT`.
- The standard OLMo-core provider imports correctly but rejects this OLMo-DDP
  checkpoint layout: it expects distributed keys starting with `model`, while
  OLMo-DDP stores `module.*.main`. This requires real provider support for DDP
  checkpoint loading, not a dependency or validation workaround.
- Canonical conversion/eval launchers: `launch_lc_hf_conversions.py` and
  `launch_lc_ruler_hf.py`.
- Generated summaries: `results/long_context_evals.md` and
  `results/long_context_evals.json`; raw metrics and predictions are cached
  under `results/cache/ruler/<beaker-experiment-id>/`.

## 2026-07-16 Long-Context Eval Refresh

All completed 275M/480M long-context checkpoints now have a finished RULER-64K
eval. Missing HF conversions were submitted as five parallel
one-H100 Jupiter tasks in Beaker experiment
[`01KXMA60MA5SK89ZKNCBMQK857`](https://beaker.org/ex/01KXMA60MA5SK89ZKNCBMQK857).
The conversion tasks are resumable and require a validated
`conversion_complete.json` marker before evaluation.

| model | RULER experiment | aggregate recall | state |
| --- | --- | ---: | --- |
| 275M baseline | [`01KXHATN799NGFT5ADBAK343MT`](https://beaker.org/ex/01KXHATN799NGFT5ADBAK343MT) | `0.1753` | finished |
| 275M integration deep | [`01KXMAF717DYMFG7NCRE9J69ND`](https://beaker.org/ex/01KXMAF717DYMFG7NCRE9J69ND) | `0.2518` | finished |
| 275M integration wide | [`01KXMAFF0TBDCA4Z83B6PB4052`](https://beaker.org/ex/01KXMAFF0TBDCA4Z83B6PB4052) | `0.2159` | finished |
| 480M baseline | [`01KXMAFPGY9QY7SK6MJKQHWTSW`](https://beaker.org/ex/01KXMAFPGY9QY7SK6MJKQHWTSW) | `0.2109` | finished |
| 480M integration deep | [`01KXMAJ765PQ9QYYYDAMSYW183`](https://beaker.org/ex/01KXMAJ765PQ9QYYYDAMSYW183) | `0.1978` | finished |
| 480M integration wide | [`01KXMAVY30QJ7J2X94H00XGBJ8`](https://beaker.org/ex/01KXMAVY30QJ7J2X94H00XGBJ8) | `0.2095` | finished |

Raw outputs for all six runs are cached under `results/cache/ruler/`, and the
six-record summaries in `results/long_context_evals.{json,md}` have been
regenerated. The 810M/1.2B integration-wide checkpoints are still training, so
their conversions and evals are intentionally not launched yet.

The V1 long-context trainers did not emit in-loop `eval/*` metrics. RULER is the
current final-checkpoint evaluation record; any LM/downstream validation
backfill would be additional work rather than a missing RULER job. The new
long-context HF exports and RULER outputs are also still Weka/S3 artifacts and
have not yet been archived under the V1 GCS `hf/long-context` and
`evals/long-context` prefixes.

Status refresh on 2026-07-17 03:39 UTC: the 810M and 1.2B integration-wide
continuations are still training at 43.24B and 33.72B of 100B tokens,
respectively. Their final checkpoints do not exist yet, so no new HF conversion
or RULER job is due. The six completed 275M/480M records above remain complete;
no long-context eval was relaunched in this refresh.

Status refresh on 2026-07-18 05:57 UTC: 810M integration-wide is at 61.95B,
1.2B integration-wide at 41.10B, and 810M baseline at 25.29B of 100B tokens.
The 1.2B baseline continuation is running in W&B run
[`8yqbbo8n`](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/8yqbbo8n)
from step 0. Its prior attempt stopped at step 759, before the first ephemeral
save at step 1,000, so there was no LC checkpoint to resume; the new worker
correctly reloaded the converted midtraining source and started a fresh LC
optimizer. None has a final checkpoint, so there is still no new HF conversion,
validation backfill, or RULER eval to launch. The existing six 275M/480M RULER
records remain complete and their result files were regenerated.

The matching 810M and 1.2B baseline continuations were queued on 2026-07-17 in
[experiment 01KXPYTPVH09F88PVR64G22HG3](https://beaker.org/ex/01KXPYTPVH09F88PVR64G22HG3).
Their HF conversion, RULER, and final-checkpoint validation remain intentionally
pending until training finishes.

## Training/evaluation separation decision

As of 2026-07-16, all new and resumed pretraining, midtraining, and
long-context jobs disable evaluator callbacks inside the training process,
including final-step evaluation. Periodic full validation caused severe
slowdowns, and evaluator execution was implicated in illegal-memory failures.
Once a final checkpoint exists, launch separate eval-only validation jobs.
Long-context checkpoints additionally follow the existing HF conversion plus
RULER/vLLM path. Record eval Beaker/W&B identities alongside the training run;
do not treat missing in-loop `eval/*` keys as missing training output.

## Current State

- A single base-eval smoke test has completed successfully from a converted HF
  checkpoint.
- The tested training checkpoint was:
  `/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx1-b256k-gpu2-ep1mb16-lr2e-3-r2/step15365`
- The converted HF checkpoint was written to:
  `/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/hf-checkpoints/olmoe3-tiny-275m-cx1-b256k-gpu2-ep1mb16-lr2e-3-r2/step15365`
- The smoke eval used HF loading with `trust_remote_code=true` and wrote the
  expected olmo-eval local artifacts under the job output directory.

## HF Code Location

The HF model code is in this repo:

- `src/olmo_core/nn/moe/v2/hf/configuration_olmo3moe.py`
- `src/olmo_core/nn/moe/v2/hf/modeling_olmo3moe.py`
- `src/olmo_core/nn/moe/v2/hf/convert_checkpoint.py`

Direct code read:

- `Olmo3MoeConfig.shared_expert_intermediate_size = None` means no shared
  expert.
- `Olmo3MoeSparseMLP` only instantiates `shared_expert` when
  `shared_expert_intermediate_size` is not `None`.
- `Olmo3MoeDecoderLayer` selects dense vs sparse MLPs from
  `dense_layers_indices`.
- The converter currently assumes at least one dense layer override and directly
  reads `block_cfg["shared_experts"]["hidden_size"]`.

That means no-shared checkpoints are likely a converter limitation, not an HF
architecture limitation. Dense0 checkpoints are also likely mostly a converter
limitation, because the HF model can represent `dense_layers_indices=[]`, but
the converter currently errors when no dense override exists. Qwen-like
checkpoints need a config-by-config audit; if they use the same OLMoE v2 module
graph, they should mostly need converter/config mapping, but any genuinely new
attention, norm, or block semantics would require HF model changes.

## Storage Layout

Use Jacob's checkpoint subtree as the eval source of truth, not Beaker's
ephemeral `/results` directory.

HF checkpoints:

```text
/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/hf-checkpoints/
  <train_run_name>/
    step<step>/
      config.json
      model.safetensors
      modeling_olmo3moe.py
      configuration_olmo3moe.py
      tokenizer files...
```

Eval outputs:

```text
/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/eval-results/
  <eval_group>/
    <model_alias>/
      step<step>/
        <eval_run_id>/
          manifest.json
          metrics.json
          predictions/
          requests/
```

The Beaker job can still write logs and normal job artifacts, but the launcher
should pass olmo-eval an output directory under the Weka `eval-results` tree.

## Tracking Files

Add lightweight jsonl indexes under:

```text
/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/eval-results/
  CHECKPOINTS.jsonl
  EVAL_RUNS.jsonl
```

`CHECKPOINTS.jsonl` should have one row per attempted conversion:

- `train_run_name`
- `step`
- `olmo_checkpoint_path`
- `olmo_config_path`
- `hf_checkpoint_path`
- `model_family`
- `variant`
- `convert_status`
- `convert_error`
- `converter_git_ref`
- `converted_at`

`EVAL_RUNS.jsonl` should have one row per launched eval:

- `eval_run_id`
- `eval_group`
- `train_run_name`
- `step`
- `hf_checkpoint_path`
- `task_suite`
- `task_specs`
- `limit`
- `beaker_experiment_id`
- `beaker_url`
- `priority`
- `workspace`
- `status`
- `metrics_path`
- `predictions_dir`
- `requests_dir`
- `launched_at`
- `finished_at`

## Scripts To Add

1. `scan_convertible_checkpoints.py`

   Scan the ladder run tracker/checkpoint tree and classify checkpoints as:
   `ready`, `needs_config_reconstruction`, `converter_gap`, or
   `architecture_gap`.

2. `convert_ladder_checkpoint.py`

   Wrap `olmo_core.nn.moe.v2.hf.convert_checkpoint`, write into
   `hf-checkpoints/<train_run>/step<step>`, validate with a tiny HF forward
   pass, and append/update `CHECKPOINTS.jsonl`.

3. `launch_ladder_eval.py` or explicit shell launchers

   Launch olmo-eval jobs against a selected HF checkpoint, set the Weka output
   directory, and append a pending row to `EVAL_RUNS.jsonl`.

4. `collect_ladder_eval_results.py`

   Scan `eval-results/**/metrics.json`, update `EVAL_RUNS.jsonl`, and write:
   `summary.csv`, `summary.json`, and `SUMMARY.md`.

## Eval Groups

Start with small, explicit groups:

- `base-eval-smoke`: one or a few tasks with `--limit`, used only to verify
  checkpoint loading and result writing.
- `base-eval-core`: the first real base-eval suite once smoke is clean.
- `long-context-smoke`: minimal long-context loading and scoring tests.
- `long-context-core`: the real long-context eval set, after the smoke path is
  proven.

## Launch Flow

1. Pick a completed ladder checkpoint and ensure it has or can reconstruct the
   original training config.
2. Convert to HF under `hf-checkpoints`.
3. Run a local or Beaker smoke eval with `trust_remote_code=true`.
4. Write eval artifacts directly under `eval-results`.
5. Aggregate into summary tables.
6. Only then launch wider sweeps.

## Near-Term Checklist

- Add config-save callback to all relevant future training launchers.
- Finish the converter fixes for:
  - no shared expert;
  - dense0 / no dense layers;
  - qwen-like configs, if they only require config mapping.
- Convert a small set of known-good checkpoints:
  - baseline 275m Cx1 observed-best;
  - one shared-expert checkpoint;
  - one dense-schedule checkpoint;
  - one qwen-like checkpoint, after audit.
- Add Weka-backed eval launchers.
- Add aggregation scripts and summary tables.

## 2026-07-03 Representative Eval Matrix

Current eval selection rule for the first systematic pass: convert and eval only
observed-best checkpoints for representative settings, excluding diagnostic
curiosities and incorrect-batch artifacts. For each setting, evaluate Cx1 and
Cx8 first so we can compare low-data and high-data behavior without immediately
launching every Chinchilla multiple.

Representative settings:

| Setting | Variant | Notes |
| --- | --- | --- |
| Baseline | `48E/top4` | Control, shared expert on, dense layer 1. |
| Expert granularity | `96E/top8` | Main granularity winner. |
| Total sparsity | `192E/top4` | Strongest sparsity variant so far. |
| Qwen-like active matched | `q3am128e8k` | Wider/shallower active-matched Qwen-like. |
| Qwen-like true 3D | `q3td128e8k` | Narrower/deeper Qwen-like. |
| Integration wide | `intw256e8k` | Combined intervention at baseline-ish depth. |
| Integration deep | `intd256e8k` | Combined intervention with deeper/narrower shape. |

Checkpoint count under this rule: 7 settings x 2 data scales = 14 checkpoints per
model size, or 56 total checkpoints across 275M, 480M, 810M, and 1.2B. As of this
plan update, 48 are ready: all 275M and 480M targets, 12/14 810M targets, and
8/14 1.2B targets. Missing 810M targets are integration Cx8 wide/deep final
checkpoints; missing 1.2B targets are total-sparsity Cx1/Cx8 and integration
wide/deep Cx1/Cx8.

Durable eval outputs should go under:

```text
/weka/oe-training-default/ai2-llm/evals/jacobm/olmoe3/olmo-base/
  <setting>/
    <model_size>/
      <cx>/
        <train_run_name>/
          step<step>/
            <suite_run>/
              metrics.json
              metrics/
              predictions/
              requests/
```

Smoke-test order for larger models: first convert the optimal baseline Cx1
checkpoints for 480M, 810M, and 1.2B, then run ARC-Easy `arc_easy:mc:olmo3base`
smokes through vLLM on Jupiter. Start with 1 engine / 1 H100 per checkpoint. If
all three load and score cleanly, test 4 engines / 4 H100s for the largest model
before launching full OLMoBase suites.

Initial larger baseline Cx1 conversion manifest:

```text
src/scripts/train/jacobm_olmoe_ladder/eval_baseline_cx1_larger_targets.jsonl
```

| Size | Cx | Train run | Step | LR |
| --- | --- | --- | ---: | --- |
| 480M | Cx1 | `m480-cx1-b256k-gpu4-ep1mb8-lr1.2e-3-r1` | 29022 | `1.2e-3` |
| 810M | Cx1 | `olmoe3-moe-a0-810m-cx1-b256k-gpu4-ep1mb4-lr6e-4-r1` | 52648 | `6e-4` |
| 1.2B | Cx1 | `olmoe3-moe-a0-1p2b-cx1-b256k-gpu8-ep1mb2-lr4e-4-r1` | 81190 | `4e-4` |

## 2026-07-02 Converter + Suite Smoke Update

Converter changes now tested locally:

- no shared expert is supported by omitting `shared_expert_intermediate_size`;
- dense0 is supported with `dense_layers_indices=[]` and
  `dense_mlp_intermediate_size=None`;
- `--config-path` can point conversion at a reconstructed training config when
  older checkpoints do not contain `config.json`.

Validated converted HF checkpoints with a tiny forward pass:

- baseline 275m Cx1 observed-best:
  `/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/hf-checkpoints/olmoe3-tiny-275m-cx1-b256k-gpu2-ep1mb16-lr2e-3-r2/step15365`
  - HF config summary: 12 layers, dense layer `[0]`, shared expert hidden size
    384.
- qwen-like narrow/deep 275m Cx1 observed-best:
  `/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/hf-checkpoints/q3-275m-cx1-q3td128e8k-lr2e-3-r1/step15365`
  - HF config summary: 16 layers, dense layers `[]`, no shared expert.

Launched OLMoBase non-code-exec suite jobs on Titan urgent, no-store, 1 GPU
each, group `olmoe3-275m-cx1-base-eval-suite-smoke`:

- baseline, failed anonymous HF API rate limit: https://beaker.org/ex/01KWGP16X51RQCCWT0KY3YZCAA
- qwen-like narrow/deep, failed anonymous HF API rate limit: https://beaker.org/ex/01KWGP1EME9G3XY0GZGSDBTENM
- baseline replacement with `jacobm_HF_TOKEN`: https://beaker.org/ex/01KWGPH7XTJ5NR5YST0S7YHK9M
- qwen-like narrow/deep replacement with `jacobm_HF_TOKEN`: https://beaker.org/ex/01KWGPHFPBPRZ8ZRHVKYD2EP5R

The first two launches failed while expanding Hugging Face datasets anonymously and hit HTTP 429s; replacements inject `jacobm_HF_TOKEN` as `HF_TOKEN`.

These jobs include eight OLMoBase suites: `mcqa_stem`, `mcqa_non_stem`, `gen`,
`math`, `easy:qa:rc`, `easy:qa:bpb`, `easy:math:bpb`, and `easy:code:bpb`
(279 expanded tasks per checkpoint). The `olmobase:code` execution suite is
still opt-in because it uses the `codex_universal` harness and Modal secrets.

Launcher added in the olmo-eval checkout:

```text
/weka/oe-adapt-default/jacobm/olmoe3/olmo-eval/scripts/olmoe/launch_275m_cx1_base_eval_suite.sh
```


## 2026-07-03 Auto-Image Old-Path Eval Smoke

The working path is the direct/old `python -m oe_eval.run_eval` Beaker spec using
`oe-eval-beaker/oe_eval_auto` (`01KPEDSKAH465STZBH0QPBC6KV`), not the newer
`olmo-eval beaker launch` path. The newer launcher failed with this image because
it tried to run the current `olmo-eval` checkout under Python 3.11 while the
checkout requires Python >=3.12.

Validated smoke jobs:

- HF backend correctness smoke: https://beaker.org/ex/01KWK6HSQYTXPH21S88Y3H53QE
  - exit 0
  - task `arc_easy:mc`, limit 8
  - primary score `0.375`
- vLLM backend smoke: https://beaker.org/ex/01KWK6C5F2XNY25N5A8DJXFCVS
  - exit 0
  - task `arc_easy:mc`, limit 64
  - primary score `0.265625`
  - model loaded via vLLM 0.11.0 V1 with `TransformersForCausalLM` fallback and
    `enforce_eager=true`.

Two fixes were required:

1. The converted HF model code must pass the current Transformers mask-helper
   kwargs: `input_embeds` and `cache_position`. This is fixed in
   `src/olmo_core/nn/moe/v2/hf/modeling_olmo3moe.py`; already-converted HF
   checkpoints need to be regenerated or patched before eval.
2. The auto image's `oe_eval` defaults do not include `enforce_eager`, so the
   direct Beaker command patches `/stage/oe_eval/default_configs.py` inside the
   container before invoking `oe_eval.run_eval`. This is container-local and does
   not mutate the image, repo, or checkpoint.

Reusable launcher:

```text
src/scripts/train/jacobm_olmoe_ladder/launch_oe_eval_auto_vllm_smoke.sh
```

Important operational details:

- Clear the cached Transformers remote-code module before each run, otherwise
  `HF_HOME/modules/transformers_modules/<step>` may keep stale copied code.
- Use the mounted Weka path (`/weka/oe-training-default/...`) for `model_path`.
- Keep `add_bos_token=false`, matching the tokenizer/config behavior used in the
  successful smoke.
- vLLM currently uses the generic Transformers fallback for `Olmo3MoeForCausalLM`;
  this works with `enforce_eager=true` but is probably not the fastest possible
  implementation.

## Open Questions

- Which exact olmo-eval base suite should become `base-eval-core`?
- Which long-context tasks are the right first smoke tests?
- Do we want eval metadata mirrored into the repo docs, or should the Weka jsonl
  files be the only source of truth?
- Should converted HF checkpoints include a copy of the exact converter git ref
  or a small `conversion_manifest.json` inside each checkpoint directory?

## 2026-07-02 Eval Throughput / Backend Follow-up

The converted baseline HF checkpoint loads correctly and can run olmo-eval, but
we should not launch the full eval batch until the fast vLLM backend is fixed in
the eval image/package stack.

Smoke results:

- Default vLLM failed because FlashInfer JIT tried to call
  `/usr/local/cuda/bin/nvcc`, which is not present in the image.
- Setting `VLLM_ATTENTION_BACKEND=FLASH_ATTN` did not affect vLLM 0.19.1 in this
  image; it still failed through the FlashInfer/nvcc path.
- Passing `attention_config.backend=FLASH_ATTN` and `enforce_eager=true` got the
  provider running quickly and started scoring around 280-325 items/sec, but then
  crashed in `quack.activation` / `vllm_flash_attn` with
  `AttributeError: module 'cutlass.cute.core' has no attribute 'ThrMma'`.
- `attention_config.backend=TORCH_SDPA` failed because that backend was not
  registered in this vLLM build.
- `attention_config.backend=TRITON_ATTN` and the direct provider kwarg
  `attention_backend=FLASH_ATTN` both hung during provider initialization/logging
  rather than producing a usable smoke.
- HF provider with 2 GPUs worked functionally, but processed only about
  0.5 items/sec on the OLMoBase suite, so it is too slow for the intended sweep.

Conclusion update: this was true for the earlier new-launcher/provider path, but
it is no longer the best current plan. As of 2026-07-03, the direct/as-is
`oe_eval.run_eval` path with `oe_eval_auto` works for vLLM smoke evals when the
converted HF checkpoint uses the current Transformers mask-helper kwargs and the
container-local `MODEL_DEFAULTS` is patched to allow `enforce_eager=true`. This
is still a Transformers fallback path, so speed should be measured on the real
suite before launching a large eval batch.

## 2026-07-03 oe_eval_auto Image Smoke

Tried the current `olmo-eval beaker launch` path with the older/internal
`oe-eval-beaker/oe_eval_auto` image, keeping the converted baseline 275M Cx1 HF
checkpoint and vLLM provider settings:

- experiment: https://beaker.org/ex/01KWK45KVFMTJZ6PZHTKG6TYVT
- image: `oe-eval-beaker/oe_eval_auto` / Beaker image
  `01KPEDSKAH465STZBH0QPBC6KV`
- task suite: `olmobase:mcqa_stem`
- provider: `vllm`, 1 GPU, TP=1, max model length 4096

This failed before vLLM startup. The image's active Python environment is
Python 3.11.14 (`/stage/.venv`), while the current `olmo-eval` branch requires
Python >=3.12. Gantry therefore failed during the overlay install:

```text
No solution found when resolving dependencies:
the current Python version (3.11.14) does not satisfy Python>=3.12
```

So `oe_eval_auto` cannot currently be used with the new launcher overlay as-is.
The next useful paths are either an image with Python 3.12 plus a compatible
vLLM/CUDA stack, or a direct/as-is launch against the older eval runtime already
baked into `oe_eval_auto`.

## 2026-07-03 Eval Backend / Speed Notes

Prompt-matched HF comparison on `arc_easy:mc`, limit 64, completed with the
same first request as the vLLM smoke but a different primary score:

- HF backend: https://beaker.org/ex/01KWK70FMMTEQSFKBN09SRZ3ZT
  - primary score `0.296875`
- vLLM backend: https://beaker.org/ex/01KWK6C5F2XNY25N5A8DJXFCVS
  - primary score `0.265625`

So the earlier HF/vLLM difference was not only a prompt subset mismatch; the two
backends need prediction-level comparison before we treat them as numerically
interchangeable.

The direct/as-is `oe_eval.run_eval` path in `oe_eval_auto` cannot run the new
`olmobase:*` suite aliases. A compatibility probe failed with
`ValueError: Task olmobase:mcqa_stem not found in the task registry`, and a
registry introspection job found old task names like `arc_easy:mc` but no
`olmobase:*` or `*:olmo3base` tasks. This means full OLMoBase evals should use
the newer `olmo-eval` suite layer once the Python/image issue is resolved, rather
than trying to force the old image path to understand new suites.

Proxy vLLM speed test on the old/direct path:

- job: https://beaker.org/ex/01KWK7E4H1G00ZXSKPK0H8GCHG
- task: `arc_easy:mc`, limit 1024, one GPU
- vLLM config: `tensor_parallel_size=1`, `enforce_eager=true`,
  `TransformersForCausalLM` fallback
- model load: about 4.7 seconds after weight loading begins
- scoring: 2,281 loglikelihood requests in about 34 seconds, roughly 66.5
  loglikelihood requests/sec

A 2-GPU proxy comparison was submitted as
https://beaker.org/ex/01KWK7E5VQ2P8V2B3WG5R723B5 but had not started when these
notes were written. For these small 275M checkpoints, the expected efficient
scaling strategy is multiple independent 1-GPU vLLM engines, not one
tensor-parallel engine. The newer `olmo-eval` path supports this via
`provider.num_instances=N` with `provider.kwargs.tensor_parallel_size=1`; the old
direct path does not expose that cleanly.


## 2026-07-03 New-Path OLMoBase Full-Suite Speed Attempt

Tried to get a real OLMoBase-suite speed number for the converted baseline 275M
Cx1 HF checkpoint on the newer `olmo-eval` path with vLLM and one GPU. This is
separate from the older `oe_eval_auto` direct path, which is fast on proxy tasks
but does not have the current `olmobase:*` suite aliases.

Attempts and current state:

- `https://beaker.org/ex/01KWKB3VCFYTBWRCEF6FF8J7K3`: fresh HF cache plus
  `enforce_eager=true`; failed during vLLM dummy forward because the HF shim was
  passing `input_embeds` into the causal-mask helper instead of
  `inputs_embeds`.
- Patched `input_embeds` -> `inputs_embeds` in the repo HF shim and in the
  converted checkpoint copy for the optimal 275M Cx1 baseline.
- `oe-eval-beaker/oe_eval_auto` through the new launcher overlay still fails at
  dependency resolution because the image Python is 3.11 while this `olmo-eval`
  branch requires Python >=3.12.
- Python-3.12 image `01KVTNKQXYACY9J3HEE265KJXA` got through package setup, but
  the full `olmobase:math` alias includes `gsm_symbolic:p1:olmo3base`, which was
  not registered/preparable in this environment.
- Relaunching the non-code-exec suite minus only `gsm_symbolic:p1:olmo3base`
  reached task/dataset generation but exited `139` before vLLM provider init
  (`https://beaker.org/ex/01KWKCCTM05ERC121H0D9ZMCVY`).

Current conclusion: we still do not have a trustworthy full-suite speed number.
The next useful debugging path is to split OLMoBase into smaller chunks, or lower
any task-prep concurrency / isolate the task that segfaults, then re-measure with
vLLM once provider initialization is reached.

## 2026-07-03 New-Launcher vLLM Engine-Death Repro Pair

Relaunched the same new `olmo-eval beaker launch` vLLM configuration that had
previously started scoring and then produced mass `EngineDeadError` task failures
on Titan/B200s. The goal is to distinguish a consistent runtime/image issue from
an intermittent engine failure, and to compare B200 vs H100 behavior.

Both jobs use image `01KVTNKQXYACY9J3HEE265KJXA`, commit `eb8da23` of
`allenai/olmo-eval`, the converted 275M Cx1 baseline HF checkpoint, the three
suites `olmobase:mcqa_stem`, `olmobase:mcqa_non_stem`, and `olmobase:math`, 2
GPUs, `provider.num_instances=2`, `tensor_parallel_size=1`, `bfloat16`,
`max_model_len=4096`, `max_num_seqs=128`, `max_num_batched_tokens=32768`,
`attention_config.backend=FLASH_ATTN`, `enforce_eager=true`, and
`batching.chunk_size=128`.

| Name | Cluster | Purpose | Beaker |
| --- | --- | --- | --- |
| `olmoe3-275m-cx1-baseline-vllm2-fa-eager-rerun-titan` | `ai2/titan` / B200 | Reproduce original B200 engine-death behavior. | https://beaker.org/ex/01KWMKBXJ0PWVGSZFX031TVQR8 |
| `olmoe3-275m-cx1-baseline-vllm2-fa-eager-rerun-jupiter` | `ai2/jupiter` / H100 | Check whether the same image/runtime fails similarly on H100. | https://beaker.org/ex/01KWMKCNSJ455QGCKMT8F34W3C |
