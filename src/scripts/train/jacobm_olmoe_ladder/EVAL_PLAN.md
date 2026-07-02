# OLMoE Ladder Eval Plan

This document tracks the near-term plan for evaluating ladder checkpoints with the
new `olmo-eval` checkout in `/weka/oe-adapt-default/jacobm/olmoe3/olmo-eval`.

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

## Open Questions

- Which exact olmo-eval base suite should become `base-eval-core`?
- Which long-context tasks are the right first smoke tests?
- Do we want eval metadata mirrored into the repo docs, or should the Weka jsonl
  files be the only source of truth?
- Should converted HF checkpoints include a copy of the exact converter git ref
  or a small `conversion_manifest.json` inside each checkpoint directory?
