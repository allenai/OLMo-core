# Prefill-wide top-k landmark retrieval

## The gap this closes

Landmark / compressive-landmark eval applies hard top-k block retrieval **only at decode**
(`GenerationConfig.landmark_top_k_fraction`, default 0.1 — see
`generation_module.py`, which prefills `input_ids[:, :-1]` and decodes the last prompt token so the
first generated token is gated). The **prefill is dense**: every prompt token still soft-gates over
*all* past blocks. So every landmark number in results-hub was produced by a model whose prompt
representations were built with full attention — the top-k is numerical window-dressing on the last
few queries, not a sparse model.

This directory makes top-k apply to **every** query position, prompt tokens included, and measures
what that costs.

## Code

| file | what |
| --- | --- |
| `../../src/olmo_core/nn/attention/landmark_prefill_topk.py` | the implementation (new, additive: no existing file changed). Eager forward-only landmark/compressive attention with per-query top-k, plus `enable_prefill_topk(model, ...)` which monkeypatches `_prefill` on a built model at eval time. |
| `test_prefill_topk.py` | GPU validation (below). |
| `bench_prefill_topk.py` | wall-clock / peak memory vs the fused Triton prefill kernel. |
| `eval_lc_native_prefill_topk.py` | **copy** of `src/scripts/ctc_eval/eval/eval_lc_native.py` + `--prefill-topk*` flags. The production eval script is untouched. |
| `run_q06b_comp_prefill_topk_sweep.sbatch` | local sweep on the 0.6B compressive contra-n20 checkpoint (mooney). |
| `run_beaker_prefill_topk_eval.sh` / `launch_beaker_prefill_topk_eval.py` | on-Beaker sweep over the contra v2 ladder for a weka-resident checkpoint. |

## Validation (`test_prefill_topk.py`, H200)

1. `top_k=None` reproduces the **fused Triton prefill kernel** for both the plain and the compressive
   variant (max abs err 7.8e-3 in bf16, rel 2.5e-3) — so the new path is the same attention the model
   was trained with when top-k is off.
2. Per-query top-k matches the **shipped decode top-k** (`_decode_one` with
   `set_landmark_eval_decode`) at landmark rows, content rows, the final position, and rows whose
   past-block count is below `k`; across `top_k ∈ {1, 2, 4, n_blocks}`, compressive on/off, and
   `nonselected_landmark_mass ∈ {0, 0.1}`. Max abs err 7.8e-3.
3. `query_tile` (the memory knob) is bit-exact invariant.

## Result 1 — Qwen3-0.6B compressive landmark, contradiction n=20 no-cot, eval_size=488

`q06b-comp-contra-n20-sft-local/step750`. Prompts are ~1015 tokens median → **~16 landmark blocks**,
so 10% ≈ k=2, 25% ≈ k=4, 50% ≈ k=8. Decode top-k stays at 10% everywhere except the first row.
Binomial SE at f1≈0.94 with 488 examples is **±0.011**.

| setting | prefill | decode | f1 | EM |
| --- | --- | --- | --- | --- |
| dense everywhere | dense | dense | **0.962** | 0.893 |
| **shipped baseline** | dense | top-10% | **0.940** | 0.844 |
| prefill top-50% (k≈8), hard drop | top-k | top-10% | 0.944 | 0.850 |
| prefill top-25% (k≈4), hard drop | top-k | top-10% | 0.949 | 0.863 |
| prefill top-25%, α=0.1 | top-k | top-10% | 0.935 | 0.824 |
| prefill k=4, hard drop | top-k | top-10% | 0.943 | 0.848 |
| prefill top-10% (k≈2), α=0.1 | top-k | top-10% | 0.906 | 0.754 |
| prefill top-10% (k≈2), hard drop | top-k | top-10% | 0.902 | 0.744 |
| prefill k=2, hard drop | top-k | top-10% | 0.902 | 0.746 |
| prefill k=1, hard drop | top-k | top-10% | 0.784 | 0.463 |

Reading:

* **Going fully sparse at the shipped 10% costs ~0.038 f1** (0.940 → 0.902), ~3.5 SE — real, but far
  from collapse. EM takes the bigger hit (0.844 → 0.744): the model still finds the contradicting
  pair most of the time and loses whole-answer exactness.
* **At 25–50% of blocks the loss is gone** (0.949 / 0.944 vs 0.940 — inside ±0.011). So on this task
  the cliff sits somewhere between k=2 and k=4 of ~16 blocks, not at the sparsity level itself.
* **α (the compressive non-selected-mass reserve) barely matters**: 0.906 vs 0.902 at 10%, and it is
  slightly *worse* at 25% (0.935 vs 0.949). Keeping a 10% mass reserve on every non-selected
  landmark buys nothing here, which is good news — the hard drop is the version that is actually
  cheap.
* Caveat: this context is short (~16 blocks), so "10%" is only 2 blocks. The 4B contra **v2 ladder**
  (2k…32k, up to ~512 blocks) is the setting where a 10% budget is a genuinely rich selection — see
  result 2.

## Result 2 — Qwen3-4B compressive landmark, contradiction v2 ladder (in flight)

`q4b-compressive-5task-32k-nocpt-fixdata/step8550` (weka), Beaker experiment
`01KYTAGRDFDBFTXMZBYX4DEH87`, rungs 2k/8k/16k/32k × {baseline, prefill 10%, 25%, 50%, 10% hard-drop}.
results-hub baselines to beat: f1 0.783 / 0.741 / 0.626 / 0.554.
