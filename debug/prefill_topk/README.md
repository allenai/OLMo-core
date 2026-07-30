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
| `../../src/olmo_core/nn/attention/landmark_prefill_topk.py` | the implementation (new, additive: no existing file changed). Two backends + `enable_prefill_topk(model, ...)`, which monkeypatches `_prefill` on a built model at eval time. |
| ↳ eager backend | `landmark_topk_prefill_attention` — readable reference, supports the compressive `nonselected_mass` (α) reserve. ~55x slower than the fused prefill kernel. |
| ↳ fused backend | `landmark_topk_prefill_attention_fast` — two passes: a landmark-only matmul (`T × n_blocks`) gives each query its cutoff, then a Triton landmark forward floors sub-cutoff blocks out of the gate softmax. α = 0 only. Selected automatically when α = 0 on CUDA. |
| `test_prefill_topk.py` | GPU validation (below). |
| `bench_prefill_topk.py` | wall-clock / peak memory vs the fused Triton prefill kernel. |
| `eval_lc_native_prefill_topk.py` | **copy** of `src/scripts/ctc_eval/eval/eval_lc_native.py` + `--prefill-topk*` flags. The production eval script is untouched. |
| `run_q06b_comp_prefill_topk_sweep.sbatch` | local sweep on the 0.6B compressive contra-n20 checkpoint (mooney). |
| `run_beaker_prefill_topk_eval.sh` / `launch_beaker_prefill_topk_eval.py` | on-Beaker sweep over the contra v2 ladder for a weka-resident checkpoint. |
| `launch_beaker_sweep.sh` | submits one Beaker job **per config** so they run concurrently. |

## The threshold trap (fused backend)

The fused path selects blocks by comparing each block's landmark score against a per-query cutoff.
The cutoff must be the **midpoint between the k-th and (k+1)-th** landmark score, *not* the k-th
value: pass 1 computes scores with a torch fp32 matmul while the kernel recomputes them with a bf16
`tl.dot`, so at `top_k=1` the arg-max block's own score lands a rounding step below its own threshold
and gets dropped — selecting **nothing**, and falling back to local-block-only attention. That bug
measured as max abs err **3.3 (rel 0.96)** against the eager reference; the midpoint takes it to bf16
noise (7.8e-3). If you port this selection rule anywhere else, port the midpoint with it.

## Speed (H200, H=32, D=128, 36 layers, per prompt)

| T | fused landmark kernel | fused top-k | eager top-k |
| --- | --- | --- | --- |
| 8k | 0.1 s | 0.2 s | 5.4 s |
| 16k | 0.4 s | 0.7 s | 20.0 s |
| 32k | 1.4 s | **2.7 s** | 81.8 s |

So prefill top-k costs ~2x the dense landmark prefill — and note that is measuring the *masked* form,
which still touches every block. A production implementation would gather only the selected blocks
and be sublinear; nothing here claims that speedup, this is an accuracy probe.

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

## Result 2 — Qwen3-4B compressive landmark, contradiction v2 ladder ✅

`q4b-compressive-5task-32k-nocpt-fixdata/step8550` (weka), rungs 2k/8k/16k/32k, eval_size 500,
all prefill configs hard-drop (α=0). This is the setting that matters: at 32k the prompt is ~512
landmark blocks, so a 10% budget is ~52 blocks — a far richer selection than the 0.6B task's 2.

Binomial SE at eval_size=500: **±0.019** at f1≈0.78, **±0.022** at f1≈0.55.

| config | 2k | 8k | 16k | 32k | Beaker experiment |
| --- | --- | --- | --- | --- | --- |
| baseline (dense prefill, decode-only top-k) | 0.783 | 0.741 | 0.626 | 0.554 | `01KYTB8FCFAV4RY3Q71AZFK1DC` |
| prefill top-50% | 0.785 | 0.740 | 0.626 | 0.557 | `01KYTBB3BFT2DNYRJQBGHHWEZQ` |
| prefill top-25% | 0.784 | 0.742 | 0.628 | 0.559 | `01KYTBA7JS7KG6R0N1PE05XFRP` |
| prefill top-10% | 0.768 | 0.742 | 0.618 | 0.552 | `01KYTB9BZ1TV6FDBDFK1JSD23C` |
| Δ (10% − baseline) | −0.015 | +0.001 | −0.008 | −0.002 | |

**Applying top-k to the entire prefill is free on this ladder.** At the same 10% budget the decode
already uses, the largest movement is −0.015 at 2k and every rung sits inside 1 SE; 25% and 50% are
indistinguishable from baseline at every rung. The landmark gate is doing its job — the blocks the
prompt tokens actually needed were in their top 10% all along, so the dense soft-gating that prefill
was doing bought nothing.

The baseline row reproduces the results-hub numbers **exactly**, so this harness copy is faithful to
the production eval and these rows are directly comparable to everything already recorded for this
checkpoint.

### Reconciling with the 0.6B result

The 0.6B contra-n20 run *did* lose 0.038 f1 at 10%. That is not a model-size effect: those prompts
are ~16 blocks, so "10%" is **2 blocks**, and its own k-sweep shows the cliff is between k=2 (0.902)
and k=4 (0.943) — i.e. the loss tracks the *absolute* number of retained blocks, not the fraction. On
the 4B ladder 10% is 6 blocks at 2k and ~52 at 32k, all comfortably past that cliff. The one rung
where 10% shows any movement at all (2k, −0.015) is the shortest, i.e. the one with the fewest
absolute blocks, which is consistent.

**Practical read:** report top-k budgets in absolute blocks, not percentages. A percentage silently
becomes a starvation regime at short context.

Results land on weka at
`checkpoints/prasanns/q4b-compressive-5task-32k-nocpt-fixdata/eval_prefill_topk/contradiction_<tag>.json`
and are echoed into each job's log.
