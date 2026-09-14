# EMO-disabled post-training SFT sweep

New independent campaign: `olmo35-small-gptoss-sft-posttrain-noemo-20260914`.
Source lineage is **EMO PT -> no-EMO MT -> no-EMO LC -> no-EMO SFT**.
This is separate from the original sweeps and the SFT-only EMO-off ablation.

- Native LC source: `01M2FV97BETNXMM08QS0WYE02G`, successful step 5961.
- LRs: 1e-5, 5e-5, 1e-4; two epochs / 1810 steps; checkpoints at 905 and 1810.
- Each trial: 8 B300s, urgent allocated, `ai2/olmo3p5-training`.
- Same data, tokenizer/Think template, assistant-only masks, seed 1729,
  524288-token batch, 65536-token packed sequence, linear decay with 3% warmup,
  fresh optimizer, BF16 precision, eager execution, block recomputation and FLA KDA.
- Config gate checks identical data and training settings against the original
  campaign. One four-step, eight-GPU source-load/restart smoke gates all three LRs.
- Native checkpoint transfer checks source weights/buffers, fresh optimizer,
  finite updates, held-out loss and full-state resume across all eight ranks.
- Both epochs automatically upload and convert; only epoch 2 is evaluated on
  Math500, IFBench, HumanEval and AlpacaEval, after unchanged inference gates.
- Resource-free urgent Phobos controllers; allocated conversion/eval workers in
  `ai2/OLMo-3-moe-experiments`; frozen commits and submit-once records.
- Independent checkpoint/upload/export/result namespaces. Existing runs and
  source files are untouched. Keep the established uploader keep-2/1h-grace policy.

The source LC export failed its long-prompt HF/vLLM probability qualification;
native training and conversion succeeded. This campaign loads **native OLMo-core**
weights, never the HF/vLLM export. The source inference failure does not bypass
any training gate or authorize weakening qualification on the new SFT exports.
Do not claim that SFT fixes inference; assess each resulting checkpoint normally.
