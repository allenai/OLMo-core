# Hybridish 7:1 vs 4:1 on the CTC suite — in-domain SFT comparison

**Goal** (prasann, 2026-09-16/17): finetune the hybridish attention-ratio ladder on a CTC-suite SFT
mix and compare 7:1 against 4:1 on high-CTC vs low-CTC tasks. Smaller scales first. Two mix
versions: 2k–4k and 2k–32k. `reorder` and `grouping_labeled` omitted from eval for speed.

Motivation: the base models are at the floor on the suite. `debug/ctc_suite_tractability/` measured
Qwen3.5-4B-Base as tractable at r2k (nq/niah/scifact 1.000 after four grader fixes), while the
hybridish curve is near zero — 275m/450m/810m mostly cannot emit an `[id]` at all, and 2p7b is
nq 0.170 / hpqa 0.220 (eval_size 200 ⚠). SFT is the question: does the suite become measurable.

## Where the checkpoints are

The released trees are a dead end for this comparison. **Every** checkpoint in
`gs://ai2-llm/yashasbls/hybridish{,-science}` and in `checkpoints/{yashasbls,tanushy}` is 4:1 —
50 runs, full attention at every 5th layer, verified by reading each run's own config rather than
its name.

The ratio axis lives in a different tree entirely:

    /weka/oe-training-default/ai2-llm/scaling-ladders/mainline/<user>/<version>/<size>-Cx<N>/<stage>/step<NNN>[-hf]

| arm | tree | 275M-Cx8 long-context |
|---|---|---|
| 7:1 | `tanushy/v0.0.1-seven_to_one_hybrid_ratio-c6e480e336d5` | 16 layers, full @ 7,15 → `step72023` |
| 4:1 | `tanushy/v0.0.1-dolma3p5_v2-584ace9eef5d` | 10 layers, full @ 4,9 → `step53839` |

Both trees carry 275M / 450M / 810M / 1.4B at Cx8 with pretrain, midtrain and long-context, in
olmo-core and `-hf`. Final long-context steps:

| size | 7:1 | 4:1 |
|---|---|---|
| 275M | 72023 | 53839 |
| 450M | 58015 | 42919 |
| 810M | 61133 | 45711 |
| 1.4B | 44124 | 34156 |

The design holds the **number of full-attention layers fixed** (275M: 2 either way) and varies the
count of gated-delta-net layers between them (14 vs 8).

## ⚠ Two confounds that must be quoted with any result

1. **Not parameter-matched.** 7:1 275M ≈ 0.29B vs 4:1 ≈ 0.23B (+26%) — the extra linear layers
   carry their own FFN. A 7:1 win is partly a bigger-model win.
2. **`scalable_softmax` differs between the arms.** The 7:1 override layers carry
   `"scalable_softmax": true`; the 4:1 override layers do not have the field at all. So
   `dolma3p5_v2` vs `seven_to_one_hybrid_ratio` measures **ratio AND SSMax together**, not ratio.
   Yashas has dedicated `v0.0.1-ssmax-*` / `ssmax_qknorm_wd` / `ssmax_wo_qknorm` trees; a matched
   baseline should come from whichever of those is 4:1 with SSMax on. Until that is checked, any
   7:1-vs-4:1 number from this pair is confounded.

## The matched pair (use this one)

| arm | tree | 275M-Cx8 long-context |
|---|---|---|
| 7:1 | `tanushy/v0.0.1-seven_to_one_hybrid_ratio-c6e480e336d5` | 16 layers, full @ 7,15 → `step72023` |
| 4:1 | `yashasbls/v0.0.1-ssmax-a04f0e8e7236` | 10 layers, full @ 4,9 → `step53839` |

Verified at 275M-Cx8: the two arms' override-layer `sequence_mixer` dicts are **byte-identical**
(`backend flash_4`, elementwise gate, rms `qk_norm` eps 1e-6, `use_head_qk_norm: true`,
`scalable_softmax: true`), the GDN base blocks match, and both ran lr 4.2e-4 / wd 0.1 / betas
(0.9,0.95) / seq 65536. The only difference is depth and override placement. `ssmax_wo_qknorm`
drops `qk_norm`; `ssmax_qknorm_wd` differs in decay; neither is the match.

## ⚠ Our olmo-core cannot build these configs

`TransformerConfig.from_dict` on either checkpoint fails:

    OLMoConfigurationError: Failed to construct 'block_overrides.7.sequence_mixer':
      class 'AttentionConfig' has no attribute 'scalable_softmax'

`gated_delta_net`, `block_overrides` and `peri_norm` all exist on `prasann/landmark` — the
divergence is confined to `AttentionConfig`.

**This is not a config field that can be added and ignored: SSMax carries LEARNED WEIGHTS.** The
7:1 checkpoint holds `model.layers.{7,15}.self_attn.ssmax_scale` of shape `[8]` — a per-head scale
on every full-attention layer. Our olmo-core has no such parameter, so a load would drop them and
run a model the checkpoint was never trained as. Both arms of the comparison have SSMax on, so
this cannot be dodged by choosing a different baseline either.

⚠ **The fork on weka is NOT the lineage that trained these.**
`/weka/oe-training-default/yashasbls/OLMo-core` (branch `val-dump`) has no `scalable_softmax`
anywhere in `src/olmo_core/nn/attention/`. It *does* carry `PackingInstanceSourceConfig` and
`NumpyDocumentSourceConfig` with `label_mask_paths`, so that lineage is compatible with our shards
— it is only the attention implementation that is missing.

⚠⚠ **The transformers plugin on weka has no SSMax either.**
`yashasbls/scaling-ladders/ladders/mainline/transformers_plugin/src/` contains no `ssmax` anywhere
(checked `modeling_`, `modular_`, `configuration_mainline_ladder.py` and the weight converter). That
weka checkout predates the 7:1 checkpoints. **This is a silent-wrong-answer hazard, not just a
blocker**: loading `...-hf/model.safetensors` with that plugin would quietly ignore every
`ssmax_scale` tensor and evaluate a model that is not the one on disk. Any eval of the 7:1 (or any
`ssmax` tree) arm must first assert that `ssmax_scale` is actually consumed.

So training needs the olmo-core that `allenai/scaling-ladders` pins (private; no `gh` on this
host). Options, in order of preference:
1. Get that pin (commit/branch) from Yashas or Tanush and train from it.
2. Port `ssmax_scale` into our `AttentionConfig` + attention forward, matching their formulation
   exactly, and verify by reproducing a forward pass against the `-hf` export before training.
   Do NOT skip that check -- a per-head scale that is applied at the wrong point is invisible in
   the loss curve and shows up only as a bad result.
3. Sidestep olmo-core entirely: SFT through HF/transformers against a *current* plugin. Our shards
   are plain uint32 token ids + a bool label mask, so an HF training loop consumes them directly.
   This is likely the fastest unblock IF a plugin build with SSMax can be obtained.

## Data

Built with the clean `ctc` package, which takes the length band as a first-class argument:

    ctc-data build --task X --split train --rungs 2k,4k[,8k,16k,32k] --train N --pool auto --out ROOT

`--pool auto` needs no GPU, index or LLM. ⚠ Synthetic tasks (`strmatch`, `cycle`, `groups4`,
`mathmatch`, `textgroups`) reject `--pool` — they have no corpus.

Then `src/scripts/data/hybridish/`:

* `build_ctc_sft_mix.py` — tags/budgets/shuffles per-task builds into one mix. Enforces one source
  per grading spec (retrieval and qdmatch would otherwise be trained from several of their own
  sources, leaving nothing to generalise to) and refuses held-out ladders outright.
* `convert_ctc_to_sft_completion.py` — plain-completion tokenizer. **Not** the Qwen chat-template
  converter: these are base models, and the ctc evaluator prompts them with a bare alpaca string.
  `corpus_reasoning_prompts.build_prompt(..., use_alpaca=True)` was verified byte-identical to the
  evaluator's `spec.build_prompt` (8841/8841 chars), so train and eval inputs match by
  construction. `--verify` scores every gold target under the evaluator's own parser and aborts
  below 1.0.
* `stage_shards_to_weka.sh` — the S3→weka two-step (an S3 push alone leaves Beaker jobs with
  MISSING paths).

Launcher: `src/scripts/train/memexpress/hybridish/hybridish-ctc-sft.py`, which reads model geometry
from each checkpoint's own `config.json`. That is what surfaced the `scalable_softmax` blocker
instead of silently training a wrong model.

### Measured (dolma2 tokenizer, 2k–4k band)

| task | role | instances | tokens | p50 len | loss tok/ex | verify |
|---|---|---|---|---|---|---|
| qdmatch_nq | high-CTC | 8000/8000 | 24.0M | 3005 | 19 | 1.0 |
| nq | low-CTC | 8000/8000 | 21.1M | 2722 | **4** | 1.0 |

⚠ **The loss signal is very thin** — 4 answer tokens per nq example, 19 per qdmatch. Per-step CE
cannot resolve small differences here; it can only rule out a gross failure to fit. Judge these runs
on the graded suite metric. The same trap produced a fake 2–4x oolong result in wave 2
([[olmo3-vs-hybrid-wave2]]).

⚠ **Train on EOS.** The first version of the converter masked the terminating EOS out of the loss,
so the model would never learn to stop — the failure mode behind both
[[eval-lc-native-nocot-fullpath-bug]] and [[native-eval-repetition-loop-bug]]. Now on by default.

### 2k–32k mix roster

`absence, contradiction, oolong, qdmatch_nq, strmatch, xabsence, nq, outlier` at 2000 examples each
across rungs 2k/4k/8k/16k/32k. ⚠ `retrieval` is drawn from **nq, not MS MARCO** as requested —
`msmarco` is not a ported retrieval ladder (MS MARCO exists in `ctc` only as `rerank`), and adding
it needs a *measured* docs-per-rung calibration, not a copy of rerank's (whose own table is flagged
"widest uncertainty band here, re-measure before quoting"). The substitution is recorded in the mix
manifest rather than left implicit.

## ⚠ absence and xabsence overshoot their 32k rung label

Measured on the dolma2 tokenizer over a 280-row multi-task probe (40/task):

| task | p50 | p90 | max | share > 32768 |
|---|---|---|---|---|
| oolong | 2532 | 11143 | 18876 | 0% |
| contradiction | 5473 | 29717 | 30338 | 0% |
| nq | 7104 | 29804 | 30258 | 0% |
| qdmatch_nq | 7941 | 30993 | 31433 | 0% |
| strmatch | 7814 | 30968 | 31067 | 0% |
| **absence** | 10152 | 32101 | **52024** | **10%** |
| **xabsence** | 15005 | 40650 | **42009** | **22%** |

Five of seven land under 32768. `absence` and `xabsence` do not: their 32k rung realizes at up to
52k and 42k tokens. This is the [[ctc-rung-labels-not-tokens]] trap in the other direction -- that
note recorded tasks running *below* their label; these two run *above* it.

Consequence: at `SEQUENCE_LENGTH=32768` with `LongDocStrategy.exclude`, **5% of instances are
dropped but 15.9% of TOKENS**, concentrated in the longest and most cross-document examples. Either
train at 65536 (the models' `max_position_embeddings`, so it is in range) or accept the loss and
quote it. The shards are tokenized at `--max-seq-len 40960` so the choice stays open; nothing is
thrown away at build time.

## Follow their recipe, not ours (user directive 2026-09-17)

**7:1 long-context launch block** (read from the checkpoint's own `config.json`, which records how
it was produced):

```
beaker_image  akshitab/olmo-core-tch2110cu130-2026-07-03
entrypoint    ladders/mainline/workloads/long_context.py train <run> ai2/holmes
lr 4.2e-4 · global_batch_size 262144 · rank_microbatch_size 131072
--model.lm_head.loss_implementation=default
1 node / 2 GPUs · preemptible=false · ws ai2/OLMo-3-moe-experiments · budget ai2/oe-other
```

⚠ That image is **deps-only** — `import olmo_core` fails inside it. Their olmo-core (the one with
`scalable_softmax`) comes from the scaling-ladders repo's own pin at runtime, so the image alone
does not supply SSMax.

**Yashas' SFT recipe** (`hybrid-small-sft-think-275M-lr{1e-4,2e-4,4e-4,8e-4}`), which is what an SFT
run here should mirror:

| knob | theirs | what I had first (wrong) |
|---|---|---|
| optimizer | `SkipStepAdamW` lr 1e-4–8e-4, betas (0.9,0.95), eps 1e-8 | AdamW-ish, lr 1e-5 |
| weight decay | **0.0** | 0.1 |
| scheduler | `LinearWithWarmup`, `warmup_fraction` 0.03, alpha_f 0.0 | fixed 50 warmup steps |
| sequence length | 32768 (`rank_microbatch_size` 32768) | 32768 |
| global batch | 2,097,152 tokens (64 x 32768) | 8 x 32768 |
| duration | **2 epochs** | fixed step count |
| data | `paths` + `label_mask_paths` + `long_doc_strategy` + `generate_doc_lengths`, `instance_source: null` | `PackingInstanceSource` |

The LR was off by ~20x and the decay, schedule, batch and duration were all invented rather than
matched. Note their seq len is 32768, which is the setting under which absence/xabsence lose the
15.9% of tokens recorded above -- they take the same hit.

## SSMax: measured, as a fallback only

Their code is the path. For the record, the formulation was also recovered empirically here, because
the plugin on weka drops the weights:

* The plugin reports `model.layers.{7,15}.self_attn.ssmax_scale` as **UNEXPECTED** and loads a model
  without them -- `MISSING keys: []`, no error. Demonstrated, not hypothesised.
* Recovered scales are per-head `[8]`, values ~7.5-13.4, on the two full-attention layers.
* Re-applying them beats dropping them by **0.31 nats** of LM loss on real CTC text
  (3.07 vs 3.38), so the reconstruction is substantively right.
  `position` (per-query prefix) edges `seqlen` 3.0699 vs 3.0733 -- too thin to call settled.
* ⚠ **It must be applied AFTER qk-norm.** These models use rms qk-norm with `use_head_qk_norm`, and
  RMSNorm divides out any scalar applied before it: scaling at `q_proj` output measured *identical*
  to dropping SSMax entirely, which reads as "the knob does nothing" rather than as a bug.

## Open

* Resolve the SSMax confound — find a 4:1 + SSMax baseline, or a 7:1 without it.
* Decide fork-vs-patch for `scalable_softmax`.
* **Eval path is unproven**: `mainline_ladder` needs the scaling-ladders transformers/vLLM plugin,
  which our ctc-suite harness has never been pointed at. Plugins live on weka at
  `yashasbls/scaling-ladders/ladders/mainline/{transformers,vllm}_plugin`.

## Results 2026-09-18/19 — and the grading bug that voided the first table

**First table was invalid; retracted.** The grader zipped shard instances to source rows by index,
but the converter drops rows over the length cap (16,000 src -> 15,183 instances, 817 dropped,
first drop at row 11), so 459/470 probed instances were scored against another example's gold. The
measured ceiling under that pairing was 0.0419 mean and the two arms "scored" 0.0278 / 0.0249 — at
the ceiling. Full diagnosis: memory `shard-src-positional-zip-bug`. Training was unaffected (it
reads only token_ids + labels_mask). The earlier single-task nq run stands: `shards_short/nq` has
zero drops, hence f1 0.757 / parse 1.0.

Fixes, all in `debug/hybridish_sft/`: converter now emits `src_index.json`;
`build_src_index.py` reconstructs it for existing shards (verified exact by element-wise
token-length match); `grade_ctc_mix.py` REQUIRES the sidecar and runs a gold self-check
(score each shard's own gold against its mapped example — must be 1.0) before any generation.

### Corrected, eval_size 500/task, 1 epoch over 15,183 instances

| task | 4:1 (1.4B) | 7:1 (2.1B) | 4:1 parse | 7:1 parse |
|---|---|---|---|---|
| nq | 0.4500 | **0.9720** | 1.000 | 1.000 |
| outlier | 0.0600 | **0.1494** | 0.452 | 1.000 |
| qdmatch_nq | 0.0007 | **0.1148** | 1.000 | 1.000 |
| oolong | 0.4140 | 0.4300 | 1.000 | 1.000 |
| contradiction | 0.0020 | 0.0204 | 1.000 | 0.996 |
| strmatch | 0.0000 | 0.0180 | 1.000 | 0.988 |
| absence | 0.0508 | 0.0113 | 0.734 | **0.102** |
| xabsence | 0.0253 | 0.0080 | 0.732 | **0.256** |

7:1 wins wherever both arms parse. The two 4:1 "wins" are 7:1 parse-rate collapses on exactly the
two tasks the 32k cap truncated hardest — formatting, not corpus tracking.

⚠ **These are TRAIN-SET numbers** — the SFT saw all 15,183 instances; the grader's holdout only
protects its own optional training loop. Upper bounds, not generalization. Also one seed, and
context length is not matched across tasks (oolong median 1,945 tokens vs strmatch 15,430).

### ⚠ SSMax is wrong under KV-cache decoding

The fork's `_apply_scalable_softmax` raises `NotImplementedError` with a KV cache. Our harness
re-attaches SSMax with a `q_norm` forward hook and generates with `use_cache=True`, so each decode
step sees T=1 and computes log(2)≈0.69 where the true value is log(prompt_len+step)≈9 — the
generated tokens are scaled ~13x too small on SSMax layers. Prefill is correct. Hits both arms
identically so the ranking should survive, but absolute values are depressed by an unknown amount.

Separately, the stock `mainline_ladder` plugin has **no SSMax support whatsoever** (zero matches in
its source; the checkpoint's `ssmax_scale` loads as UNEXPECTED and is ignored). Any olmo-eval run
against the stock plugin silently evaluates both arms with SSMax off. Port it before trusting
suite numbers.

### Next: the real suite, on Beaker

Use olmo-eval branch `prasann/ctc-suite` — 22 tasks, held-out, **500 per rung**, public HF dataset
`PrasannSinghal/ctc-suite-eval` (no weka needed), and `ctc_contradiction` already on the correct
IID realistic ladder. All 8 mix tasks map to roster rows. Blockers: the SSMax port above, and the
two checkpoints are node-local on horton (~2.8GB + 4.3GB, needs the S3->weka two-step).

## The real suite on Beaker (2026-09-19) — how it was wired

olmo-eval branch `prasann/ctc-suite-grader-fixes` (pushed, `69729fa`), 8 tasks x 2k-32k, 500/rung,
held-out, against the PUBLIC HF dataset `PrasannSinghal/ctc-suite-eval` (no weka needed for data).
39 task x rung runs per arm, sharded one Beaker job per (arm, task) = 16 jobs; serial would not
have been an overnight job. Launcher `debug/hybridish_sft/run_olmo_eval_ctc.sh`, harvest
`harvest_sweep.py`.

**The checkpoints are self-contained.** `make_self_contained_ckpt.py` copies the SSMax-patched
config+modeling into the checkpoint and sets `auto_map`, so `trust_remote_code=True` is the entire
integration -- no plugin install, for olmo-eval or anyone else. `scalable_softmax` is set from the
WEIGHTS, never a flag, so a non-SSMax model cannot be mislabelled. Verified loading with
`transformers_plugin` NOT importable, ssmax_scale live, and real generation.

Five failure modes, each now a comment in the launcher that hit it:

1. `pip install -e ".[hf]"` -> `ModuleNotFoundError: olmo_eval.cli`. Use a NON-editable install.
2. A torch-less base image -> `Not enough GPUs. Need 1 ... but only 0 available` **even though
   Beaker allocated one**: olmo-eval sizes its plan from `torch.cuda.device_count()`, and `.[hf]`
   pulls only transformers. Use a torch image and assert the count at install time.
3. gantry pins the job to a PUSHED commit; a fresh local commit fails `not our ref`. `--ref`.
4. No AWS credentials in the container -> `Unable to locate credentials`, after a clean setup.
   `--env-secret` pair + writing `~/.aws`.
5. Nested `bash -c` loop quoting shipped a literal `$c` and synced into a directory of that name.
   Unroll the loop.

Smoke test (8 instances/task) passed end to end: `ctc_nq:r2k` f1 **1.0000**, `ctc_contradiction:r2k`
f1 **0.5417** -- against 0.0020 for contradiction on the old in-house harness, which was grading
train data on the wrong ladder with SSMax broken under decode.

## Reference recipe

The runnable end-to-end path (mix -> shards -> SFT -> export -> self-contained ckpt -> olmo-eval -> harvest), with each trap named at the step that avoids it: **`src/scripts/train/hybrid-small-suite/README.md` on branch `prasann/ctc-sft-hybridish`** (that branch is self-contained: recipe, SFT script and tooling).

Related: [[ctc-final-suite-22-tasks]], [[olmo3-vs-hybrid-wave2]], [[ctc-rung-labels-not-tokens]],
[[eval-size-and-error-bars]].
