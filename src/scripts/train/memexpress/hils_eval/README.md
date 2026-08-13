# hils_eval — scoring third-party HF models on our long-context ladder

Evaluates **HuggingFace-format** checkpoints on the same eval bundle, ladder and metrics as our own
runs. Built for [HiLS-Attention](https://github.com/abertsch72/HiLS-Attention)
(`tencent/HiLS-Attention-7B`) and its control, the `allenai/Olmo-3-1025-7B` base it was
continued-pretrained from.

Everything here is the **hf backend** of the existing eval path — `eval_lc_native.py --backend hf`.
Ladder resolution, prompt construction and scoring are shared code; only model construction and the
`generate` call differ. That is deliberate: a third-party number is only worth having if it was
produced by the same measurement as the numbers it will sit beside.

| File | What |
|---|---|
| `build_hils_env_weka.sh` | Builds the HiLS runtime once, as a py3.11 venv on weka. Run before anything else. |
| `hils_env_setup.sh` | Activates that venv + checks out the HiLS repo. `source` it — it exports `$HILS_REPO`. |
| `smoke_test_hils.py` | GPU smoke test: imports, load, short generate, long-prefill probe with timing + peak memory. |
| `run_beaker_hf_eval.sh` | On-node runner (the hf twin of `singletask_ladder/run_beaker_multirung_eval.sh`). |
| `run_hf_beaker_eval.py` | Beaker launcher: one job per `(model, task)`. |

Model loading lives in `src/scripts/ctc_eval/lib/hils_loader.py`; the per-task rung table is shared
with the olmo_core runner via `singletask_ladder/ladder_rungs.sh`.

## Why HiLS needs any of this

The released 7B is an HF checkpoint that **cannot** be loaded with `AutoModelForCausalLM`:

* the HF repo ships **no `auto_map`**, so `trust_remote_code=True` finds nothing;
* `model_type` is `olmo_hils`, which no transformers version knows;
* the modeling code is out-of-tree in the HiLS repo, and imports `tilelang` (JIT CUDA kernels,
  called on **every forward** — not optional even for inference) and `veomni`.

`hils_loader.register_hils()` puts the repo on `sys.path` and registers the classes; that is the
one place that knows this.

`veomni` is installed `--no-deps` on purpose: the modeling code imports ~7 trivial symbols from it
(logging, parallel state, a checkpointing base class), while its full dependency set pins
torch/transformers and would rebuild the environment underneath us.

## Prerequisites

### 1. Build the runtime (once)

The OLMo-core image's python is **3.12**, on which the HiLS stack does not run at all — `veomni`
declares `requires-python <3.12`, and `tilelang` 0.1.9 raises on import
(`AttributeError: attribute '__dict__' of 'type' objects is not writable`). HiLS pins python 3.11 +
torch 2.8.0, so we reproduce that in a venv on weka instead of fighting it:

```bash
gantry run --name build-hils-env -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/neptune-cirrascale --cluster ai2/ceres-cirrascale \
  --cluster ai2/saturn-cirrascale --cluster ai2/jupiter-cirrascale-2 \
  --gpus 0 --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --weka oe-training-default:/weka/oe-training-default \
  --no-python --allow-dirty --timeout 0 --yes -- \
  bash src/scripts/train/memexpress/hils_eval/build_hils_env_weka.sh
```

Lands at `/weka/oe-training-default/amandab/envs/hils-py311`, with the managed CPython alongside it
at `envs/pythons` — that placement matters, since a container-local interpreter would leave the
venv dangling in the next job. `REBUILD=1` forces a rebuild.

`flash-attn` is intentionally absent (no wheel for this torch/python pair; pip would compile for
30+ minutes). Nothing needs it: the HiLS sparse path is tilelang, and the interleaved dense layers
run on `sdpa`.

### 2. Stage the weights

Stage the weights to weka first — never pass a Hub id, or every job depends on huggingface.co being
reachable at startup:

```bash
gantry run --name stage-hf-models -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/neptune-cirrascale --cluster ai2/ceres-cirrascale \
  --cluster ai2/saturn-cirrascale --cluster ai2/jupiter-cirrascale-2 \
  --gpus 0 --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --weka oe-training-default:/weka/oe-training-default \
  --python-manager conda --system-python \
  --env HF_HUB_DISABLE_XET=1 --install true --timeout 0 --yes -- \
  python src/scripts/data/stage_hf_models_weka.py \
    --repos tencent/HiLS-Attention-7B,allenai/Olmo-3-1025-7B
```

`--python-manager conda --system-python` is load-bearing: without it the job runs in gantry's own
venv, which has no `huggingface_hub`.

Staged at `/weka/oe-training-default/amandab/hf_models/<repo with '/' -> '__'>`.

## Launching

Two passes per model per prompt condition (the `run-evals` standing rules apply here too):

```bash
# Pass A -- base ladder, all 9 tasks including the 4 OOD ladders
PYTHONPATH=src python src/scripts/train/memexpress/hils_eval/run_hf_beaker_eval.py \
    ai2/jupiter-cirrascale-2 --task all \
    --model /weka/oe-training-default/amandab/hf_models/tencent__HiLS-Attention-7B \
    --model-name hils-7b --prompt-format raw --eval-tag base-raw

# Pass B -- xlong rungs, in-distribution tasks only
PYTHONPATH=src python src/scripts/train/memexpress/hils_eval/run_hf_beaker_eval.py \
    ai2/jupiter-cirrascale-2 --task contra,nq,outlier,rerank,oolong \
    --model .../tencent__HiLS-Attention-7B --model-name hils-7b \
    --prompt-format raw --xlong --xlong-only --xlong-rungs 64k,128k --eval-tag xlong-raw
```

`--eval-tag` is effectively required: every model runs in at least two prompt conditions, and
without a tag the second one overwrites the first's result files.

## Traps specific to this path

**These are BASE models.** Neither `Olmo-3-1025-7B` nor `HiLS-Attention-7B` is instruction-tuned,
and neither tokenizer ships a `chat_template`. Consequences:

* `--prompt-format raw` is the honest default. Absolute numbers will be well below our SFT'd
  arms for reasons that have nothing to do with attention, so these rows are comparable to *each
  other*, not to the SFT ladder.
* `--prompt-format chat` needs an explicit template. The runner defaults to
  `ctc_eval/lib/chat_templates/olmo3_chatml.jinja` — plain ChatML over the OLMo-3 vocab's real
  `<|im_start|>`/`<|im_end|>` tokens. It is **not** `Olmo-3-7B-Instruct`'s shipped template, which
  injects a function-calling system preamble into every prompt when the caller supplies no system
  message. Attaching the same file to both models is what makes "chat" mean one thing across the
  comparison.

**The two models do not share a position ceiling.** HiLS-7B ships
`max_position_embeddings=131072`; Olmo-3-1025-7B ships **65536**. So the 128k xlong rung is
in-ceiling for HiLS and pure extrapolation for the control — the runner warns, and any 128k
comparison has to say so. HiLS's whole claim is extrapolation past the trained length (HoPE), so
this asymmetry is a *finding to report carefully*, not a bug to paper over.

**HiLS runs at batch size 1.** Its chunk grid and sliding window are tied to absolute position,
exactly like our landmark/compressive variants: left-padding a batch shifts every chunk boundary,
so batching changes the mask rather than just the speed. The runner forces this and says so.

**No chunked prefill on the hf backend.** `PREFILL_CHUNK_SIZE` is an olmo_core generation-module
feature. An hf-backend rung at ≥256k is a one-shot prefill and will very likely OOM on one 80GB
card; 64k/128k are the supported xlong range here.
