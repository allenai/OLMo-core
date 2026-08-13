# Running HiLS-Attention-7B on our long-context ladder

**2026-08-13.** How a third-party model that olmo_core cannot express gets scored on our eval
suite, and the eight failures it took to get there. Code: `src/scripts/train/memexpress/hils_eval/`
(family README has the launch commands); backend: `eval_lc_native.py --backend hf`.

## What HiLS is

[HiLS-Attention](https://github.com/abertsch72/HiLS-Attention) (upstream `Tencent-Hunyuan`,
arXiv 2607.02980) is chunk-wise sparse attention that learns chunk selection end-to-end under the
LM loss. The released `tencent/HiLS-Attention-7B` is **continued pre-training of
`allenai/Olmo-3-1025-7B`** on ~50B tokens: `chunk_size=64`, `hils_topk=32`,
`full_attn_interleave=4`, `sliding_window=512`, HoPE positional encoding,
`max_position_embeddings=131072`, OLMo-3 vocab (100278).

Because it is a CPT of Olmo-3-1025-7B, **that base is the only honest control** — it is the one
comparison that isolates the attention mechanism rather than the model family.

## Why it needed a new backend

The HF repo cannot be loaded with `AutoModelForCausalLM`:

* no `auto_map`, so `trust_remote_code=True` finds nothing;
* `model_type: olmo_hils`, which no transformers version knows;
* the modeling code is out-of-tree in the HiLS repo and imports `tilelang` (JIT CUDA kernels called
  on **every forward**, not just training) and `veomni`.

`ctc_eval` has no olmo_core dependency, so the coupling in `eval_lc_native.py` was four lines. The
hf backend swaps model construction and the `generate` call; ladder resolution, prompt construction
and scoring are shared. The per-task rung table moved to `singletask_ladder/ladder_rungs.sh` so the
two backends cannot drift onto different ladders.

## The runtime recipe (`build_hils_env_weka.sh`)

A Python **3.11** venv on weka at `/weka/oe-training-default/amandab/envs/hils-py311`, plus a CUDA
12.8 prefix at `envs/cuda12`. Built once; every eval job activates it via `hils_env_setup.sh`.
Verified contents: python 3.11.15, torch 2.8.0+cu128, transformers 4.57.3, **tilelang 0.1.13**,
veomni @441e1b2 (+torchdata, datasets, diffusers, liger_kernel), flash-attn 2.8.3.

Each line below is a failure that actually happened, in order:

| Symptom | Cause | Fix |
|---|---|---|
| `veomni requires a different Python: 3.12.12 not in '<3.12,>=3.11'` | The beaker image is py3.12 | Build a py3.11 venv (what HiLS pins) |
| tilelang import: `attribute '__dict__' of 'type' objects is not writable` | tilelang on py3.12 | same |
| `ValueError: Type 'ffi.Tensor' already has a registered class` | tilelang declares `apache-tvm-ffi>=0.1.11,<0.1.13` — **a range PyPI does not contain** (0.1.9 → 0.1.13.post3); the resolver picks something untested and it collides with the tvm bundled in the wheel | Use tilelang 0.1.13 with its resolved tvm-ffi (older tilelang needs `apache-tvm-ffi==0.1.9`) |
| `No CUDA or HIP or MPS available on this system` on a working H100 | `libnvrtc.so.12` is vendored by torch's cu128 wheels but not on `LD_LIBRARY_PATH` | `hils_cuda_paths.sh` |
| `No such file or directory: .../nvidia/cuda_nvcc/bin/nvcc` | tilelang JITs every kernel and needs a real `nvcc`. There is **no pip route on CUDA 12** — tilelang's own `env.py` says "only `nvidia-cuda-nvcc>=13.0` works. `nvidia-cuda-nvcc-cu12`, etc. only installs `ptxas`" (verified: the 12.9.86 wheel ships exactly one binary, `ptxas`) | Unpack NVIDIA's CUDA 12.8 redist (`cuda_nvcc` + `cuda_cudart` + `cuda_cccl`) into `envs/cuda12` |
| `fatal error: cuda_runtime.h` / `fatal error: nv/target` | cudart / cccl components missing | add both; guard the prefix on a header from each |
| `ModuleNotFoundError: diffusers` at model load, after a green build | `import veomni` is a weak check — the modeling code imports veomni **submodules**, which pull in more | Probe by importing `models.FlashHiLS.modeling_olmo_hils` itself |
| Every 8-way `torchrun` job dies on the first generate: `ValueError: The product of parallel sizes should be equal to the world size` | HiLS's forward calls `veomni.get_parallel_state()`; with nothing initializing it that builds a default `ParallelState` asserting `pp*dp*cp*ulysses*tp == world_size`, all 1 → holds at world=1, fails otherwise | `init_veomni_parallel_state()` in `hils_loader.py`: `dp_replicate_size=world_size` |

Two of those were **self-inflicted and produced wrong verdicts**, which is the more useful lesson:

* The tilelang candidate list was built with `sorted()` on version strings, which puts `0.1.13`
  **before** `0.1.6` lexicographically — so the bisect silently skipped the four newest releases,
  and 0.1.13 is the one that works. Sort versions with a version key, never lexicographically.
* A CUDA prefix left half-populated by an earlier failure passed an `nvcc`-only guard, so the
  install was skipped and four candidates were rejected for a missing header that had nothing to do
  with tilelang. An install trigger and its verification must check the **same** file list.

## Operational facts

* **flash-attn is required, not optional.** Without it the dense layers fall back to sdpa and
  transformers materializes a `(B,1,T,T)` mask for padded prompts — ~17 GB per attention call at
  32k. Use the prebuilt wheel for the exact torch/python/ABI (detect ABI from
  `torch._C._GLIBCXX_USE_CXX11_ABI`); building from source is 30+ minutes per attempt.
* **FA3 is absent**, so the checkpoint's `_attn_implementation: flash_attention_3` must fall back.
  The harness probes fa3 → fa2 → sdpa. This only selects the kernel for the interleaved *dense*
  layers — the sparse path is tilelang regardless — so it is a speed choice, not a semantic one.
* **batch size 1.** HiLS ties its chunk grid to absolute position, so left-padding a batch changes
  the mask (same constraint as our landmark/compressive variants). Separately, Olmo-3-7B is MHA
  (~0.5 MB/token of KV) and batch 4 at the 16k rung exhausts an 80 GB H100.
* **Per-rank `TILELANG_CACHE_DIR`**, or 8 ranks race to JIT-compile into one `$HOME` cache.
* Measured: 7.31B params bf16; 7,969-token prefill in 47.6 s (first call includes JIT), peak
  18.4 GiB, needle retrieved.

## Reading the results

Both models are **BASE** — no instruction tuning, and neither tokenizer ships a `chat_template`.

* `raw` is the honest condition. These rows are comparable to *each other*, **not** to our SFT'd
  Qwen3.5 arms, which were trained on these tasks.
* `chat` uses a supplied template: plain ChatML over the OLMo-3 vocab's real `<|im_start|>` /
  `<|im_end|>` tokens (`ctc_eval/lib/chat_templates/olmo3_chatml.jinja`), attached identically to
  both models. Deliberately **not** Olmo-3-7B-Instruct's shipped template, which injects a
  function-calling system preamble into every prompt.
* **Ceilings differ**: HiLS-7B is 131072 positions, the Olmo-3 control is 65536. The 32k base
  ladder is inside both. A 128k xlong rung would be native for HiLS and pure extrapolation for the
  control — which is exactly HiLS's headline claim, so it has to be labelled, not averaged over.
* The hf backend has **no chunked prefill** (that is an olmo_core generation-module feature), so
  ≥256k rungs are one-shot prefills and will OOM. 64k/128k is the supported xlong range here.
