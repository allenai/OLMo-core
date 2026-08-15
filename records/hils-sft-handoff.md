# HiLS-Attention vs Olmo-3: SFT comparison — handoff

**Branch:** `amandab/hils-eval` @ `530a39e89` (everything below is committed and pushed)
**As of:** 2026-08-15
**Companion doc:** `records/hils-attention-eval-integration.md` — the eval backend, the runtime
recipe, and why the zero-shot numbers read the way they do. Read it before reporting any Phase 1
number.

---

## The question

Does [HiLS-Attention](https://github.com/abertsch72/HiLS-Attention)'s chunk-wise sparse attention
buy long-context ability, measured on **our** ladder rather than the paper's?

`tencent/HiLS-Attention-7B` is a ~50B-token continued-pretrain of `allenai/Olmo-3-1025-7B`. That
base is therefore the only honest control — HiLS-vs-Olmo-3 isolates the attention mechanism,
HiLS-vs-anything-else does not.

Phase 1 (zero-shot) is complete but measures **answer-format compliance, not long context** — the
models are base models and mostly fail to produce parseable answers. That is the entire reason
Phase 2 exists: SFT the two models identically, then re-run the ladder, and the comparison finally
becomes about context.

---

## Status right now

### Three arms, one pack

| Arm | Stack | Purpose | Experiment | State |
|---|---|---|---|---|
| `hils-7b` | veomni | treatment | `01M01CPCD7HS7WV40MEJ6JMYYG` | ✅ **complete** — 3106 steps / 360.9 min |
| `olmo3-7b` | veomni | control | `01M01CPHH0A6X6W8H32EXP71SY` | ❌ crashed on first backward |
| `olmo3-...-olmocore` | olmo_core | bridge | `01M018WY9MKTWV3S7VQYE08KEW` | ❌ crashed at data-loader build — **fix pushed, needs relaunch** |

**The treatment arm finished and saved:**
`/weka/oe-training-default/amandab/sft_runs/hils-7b-sft-5task-dolci25-32k/step3106`
Loss fell 0.78 (step 500) → 0.75 (1500) → 0.50 (3000). This is a loadable HF directory — veomni
writes weights and the task copies `config.json`/tokenizer files alongside, so **no DCP→HF
conversion step is required** (an earlier plan said otherwise; it is obsolete).

**One completed arm is not a result.** HiLS alone says nothing without its control. Do not report
anything from this checkpoint until `olmo3-7b` lands.

### All three read ONE materialized pack

This is the load-bearing design decision, and the reason the arms are comparable at all.
olmo_core's composable loader would re-mix and re-pack the same corpus with its own mixer and
packer, producing **different windows** — the arms would train on different data while every config
still looked identical. So the mixture is built and packed exactly once:

- `sft_shard_dataset.py` mixes to target **token** shares, packs to 32k windows, writes them out.
- Every shard length is an exact multiple of `max_seq_len`, so olmo_core recovers the identical
  windows by fixed-length chunking (`ConcatAndChunkInstanceSource`), and veomni reads them back
  verbatim (`prepacked=True`, which refuses a mismatched window rather than silently re-windowing).
- Consequence: token-matched and data-matched **coincide**. A step is a window is a document
  multiset.

Pack: `/weka/oe-training-default/amandab/sft_olmo3/packed_32k` — 24,849 windows, 700M content
tokens, 86.0% packing efficiency. Realized shares, all within 0.001 of target: contra .265,
dolci .250, rerank .137, outlier .137, oolong .119, nq .091. `oolong` is the scarcest source at
×5.50 repetition (only 15M tokens exist); the lever if that is too aggressive is its weight or the
700M budget.

---

## The two open bugs

### 1. Control arm: `out.loss` is a tuple

```
File "train_sft_veomni.py", line 247, in main
    (out.loss / accum).backward()
TypeError: unsupported operand type(s) for /: 'tuple' and 'int'
```

**This is not a simple crash, and the obvious one-line fix would paper over something that matters
for the comparison.** The log shows veomni monkey-patches transformers' loss function, and the two
arms take *different branches of it*:

```
hidden_states or weights is None, use eager loss implementation.
To enable fused linear cross entropy loss, please patch modeling.py `forward`
function to pass `hidden_states` and `weights` to `loss_function`.
```

HiLS's out-of-tree modeling code passes `hidden_states`/`weights` and so takes veomni's **fused
linear CE** path, which returns a scalar. Stock Olmo-3 modeling does not, falls back to the
**eager** path, and that path returns a tuple.

So before fixing: **confirm what the tuple contains** (read veomni's patched loss in
`/weka/oe-training-default/amandab/envs/hils-py311/`). Then decide deliberately between

- unpacking the tuple — cheap, but leaves the arms on two different loss implementations, or
- patching Olmo-3's forward to pass `hidden_states`/`weights` — puts both arms on the fused path,
  which is the choice consistent with why we set `MODELING_BACKEND=hf` in the first place.

Fused and eager CE should agree numerically, so this likely does not invalidate a comparison — but
it is an asymmetry between treatment and control, and it should be a recorded decision rather than
an accident.

**Why the treatment arm survived a bug the control did not:** the arms differ in exactly one
place — the model — and that place decides which loss branch runs. This is the class of thing a
smoke test on one arm cannot catch.

### 2. Bridge arm: fixed, not yet relaunched

`ComposableDataLoaderConfig.build()` requires `tokenizer`, `work_dir` **and** `global_batch_size`;
the bridge arm set only the latter two. The assertion fires *after* the model and optimizer are
built, ~4 minutes in. Fixed in `530a39e89` by passing the base `tokenizer_config` (not the
bos-stripped `doc_tokenizer_config` — the loader uses it solely for the collator's `pad_token_id`
and `padded_vocab_size`).

Relaunch (note `allow_dirty`, since the working tree carries unrelated WIP):

```bash
PYTHONPATH=src python src/scripts/train/memexpress/hils_sft/Olmo3-7B-sft-5task-dolci25-32k-olmocore.py \
    launch olmo3-7b-sft-5task-dolci25-32k-olmocore-v2 ai2/jupiter-cirrascale-2 \
    --launch.follow=false --launch.step_soft_timeout=null --launch.allow_dirty=true
```

Use a **fresh run name**: old checkpoints in `save_folder` trigger a silent resume.

---

## What's left, in order

1. **Fix the control's loss path** (decision above), relaunch `olmo3-7b`. ~6 h.
2. **Relaunch the bridge arm** — command above. Can run concurrently with (1).
3. **Re-run the eval ladder on the SFT'd models.** Same launcher as Phase 1 but
   `--prompt-format chat`: the SFT data was rendered with `olmo3_chatml.jinja`, and **train and
   eval templates must match**. The bridge arm saves distcp, so it uses the olmo_core eval path
   instead of `--backend hf`. Write launch ledgers per the `run-evals` skill.
4. **Success criterion:** non-floored contra/nq numbers. That is the signal SFT fixed the format
   problem and the HiLS-vs-Olmo-3 contrast is finally about long context rather than parseability.

### Open decisions for the owner

- **Pass B (xlong): 64k only.** Measured 0.617 MiB/token ⇒ 128k needs ~93 GiB. HiLS's built-in CPU
  offload does not help (it copies the whole cache back to GPU each update). 64k is also the last
  rung where the control is within its position ceiling.
- **The models do not share a position ceiling** — HiLS 131072, Olmo-3 65536. A 128k rung is native
  for one and extrapolation for the other.
- **contra@32k is unmeasurable** on this backend: prompts reach 108k tokens (3.3× the label), ~79
  GiB with weights on an 80 GB card. Both models hit it identically, so it is a missing cell rather
  than a confound.
- **A few-shot pass** would make Phase 1's format-heavy tasks measurable without SFT. Not launched;
  it is a third prompt condition beyond the raw+chat that were requested.

---

## Code map

| File | Role |
|---|---|
| `src/scripts/ctc_eval/eval/eval_lc_native.py` | `--backend hf`, `--attn-impl`, `--chat-template`; olmo_core imports are backend-conditional |
| `src/scripts/ctc_eval/lib/hils_loader.py` | Registers HiLS's out-of-tree classes; per-rank tilelang cache; veomni parallel-state init |
| `src/scripts/train/memexpress/hils_eval/` | Runtime build, on-node runner, Beaker launcher, smoke test, README |
| `src/scripts/train/memexpress/hils_sft/sft_shard_dataset.py` | Mixing, `materialize()`, prepacked reader, CLI |
| `src/scripts/train/memexpress/hils_sft/build_olmo3_sft_data.py` | Manifest-driven per-task conversion to the OLMo-3 vocab |
| `src/scripts/train/memexpress/hils_sft/train_sft_veomni.py` | The SFT task — **one file drives both veomni arms** |
| `src/scripts/train/memexpress/hils_sft/run_sft_beaker.py` | Launcher; arms in an `_ARMS` dict |
| `src/scripts/train/memexpress/hils_sft/Olmo3-7B-sft-5task-dolci25-32k-olmocore.py` | Bridge arm |
| `src/test/scripts/hils_sft/sft_shard_dataset_test.py` | 23 tests, no GPU needed |

### Artifacts on weka

| Path | What |
|---|---|
| `amandab/envs/hils-py311` | The runtime (py3.11, torch 2.8.0+cu128, tilelang 0.1.13, veomni, flash-attn 2.8.3) |
| `amandab/envs/cuda12` | CUDA 12.8 nvcc — tilelang JITs kernels and needs a real compiler |
| `amandab/hf_models/` | HiLS-7B, Olmo-3-1025-7B (+ Olmo-3-7B-Instruct for its template) |
| `amandab/olmo3-7b-base-olmocore/` | olmo_core distcp base, parity-verified (top-1 agreement 0.9934) |
| `amandab/sft_olmo3/{contra,nq,rerank,outlier,oolong,dolci}` | Per-task OLMo-3-vocab shards |
| `amandab/sft_olmo3/packed_32k` | **The pack** — 24,849 windows / 700M content tokens |
| `amandab/sft_runs/hils-7b-sft-5task-dolci25-32k/step3106` | ✅ the finished treatment checkpoint |

---

## Traps

1. **The eval/SFT runtime is a weka venv, not the Beaker image.** The image's Python is 3.12; the
   HiLS stack needs 3.11. Always `source hils_eval/hils_env_setup.sh`. Rebuild only via
   `build_hils_env_weka.sh` (GPU node; `TILELANG_ONLY=1` re-bisects tilelang without wiping the env).
2. **The SFT shards are RAW binary despite the `.npy` extension** (`ndarray.tofile`). Use
   `sft_shard_dataset.read_shard()`. `np.load` fails on them; `np.save` would prepend a header that
   olmo_core's raw reader consumes **as tokens**. A test pins this.
3. **Data selection is booby-trapped, twice.** NQ must be the p10 build
   (`nq_train_k100_hn10_2500.jsonl`; `hn` is the hard-negative *count*, and the banned 98%-hard
   family sits under near-identical names). rerank must be the CE-graded `msmarco_trainhn_*`, filed
   under the overloaded `retrieval` manifest task — the files under manifest task `rerank` are a
   deprecated format `build_prompt` refuses. `build_olmo3_sft_data.py` encodes both rules.
4. **The converter exits 0 after skipping every row.** Never treat exit status as the success
   signal; the builder fails the run when a task wrote no shards.
5. **`convert_checkpoint_from_hf.py` exits 1 on a correct conversion** — its `atol=1e-4` logit check
   is unmeetable in bf16 at 7B. The checkpoint is written before the assert. Validate with
   `ctc_suite/olmo3_parity_check.py` (CE + top-1 agreement) instead.
6. **veomni parallel state:** `init_veomni_parallel_state(shard=False)` for eval (replicated),
   `shard=True` for training (FSDP). The wrong choice is a silent ~70 GB OOM, not an error.
7. **HiLS runs at batch size 1** — its chunk grid is tied to absolute position, so left-padding a
   batch changes the mask, not just the speed.
8. **Beaker log fetches for these jobs take minutes.** `beaker experiment logs <id> > file` once,
   then grep the file, rather than re-fetching per pattern.

## Verification

```bash
# unit tests (fast, no GPU)
python -m pytest -q src/test/scripts/hils_sft/sft_shard_dataset_test.py    # 23 pass

# the rung table both eval backends share
for t in contra nq rerank oolong fiqa; do
  TASK=$t bash -c '. src/scripts/train/memexpress/singletask_ladder/ladder_rungs.sh; echo "$TASK $RUNGS $LTASK"'
done

# HiLS loads and retrieves at length (GPU)
source src/scripts/train/memexpress/hils_eval/hils_env_setup.sh
PYTHONPATH=src/scripts:src python src/scripts/train/memexpress/hils_eval/smoke_test_hils.py \
    --model /weka/oe-training-default/amandab/hf_models/tencent__HiLS-Attention-7B
```
