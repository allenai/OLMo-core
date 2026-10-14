# recipes/train/setA/ — dense Qwen3.5-4B SFT on the CTC setA mix, 32k vs 256k

Dense (full-attention) Qwen3.5-4B SFT on **setA**, the 11-task CTC mix built IID with the
olmo-eval CTC suite at every rung from 2k to 256k. It is a controlled pair on maximum training context:

| Arm | Window | Examples | Geometry | Script |
|---|---|---|---|---|
| `32k` | 32,768 | only those ≤32,768 tokens; longer ones **dropped**, not truncated | 1 node, no CP, DP=8, grad-accum 4 | `Qwen3.5-4B-dense-ctc-setA-32k-SFT.py` |
| `256k` | 262,144 | all | 1 node, Ulysses CP=8, DP=1, grad-accum 4 | `Qwen3.5-4B-dense-ctc-setA-256k-SFT.py` |
| `256k-2node` | 262,144 | all | 2 nodes, CP=4, DP=4 (amandab's exact geometry) — fallback | `Qwen3.5-4B-dense-ctc-setA-256k-2node-SFT.py` |

Everything else is shared, from one builder (`_qwen35_setA_common.py`):

- Base checkpoint: `q35-4b-dense-256k-fix/step2385`, loaded weights-only.
- Data files, one epoch, seed 34521.
- 1,048,576 tokens per step at LR 4e-5 with 3% linear warmup.

The 32k arm trains on a strict subset of the 256k arm's data, so it takes fewer steps.

## Reproduce

Every step runs on Beaker and is pinned to a pushed commit. The jobs are unallocated (workspace
`ai2/flex2`, budget `ai2/oe-other`), run at priority `urgent`, and use jupiter H100s.

```bash
L=recipes/train/setA/launch_setA_sft.py
python $L prep       # CPU: shard checks, drops per window, base ckpt + marker audit, packed windows/steps
python $L 32k        # 1 x 8 H100
python $L 256k       # 1 x 8 H100   (or: python $L 256k-2node)
```

Add `--dry-run` to print the gantry command without submitting.

### Task set, data root, base checkpoint

`launch_setA_sft.py` takes three optional flags (defaults in the import-free `_setA_defaults.py`):

| Flag | Env var exported to the job | Default |
|---|---|---|
| `--tasks nq,hotpotqa` | `SETA_TASKS` | the 11 setA tasks below |
| `--data-root <set root>` | `SETA_DATA_ROOT` | `/weka/.../prasanns/ctc_sft_sets/setA_max20_evaliid` (shards read from `<root>/shards_qwen35_256k/<task>/`) |
| `--base-checkpoint <model_and_optim>` | `SETA_BASE_CKPT` | `q35-4b-dense-256k-fix/step2385/model_and_optim` |

The training scripts are driven by the olmo-core CLI (`<script> train <run-name> <cluster>`), so the
launcher passes these as env vars (`gantry --env`) and `_qwen35_setA_common.py` reads them; `prep`
sees the same values. A non-default task set goes into the run name, e.g.
`python $L 32k --tasks nq` → `q35-4b-dense-ctc-setA-nq-32k-<stamp>` (more than three tasks →
`<k>tasks-<hash6>`). Running a training script directly (without the launcher) honours the same
env vars.

Single-task debugging: `python $L prep --tasks nq` then `python $L 32k --tasks nq`.

- Checkpoints go to `/weka/oe-training-default/ai2-llm/checkpoints/prasanns/<run-name>/`.
- Logs go to wandb `prasanns-allen-institute-for-ai/memory-networks`, grouped by run name.
- Evaluate with the olmo-eval launcher, **with `--doc-markers`**:
  `scripts/ctc_suite/launch_ctc_suite.py --arm full --doc-markers --ckpt <step dir>` (branch
  `prasann/ctc-suite-launcher`). It renders prompts in the chat template, with each document
  wrapped in the same markers as these shards. With the default `Qwen/Qwen3.5-0.8B` tokenizer each
  eval prompt is an exact token prefix of its training sequence, ending where the answer starts.
  Do **not** pass the 4B tokenizer: its template opens `<think>\n` for a generation prompt, a split
  the training data never contains. The r256k rung runs with YaRN x2 + chunked prefill by default:
  its prompts run to ~272k tokens and would otherwise be left-truncated.

## Qwen3-4B-Base arm (`--model qwen3-4b`, length generalization)

The same recipe on the full-attention `Qwen/Qwen3-4B-Base` (native context 32,768), trained on
exactly the baseline `32k` arm's examples and evaluated at 64k/128k with YaRN. `--model` is exported
as `SETA_MODEL` only when it is not the default, so every Qwen3.5 command and config is unchanged
(launcher dry-runs and `build_setA_experiment(...).as_dict()` for all three arms byte-identical).

```bash
python $L shards --model qwen3-4b       # CPU: Qwen3 shards (build_qwen3_setA_shards.py)
python $L base-audit --model qwen3-4b   # CPU: marker audit + provenance (audit_qwen3_base.py)
python $L prep --model qwen3-4b
python $L 32k --model qwen3-4b --smoke-steps 20   # then without --smoke-steps
```

What differs from the Qwen3.5 `32k` arm, and why (everything else is `_apply_model`'s untouched
inheritance from amandab's builder):

| | Qwen3.5 32k | Qwen3-4B 32k | Why |
|---|---|---|---|
| model | `qwen3_5_4B` (24 GDN + 8 attn) | `qwen3_4B` (36 attn, 32q/8kv x 128, theta 1e6, untied head) | the arm |
| base | `q35-4b-dense-256k-fix/step2385` (256k CPT) | `ctc_suite/bases/qwen3-4b-base-trainedmark` (HF base + marker repair) | the arm |
| tokenizer | Qwen3.5 (EOS 248044) | Qwen3 (EOS 151643), `bos=None` likewise | the model |
| shards | `shards_qwen35_256k` | `shards_qwen3_q35le32k`: same rows, same converter args, cut to Qwen3.5 length <= 32,768 | same examples |
| window / grad-accum | 32,768 / 4 | 65,536 / 2 (1,048,576 tokens/step either way) | a few kept examples are > 32,768 Qwen3 tokens; the packer needs a power of two |

RoPE positions restart at every packed example (`cu_doc_lens`), so the larger window does not
show the model positions past its longest example.

**Label mask.** The converter labels everything after the template's generation prompt. Qwen3.5-4B's
ends `<think>\n`, so the baseline's labels start at `</think>`; Qwen3's ends `assistant\n`, which
would also label `<think> \n\n`. `build_qwen3_setA_shards.py` unlabels those two tokens (asserted on
every row), so the labeled span is the baseline's: `</think> \n\n <answer> <|im_end|> \n`.

**Eval tokenizer.** Qwen3's template renders a generation prompt as `<|im_start|>assistant\n` (a
token prefix of the training sequence, so olmo-eval's preflight passes it) but the training sequence
continues `<think>\n\n</think>\n\n` before the answer. Evaluate with
`/weka/.../prasanns/hf_tokenizers/Qwen3-4B-Base-nothink` (written by `audit_qwen3_base.py`: the
generation prompt closes the empty thinking block unless `enable_thinking=True`, as Qwen3.5-0.8B's
does). `check_qwen3_chat_parity.py` checks both on real eval rows.

## Research variants on the canonical recipe

`--variant <name> [--variant-opts k=v,...]` trains a research arm from `recipes/train/variants.py`
on this recipe instead of dense; it is exported as `SETA_VARIANT` / `SETA_VARIANT_OPTS`, validated
by the launcher first, and replaces `dense` in the run name (`q35-4b-ffnmoe-ctc-setA-32k-<stamp>`).
The variant's hooks are the same ones `research_trainer/train_ctc_suite.py` calls, so an option
means the same thing in both (option names = that trainer's argparse dests, e.g.
`ffn_moe_target=0.3`; use `;` between options when a value holds commas).

```bash
python $L 32k --variant ffnmoe --variant-opts "ffn_moe_target=0.3,total_steps=<packed steps>"
python $L 32k --variant chunked-mix --variant-opts "mix_start_p=0.8,total_steps=<packed steps>"
python $L 256k --variant ffndrop --variant-opts "ffn_drop_max_rate=0.3"
```

| Variant | Changes vs dense | Arms |
|---|---|---|
| `chunked` / `chunked-mix` | document-chunked mask on the attention blocks (+ mask-mix curriculum) | 32k (no CP) |
| `sparselandmark` | sparse-landmark mixer + landmark-packed windows (BFD); base needs a repaired landmark row | all |
| `ffnmoe` | nested-width FFN router, router LR group, `ffn_moe` callback, non-strict base load | all |
| `kvroute` | KV keep/drop router, router LR group, `kv_route` callback, non-strict base load | 32k (no CP) |
| `flexcompute` | FFN + KV routers, optional joint budget (block skip is broken on this branch in both trainers: `BlockSkipHolder` rejects `compact=`) | 32k (no CP) |
| `ffndrop` | drop-CPT random null-FFN token drop, `ffn_drop` callback | all |
| `pooledkv`, `softtoken` | **not on setA**: they need one example per padded row; setA rows are BFD-packed | — |

Every non-dense arm also gets the method-aware `flop_meter` callback. `total_steps=` sets the
horizon the schedules anneal over (`prep` prints the packed steps per arm); without it the horizon
is estimated from the shards' token counts, an upper bound. `seed=` sets the router / drop seeds
(default 0, as in `train_ctc_suite.py`). Unset, the dense config is byte-identical to before
(`recipes/train/check_variant_parity.py`). The recipes still legitimately differ in batch, LR, CP,
AC and attention backend (flash_3 here, flash_2 there).

## Recipe source

The recipe is **amandab's validated dense 256k SFT** (formerly `sft_xlong256k/`; its builder is
vendored here as `_qwen35_xlong5_dolci25_256k_common.py`), run
`q35-dense-contra-3ep-256k-min1h-20260914`. This branch (`prasann/ctc-setA-sft`) is cut from hers
(`amandab/contradiction-only-256k-20260914`), which carries CP fixes that `prasann/landmark` lacks:
Ulysses KV replication, contiguous all-to-alls, and glob-expansion truncation.

`_qwen35_setA_common.py` calls her `build_qwen35_xlong5_experiment` and keeps her settings for the
model, optimizer, AC and CP (Ulysses, since GDN rejects ring CP). It replaces only the data, the
budget and the node/CP geometry. Her builder's docstring explains each of those settings.

**One node at 256k.** FSDP shards over the flattened DP×CP mesh (`_get_model_mesh`), so CP=8 on one
node still shards the 4B params and optimizer state 8 ways. Each rank holds 32,768 tokens of
activations, half of what her CP=4 run held. CP=8 is legal: 8 divides the 16 attention heads and the
32 GDN value heads. This geometry is new, so if it misbehaves, `256k-2node` is her exact tested
layout at the same batch and LR.

## Data

`/weka/.../prasanns/ctc_sft_sets/setA_max20_evaliid/shards_qwen35_256k/<task>/` holds 11 tasks:

- nq, hotpotqa, qdmatch_nq, outlier, oolong, contradiction;
- xabsence (one-sided exact-copy);
- reorder, rerank, strmatch;
- grouping (unlabeled OpenAlex).

textgroups is excluded because of its shortest-document shortcut. The shards are built by
`recipes/data/ctc_sft/build_ctc_sft.py` (see `records/ctc-fast-suite-eval.md`
for the IID audit against the eval's own rows).

- **Document markers.** The shards wrap each document in `<|box_start|>`/`<|box_end|>`; oolong wraps
  each `||` item line instead. The eval matches this through olmo-eval's `CTC_SUITE_DOC_MARKERS=1`
  (the launcher's `--doc-markers`). On real eval rows of all 11 tasks, its prompts are exact token
  prefixes of the converter's training sequences, ending where the labeled answer starts (66/66,
  `recipes/eval/setA_audit/check_doc_marker_parity.py`). The check's first version
  compared against the prompt-only render with the 4B template and so passed an out-of-format
  generation prompt; fixed 2026-09-28. Without the flag, the eval prompts would be out of format.
- **Marker embeddings.** Qwen never trains the marker rows (see CLAUDE.md). `prep` audits the base's
  rows with `fix_marker_embeddings.py`'s cosine/norm gates and prints the verdict.
- **Packing.** Whole examples are packed Best-Fit-Decreasing into windows, across all tasks, with
  block-diagonal masking at EOS. `LongDocStrategy.exclude` drops over-window examples, which is how
  the 32k arm caps length.

## Runs

Prep ([01M386CQPAHG0C8SXPB4BXB9HR](https://beaker.org/ex/01M386CQPAHG0C8SXPB4BXB9HR), 2026-09-23):
base marker audit PASS (cos +0.24, norms 0.57x / 0.84x the median). The table gives
examples / tokens kept at each window:

| Window | Examples | Tokens | Dropped | Packed windows | Steps (1 epoch) |
|---|---:|---:|---:|---:|---:|
| 262,144 | 212,786 | 1,671,337,712 | 0 | 6,380 | 1,595 |
| 32,768 | 203,156 | 925,280,412 | 9,630 (4.5%) | 28,425 | 888 |

| Arm | Run name | Beaker | Commit |
|---|---|---|---|
| `32k` | `q35-4b-dense-ctc-setA-32k-20260923T1537` | [01M386N34ADGF0DVCNH5V0DNY6](https://beaker.org/ex/01M386N34ADGF0DVCNH5V0DNY6) | `2582480a8` |
| `256k` | `q35-4b-dense-ctc-setA-256k-20260923T1537` | [01M386N7CM9Y66RH505F6A9T2Z](https://beaker.org/ex/01M386N7CM9Y66RH505F6A9T2Z) | `2582480a8` |

Both runs finished one epoch: `32k` at `step888` and `256k` at `step1595` (resumed from `step1200`
after two preemptions).

Evals (olmo-eval launcher, `--arm full --doc-markers`; per-cell values in the run's results):

| Run name (launcher) | What |
|---|---|
| `q35-4b-dense-ctc-setA-{32k-step888,256k-step1595}-full-dm` | 12 setA rows, every rung 2k-256k. **Their r256k cells were left-truncated (43-100% of rows): do not use them** |
| `q35-4b-dense-ctc-setA-{...}-r256k-fit` | r256k with `--fit-window` (6 rows; not oolong) |
| `q35-4b-dense-ctc-setA-{...}-r256k-yarn2-chunked` | r256k with YaRN x2 + chunked prefill (7 rows) -- the launcher's default for this rung |
