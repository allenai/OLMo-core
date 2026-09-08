# sft_interleaved256k/ — 256k SFT of the interleaved sparse/regular landmark CPT arms (Qwen3.5)

SFT of each `cpt/interleaved/` Qwen3.5-4B arm on 75% the xlong5 2k→256k 5-task mix (the **qboth**
build) / 25% `allenai/Dolci-Instruct-SFT`, at a 262,144 landmark window. Beaker only, 2 nodes,
`urgent`, `minRuntime=1h`.

> [!WARNING]
> **These configs have not been dry-run.** They were written against the branch but never executed:
> `torch` publishes no macOS-x86_64 wheels at the pinned version, so `olmo_core` cannot be imported
> on the machine they were authored on. Everything below is verified by reading and by static
> checks (imports resolve; shared constants compared field-by-field against `sft_xlong256k/`).
> Run the `dry_run` and `launch_prep` steps on a Linux box before launching anything — the
> **Before you launch** checklist at the bottom lists what those runs still have to establish.

| Script | Arm | Layer layout over the 8 full-attention layers (3, 7, …, 31) | CPT base |
|---|---|---|---|
| `Qwen3.5-4B-interleaved-reg-first-…-SFT.py` | `reg-first` | `reg`, then `sparse` ×7 | `q35-4b-il-regfirst-256k/step2385` |
| `Qwen3.5-4B-interleaved-reg-last-…-SFT.py` | `reg-last` | `sparse` ×7, then `reg` | `q35-4b-il-reglast-256k/step2385` |
| `Qwen3.5-4B-interleaved-sparse-reg-…-SFT.py` | `sparse-reg` | alternating from `sparse`, ends `reg` | `q35-4b-il-sparsereg-256k/step2385` |
| `Qwen3.5-4B-interleaved-reg-sparse-…-SFT.py` | `reg-sparse` | alternating from `reg`, ends `sparse` | `q35-4b-il-regsparse-256k/step2385` |

`reg` is `fast_compressive_landmark`, `sparse` is `sparse_landmark` — the definitions come from
`cpt/interleaved/_qwen35_interleaved_landmark_256k_common.py`, imported, not restated.

## The control is in the other family

Everything except the architecture is copied from
`sft_xlong256k/_qwen35_xlong5_dolci25_256k_common.py`: the qboth 5-task root, the 75/25 blend, the
2.0/1.5/1.5/1.0/1.0 within-mix weights, the 262,144 window, 2 nodes × CP=4 → DP=4 → 1,048,576
tok/step, LR 4e-5, 2,240 steps = 2.35B tokens, seed 34521. So **`q35-4b-dense-xlong5-qboth-dolci25-256k`
is the architecture control** for every arm here, and the arms are paired with it and with each
other rather than being independent draws.

Read those arms with the qboth rendering — the **default**, not `--query-position after`. This
family never uses the qafter root.

## Why a separate family rather than rows in `sft_xlong256k/_ARMS`

That family advertises, in its README and its module docstring, that everything except the 5-task
shard root is shared by construction — it is a controlled pair on *query position* over one fixed
dense base. An interleaved arm changes the base checkpoint **and** the model architecture. Adding
one there would make that claim false for the whole table. This family mirrors the constants
instead, and imports them where they are literally shared.

## The one axis that is not matched — read before interpreting

Landmark training inserts one landmark token per `MEM_FREQ = 63` content tokens, so a 262,144-token
window holds **258,048** tokens of original content against the dense control's 262,144.

1. **~1.56% less content per window**, hence ~1.56% less original data at the shared 2,240-step
   budget. The budget is matched in *window* tokens (the compute-matched reading), not content
   tokens.
2. **The long tail is dropped, and only here.** `sft_xlong256k`'s README records qboth's longest
   example at 262,072 tokens — inside its dense window, which is why its "`LongDocStrategy.exclude`
   drops nothing" assertion holds. Against a *content* capacity of 258,048 that example does not
   fit. The landmark packer drops it and everything else over 258,048, concentrated in exactly the
   128–256k band these arms exist to measure.

The 32k family solved the equivalent problem by widening the landmark window (40,960 vs dense
32,768) so content capacity met the dense window. **That option is not available at 256k**: 262,144
is the length the CPT arms actually trained at, and widening past it would evaluate the model
outside the context it was continued-pretrained on — a larger confound than 1.56% of data.

Both effects push the same way (these arms see marginally less, and marginally shorter, data), so
an interleaving *deficit* on the longest rungs is confounded with them; an interleaving *advantage*
there is not.

## Reading the sweep

- **Each arm minus dense `qboth`** on the long rungs isolates interleaving, subject to the caveat
  above.
- **`sparse-reg` minus `reg-sparse`** — the same 4/4 split in opposite phase — separates *how many*
  regular-landmark layers from *where in the stack* they sit.
- **`reg-first` minus `reg-last`** does the same at the 1/7 extreme.
- The all-regular control for the CPT sweep is
  `Qwen3.5-4B-fast-compressive-landmark-longmino512k.py`; it has no arm in this family yet.

## Before you launch

Unresolved items, in the order they have to be settled:

1. **`reg-last` has no CPT checkpoint.** `q35-4b-il-reglast-256k` failed on 2026-08-11 and again on
   its 2026-08-31 relaunch (NCCL IB transport fault, not a config error), so
   `…/q35-4b-il-reglast-256k/step2385` does not exist. Its SFT arm is written and correct but
   **cannot run** until that CPT run completes. The other three arms are unblocked.
2. **`dry_run` every arm** on Linux and diff the dumps pairwise: they must differ **only** in the
   run name, the launch description, `trainer.load_path`, and the model's per-layer
   `attn.sequence_mixer.layer_types`. Any other difference is a bug in this file.
3. **`launch_prep` every arm** and record, per arm, the `MixingInstanceSource: NNB tokens` line, the
   `LandmarkPackingInstanceSource packed N windows` line, and the long-document drop count.
   `warn_drop_fraction` is set to `0.0` precisely so that drop is always logged. Put the numbers in
   a table here — the drop count is **not** known yet, and no arm-vs-dense long-rung delta should be
   reported before it is.
4. **Confirm 2,240 steps is ≤ one epoch on the packed count.** The dense arms land at 8,971 windows
   → 2,242.8 steps/epoch. The landmark packer will produce a different count; if it comes out under
   ~8,960, these arms repeat data the dense control does not, and `MAX_STEPS` needs revisiting.
5. **Memory.** Same untested corner as the dense pair — `shard_degree` is 4, so optimizer state
   spreads over 4 ranks instead of the legacy runs' 16, at unchanged per-rank activations. If an
   arm OOMs it will do so in the first few steps; the fix that preserves the experiment is 4 nodes
   at CP=8 (DP stays 4, so batch/LR/steps are unchanged).

## Commands

```bash
S=src/scripts/train/memexpress/sft_interleaved256k/Qwen3.5-4B-interleaved-reg-sparse-xlong5-dolci25-256k-SFT.py

PYTHONPATH=src python $S dry_run q35-4b-il-regsparse-xlong5-dolci25-256k ai2/jupiter-cirrascale-2

PYTHONPATH=src python $S launch_prep q35-4b-il-regsparse-xlong5-dolci25-256k-prep \
    ai2/jupiter-cirrascale-2

PYTHONPATH=src python $S launch q35-4b-il-regsparse-xlong5-dolci25-256k \
    ai2/jupiter-cirrascale-2 --launch.follow=false --launch.step_soft_timeout=null
```
