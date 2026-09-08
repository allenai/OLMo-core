# sft_interleaved256k/ — 256k SFT of the interleaved sparse/regular landmark CPT arms (Qwen3.5)

SFT of each `cpt/interleaved/` Qwen3.5-4B arm on 75% the xlong5 2k→256k 5-task mix (the **qboth**
build) / 25% `allenai/Dolci-Instruct-SFT`, at a 262,144 landmark window. Beaker only, 2 nodes,
`urgent`, `minRuntime=1h`.

> [!NOTE]
> Validated on 2026-09-08 against commit `61e956ba`: `dry_run` passed for all three launchable arms
> (each resolving its own CPT checkpoint, and each producing the correct 32-layer pattern with
> `reg`/`sparse` in the right slots and every GDN slot left at `default`), and the shared `prep` job
> completed. Measured numbers are in **Prep readout** below. `reg-last` remains unvalidated because
> it has no CPT checkpoint.

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
2. **Memory.** Same untested corner as the dense pair — `shard_degree` is 4, so optimizer state
   spreads over 4 ranks instead of the legacy runs' 16, at unchanged per-rank activations. If an
   arm OOMs it will do so in the first few steps; the fix that preserves the experiment is 4 nodes
   at CP=8 (DP stays 4, so batch/LR/steps are unchanged).

## Prep readout (measured 2026-09-08, Beaker `01M20FXCTYN403D1ARAS23JX5Q`)

One prep job serves all arms: they share the data root, blend, weights, window and seed, and the
packing cache is keyed by content under `get_work_dir(root_dir)`, not by run name.

| | this family (landmark) | dense control (`qboth`) |
|---|---|---|
| packed windows | **9,195** | 8,971 |
| steps per epoch at DP=4 | 2,298.8 | 2,242.8 |
| epochs at the shared `MAX_STEPS` = 2,240 | **0.974** | 0.999 |
| content tokens packed | 2.341B | — |
| documents kept | 1,030,523 / 1,030,564 | — |
| **documents dropped (over 258,048 content)** | **41 (0.004%)** | **0** |
| non-content tokens per window (landmarks + block padding + tail) | 7,562.3 (2.88%) | — |

Both open questions from the design come out benign:

- **The long-tail drop is 41 documents**, not the larger loss the 262,072-vs-258,048 gap allowed for.
  That is the same order as the 112-instance asymmetry the dense pair already tolerates between its
  own arms, so it does not by itself confound an arm-vs-dense long-rung delta. It is still a
  one-directional asymmetry concentrated in the long band — name it if a long-rung delta is small.
- **2,240 steps stays inside one epoch** (0.974), so these arms repeat no data. The landmark packer
  produced *more* windows than the dense one (9,195 vs 8,971) because landmark tokens and
  block padding inflate each document — the ~1.56% content deficit per window is paid in extra
  windows, and the budget being matched in window tokens is what leaves it at 0.974 of an epoch.

## Commands

```bash
S=src/scripts/train/memexpress/sft_interleaved256k/Qwen3.5-4B-interleaved-reg-sparse-xlong5-dolci25-256k-SFT.py

PYTHONPATH=src python $S dry_run q35-4b-il-regsparse-xlong5-dolci25-256k ai2/jupiter-cirrascale-2

PYTHONPATH=src python $S launch_prep q35-4b-il-regsparse-xlong5-dolci25-256k-prep \
    ai2/jupiter-cirrascale-2

PYTHONPATH=src python $S launch q35-4b-il-regsparse-xlong5-dolci25-256k \
    ai2/jupiter-cirrascale-2 --launch.follow=false --launch.step_soft_timeout=null
```
