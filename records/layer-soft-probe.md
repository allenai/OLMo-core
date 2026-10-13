---
title: Soft LAYERS — does the pooled-doc soft token have to be applied at every layer? (eval only)
date: 2026-09-14
updated: 2026-09-14
experiment: pooled_kv
area: token_routing
kind: result
status: eval-only probe
superseded_by: 
summary: Does the pooled soft token need to be applied at every layer?
---
# Soft LAYERS — does the pooled-doc soft token have to be applied at every layer? (eval only)

Date: 2026-09-14. Eval only, no training. Probe:
`research/token_routing/pooled_kv/layer_probe/layer_soft_probe.py` (imports its helpers from
`research/token_routing/pooled_kv/eval_side_slot_probe.py`, which was not modified). Collector:
`research/token_routing/pooled_kv/layer_probe/collect.py` → `research/token_routing/pooled_kv/layer_probe/results.csv`
(column `prefix_real` says which baseline each row belongs to). Token-role split:
`research/token_routing/pooled_kv/layer_probe/analyze_tokens.py`. Launchers: `run_smoke.sbatch`,
`run_tokenize.sbatch`, `run_sweep.sbatch`, `run_cutscan.sbatch`, `run_twolevel.sbatch`,
`run_twolevel_gold.sbatch`, `run_sparse_slots.sbatch`, `run_droppooled.sbatch`,
**`run_header_real.sbatch`** (the decisive one). `run_stale_and_oolong.sbatch` is kept for the
record but was cancelled mid-flight and its steps were re-run inside `run_twolevel.sbatch`.
Raw probe JSONs and logs: `/net/sneetches/data/prasann/layer_probe/*.{json,log}`.

> **⚠ ANSWER FIRST — and a correction to an earlier draft of this record.** The first pass ran
> against the **plain mean slot with no header**, which is the construction the 2026-09-08 study
> already knew is NOT at parity. Against that weak baseline, a slot in only 1 of 32 layers looked
> like a large win. Re-run against the **header-real** baseline that actually reached parity
> (`--prefix-real stop1` / `stop3`, i.e. `--st-header-stop-id 25 --st-header-stop-count {1|3}`;
> see `records/soft-kv-slot-probe-handoff.md`), **the fidelity win disappears**: SOFT with the slot
> in every layer is already at full-attention parity, and sparse slot layers buy only ~1.5–2 points
> of extra FLOPs/memory at equal fidelity. The mid-stack-cut catastrophe also **vanishes** at 8k —
> it was an id-readout artefact of the header-less slot. Every header-less number below is retained
> in §5, clearly marked, because it is what the correction is against.

**⚠ Every number is a PROBE at eval_size = 200 (contradiction / oolong @8k) or eval_size = 64
(contradiction @32k), under the 500-example bar. SEs are inline; treat differences below ~2 SE of
the difference as noise.**

## Setup

Reused exactly the checkpoints, rows and node of `records/soft-kv-slot-probe-handoff.md`:

- Model: dense-trained Qwen3.5-4B hybrid ladder checkpoints on sneetches node-local disk,
  `/data/prasann/dense_ckpts/tsl-full-contradiction-s56M-4b-.../model_and_optim` and
  `.../tsl-full-oolong-s80M-4b-.../model_and_optim`. 32 layers: 24 GatedDeltaNet (linear attention)
  and 8 softmax-attention (indices 3, 7, 11, 15, 19, 23, 27, 31).
- Rows: first 200 held-out examples of
  `/data/prasann/ctc_suite_staged/eval_rungs/contradiction_iid/rung_8192.jsonl` and
  `.../oolong/rung_8192.jsonl`, plus 64 of `contradiction_iid/rung_32768.jsonl`, tokenised in job
  with the marker converter (`run_tokenize.sbatch`). Median length ~8.4k / ~33k tokens.
- Keep plan: gold + a random fraction of the non-gold documents (`keep` = 1/12 and 1/36 on
  contradiction, 1/3 on oolong — the handoff's parity points).
- **Header-real pooling** (`--prefix-real stopK`, `--prefix-stop-id 25`): each document's header —
  the tokens after `doc_start` through the K-th `:` — stays REAL and the slot stands for the body
  only. K = 1 for contradiction (`Claim N:`), K = 3 for oolong (`Date: … || User: … || Instance:`).
  Implemented by reusing `install_prefix_real_patch()` from `eval_side_slot_probe.py` verbatim, so
  the compaction, the chunk-id roles and the keep plan are bit-for-bit the ones the parity study
  used. Header-real compacts less: on a contradiction row at keep 1/12,
  T = 8036 → T2 ≈ 2900 (vs 1271 header-less); on oolong the header is over half the line.
- Node: sneetches (H200), partition jsteinhardt / QOS preemptive_high, 1 GPU per job, node-local
  interpreter `/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python`, logs and outputs on
  `/data/prasann/layer_probe/`. ≈ 4.5 GPU-hours, 7 sbatch jobs, 34 probe runs.

### What "soft-only layer" means here

The model is held fixed. Per layer we choose one of two sequences:

- **dense** — the full row, exactly the FULL baseline;
- **soft-only** — the COMPACTED row (kept real tokens + every document's header when header-real is
  on + one slot per pooled doc at the doc's centre RoPE position, `position_ids` carrying original
  positions). The pooled documents' body tokens are not computed in that layer at all; their hidden
  state passes through by the residual identity and nothing in that layer attends to them (GDN
  layers: nothing writes the recurrent state from them either).

`--stale-keys mask`: once a soft layer has run, a pooled doc's dropped tokens are frozen and never
read again, so every LATER dense layer excludes them with `block_keep` — the same kwarg
`olmo_core.nn.block_skip` uses for a skipped token (that module's `compact=True` gather/scatter is
the mechanism this probe generalises; it has no slot insertion, hence a separate script). Priced at
what a real implementation would pay: from the first soft layer on, those tokens are dropped.

Slot sources: `embed_mean` (mean input embedding, the training construction, carried through the
stack and frozen across dense layers) and `layer_input_mean` (rebuilt at the first soft layer of
each soft run from the mean of the doc's real hidden states at that layer's input).
`droppooled` = the control with NO slot anywhere (headers real, bodies simply absent from the whole
stack, same cost as one slot layer).

**Validation** (`--selftest`): no-soft-layers reproduces `model.forward` to max |Δlogit| = 0.0000;
all-layers-soft reproduces the ordinary compacted soft forward to KL 0.0000–0.0011 against a
call-to-call noise floor of KL 0.0000–0.0022 — **the compacted bf16 GDN path is not deterministic
against itself**, so the reductions are exact up to kernel noise, not bitwise. The two-level mode's
ultra-on-no-layers likewise reproduces the compacted forward (KL 0.0002).

## 1. HEADER-REAL (the parity baseline) — the decisive table

Contradiction @8k, keep 1/12, `--prefix-real stop1`, `--stale-keys mask`, eval_size = 200.
"slot present in" = the layers where the pooled documents exist as a slot; in every other layer
their bodies are absent. FLOPs/memory saved are against FULL.

| condition | slot layers | CE | ΔCE vs SOFT | top-1 = FULL | EM | FLOPs saved | mem saved |
|---|---|---|---|---|---|---|---|
| FULL | — | 0.019 ± 0.003 | −0.000 | 1.000 | 0.860 ± 0.025 | 0.000 | 0.000 |
| `top25` (compacted in the last 8 layers only) | — | 0.018 ± 0.003 | −0.001 | 0.999 ± 0.000 | 0.875 ± 0.023 | 0.184 | 0.181 |
| **SOFT — slot in every layer (today's scheme)** | **32** | **0.019 ± 0.002** | **0** | 0.994 ± 0.001 | **0.865 ± 0.024** | **0.738** | **0.724** |
| `every4` | 8 | 0.021 ± 0.002 | +0.001 | 0.993 ± 0.001 | 0.860 ± 0.025 | 0.753 | 0.739 |
| `every8` | 4 | 0.021 ± 0.002 | +0.002 | 0.993 ± 0.001 | 0.860 ± 0.025 | 0.755 | 0.741 |
| layer 0 only | 1 | 0.023 ± 0.002 | +0.003 | 0.993 ± 0.001 | 0.820 ± 0.027 | 0.757 | 0.743 |
| no slot at all (`droppooled`) | 0 | 0.038 ± 0.004 | +0.018 | 0.989 ± 0.001 | 0.780 ± 0.029 | 0.757 | 0.744 |

**Answer to the question posed: only the marginal FLOP/memory saving, and it is small.** Once the
baseline is at parity, today's every-layer slot is already as good as full attention
(EM 0.865 ± 0.024 vs 0.860 ± 0.025, ΔCE 0.000) at 0.738 of FULL's FLOPs. Making the slot sparse in
depth does not improve fidelity — `every4`/`every8` match SOFT within noise and buy **+1.5 to +1.7
points** of FLOPs and memory; one slot layer costs EM 0.820 ± 0.027 (−0.045 against SOFT, SE of the
difference ≈ 0.036, i.e. ~1.3σ) for +1.9 points; and removing the slot entirely costs a clear
−0.085 EM. The slot is doing real work in the header-real regime too, but the *number of layers it
occupies* is no longer a fidelity lever.

Same at keep **1/36** (contradiction @8k, eval_size = 200): FULL 0.019 ± 0.003 / EM 0.860 ± 0.025;
SOFT 0.023 ± 0.003 / EM 0.845 ± 0.026 at 0.780 FLOPs; `every4` 0.024 ± 0.002 / EM 0.810 ± 0.028 at
0.795; layer 0 only 0.030 ± 0.003 / EM 0.760 ± 0.030 at 0.800; `droppooled` 0.042 ± 0.004 /
EM 0.740 ± 0.031. Sparser-in-depth is now monotonically *worse*, still for ~2 points of FLOPs.

Contradiction @32k, keep 1/12, eval_size = **64** (EM ±0.06, so read this as directional):
FULL 0.055 ± 0.012 / EM 0.641 ± 0.060; SOFT 0.049 ± 0.010 / EM 0.609 ± 0.061 at 0.790 FLOPs;
`every8` 0.045 ± 0.009 / EM 0.719 ± 0.057 at 0.806; layer 0 only 0.046 ± 0.009 / EM 0.719 ± 0.057
at 0.808; `droppooled` 0.287 ± 0.018 / EM 0.062 ± 0.030. The sparse arms are nominally ahead of
SOFT at 32k (ΔEM +0.11, SE of the difference ≈ 0.084 — **not resolvable at eval_size 64**); the
no-slot collapse is unambiguous.

oolong @8k, keep 1/3, `--prefix-real stop3`, eval_size = 200: FULL 0.309 ± 0.027 / EM 0.615 ± 0.035;
SOFT 0.345 ± 0.030 / EM 0.595 ± 0.035 at only **0.392** FLOPs saved (oolong's header is over half
the line, so header-real compacts little); `every4` 0.336 ± 0.027 / EM 0.595 ± 0.035;
layer 0 only 0.341 ± 0.028 / EM 0.585 ± 0.035; `droppooled` 0.499 ± 0.032 / EM 0.475 ± 0.035;
`top25` 0.309 ± 0.027 / EM 0.605 ± 0.035 at 0.098. Same shape: slot-layer count is flat, no-slot is
clearly worse.

## 2. The mid-stack cut catastrophe was an id-readout artefact

`fromL` = layers ≥ L soft (read the documents densely for L layers, then compress). Header-less,
the L = 7–14 band was 4–5× worse than compressing from layer 0. **With headers real it is gone at
8k.** Contradiction @8k, keep 1/12, eval_size = 200, CE (EM):

| cut L | 4 | 6 | 8 | 10 | 12 | 16 | SOFT (L = 0) |
|---|---|---|---|---|---|---|---|
| header-real | .016 (.895) | .016 (.900) | **.018 (.870)** | .028 (.760) | .030 (.745) | .018 (.900) | .019 (.865) |
| header-less (§5) | .142 (.415) | .056 (.660) | **.604 (.005)** | .582 (.015) | .558 (.035) | .070 (.585) | .133 (.415) |

(SE 0.002–0.016.) At 8k the worst header-real cut is CE 0.030 ± 0.003 / EM 0.745 ± 0.031 — a bump,
not a cliff. At 32k a **residue survives**: `from12` 0.102 ± 0.010 / EM 0.297 ± 0.058 and `from10`
0.103 ± 0.011 / EM 0.344 ± 0.060 against SOFT 0.049 ± 0.010 / EM 0.609 ± 0.061 (eval_size = 64).
oolong shows no cliff at any length (worst cut 0.354 ± 0.028 vs SOFT 0.345 ± 0.030). So: mostly an
artefact of asking a header-less slot to carry a document id, with a real but much smaller
mid-stack penalty remaining at long context.

## 3. Two-level "soft layers" — can we be sparser on the token axis as well?

Mode `--ultra-patterns`: the per-layer default is the SOFT compacted row, and the selected ("ultra")
layers run on a further-compacted row in which the KEPT documents are pooled to one slot each too
(headers stay real there as well when `--prefix-real` is on; kept-doc slot = `layer_input_mean`).
Deltas are against SOFT-header-real, eval_size = 200.

Contradiction @8k, keep 1/12, header-real:

| ultra layers | CE | ΔCE vs SOFT | EM | FLOPs saved vs SOFT | mem saved vs SOFT |
|---|---|---|---|---|---|
| none (= SOFT) | 0.019 ± 0.002 | 0 | 0.865 ± 0.024 | 0.000 | 0.000 |
| `top25` | 0.018 ± 0.002 | −0.001 | 0.855 ± 0.025 | 0.085 | 0.084 |
| `top50` | 0.043 ± 0.005 | +0.023 | 0.730 ± 0.032 | 0.169 | 0.167 |
| `all` | 1.194 ± 0.007 | +1.175 | 0.000 | 0.339 | 0.334 |

Ultra on the top quarter is now **free** on contradiction (header-less it cost +0.065 CE and half
the EM) but buys only 8.5 % of SOFT's remaining compute; beyond that it degrades, and ultra
everywhere still annihilates the task — the kept documents' bodies are the evidence.

oolong @8k, keep 1/3, header-real: `top25` 0.344 ± 0.030 / EM 0.605 ± 0.035 (ΔCE −0.001) at 0.084;
`top50` 0.347 ± 0.030 / EM 0.600 ± 0.035 at 0.167; **`all` 0.417 ± 0.031 / EM 0.490 ± 0.035
(ΔCE +0.072, ΔEM −0.110) at 0.335**. This **revises the header-less finding**: against the weak
baseline, ultra-everything on oolong looked free (ΔCE −0.027); against the parity baseline it costs
~2σ of EM for a third of SOFT's remaining compute.

## 4. Attention vs GDN layers, and the top-quarter result (header-less, but robust)

These two conclusions from the header-less pass survive qualitatively under header-real, because
they are about *which* layers are compacted rather than about the slot's readability:

- Compacting only the **top quarter** of layers is at parity with full attention on both tasks and
  both baselines (header-real: contradiction 0.018 ± 0.003 / EM 0.875 ± 0.023, oolong
  0.309 ± 0.027 / EM 0.605 ± 0.035) — but it saves far less than compacting everywhere (0.18 / 0.10
  of FULL's FLOPs vs 0.74 / 0.39), so it is not a competitive operating point now that SOFT itself
  is at parity.
- At equal damage, **GDN layers are the cheaper ones to soften**: header-less, `gdn_top50`
  (12 GDN layers) and `attn_top50` (4 attention layers) cost the same CE (0.059 ± 0.005 vs
  0.055 ± 0.006) but save 0.300 vs 0.130 of FLOPs. There is no sign that skipping real tokens is
  qualitatively different in a recurrent layer than in a softmax one.

## 5. The header-less pass (NOT the parity construction) — retained for the record

All rows with `prefix_real = none` in `results.csv`. With the plain mean slot, SOFT is far from
parity (contradiction @8k keep 1/12: CE 0.133 ± 0.011 / EM 0.415 ± 0.035 vs FULL 0.019 ± 0.003 /
EM 0.860 ± 0.025), and **96–99 % of all CE mass sits on the answer's document-id digit tokens**
(Qwen3.5 ids 15–24; 51 % of answer tokens; FULL itself 88 %) — the id is unreadable from a
body-only slot, which is exactly why the header matters. Against that baseline the layer axis looks
decisive: one slot layer gave CE 0.045 ± 0.005 / EM 0.730 ± 0.032 at 0.879 FLOPs saved
(vs SOFT 0.861), `every8` 0.043 ± 0.005 / EM 0.730 ± 0.032, `droppooled` 0.065 ± 0.006 /
EM 0.600 ± 0.035, and at 32k one slot layer was 3× SOFT's exact-match (0.250 ± 0.055 vs
0.078 ± 0.034). Also header-less only: the L = 7–14 cut catastrophe (§2), `top75` 0.604 ± 0.015,
`alt` 0.429 ± 0.011, and the oolong two-level "ultra everywhere is free" result (§3). **None of
these should be quoted** — they measure how much damage an unreadable slot does and how sparsity in
depth limits that damage, not a property of the method as it is actually configured.

## Interpretation

The corrected answer is that the layer axis is close to a dead end for this scheme. Once the soft
token is built the way the 2026-09-08 parity study built it — document header real, slot for the
body only — applying it at **every** layer already reproduces full attention on contradiction
(EM 0.865 ± 0.024 vs 0.860 ± 0.025, ΔCE 0.000 at 0.738 of FULL's FLOPs) and comes within one SE on
oolong, so there is no fidelity left for a layer schedule to recover. Making the slot sparse in
depth is fidelity-neutral at keep 1/12 (`every4`/`every8` within noise) and mildly harmful at keep
1/36 and at one slot layer, while buying only 1.5–2 points of additional FLOPs and memory, because
with headers real most of the sequence that remains is header tokens rather than slots. What the
sparse-slot arms do show is that the slot's *content* matters and its *ubiquity* does not: dropping
the pooled bodies entirely (no slot anywhere) costs 0.085 EM at 8k and collapses at 32k, while
moving the slot from 32 layers to 4 costs nothing measurable. The mid-stack-cut catastrophe was
largely an artefact of the header-less slot — reading densely for 7–14 layers and then compressing
is fine with headers real at 8k, though a real penalty (EM 0.297 ± 0.058 vs 0.609 ± 0.061 at
`from12`) survives at 32k. And pushing sparsity onto the kept documents remains task-limited: free
on the top quarter of layers on both tasks, but ultra-everywhere annihilates contradiction and now
also costs oolong ~2σ of EM, which retracts the earlier "free on oolong" claim.

## What I would try next

1. **Don't take the sparse-slot recipe to training** on the strength of this. Its fidelity case
   evaporated against the parity baseline; the remaining case is ~2 points of FLOPs/memory, which
   is inside the run-to-run noise of a training comparison.
2. **Spend the layer axis on the header instead.** Header-real is what closes the gap and it is also
   what stops the compaction (T2 2900 vs 1271 at 8k; oolong 0.39 vs 0.66 FLOPs saved). The valuable
   question is whether headers need to be real in *every* layer, or whether they could be real in
   the first few layers and pooled thereafter — the direct analogue of this study, run on the part
   of the sequence that is actually expensive.
3. **Confirm the 32k signals at a usable eval size.** Two results there are ~1.3σ at eval_size 64
   (sparse slots nominally beating SOFT; the surviving `from12` penalty). Both are cheap to re-run
   at 200+ rows and would change the recommendation if real.
4. **nq and outlier could not be run at all** — their eval JSONLs are weka-only and not staged on
   sneetches, so every conclusion here rests on two tasks.
5. **Raise eval_size to ≥ 500** before any number leaves a probe context.

## Traps hit

- **Benchmarking against a baseline you have not checked is the parity one.** The whole first pass
  measured how much damage a header-less slot does; its headline reversed once the baseline was
  fixed. `records/soft-kv-slot-probe-handoff.md` says plainly that parity needs header-real — read
  the handoff's *result*, not just its tooling.
- **Editing the probe script while a multi-step sbatch is running.** Each step starts a fresh python
  that re-reads the file from NFS, so a mid-flight edit killed one step with a `NameError` and
  silently changed what another step ran. Write via a temp file + `os.replace`, or wait.
- A function appended after `if __name__ == "__main__": main()` is defined too late to be called.
- The compacted bf16 GDN forward is **not deterministic against itself** (KL ~0.001–0.002, per-row
  CE wobble ~0.01, EM reproducibility ±0.015 on repeated identical runs). Any "is my path exact?"
  assertion must be against that measured floor, not zero.
- The staged shards under `/scratch/users/prasann/slot_probe/` hold only the first **24** rows of
  each eval file (the old probe's `--rows 24`); reaching eval_size 200 meant re-running the
  converter on the staged `eval_rungs` JSONLs.
- `install_prefix_real_patch()` rebinds `olmo_core.nn.transformer.model.build_chunk_ids_from_tokens`.
  Any *other* module that imported that symbol directly keeps the unpatched one — this probe now
  calls it through the module (`_tm.build_chunk_ids_from_tokens`) so doc membership matches the
  compaction.
