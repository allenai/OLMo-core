# Plan: flat-softmax inference variant (remove gate-score reweighting) + regression eval

**Status (2026-07-13): NOT STARTED.** This is a distinct follow-up to the gate-score *distribution*
study (`analysis/in_progress_gate_distribution.md`). That study only *observed* the gate scores. This
task *changes inference*: add an inference-only flag that removes the gate-score reweighting, then
measure how much eval quality regresses on HELMET, RULER, and the SFT datasets (full datasets) for the
same three checkpoints.

This note is self-contained for a cold-start Claude. Read the two referenced decode functions before
coding — line numbers drift, so locate by name.

---

## 0. TL;DR

1. Implement an inference-only flag (`landmark_flat_softmax`, env `OLMO_LANDMARK_FLAT_SOFTMAX`) that,
   at **decode time only**, replaces the landmark **grouped/gated softmax** with a **plain (flat)
   softmax over exactly the currently-visible key positions** — i.e. keep the hard top-k block
   selection, but drop the per-block gate-weighting (§1, §3).
2. Add a unit test: flag-on == a hand-computed flat softmax over the selected support; flag-off is
   bit-identical to today (§3.4).
3. Thread the env flag into the three eval harnesses (§4) and run **baseline (gated) vs variant
   (flat)** on **HELMET + RULER + SFT, full datasets**, for the three checkpoints, at the same top-k
   operating point (fraction 0.1) (§5).
4. Report the regression: per checkpoint × harness × task/length, baseline vs flat, delta (§6). Tie it
   back to the peakiness measured in the distribution study.

Code baseline to branch from: commit **`f3eecf6cb`** on **`amandab/remove-softmax-scaling`** (has the
gate-logging hook and the eval launchers). Make a new branch for this change.

---

## 1. The change, precisely

Current landmark decode (per step, per head), after the hard top-k selects which past blocks are
visible, uses a **two-level grouped softmax**:
- a cross-block **gate** softmax over {selected-block landmarks + local section} → a weight `G_b` per
  selected block, and
- a **within-block** softmax over each block's tokens → `f_n`,
- so each visible past-block key gets `G_b · f_n`, and local-section keys keep their gate weight.

**Variant:** keep the hard top-k selection unchanged, but allocate mass with a **single flat softmax
over the raw attention logits of exactly the key positions that currently receive nonzero weight** —
no `G_b`, no within-block split. "Let the regular softmax decide across all selected regions."

**The visible support (what the flat softmax is over) — this differs by model type:**
- **Non-compressive landmark** (checkpoint #2, `fast_landmark`): landmark/memory tokens are pure gates
  and contribute **no value** today. Flat support = **content tokens of selected blocks + local
  section**. Exclude ALL landmark positions.
- **Compressive landmark** (checkpoints #1, #3): each block's landmark is a learned **compression
  token that does carry a value** today. Flat support = **selected blocks' content tokens + selected
  blocks' landmark tokens + local section**. Non-selected blocks (including their compression token,
  i.e. the `nonselected_landmark_mass` α reserved for them) are **excluded** — the variant zeroes
  non-selected regions entirely.

The point of matching the support exactly is that this becomes a clean apples-to-apples ablation:
same tokens visible, only the *weighting* changes (gated → flat).

Prefill is unchanged (it is already dense with no top-k selection; "selected regions" only exist at
decode). So this is genuinely **decode/inference-only**.

---

## 2. Why (and how it connects to the distribution study)

The distribution study measures how peaky the gate scores are. This task measures whether that
peakiness *matters for quality*: if removing the gate reweighting barely regresses, the gate weights
were doing little work (the flat variant is a cheaper/simpler equivalent); if it regresses a lot, the
gate weighting is load-bearing. Expect the two results to correlate — peaky gate distributions (low
selected-softmax entropy, high top-1 mass) should predict larger regression under flattening. State
that correlation in the writeup.

---

## 3. Implementation

Branch: `git switch -c amandab/landmark-flat-softmax` off `f3eecf6cb`.

### 3.1 The flag (GenerationConfig field + env fallback)
- `src/olmo_core/generate/generation_module/config.py`: add
  `landmark_flat_softmax: bool = False` with a docstring (inference-only; keep top-k, drop gate
  reweighting; see this note). Default False = no behavior change.
- `src/olmo_core/generate/generation_module/transformer/generation_module.py`: in the
  `if landmark_active:` block (right where `_set_landmark_eval_decode(...)` is called), also set on
  every landmark attention module:
  ```python
  flat = generation_config.landmark_flat_softmax or bool(os.environ.get("OLMO_LANDMARK_FLAT_SOFTMAX"))
  for _, block in self.model.blocks.items():
      attn = block.attention
      if isinstance(attn, _LANDMARK_ATTENTION_TYPES):
          attn._eval_flat_softmax = flat
  ```
  The env fallback is what lets RULER/HELMET enable the variant **without any oe-eval/HELMET model-arg
  or MODEL_DEFAULTS change** — just set the env var in the container (§4). Add `_eval_flat_softmax =
  False` init alongside the existing `_eval_top_k` init on the attention base classes, and clear it in
  `clear_landmark_eval_decode`.

### 3.2 Non-compressive decode — `landmark_fast.py::_decode_probs`
Find the `probs = landmark_grouped_softmax(scores, ...)` call. Replace the return with:
```python
probs = landmark_grouped_softmax(scores, dim=-1, is_mem=..., last_section_mask=...)
if getattr(self, "_eval_flat_softmax", False):
    # Flat over exactly the value-contributing support of the gated scheme (content of selected
    # blocks + local section; landmark positions carry no value here so are already 0-weight).
    visible = probs > 0
    neg = torch.finfo(scores.dtype).min
    probs = torch.softmax(scores.masked_fill(~visible, neg), dim=-1)
```
The `probs > 0` support is EXACT here: non-compressive `landmark_grouped_softmax` gives zero weight to
non-selected blocks and to all landmark positions, so the surviving support is exactly
{selected content + local}. (`scores` already has non-selected landmarks at −inf from top-k; masking
to `visible` also drops non-selected *content*.)

### 3.3 Compressive decode — `landmark_compressive.py::_compressive_decode_probs`
This function computes `gate_set = selected | last_section_b` and `gate_w = softmax(...)`, then
combines gate × within-block into `final`. Do NOT reuse the `probs>0` trick here — with
`nonselected_landmark_mass` (α) > 0 the gated output gives nonzero weight to non-selected compression
tokens, which the variant must exclude. Build the support explicitly from the masks already in scope
(`selected`, `last_section_b`, `S`, `Lb`, `block_landmark_pos`):
```python
if getattr(self, "_eval_flat_softmax", False):
    # Selected blocks' content+landmark, + local section; non-selected fully excluded (no α).
    flat_visible = last_section_b.clone()
    if S > 0:
        block_sel = selected[..., block_landmark_pos]              # (B,H,1,n_past_blocks) bool
        flat_visible[..., :S] |= block_sel.repeat_interleave(Lb, dim=-1)
    return torch.softmax(scores.masked_fill(~flat_visible, neg_inf), dim=-1)
# else: existing gated path unchanged
```
(Place this right after `gate_set`/`selected` are known and before the gated `final` is assembled;
return early. Double-check `block_landmark_pos` indexes the landmark within each of the `S//Lb` past
blocks — it's defined a few lines below as `arange(Lb-1, S, Lb)`; hoist it if needed.)

### 3.4 Sparse
`landmark_sparse.py` has its own decode path. **None of the three target checkpoints are sparse**
(they are `fast_landmark` / compressive), so you can skip it. If a sparse model is ever in scope, apply
the same idea to its `_apply_topk_landmark_retrieval` + grouped softmax.

### 3.5 Tests (`src/test/nn/attention/`)
Add a decode test (mirror `landmark_decode_test.py` / `landmark_gated_test.py` style):
- **flag-off unchanged:** with `_eval_flat_softmax=False`, decode output is bit-identical to current.
- **flag-on == reference:** with the flag on, the decode probs equal an independent flat softmax over
  the selected support (build the reference by hand from the top-k `keep` mask). Cover both a
  compressive and a non-compressive tiny config, on CPU (eager) so no GPU is needed. Assert non-selected
  blocks get exactly 0, and (non-compressive) landmark positions get 0.
Run: `~/miniconda3/envs/olmo-core/bin/python -m pytest -q src/test/nn/attention/landmark_flat_softmax_test.py`.

### 3.6 Verify end-to-end locally (no GPU needed)
The compressive/sparse EAGER decode runs on CPU. Reuse the harness in
`analysis/in_progress_gate_distribution.md` §... or the snippet that builds a `SparseLandmarkAttention`
/ fast landmark and runs prefill+decode with `set_landmark_eval_decode(..., top_k=...)`; set
`attn._eval_flat_softmax=True` and confirm the decode probs put zero mass on non-selected blocks and
are flat within the selected support. Commit + push the branch (eval jobs install olmo_core at the
pushed commit).

---

## 4. Threading the flag into the three harnesses

Because the attention module reads `OLMO_LANDMARK_FLAT_SOFTMAX` (via §3.1's env fallback), each harness
only needs to **set that env var in the container** — no model-arg / MODEL_DEFAULTS / backend changes.
Add these small passthroughs (all mirror the existing gate-log env plumbing):

- **RULER** — `scripts/launch_long_context_evals.sh`: add a knob, e.g.
  `OLMO_CORE_FLAT_SOFTMAX="${OLMO_CORE_FLAT_SOFTMAX:-}"`, and in the RULER loop where
  `gate_log_gantry_args` is built, append (independent of gate logging):
  `,env##5=OLMO_LANDMARK_FLAT_SOFTMAX=1` when it's set. (The `env##N` convention renders as repeated
  gantry `--env`; keep N unique.)
- **SFT** — `src/scripts/train/sft/singletask_ladder/run_q4b_beaker_multirung_eval.py`: add a
  `--landmark-flat-softmax` flag that appends `BeakerEnvVar("OLMO_LANDMARK_FLAT_SOFTMAX","1")` to
  `launch_config.env_vars` (exactly like the existing `--gate-log-all`/top-k env injection). Expose it
  through `launch_gate_score_eval.sh` **or** just call the python launcher directly (§5).
- **HELMET** — `../ai2-helmet/gantry_eval.sh` (sibling checkout; branch `amandab/qwen-landmark-eval`):
  forward the env var to the gantry job (add a `--env OLMO_LANDMARK_FLAT_SOFTMAX=1` when a
  `FLAT_SOFTMAX=1` shell var is set, next to where it already forwards `OLMO_CORE_*`). HELMET's
  olmo_core backend uses the same generation module, so the module-level env read applies.

**Sanity-check the env actually lands in the container:** before trusting a full run, do one small
job (e.g. RULER 4k limit 50 with the flag) and confirm from the model that the flat path is active —
easiest is to also enable the gate-log hook and diff a couple of decode steps, or add a one-time
`log.info` when `_eval_flat_softmax` is set.

---

## 5. Eval plan — baseline vs variant, full datasets

Same three checkpoints (see §8). For **each checkpoint**, run **each harness twice**: baseline
(gated, flag OFF) and variant (flat, flag ON), on the **full datasets**, at the **same top-k operating
point (fraction 0.1)** so the only difference is gated-vs-flat.

- **First check for existing baselines.** These checkpoints may already have full-dataset gated eval
  results (RULER/HELMET dashboards, `results/*.csv`, SFT `results/*_multirung.json`). If a clean
  full-dataset gated baseline at fraction 0.1 exists, you can skip re-running baseline and only run the
  variant. Otherwise run both.
- **Use distinct dashboards / name-suffixes** for gated vs flat so results never collide (see memory
  `topk-eval-per-topk-suffixes`): e.g. suffix `<label>-gated` vs `<label>-flat`, dashboard
  `flat-softmax-ablation`.

**RULER** (`scripts/launch_long_context_evals.sh`, cookbook env `cookin-lc-olmo3` — NOT `aman-launch-2`,
whose cookbook is too old; see the other note):
```
# full datasets => DROP RULER_LIMIT. fraction 0.1 per-length. SKIP_HELMET (HELMET run separately).
SKIP_HELMET=1 OLMO_CORE_LANDMARK_TOP_K_FRACTION=0.1 \
RULER_DASHBOARD=flat-softmax-ablation RULER_NAME_SUFFIX=<label>-flat \
OLMO_CORE_FLAT_SOFTMAX=1 \
scripts/launch_long_context_evals.sh <weka_ckpt>
# baseline: same but RULER_NAME_SUFFIX=<label>-gated and NO OLMO_CORE_FLAT_SOFTMAX.
```

**SFT** (native, `olmo-core` env). The 5 tasks run through the multirung launcher; use the **full**
per-rung sample count (its default `--max-test 600`, i.e. drop the 50 cap). Simplest is to call the
python launcher directly with the flat flag, prompt-format per model (base #1/#2 → `raw`, SFT #3 →
`chat`), variant per model (compressive / landmark):
```
PYTHONPATH=src python src/scripts/train/sft/singletask_ladder/run_q4b_beaker_multirung_eval.py \
  <label>-flat ai2/jupiter --task all --variant <compressive|landmark> --ckpt <weka_ckpt> \
  --prompt-format <raw|chat> --max-test 600 --priority urgent \
  --results-dir /weka/.../amandab/flat_softmax_ablation/<label>/sft_flat \
  --landmark-flat-softmax
# baseline: same without --landmark-flat-softmax, results-dir .../sft_gated.
```
(`--task all` submits one Beaker job per task. Note the SFT native eval already decodes at
`landmark_top_k_fraction=0.1` by default, matching RULER's fraction.)

**HELMET** (`../ai2-helmet/gantry_eval.sh`, backend olmo_core). One run covers 8k–128k; the
GenerationConfig default fraction 0.1 is length-adaptive so a single job is fine. Set the flat env
(§4) for the variant run; unset for baseline. Use `EVAL_NAME_SUFFIX=<label>-flat|-gated`. **Verify**
HELMET's olmo_core backend actually applies fraction-0.1 top-k by default (same open question the
launch scripts flag for "empty top_k"); if it defaults to dense soft-gating instead, pass an explicit
top-k so baseline and variant share the same selection. Confirm this on a single length before the full
suite.

Priorities: `urgent` everywhere except `ai2/holmes` (see memory `holmes-cluster-priority-exception`).

---

## 6. Deliverable — regression report

For each checkpoint × harness, build a table: task/length, **baseline (gated)** score, **variant
(flat)** score, **Δ (flat − gated)**, and a per-harness mean Δ. Save figures/tables under
`analysis/` (e.g. `analysis/flat_softmax_results.md` + CSVs).

Answer:
- **How much does removing gate reweighting regress each checkpoint** on HELMET / RULER / SFT? Where is
  it worst (which tasks, which context lengths)?
- **Compressive (#1, #3) vs plain landmark (#2)** and **base (#1, #2) vs SFT (#3)** — does the variant
  hurt one type more?
- Bottom line: is "top-k cutoff + flat softmax over selected regions" a viable simplification (small
  regression) or does the gate weighting carry real quality?

---

## 7. Gotchas

- **Decode-only:** do NOT touch prefill or the training-time grouped softmax — only the single-query
  decode probs. A regression in *training* numerics means you edited the wrong path.
- **Support must match the model type** (§1): non-compressive excludes landmark positions; compressive
  includes selected landmarks and excludes non-selected (ignore α when the flag is on).
- **Env must reach the worker process.** Under torchrun/gantry the env is per-container; the module
  reads it at generation start. Verify with a tiny job before committing to full runs (§4).
- **RULER cookbook env:** use `~/miniconda3/envs/cookin-lc-olmo3/bin`; `aman-launch-2`'s cookbook lacks
  `--model-backend olmo_core`. SFT launcher needs the `olmo-core` conda env (torch+olmo_core+beaker).
- **RULER datalake flakiness:** a job can exit nonzero at upload yet have valid scores — pull via
  `beaker experiment results` rather than rerunning (memory `ruler-datalake-upload-failure-recovery`).
- **Base models on SFT tasks** run off-distribution (raw prompt); the SFT regression for #1/#2 is
  weaker signal than RULER/HELMET. Keep them for completeness but weight the conclusion toward
  RULER/HELMET for the base models.
- **Full HELMET/RULER are expensive.** Consider a length/task subset first to sanity-check the Δ sign
  and env wiring, then scale to full.

---

## 8. Reference

- **Baseline code:** `f3eecf6cb` on `amandab/remove-softmax-scaling`. New work on a fresh branch.
- **Three checkpoints** (`.../` = `/weka/oe-training-default/ai2-llm/checkpoints/`):
  | # | label | checkpoint | variant | type | SFT prompt |
  |---|-------|-----------|---------|------|------------|
  | 1 | q4b-base-fastcomplm-s2385 | `.../amandab/q4b-base-fast-compressive-landmark-8node/step2385` | compressive | base | raw |
  | 2 | q4b-base-fastlm-s2385 | `.../amandab/q4b-base-fast-landmark-8node/step2385` | landmark | base | raw |
  | 3 | q4b-comp-5task-s8550 | `.../prasanns/q4b-compressive-5task-32k-nocpt-fixdata/step8550` | compressive | SFT | chat |
- **Files to change:** `generate/generation_module/config.py` (+ flag),
  `generate/generation_module/transformer/generation_module.py` (set `_eval_flat_softmax` from flag/env),
  `nn/attention/landmark_fast.py::_decode_probs`, `nn/attention/landmark_compressive.py::_compressive_decode_probs`,
  attention base init/clear for `_eval_flat_softmax`, a new test under `src/test/nn/attention/`, and the
  three harness env passthroughs (§4).
- **Related:** `analysis/in_progress_gate_distribution.md` (the observational study), memory
  `landmark-gate-score-capture` (the launchers, envs, and gate-log plumbing this reuses).
```
