# CTC Suite Scaling Plan — Figure-4 at ~25 tasks, 2k–32k token ladders

**Status: CONFIRMED 2026-07-18** — decisions Q1–Q4 resolved (see §11); executing.
Owner: prasann + Claude. Created 2026-07-18.

## 1. Goal

Produce a large-scale version of TimeToPayAttention Figure 4: for every task in the current
corpus-reasoning suite (~25 catalog rows), plot **context length in tokens (2k → 32k)** on the
x-axis vs. the task metric, with two arms per task:

- **full** — standard causal attention (upper bound / expensive O_A(N²)).
- **chunked** — document-chunked attention on the non-GDN layers of Qwen-3.5, trained with
  curriculum mask mixing where it helps, **always evaluated with the pure chunked mask**.

The headline question stays the paper's: on which tasks does the full-vs-chunked gap stay flat
with scale (O(N)-like) and on which does it widen (O(NM)/O(N²)-like) — now across the whole
suite, on token-denominated ladders, with one jointly-trained model per (task, arm) instead of
per-N models.

## 2. Deltas vs. the paper (all intentional)

| Paper | This effort | Why |
|---|---|---|
| Axolotl + HF, LoRA r=32 on frozen base | olmo-core, **full fine-tune** (established memexpress SFT path) | requirement (a); full-FT is the battle-tested path in this repo. LoRA via `olmo_lora.py` exists but is unproven at scale — decision point Q1 below |
| x-axis = N documents | x-axis = **tokens** (2k/4k/8k/16k/32k rungs); N derived per task from per-doc token size | requirement (d) |
| one training run per (task, N) | **one joint run per (task, arm)**: train docs-count n drawn uniformly over [n(2k), n(32k)] | requirement (e) |
| Qwen-3.5 0.8B/2B/4B/9B | Qwen-3.5 **0.8B / 4B / 9B** (olmo-core factories `qwen3_5_0_8B/4B/9B`; no 2B factory) | requirement (c) |
| static mask-mix p=0.10 | **curriculum mix 0.8→0.0** (project default, two-divisor anneal fix in place), pilot-compared against pure chunked per task | requirement (b) |
| results scattered per-run | **all results as JSONs in `results/ctc_suite/`** + flat `all_results.jsonl` (results-hub ingestible, full provenance) | requirement (f) |
| 6 main-figure tasks | ~25 tasks | the point |

Non-deltas: same chunk semantics (context↔context edges restricted; query/CoT/answer tokens
FREE), eval always pure chunked mask for the chunked arm, eval sets ≥ ~500 (488 for
contradiction), each task keeps its catalog CoT mode (labeled long/short/no-cot in results).

## 3. Task suite and ladder mapping

Source of truth: `results/task_suite.md` + `results/task_suite_README.md` (corpus-reasoning),
generators identical in-tree at `src/corpus_reasoning/data/`. Per-doc token sizes are from paper
Table 4 + suite README; **rung doc-count n(T) ≈ T_tokens / per-doc-tokens** (minus ~100–400
query/CoT overhead), audited per task in Stage 1.

Model-scale prior column = starting scale, from paper Table 5 where the task existed; every task
gets a 2k-rung full-attention pilot and escalates 0.8B → 4B → 9B until the small-context metric
clears its floor (see §5).

| # | Task (`--task`) | Class | Metric | ~tok/doc | n at 2k→32k | scale prior | CoT |
|---|---|---|---|---|---|---|---|
| 1 | NQ retrieval (`retrieval`) | O(N) | gold-ID F1 | 130 | 15→240 | 0.8B | no |
| 2 | HotpotQA (`retrieval`/`cot_retrieval`) | O(N) | gold-ID F1 | 130 | 15→240 | 0.8B | short |
| 3 | NIAH-contradiction (`retrieval`) | O(N) | set-F1 | 45 | 40→700 | 0.8B | no |
| 4 | OOLONG (`oolong`) | O(N) | partial credit | line-mode | auto | 4B | short |
| 5 | HELMET NarrativeQA (`qa`) | O(N) | token-F1 | — | `--lengths` | 4B | no |
| 6 | HELMET GovReport (`summarization`) | O(N) | ROUGE | — | `--lengths` | 4B | no |
| 7 | BEIR SciFact (`retrieval`) | O(N) | set-F1 | ~180 | 10→175 | 0.8B | no |
| 8 | BEIR FiQA (`retrieval`) | O(N) | set-F1 | ~130 | 15→240 | 0.8B | no |
| 9 | MS MARCO (`retrieval`) | O(N) | set-F1 | ~80 | 25→400 | 0.8B | no |
| 10 | MS MARCO rerank (`rerank`) | O(N) | MRR@10 | ~80 | 25→400 | 0.8B | no |
| 11 | outlier wiki scale-k (`outlier`) | O(NM) | set-F1 | 130 | 15→240 | 4B | short |
| 12 | outlier Amazon (`outlier`) | O(NM) | set-F1 | 80 | 25→400 | 0.8B | short |
| 13 | grouping OpenAlex (`grouping`) | O(NM) | pairwise-F1/ARI | 200 | 10→160 | 4B | short |
| 14 | grouping labeled (`grouping_labeled`) | O(NM) | F1 | 200 | 10→160 | 4B | short |
| 15 | textgroups (`textgroups`) | O(NM) | ARI/NMI | ~150 | 13→210 | 4B | short |
| 16 | contradiction PubMed (`contradiction`) | O(N²) | set-F1 | 45 | 40→700 | 4B (repro also 0.8B) | no |
| 17 | redundancy (`redundancy`) | O(N²) | F1 | 45 | 40→700 | 0.8B | no |
| 18 | absence PubMed (`absence`) | O(N²) | F1 | 45 | 40→700 | 0.8B | no |
| 19 | absence official (`absence`) | O(N²) | F1 | varies | fixed sets | 4B | no |
| 20 | strmatch (`strmatch`) | O(N²) | F1 | ~60 | 30→500 | 0.8B | no |
| 21 | qdmatch NQ / HPQA / ObliQ (`qdmatch`, 3 rows) | O(N²) | pair-F1 | 130 | q=8→120 | 4B | no |
| 22 | xabsence (`xabsence`) | O(N²) | F1 | ~45 | P=4→60 | 4B | no |
| 23 | mathmatch (`mathmatch`) | O(N²) | F1 | ~30 | 60→1000 | 0.8B | no |
| 24 | reorder Gutenberg (`reorder`) | O(N²) | Kendall-τ | 130 | 15→240 | 9B | short |
| 25 | cycle (`cycle`) | O(N³) | F1 | ~40 | 45→750 | 4B | short |
| 26 | groups4 (`groups4`) | O(N³) | F1 | ~30 | 60→1000 | 4B | short |

Notes:
- Rows 21 counts as 3 catalog rows (NQ/HPQA/ObliQ variants) → ~28 rows total; we can trim to
  one qdmatch variant if compute gets tight (decision Q3).
- ruler / matching_ngram / musique / arithmetic generators exist but are not in the catalog's
  core rows; excluded unless requested.
- outlier wiki runs **scale-k only** in the main grid; the fixed-M control is reproduced once in
  the trend-repro gate (§6), not across the whole grid.
- HELMET tasks (5, 6) have native `--lengths` control; GovReport ROUGE is the one non-ID metric
  — flagged as lower-priority (Q3).

## 4. Method configuration

### Model / attention
- Qwen-3.5 hybrid via `TransformerConfig.qwen3_5_{0_8B,4B,9B}` — 3:1 GDN:attn pattern; with
  `document_chunked=True` only the attention blocks get the chunked mask
  (`config.py:1577-1587`); GDN layers ignore `chunk_ids` (matches paper: GDN state crosses
  chunk boundaries by design).
- Doc boundaries: `<|box_start|>`=248049 / `<|box_end|>`=248050, eos 248044 (Qwen-3.5 ids).
- **Stage 0 gate: marker-embedding audit on every Qwen-3.5 base scale** (cosine AND norm — both
  Qwen3 marker bugs; `records/document-chunked-marker-embeddings.md`,
  `records/n100-chunked-marker-position-bug.md`). Extend `fix_marker_embeddings.py` for the
  Qwen-3.5 vocab if the audit fails; train only from repaired `-fixmark` bases.
  - **0.8B AUDITED 2026-07-18: PASS.** On `q35-08b-base-modelonly`: cos(248049, 248050) = +0.29,
    norms 0.458/0.540 vs trained-row median 0.629 — markers are trained and distinct (Qwen-3.5
    uses box tokens for multimodal grounding), so **no fixmark needed** for docchunk. Caveat:
    rows 248200/248319 (landmark-region padding) are still cos = 1.0000 — fine here (no landmark
    variant in this plan), but any future Qwen-3.5 landmark run must repair them first.
    Repeat this audit when the 4B/9B bases are converted (audit snippet: `load_keys(path,
    ["model.embeddings.weight"])` then cos/norm per the marker records).
  - **4B AUDITED 2026-07-19: PASS.** Converted HF `Qwen3.5-4B-Base` (cubbins
    `/data/prasann/hf_models/Qwen3.5-4B-Base`) → distcp
    `/data/prasann/ctc_suite/bases/q35-4b-base-modelonly` (13G, `.metadata` verified). cos(248049,
    248050) = +0.226, norms 0.431/0.624 vs trained-row median (first 150k rows) 0.659 — ratios
    0.65×/0.95× — trained and distinct, no fixmark needed.
  - **9B AUDITED 2026-07-19: PASS.** Converted HF `Qwen3.5-9B-Base` (cubbins
    `/data/prasann/hf_models/Qwen3.5-9B-Base`) → distcp
    `/data/prasann/ctc_suite/bases/q35-9b-base-modelonly` (23G, `.metadata` verified). cos(248049,
    248050) = -0.001, norms 1.031/0.897 vs trained-row median 0.929 — ratios 1.11×/0.96× —
    trained and distinct, no fixmark needed.
  - 0.8B base also copied node-local to cubbins `/data/prasann/ctc_suite/bases/q35-08b-base-modelonly`
    (2.6G, `.metadata` verified) so all three scales now read from `/data` (no more `/scratch` NFS
    reads at train time). Conversion driver:
    `src/scripts/train/memexpress/ctc_suite/convert_qwen35_base.py` (+
    `convert_qwen35_bases.sbatch`) — bypasses `convert_hf_to_olmo.py`'s unconditional
    `snapshot_download` call to convert directly from an already-local flat HF dir, reusing
    `corpus_reasoning.train.convert_hf_to_olmo._convert_qwen3_5` / `convert_qwen3_5_state_from_hf`
    and running the marker audit inline (writes `marker_audit.json` next to each converted base).
- One shared converted distcp base per scale (model-only resave), so full/chunked arms of a task
  start bit-identical. Fresh `--save-folder` per run (silent auto-resume trap).

### Mask mixing (chunked arm only)
- Default: curriculum `mix_start_p=0.8 → mix_end_p=0.0`, `mix_seed=42`, with
  `mix_total_forwards` derived from examples/epochs ÷ world_size ÷ micro-batch (both known
  divisor bugs; script hard-fails if the anneal can't reach 0).
- Pilot stage runs each task chunked **{pure, curriculum}** at 0.8B/small-ctx; the winner is the
  task's chunked recipe at scale (requirement (b): mixing only where it helps). Paper prior:
  helps grouping/contradiction/reorder, neutral-to-negative on outlier.
- Eval never mixes.

### Training recipe (from working memexpress scripts, sanity-checked per task in pilots)
- Full fine-tune, bf16, AdamW, cosine + warmup, seq_len 40960 (fits the 32k rung + overhead),
  grad-accum tuned per scale; hyperparams inherited from
  `attn_explore/Qwen3.5-0.8B-docchunk-mask-mix-contradiction-SFT-local.py` and
  `sft_docchunk/_docchunk_5task_32k_nocpt_common.py`, adjusted only if pilot loss curves demand.
- Train data per (task, arm): **~20k examples for final runs** (user requirement 2026-07-18),
  doc-count n ~ Uniform[n(2k), n(32k)] (uniform interpolation per requirement (e)); pilots use
  ~2–3k subsets. Per-task pool-capacity audit in Stage 1 flags any task that can't reach 20k
  without excessive gold reuse (achievable count recorded instead). ≈350M tokens/epoch at 20k ×
  ~17k avg tokens; epochs tuned so total tokens stay ≈300–500M/run.
- 0.8B: 1 node × 8 GPU (local H200 / lambda A100 / VESSL). 4B: 2–4 nodes (Beaker jupiter,
  urgent). 9B: 4 nodes (Beaker jupiter, urgent). Wandb group link surfaced at launch.

### Eval
- Per-rung eval files at 2k/4k/8k/16k/32k, **eval_size ≥ 500** each (488 for contradiction;
  anything smaller flagged inline with binomial SE).
- **Fixed-across-rungs eval sets** (user requirement 2026-07-18): the same 500 underlying
  examples (identical queries + golds) at every rung, with only corpus/filler size varying —
  rung-to-rung deltas must not include eval-set resampling noise. Generators that can't hold
  the example set fixed while scaling n get a small ladder-mode patch (Stage 1 ACTION items;
  `build_v2_eval_ladders.py`/`build_xlong_rungs.py` in the parallel repo may already do this).
- **Speed: vLLM wherever possible** (user requirement 2026-07-18). Full-attention arms on
  vLLM if it serves Qwen-3.5 hybrids in our env (Stage 0 verification); for chunked arms,
  check the corpus-reasoning vLLM-chunked setup — if none works, `chunked-sdpa` batched path
  with split jobs and per-rung timing checks. Getting chunked evals fast is a Stage-0/2
  deliverable, not an afterthought.
- olmo→HF export via the in-tree export path. Native olmo evaluators
  (`eval_lc_native_docchunk_ladder.py`) used where they already support the task.
- Known traps enforced: `--max-length ≥ rung + max_new_tokens` (maxlen-truncation bug), no-cot
  EOS/truncate-at-`]]` handling, think-strip, per-rung timing sanity check, dump generations
  whenever a trained model scores ~0.

## 5. Model-scale selection ("effective at small context")

For each task: 2k-rung pilot, full attention, 0.8B, ~500 train examples-equivalent short run.
Pass = metric clears `max(2× random/floor baseline, 0.3 absolute)` (per-task floor recorded in
the plan's audit sheet). Fail → escalate 4B, then 9B. The chosen scale is used for **both arms**
of that task (matched-scale comparison, as in the paper). Scale choices logged in
`results/ctc_suite/scale_selection.json`.

## 6. Trend-reproduction gate (must pass before full fan-out)

Reproduce the paper's key qualitative trends **with the new pipeline** on the 6 paper tasks
(NQ, HotpotQA, outlier-wiki, grouping, contradiction, reorder), using the new joint-training
protocol, plus one per-N control:

| Criterion | Pass condition |
|---|---|
| T1 CTC separation | NQ/HPQA full-vs-chunked gap roughly flat across rungs; grouping/contradiction/reorder gap grows ≥2× from smallest to largest rung |
| T2 mask-mixing lift | on contradiction (and reorder if cheap): chunked+mix ≫ pure chunked, ordering pure < mixed ≤ full |
| T3 outlier fixed-M control | fixed-M outlier gap stays flat while scale-k widens (one extra run pair) |
| T4 reorder floor | chunked τ → ~random at large rung while full stays above |
| T5 scale pushback (spot check) | contradiction at 0.8B vs 4B: 4B chunked degrades later; 0.8B-full ≥ 4B-chunked at the largest rung |
| T0 per-N control | contradiction per-N (n=20/50/100, 0.8B, both arms) reproduces the paper's widening-gap shape — isolates protocol (joint vs per-N) from pipeline bugs if T1 fails |

Qualitative match (ordering + gap growth), not absolute numbers, is the bar — data is
regenerated and the protocol differs. Any failed criterion stops the fan-out and gets a
diagnosis first (the repo history says failures here are usually data/embedding bugs, not
modeling results).

## 7. Compute allocation

**Cluster-access correction (user, 2026-07-19):** Berkeley slurm access is **only `berkeleynlp`
and `jsteinhardt`** partitions (NOT songmei/feanor, NOT yss/beren/luthien). `preemptive` and
`preemptive_high` QOS both allowed; prefer plain `preemptive` for non-urgent pilot/eval work.
Concurrency reality: `jsteinhardt` has an **8-GPU/user cap** → effectively ONE 8-GPU job at a
time across the whole partition (any of cubbins/mooney/mcfuzz/sneetches). `berkeleynlp` (horton +
lorax) is a separate quota → ~2 more concurrent H200 jobs. So Berkeley = **~3 concurrent training
slots total**. Cross-node work needs the base+shard staged to that node's `/data` (scp via a
sleeper job on the target; pam_slurm_adopt blocks ssh without an active job there). **Width for
the full fan-out therefore comes from Beaker (jupiter, wide) + lambda (A100, small-scale); Berkeley
handles the repro-gate exemplar pair + quick pilots.**


| Resource | What runs there | Notes |
|---|---|---|
| Berkeley `berkeleynlp`/jsteinhardt (H200×8) | Stage-0 audits, data builds, all 0.8B pilots + several 0.8B finals, native evals | `/data` staging + NFS log rules per `local_cluster.md`; preemptive_high QOS |
| Berkeley `lambda` partition (A100) | 0.8B finals + eval overflow | tokenize on horton → rsync via existing `jobs/*lambda*.sh` flow |
| Beaker `ai2/jupiter` (H100, **urgent**) | all 4B and 9B training runs | `sft_docchunk` common launcher pattern, 2–4 nodes, weka staging for data |
| VESSL (`cluster-betelgeuse`, A100 $1.55/hr) | pilot fan-out (25×2 short runs), eval overflow, stragglers | existing `vessl_submit.sh` + orchestrator; est. ≲$300 for pilots+evals, hard cap Q4 |

Rough final-grid budget: ~28 rows × 2 arms ≈ 56 joint runs (≈30 at 0.8B, ≈20 at 4B, ≈6 at 9B)
+ ~60 pilot/smoke shorts + repro gate (~16 runs) + evals. Order 1.5–2.5k H100/H200-hours —
in line with the paper's reproduction estimate scaled to 4× tasks, spread over 4 pools.

## 8. Results storage (requirement f)

`results/ctc_suite/` in OLMo-core (committed):

```
results/ctc_suite/
  all_results.jsonl                 # flat, one line per (task, arm, rung) — plotting + results-hub
  scale_selection.json              # per-task chosen scale + pilot numbers
  <task>/<model>_<arm>/rung_<T>.json
```

Per-result JSON schema (every eval writes this; nothing hand-copied):

```json
{
  "task": "contradiction", "complexity_class": "N2",
  "model": "qwen3.5-4b", "arm": "chunked",
  "mask_mix": {"mode": "curriculum", "start_p": 0.8, "end_p": 0.0} ,
  "trained_ctx": "2k-32k-joint-uniform",
  "rung_tokens": 8192, "n_docs": 180,
  "metric_name": "set_f1", "metric_value": 0.71,
  "aux_metrics": {"precision": 0.7, "recall": 0.72, "parse_rate": 0.99},
  "eval_size": 500, "cot_label": "no-cot",
  "provenance": {"git_commit": "...", "data_path": "...", "ckpt_path": "...",
                  "eval_backend": "chunked-sdpa", "launcher": "...", "date": "..."}
}
```

Plot generator (`src/scripts/eval/plot_ctc_suite.py`, new): Figure-4-style grid, one panel per
task grouped by complexity class, full vs chunked lines with binomial error bands, gap-growth
annotation per panel. Also ingest `all_results.jsonl` into results-hub.

## 9. Stages, gates, timeline — **compressed to 2–3 days (user directive 2026-07-18)**

Stages are gates, not calendar phases: everything that can overlap, overlaps. If throughput
threatens the schedule, optimizing the pipeline (faster evals, fewer epochs, packing, more
nodes) is in-scope before cutting coverage.

| When | Running concurrently | Gate |
|---|---|---|
| Day 0 (now → overnight) | Stage 0 finishes (marker audit ✅ 0.8B, eval-backend verdict, 4B/9B base pulls+conversion); BUILD_MATRIX done; **all P0+P1 data builds launched overnight** (CPU fan-out); joint-ladder launcher + eval harness + results-JSON writer authored | contradiction end-to-end at 0.8B works |
| Day 1 AM | Data audits (P0/P1); pilot wave launches across local+lambda+Beaker — the first pilot per pool doubles as its smoke test; P2 data builds run behind | audit sheet green per launched task |
| Day 1 PM | Trend-repro runs (P0, 2k–8k, joint + per-N control) launch as soon as their shards pass audit — **not** waiting for all pilots; remaining pilots stream; vLLM eval throughput validated at scale | scale_selection.json filling; repro evals overnight |
| Day 2 AM | Repro gate T0–T5 checked from overnight evals → **go/no-go**; on go: full fan-out waves launch everywhere (VESSL joins here, $500 cap), P0/P1 finals first | T0–T5 pass or user consulted |
| Day 2 PM → Day 3 | Fan-out completes in waves; per-rung evals stream as runs finish (eval jobs pre-templated, auto-launched); 16k/32k rung extension for tasks whose 2k–8k trends are sane | all rungs evaluated, JSONs in repo |
| Day 3 | Stragglers; aggregate: plots, gap-growth table, results-hub ingest, writeup | figure + writeup |

**Data-generation speed requirement (user, 2026-07-19): ≤10 min per task for 20k datapoints.**
Enforce at the post-reset wake: profile each generator; parallelize via wide sharding across CPU
nodes (NQ-style shard splits but 16–64-way), run CE/embedding filter steps on a GPU instead of
CPU (the NQ regen's >1h CPU CE-filter is the known violator), pre-stage retrieval indices/pools
on node-local /data (never /scratch), and template one parallel-launch wrapper all tasks share.

Throughput levers (applied as needed to hold the schedule): 1 epoch at 20k examples before
considering more; sequence packing for short rungs; eval max_new_tokens per task minimum;
vLLM for every arm that can take it; per-rung eval jobs split and right-sized; extra Beaker
nodes (urgent preempts) over queue-waiting.

## 10. Risks / open questions

- **R1** Qwen-3.5 marker rows may be untrained like Qwen3's — Stage 0 audits before anything trains.
- **R2** vLLM may not serve Qwen-3.5 hybrids in our pinned env → full-arm eval falls back to
  `standard` HF backend (slower; budget covered by splitting eval jobs).
- **R3** Joint 2k–32k training could mask per-N trends (train-length interpolation is new) —
  T0 per-N control isolates this.
- **R4** Some tasks may have thin data pools at 32k (grouping abstracts, xabsence pairs) —
  Stage-1 audit flags; rung dropped (and logged) rather than silently under-filled.
- **R5** GovReport ROUGE and OOLONG partial-credit metrics are noisier than ID-F1 — flagged in
  plots; not used for headline gap-growth claims.

## 11. Confirmed decisions (user sign-off 2026-07-18)

- **Q1 → Full fine-tune** everywhere (no LoRA).
- **Q2 → Joint + per-N control**, with a modification: the repro gate (and initial sanity work
  generally) **starts on 2k–8k rungs only**; extend to 16k/32k once the short-rung trends look
  sane. So Stage 4 first pass = 2k/4k/8k rungs, then the 16k/32k extension.
- **Q3 → All ~28 rows, ordered**: within each complexity class, run the strongest new
  representatives first; handle new (non-paper) tasks **iteratively** — for every new task,
  full attention must work well at short context (the §5 pilot bar) before any scale-up or any
  chunked-arm training on it.
- **Q4 → VESSL cap $500**, and VESSL is reserved for the **final large-scale stage only**
  (Stage 5 fan-out overflow) — no pilots/debugging there. Pilots move to local + lambda.
