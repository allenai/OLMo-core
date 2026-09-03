# FLOP-scaling study: are the FFN-routing and pooled-KV methods compute-optimal vs dense?

Brief (Prasann, 2026-09-02): "now that we have a decent approach with v12 (FFN routing) and the
KV-cache training compression (the 128 and 256 ones), do FLOP scaling experiments to see if they
are more compute-optimal than dense training, on short-heavy mix data, on several tasks (outlier,
oolong, contradiction, nq). Don't combine methods. Go up to 32k. Clear data- and FLOP-scaling
curves; are the methods scalable and applicable to diverse tasks."

Ledger: `debug/flop_scaling/LAUNCH_LEDGER.tsv`. Driver scripts: `debug/flop_scaling/`.
Results: `results/flop_scaling/`.

## 1. Question and axes

For each task, plot held-out accuracy (per eval rung, and a mean over rungs) against

1. **training data** (tokens seen, short-heavy mix), and
2. **training FLOPs** (actual, method-aware -- see §5),

for four arms. A method is "compute-optimal" on a task if its accuracy-vs-FLOPs curve lies above
dense's; it is "data-optimal" if accuracy-vs-tokens matches dense (i.e. the FLOP saving is free).
Both plots are needed: the methods are cheaper per token, so a curve that loses on the data axis can
still win on the FLOP axis.

## 2. Arms (no combinations)

| arm | what | per-token training cost vs dense (whole model, 6k / 32k ctx) |
|---|---|---|
| `dense` | plain causal Qwen3-4B | 1.0 |
| `ffnmoe` | nested-FFN router, **v12 recipe**: stage 1 = layers 12–35 routed with the 7-rung ladder `1,16,64,256,1024,9728` (width-1 rung) at target 0.01; stage 2 = warm-start from stage 1, all 36 layers, target 0.02, hinge active from step 0 | stage 1 ≈ 0.66 / 0.77; stage 2 ≈ 0.47 / 0.66 (FFN → ~0.02 of dense on routed layers; attention unchanged). Both stages count. |
| `kv17` | pooled-KV soft-token, `gold_plus_random`, detach-soft-KV, no distill (the v20/v22 recipe) with a **fixed keep fraction of 1/6** of the non-gold docs at every context length (Prasann 2026-09-02: fixed percentage, not a fixed count; 1/6 ≈ the 128-of-~760 arm at 32k) | ≈ (gold + 1/6 of docs) real + one soft token per pooled doc: sequence ≈ 1/5 |
| `kv33` | same with keep fraction **1/3** (≈ the 256 arm at 32k) | sequence ≈ 1/2.7 |

On oolong there is no gold set; the KV arms keep the same fractions gold-blind (`--st-gold-blind --st-keep-prob 0.17/0.33`).

Source recipes: `src/scripts/train/memexpress/ffnmoe/README.md` (v10/v12),
`records/pooled-doc-kv-handoff.md` (v20/v22). Model: Qwen3-4B (dense attention; the marker-repaired
base that both methods were developed on). Query position `after`, no-CoT, 1 epoch per stage.

## 3. Tasks and data

Four tasks, one **short-heavy length mix** each (45/27/16/8/4 % of TOKENS at the 2k/4k/8k/16k/32k
rungs -- the standing directive from the outlier length-mix campaign), built as **nested prefixes**
at five token budgets:

| budget | 8M | 16M | 32M | 64M | 128M |
|---|---|---|---|---|---|
| examples at 2k (45 %) | ~1.8k | ~3.6k | ~7.2k | ~14k | ~29k |
| examples at 32k (4 %) | ~10 | ~20 | ~40 | ~80 | ~160 |

Pools come from the CTC-suite generators (`ctc-data build --pool auto`, per-rung document counts
from each task's calibration table), tokenized for **Qwen3** (`--marker-set qwen3`, the outlier
campaign's arms are Qwen3.5-tokenized and cannot be reused as-is), with a `gold_fingerprints.json`
sidecar per arm for the soft-token keep policy. Per task:

- **outlier**: per-rung pools already exist (`/data/prasann/outlier_lengthmix/` on mooney,
  n14/28/57/111/220 + ext pools); re-tokenize for Qwen3. Gold = the outlier docs.
- **contradiction**: gold = the contradicting pair. The mined-pair seed pool caps distinct examples
  at ~18k, so contradiction's budgets are **8M/16M/32M/48M** (the 128M short-heavy arm alone
  would need 28k 2k-examples). Extending it means mining more pairs with an LLM.
- **nq**: gold = the answer doc (p10 hard-neg pipeline; NOT the retired 98%-hard pool).
- **oolong**: aggregation over the whole corpus -- there is **no gold subset**. The soft-token arms
  run gold-blind (`keep_prob` breadth only), which is exactly the applicability test for KV
  pooling on a task whose answer depends on every document.

Eval: the fixed held-out rung sets (same 600 examples every rung), native evaluator on Beaker
(`run_q4b_beaker_multirung_eval.py`), rungs per task: contra 2k/8k/16k/32k, nq 3k/8k/16k/32k,
outlier 3k/8k/16k/32k, oolong 8k/16k/32k. The `ffnmoe` arm is scored **with routing on** (that is
the deployed model); the soft-token arms with plain full attention (zero-shot transfer, as designed).

## 4. Run matrix

4 tasks × 5 budgets × (dense 1 + ffnmoe 2 stages + kv17 1 + kv33 1) = **100 training jobs**,
**80 eval jobs** (one multi-rung eval per (task, budget, arm)). Beaker `ai2/jupiter`, 1 node × 8
H100 per training job, `urgent`. Rough cost: dense 128M tokens at seq 40960 ≈ 1 h on 8 H100; the
whole grid ≈ 4 tasks × 248M tokens × ~2.2 arm-equivalents ≈ 15–20 node-hours plus evals. Pilots
first: one (task=contradiction, budget=16M) job per arm, end to end through eval, before the
grid (contradiction/oolong data landed first).

### Batching (decided from the smokes, 2026-09-02)

`dense` and `ffnmoe` train **packed** (`--pack`, seq 65536 -- the packer needs a power of two --
8 packed rows per optimizer step ≈ 524k tokens). The soft-token arms need one example per row
(per-row content fingerprints resolve the gold set), so they run the **padded** path at seq
40960 with a 160-example global batch (≈ the same 524k tokens per step on average); their
compaction drops padding before any compute, so no FLOPs are wasted on it. The FLOP meter counts
NON-pad tokens (rows are padded with EOS) so both paths are charged for the same real tokens.

## 5. FLOP accounting (the x-axis)

A `FlopMeterCallback` records, every step, the tokens actually processed and the model-aware cost:

- **dense**: `6·N_params·tokens + attention(seq)` per the trainer's own `num_flops_per_token`.
- **ffnmoe**: FFN share replaced by `mean_cost` from the router's hard routing that step (already
  logged by `NestedFFNMoECallback`); attention/LM-head/other unchanged. Integrated over training.
  Stage 1 + stage 2 are summed for the stage-2 checkpoint's x-coordinate.
- **kv17/kv33**: the trainer runs on the **compacted** sequence; FLOPs are the dense formula
  applied to the compacted length (all terms shrink, attention quadratically). Tokens seen for the
  DATA axis are the original (uncompacted) tokens.

Wall-clock (TPS) is recorded alongside, for a secondary "GPU-hours" x-axis, since the FFN saving is
known not to convert to wall-clock at 6k (`src/scripts/train/memexpress/ffnmoe/README.md`).

## 6. Infrastructure work (in order)

1. `train_ctc_suite.py`: add `--variant ffnmoe` and `--variant softtoken` (ported from the two
   local trainers), tolerant base loading (`strict=False` so router/projector keys initialize
   instead of needing baked bases -- no per-method bases on weka), `--ffn-moe-warm-start` for
   stage 2, the FLOP meter, and the routing callback.
2. `beaker_ctc_suite.py`: pass the new variants/args through; Qwen3-4B base default.
3. Data: `debug/flop_scaling/build_shortheavy_arms.sbatch` -- per-task per-rung pools →
   short-heavy nested arms at the five budgets → Qwen3 tokenization → gold sidecars → S3 → weka.
4. Eval: FFN routing flags in the bundled `eval_lc_native.py`; re-upload the eval bundle.
5. Pilots (4 arms on outlier/16M), then the grid via `debug/flop_scaling/launch_grid.py`, then
   `collect_results.py` → `results/flop_scaling/` and the two plots per task.

## 7. Assumptions taken (say if any should change)

- Qwen3-4B, not Qwen3.5 (the methods' recipes and bases are Qwen3-4B; Qwen3.5 would need the
  soft-token/FFN code validated on the GDN hybrid first).
- Token budgets 8M–128M (2× spacing). 128M short-heavy needs ~29k 2k-examples per task, which the
  generators can produce.
- 1 epoch per stage; the ffnmoe arm's stage 1 uses the same arm data as its stage 2 (so it sees
  the data twice, and both passes are charged).
- The soft-token arms keep a fixed FRACTION (1/6, 1/3) of non-gold docs real at every length (`--st-keep-frac`), per Prasann's instruction; the 128/256 absolute-count recipes are the reference points at 32k.

## 8. KV iteration (Prasann 2026-09-02: "iterate especially the KV idea to get good FLOP-optimal performance for all tasks")

After the baseline grid (kv17 / kv33 at fixed keep fractions), the KV arm is iterated per task on
the FLOP axis. Pre-registered variants, one change each, launched at the budgets where the
baseline curves separate (typically 16M–64M), evaluated identically:

| variant | what it tests |
|---|---|
| `kv08` / `kv50` (keep 1/12, 1/2) | the keep-fraction frontier: where does each task's curve bend? |
| `kv-blind` (gold-blind at the same fraction, all tasks) | does the method need gold knowledge at train time? (deployment-realistic; oolong is already blind) |
| `kv-range` (`--st-n-random-range` log-uniform between 1/12 and 1/2 of docs) | scale-invariant breadth randomization (the v25 mechanism) |
| `kv-mix` (`--st-mix-start-p 0.5 --st-mix-end-p 0`) | compression-mixing curriculum: some uncompressed rows early |
| `kv-nodetach` | the detach ablation (co-drift control) |
| `kv-long` (short-heavy mix shifted longer, or 32k-only tail) | whether the KV saving, which grows with context, buys more at longer training lengths |

Selection rule: for each task, the variant with the best mean-f1 at matched actual PFLOPs wins;
ties go to the cheaper one. Oolong is the hard case (no gold subset) and gets the blind /
range / mix variants first.

## 9. PIVOT to Qwen3.5-4B (Prasann 2026-09-02 ~23:30: "stick to 3.5 for everything if possible")

The prior dense campaigns (`debug/taskscale_lengthmix`, `debug/outlier_lengthmix_scaling`) already
give dense Qwen3.5-4B data-scaling points on outlier (16–640M), nq (16/32/48M), contradiction
(14/28/56M) and oolong (20/40/80M), all short-heavy mixes, base `q35-4b-base-markerfix`, seq 65536
packed, lr 5e-6, batch 8. The method arms therefore train on THOSE arms with THOSE settings and
the dense arm is not retrained (the Qwen3-4B grid above was cancelled after 5 jobs):

- **ffnmoe** (stage 1 L12+ fine ladder t0.01 → stage 2 all layers t0.02) on the dense arms as-is
  (no markers needed). Evals through `beaker_native_lengthmix_eval.py` exactly as the dense runs
  (prompt-format chat, query after, per-task ladders), routing enabled from the checkpoint config.
- **kv17 / kv33** need the same JSONL arms re-tokenized WITH document markers (marker set
  qwen3_5) + gold sidecars: `build_kv35_shards.sbatch` (outlier/nq, mooney -> S3 -> weka) and
  `build_kv35_shards_beaker.sh` (contradiction/oolong, weka-native) -> `flop_scaling35/shards/<task>_s<B>_mk`.
  The hybrid's GDN blocks see the compacted sequence; attention blocks keep original positions.
  Mechanism validated locally first (fs35-smoke-*).
- Launcher: `debug/flop_scaling/launch_grid35.py`; dense points from `points.json` join the fits.
- FLOP accounting on the hybrid: `num_flops_per_token` covers GDN + attention + FFN; the FFN
  share is lower than on dense Qwen3-4B, so the FFN arm's model-wide FLOP saving is smaller.

## 10. Overnight log, 2026-09-02 03:30–06:10 (what broke, what it changed)

Four evaluator/recipe defects surfaced as soon as the first method evals landed. Every method
number reported before 06:00 was affected; the ledger (`debug/flop_scaling/LAUNCH_LEDGER.tsv`)
has the per-job trail.

1. **Marker-aware (KV) evals died at launch**: `run_beaker_multirung_eval.sh` lost its v2
   `EVAL500` default when v3 was added, so every v2 launch that did not export it hit
   `set -u`. Restored (3c8395692).
2. **The docchunk ladder evaluator hardcoded Qwen3 ids** (doc markers 151648/151649, eos 151643,
   `<|im_end|>` 151645). A Qwen3.5 checkpoint was therefore prompted with arbitrary vocab tokens
   around each document and never hit a stop id. Now resolved per family from
   `RESERVED_IDS` + the tokenizer (`--family auto`; abca39b21). Its `DC_RUNG_FILES` override
   also sat after the v2 `return`, i.e. never ran: every "override" eval scored the default
   v2 files. Moved in front of the return in the same commit.
3. **Stage-1 FFN recipe undershoots its budget ~4x on these 27–90-step runs.** One-sided hinge:
   the only routing pressure is downward, so nq 16M ended at mean cost 0.0026 (target 0.01)
   with 52% of routed-layer tokens on the null/width-1 rungs; 32k f1 0.14 vs 0.76 dense;
   contradiction 14M ended near f1 0 at every rung. Added `--ffn-moe-two-sided`
   (`|cost - target|`; bb35999da): the t10 probe lands at 0.0997 for target 0.10.
4. **Every FFN run started from a RANDOM router (root cause of CE 8.86 at step 1).** First
   blamed on exploration (399167447 turned it off; the no-explore run reproduced 8.860/7.623
   exactly). The real defect: `Transformer.init_weights` calls `reset_parameters` on every
   module, parent then child, so the router's inner `nn.Linear` re-ran torch's kaiming init
   right after the router's one-hot "full rung" init -- on every meta-built (FSDP/Beaker, and
   the local train_ctc_suite path) run the router was random and tokens landed on random rungs
   from step 1. The per-rung gain vector was `torch.empty` on meta and never reset at all.
   Fixed in 545c07b0d (router Linear owns the one-hot reset, gains reset, tolerant load
   re-inits missing keys and logs `gain=1.00..1.00 bias0=10.0`); a meta-built nested model now
   equals the base model exactly (CPU check, logit diff 0.0). Every s1/s2/t10/a10 number before
   06:45 came from a scrambled start and was discarded (`results/flop_scaling/evals35_garbage_router`);
   all FFN arms relaunched at 06:45. The Qwen3-4B v-series (records in the nested-FFN memory)
   ran through the same meta path and carried the same handicap, healed by their longer horizons.

KV: with correct ids, outlier kv17 16M scores 8k 0.13 / 16k 0.05 (dense 0.51 / –) while its
train CE reached 0.17 — the gold_plus_random keep set leaks the answer on id-answer tasks (gold
docs are always among the few real ones). Gold-blind arms `kvb33` (all budgets) and `kvb17`
(smallest budget) were added for outlier/nq/contradiction (800f43769); oolong was blind already.

Dense baseline gap: the outlier dense campaign scored only the 8k rung at 16M/32M. Their 16k/32k
rungs were launched by hand (`dense_extra_evals.tsv`); rows whose mean covers a rung subset carry
`partial=1` and are excluded from the fits.

## 11. Phase 2 (overnight 2026-09-02/03): speed at fixed cost levels, and the model-scale ladder

**(a) Speed.** `debug/flop_scaling/bench_ffn_speed.py` forces every layer's router to one rung
(cost 1, 1/4, 1/16, 1/64, 1/256, 8/H, 1/H, null) and times train / prefill / decode-shaped
forwards on Qwen3.5 0.8B–27B plus a 70B Qwen3.5-like geometry (2- vs 4-layer probes for sizes
that do not fit). Results: `results/flop_scaling/ffn_speed_a100.{json,md}` (A100-80GB).
Measured wall-clock speedups saturate by cost 1/16 and reach 1.1x (0.8B) → 1.4x (4B) → 1.9x
(9B) → 2.0x (27B) → 2.8x (70B geometry) at 8k, against FLOP theory of 1.5x → 2.2x → 2.9x → 3.7x
→ 5.8x: the non-FFN work (GatedDeltaNet, attention, head) runs at lower utilization than the
wide FFN GEMMs, so its wall-clock share exceeds its FLOP share. 8/H and 1/H buy nothing over
1/16 in wall-clock. H100 (training image, `results/flop_scaling/ffn_speed_h100.md`): model-level 1.05x / 1.14x / 1.22x / 1.41x / 1.61x / 2.2x (0.8B / 2B / 4B / 9B@32k / 27B / 70B-geometry, train) — smaller than on A100 because Hopper speeds the FFN GEMMs most; FFN-only at cost 1/16 delivers 1.8x (0.8B/8k) to 11.4x (70B-geometry/32k) of the theoretical 16x, with a per-token routing floor of ~1.3–2 ms per 32k-token layer.

**(b) Model scale.** `orchestrate_scale.py`, `collect_scale.py`, `results/flop_scaling/scale_report.md`.
Contradiction: the KV gold+1/3 multiplier vs dense rises 0.87 (2B) → 1.00 (4B) → **1.66 (9B)**;
routed FFN 0.76 → 0.81 → 0.91. Oolong: KV 1/6 saturates at f1 ≈ 0.66 at every size while dense
climbs, so its win at small scales (2.9x at 0.8B, 1.6x at 2B) becomes a tie at 4B and a loss at
9B. Small budgets lose everywhere. Bases: 0.8B/2B repaired from the modelonly conversions, 9B
converted from Hugging Face and repaired (`prep_bases_beaker.sh`). Traps hit: 0.8B needs
`--activation-checkpointing full` at 65k on 80GB; on 8-rank FSDP the tolerant loader's stats
read an empty local DTensor shard (fixed: `full_tensor()`); a watchdog double-start launched
11 duplicate runs (cancelled at job level; lock file added).
