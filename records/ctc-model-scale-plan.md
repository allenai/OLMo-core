# Model scale × CTC class, and length generalization — plan + live state

**Started 2026-08-12.** Two questions, one set of runs.

1. **Model scale** — extend the paper's `fig:modelscale` (`figures/section4_scale.pdf`, §4 of the
   V2 draft: *"Increasing model scale within the Qwen-3.5 family (solid 4B, dashed 2B, dotted
   0.8B) pushes back the point at which the chunked variant degrades, but small full attention
   models can beat larger chunked attn"*) from **contradiction only** to **four tasks spanning the
   CTC classes**, at 0.8B / 2B / 4B × {dense, chunked}.
2. **Length generalization** — do high-CTC and low-CTC tasks degrade differently when a dense model
   is pushed past the context length it was trained on?

Task set (user, 2026-08-12): **contradiction, reorder, hotpotqa, qdmatch_nq**.
Complexity classes from `records/ctc-suite-scaling-plan.md` §3: contradiction O(N²), reorder O(N²),
qdmatch O(N²), hotpotqa O(N) — hotpotqa is the low-CTC anchor.

Reference numbers to compare against: `results/ctc_suite/dense_vs_chunked_table.md` (Qwen3.5-4B,
eval_size=500/rung, vLLM).

## 1. What was already available (nothing was rebuilt)

Per the user's "reuse train data" requirement, **no shard was regenerated**. The existing 20k
per-task shards from the July fan-out are reused verbatim:

| task | shard | instances | max_example_len | seq_len used |
|---|---|---|---|---|
| contradiction | `contradiction_train` | 19,366 | 40,957 | 40960 |
| reorder | `reorder_train` | 19,944 | 40,957 | 40960 |
| hotpotqa | `hotpotqa_train` | 20,000 | 25,742 | 26112 |
| qdmatch_nq | `qdmatch_nq_train` | 20,000 | 33,398 | 33792 |

`seq_len` and `epochs=1` are copied from the 4B fan-out so the new scales are drop-in comparable.

## 2. The 2B rung had to be built (it did not exist in olmo-core)

The paper's figure uses 0.8B/2B/4B, but olmo-core only had `qwen3_5_{0_8B,4B,9B,27B}` factories.
Added this session:

- `TransformerConfig.qwen3_5_2B` — d_model 2048, 24 layers, 8 heads / 2 kv, head_dim 256,
  intermediate 6144, linear 16/16 key/value heads. **2.390B params (1.882B non-embedding).**
- `(2048, 24) -> qwen3_5_2B` in `corpus_reasoning/lib/olmo_models.py`'s hybrid resolver.
- `"2b"` in `train_ctc_suite.py`'s `MODEL_FACTORIES`, and added to `_LARGE_SCALES` so it defaults
  to full activation checkpointing + FSDP param sharding (it runs on 80GB A100s, where even 0.8B
  needs both at 40960).

Base conversion: `debug/ctc_modelscale/fetch_convert_2b.sbatch` (download + convert in one job;
compute nodes do have internet, verified). Result:
`cubbins:/data/prasann/ctc_suite/bases/q35-2b-base-modelonly`.

- **Converter reported `strict load OK (399 tensors)`** — that is the real validation that the
  factory shape matches the released checkpoint, not the param count.
- **Stage-0 marker audit: PASS.** cos(248049, 248050) = **0.233**, norms 0.435 / 0.602 against a
  trained-row median of 0.675 (ratios 0.64× / 0.89×). Markers are trained and distinct, so **no
  `fix_marker_embeddings.py` repair is needed** — same verdict as 0.8B/4B/9B.

## 3. Training matrix

16 small-scale runs — 4 tasks × {full, chunked-mix} × {0.8b, 2b} — on the **lambda** A100 cluster
(`debug/ctc_modelscale/launch_lambda_wave.sh`), plus 3 backfills on Berkeley.

Fair share is enforced structurally, not by hand: three lanes (`ctcms-laneA/B/C`) at
`preemptive_high` / `preemptive` / `normal`, each serialized with `--dependency=singleton`, so the
*preempting* footprint never exceeds one 8-GPU node per preempting QOS — the two-node cap in
`lambda_cluster.md`. Checkpoints go to node-local `/tmp` (lambda's `/data` is root-owned and the
NFS quota is 1.30T); the harvest must node-pin to the HOST in each job's log banner.

### 4B backfill — why three runs are missing from the reference table

- **qdmatch_nq (both arms)** — launched in July, never landed in the table.
- **hotpotqa dense** — its `-full` checkpoint was corrupt on S3 (missing distcp shard), which is
  why the table's hotpotqa row is marked CHUNKED-ONLY. Without this there is no 4B dense point for
  the low-CTC anchor.

Run on jsteinhardt `mooney` (H200, has the audited q35-4b base node-local), serialized with
`--dependency=singleton` because the jsteinhardt per-user cap is 8 GPUs = one node.

## 4. Traps hit this session (all real, all cost time)

- **`contradiction_train` was never staged to lambda.** The launcher's `metadata.json` guard caught
  it and both contradiction jobs exited rc=3 in 2 seconds rather than training on nothing.
- **mooney's node-local `qdmatch_nq_train` is a 2,500-example PILOT STUB** (max_example_len 2,664
  against the real shard's 33,398). `run_ctc_local.sbatch` stages with `cp -n`, which does **not**
  overwrite, so a same-named copy would have silently trained the 4B backfill on 2.5k examples at
  1/13th the length. Worked around with a distinctly-named hardlink farm
  (`qdmatch_nq_train_20k`) — **check `num_instances` on any node-local shard before reusing it.**
- **The qdmatch_nq 20k shard has been TRUNCATED since the day it was built — everywhere.** It
  carries all five `labels_mask_part_*.npy` (245,984,267 bytes of bool = exactly the metadata's
  `num_tokens`) but only three of five `token_ids_part_*.npy` (720MB of uint32 against the 984MB
  that `num_tokens` implies) — the 2026-07-19 conversion died partway through writing token parts.
  The loader refuses it with `'label_mask_paths' should have the same length as 'source_paths'`, so
  **every qdmatch_nq training run since has died at startup**, which is almost certainly the real
  reason qdmatch_nq never appears in `dense_vs_chunked_table.md` (it was previously filed under
  "genuine underfitting"). The `/scratch`, lambda, and S3 copies are byte-identical, so this is not
  a transfer fault. The source `qdmatch_nq_train_20k_mixedn.jsonl` is intact, so the fix is a
  re-tokenization of the same input, not a data rebuild:
  `debug/ctc_modelscale/repair_qdmatch_nq_shard.sbatch`.
- **Rung labels are not token counts.** Measured through the real tokenizer
  (`debug/ctc_modelscale/measure_rung_tokens.py`), document tokens per rung come out at
  **1.5× below label for contradiction and 2.9× below for niah**; nq and hotpotqa are accurate to
  within ~1.2×. The published 4B x-axis is therefore per-task-scaled, not absolute tokens.
  Trends within a task are unaffected; cross-task token comparisons are not meaningful.
- **`contradiction/rung_131072.jsonl` is unusable for a length curve.** Its gold pairs are a proper
  nested superset of the 32k rung (verified), but its distractors average **47.1 tok/doc against
  15.6 for every rung 2k→32k** — the FEVER/wiki filler-glob leak documented in
  `build_xlong_rungs.py` and `contra-fever-filler-leak`. It is a different corpus, not a longer
  context.

## 5. Length generalization

Built fresh with `debug/ctc_modelscale/expand_ctc_rung.py`, which grows a task's **own** rung file:
keeps every existing document (so each long rung is a strict nested superset of the source — same
gold, same original hard negatives, length is the only added variable), appends distractors drawn
only from documents that are gold in *no* example, reshuffles, and remaps every index field.
Targets are **measured** token budgets (65,536 / 131,072), both under Qwen3.5's 262,144 native
position ceiling, so these are native-RoPE measurements and need **no YaRN serving copy**
(`run-evals` rule 2 kicks in only at 256k).

Evaluated on the **4B dense** checkpoints, which already have validated vLLM serving copies on
cubbins/mooney/sneetches — no retraining, no re-export. High-CTC (contradiction) vs low-CTC
(niah, hotpotqa, nq).

**The confound to control for:** these ladders scale *N* (document count) with the token budget, so
a longer rung is also a bigger corpus. For a high-CTC task, bigger N is intrinsically harder — that
is the CTC thesis itself. So a steeper drop at 64k/128k does **not** by itself demonstrate a length
effect. The analysis is therefore the *excess* drop beyond what each task's own in-training-range
N-scaling trend (2k→32k) predicts by extrapolation; that residual is the length-generalization
term.

## 6. Evaluator comparability — must stay on vLLM

The 4B reference table was produced with **vLLM**. `results/ctc_suite/vllm_eval_status.md` records
a measured native-vs-vLLM drift of **0.079 f1** on contradiction/full at 2k (~2.8× the combined
binomial SE, vLLM consistently higher, attributed to the Gated-DeltaNet recurrent state diverging
between olmo-core's kernel and vLLM's). That drift is **larger than several of the scale effects
this study is trying to resolve**, so the new 0.8B/2B checkpoints must be scored through the same
vLLM path, not the native evaluator.

The serving-copy recipe is scale-agnostic (`make_vllm_serving_copy.py` takes the matching VL base
snapshot), so 0.8B and 2B need their own `Qwen3.5-{0.8B,2B}-Base` HF snapshots as
`--base-snapshot`. 2B is already on cubbins.

## 7. Added workstreams (user, 2026-08-12)

### 7a. OOLONG — fixed chunked-vs-dense

The suite's oolong row is `CHUNKED-ONLY` *and* its chunked number came from a leaky shard, so both
arms are new. The `--item-regex` fix is already in the repo (converter rejects an empty-matching
regex, records `item_regex`/`query_position`, regression test, corrected conversion script) —
only the 2026-07-19 artifact on disk was stale. Verified by a controlled A/B on the same source
with only the bad flag removed (`scan_oolong_chunks.py`):

| shard | chunks/ex | FREE gaps/ex | affected | verdict |
|---|---|---|---|---|
| 2026-07-19 `oolong_train` | 350.9 | **5.000** | 400/400 | **LEAK** |
| re-tokenized, no `--item-regex` | 345.9 | **0.000** | 0/400 | **CLEAN** |

The known-clean `xlong5_qafter/shards_chunked/oolong_train` is **not** a drop-in: `max_example_len`
249,226 (needs ~250k seq + CP) and `query_position=after` — a different ladder. So a new
CTC-structure dataset is built (`gen_oolong_ctc_bands.sbatch` → `finalize_oolong_ctc.sbatch`):
**same generator / offline item substrate / band edges / `--pool-max-ctx` / tokenizer as the 5-task
mix**, but 5 bands 2k–32k only (the CTC eval-rung coverage) with counts **evenly split**, against
the 5-task build's 7 bands to 256k with decreasing counts. `query_position=after` for both arms
(matches the 5-task setup and the standing new-run directive) → eval must render `after`, and the
numbers are NOT comparable to the July `both`-mode row.

Useful side-finding: **oolong's rung labels ARE honest token counts** (measured label/actual
0.95–1.00), unlike contradiction (~1.5×) and niah (~2.9×).

### 7b. Outlier fixed-M — the T3 control

Planned in §6 (T3: *"fixed-M outlier gap stays flat while scale-k widens"*), never built. The
existing row is scale-K: `--majority-mode articles` fills with whole articles so K *emerges* and
grows with N (measured on the eval ladder: n=22→K=3, 55→7, 110→13, 220→25). The control pins
**K=3** — the scale-K ladder's value at its smallest rung — so both curves start together and
differ only in whether M scales, matching the paper's dashed-vs-solid Fig 4 ablation. Eval rungs
reuse the scale-K doc counts (n = 14/28/57/111/220) and rung labels, so N is matched rung-for-rung.

Pinned three ways (`--mixed-min-k = --mixed-max-k = --mixed-max-k-cap = 3`, plus
`--simple-ratio 0`, since simple mode is a single majority article = K=2). Pilot (400 ex) confirmed
the pin holds and the pool can supply it — K exactly 3 in every n band while docs/topic grows
6.5 → 95.0.

### 7c. Cluster disruption, 2026-08-12

**cubbins and mooney both entered `drng` ("Prolog error")** mid-session. Consequences and fixes:
the 4B qdmatch runs were re-targeted mooney → sneetches; the outlier article pool (mooney-only)
was copied to sneetches; the oolong generation lost one of 20 array tasks to `JobHeldAdmin` (admin
hold, cannot be released), so the combine step now **enforces the even split by subsampling every
band to the minimum band count** rather than trusting all shards to exist; generated oolong shards
were rescued from mooney's local disk to `/scratch`. Length-gen evals were pinned to cubbins for
the HF exports and had to be re-homed — note cubbins' and horton's contradiction exports are
**different checkpoints** (different byte sizes) and only cubbins' backs the published table.

## 7d. Train-step audit (steps must be the comparison's constant, not a free variable)

Every run is `epochs=1`, `global_batch=8`, so **steps = instances / 8**. Two places this could have
drifted, both checked rather than assumed:

- **`beaker_ctc_suite.py` defaults to `--num-nodes 2`**, and it derives
  `global_batch = opts.global_batch or world_size` — so a 2-node launch silently doubles the batch
  and *halves* the step count. Every launch here passes `--num-nodes 1`; the dry run confirms
  `nodes=1 world_size=8 global_batch=8`.
- **The published 4B rows had to match.** `ctc-4b-hotpotqa-full` (July) ends at **step 2500**,
  identical to the new `ctcms-hotpotqa-full-4b`, and `ctc-4b-oolong-full` at 2625 = 21,000/8 —
  confirming the published runs were `global_batch=8` too.

| shard | instances | steps |
|---|---|---|
| contradiction_train | 19,366 | 2,420 |
| reorder_train | 19,944 | 2,493 |
| hotpotqa_train | 20,000 | 2,500 |
| qdmatch_nq_train_fixed | 20,000 | 2,500 |
| outlier_train (scale-K, published) | 19,986 | 2,498 |
| outlier_fixedM_train | 20,000 | 2,500 |

**Within a task, steps are identical across scale and arm** — which is the constant the model-scale
and dense-vs-chunked comparisons actually require. Across tasks they differ by <4%, which is fine
because tasks are never compared to each other directly. The fixed-M outlier control lands within
2 steps (0.08%) of the scale-K row it is the control for.

**The one outlier was oolong.** Its first build came out at 14,463 instances = **1,807 steps, 28%
short**, because the even-split combine floors every band at the smallest band and band 2 was
missing a shard (admin-held during the cubbins/mooney drain). Regenerating that shard restores
4,000/band → 20,000 → ~2,410 steps after the decontamination drop, in line with contradiction's
2,420. Worth noting the decontamination is not cosmetic: it removed 537 examples (3.58%) that
collided with the CTC oolong eval rungs.

## 8. Live state

Launch ledger: `debug/ctc_modelscale/LAUNCH_LEDGER.tsv` (one row per submitted job).
