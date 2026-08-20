# Reference runs — the main experiments as node-local commands

Recovered 2026-08-20 from the pre-migration tree's launchers and ledgers
(`src/scripts/train/memexpress/{ctc_suite,sft_5task,sft_docchunk,singletask_ladder,cpt}`,
`records/ctc-model-scale-plan.md`, `debug/ctc_modelscale/LAUNCH_LEDGER.tsv` — all old-repo paths),
re-expressed against this tree's one recipe (`src/scripts/ctc/train/`). The public twin of this
file is `REPRODUCING.md` on `prasann/ctc_public`, which uses only public resources; this one keeps
the internal checkpoint/data paths and the caveats that came out of the recovery.

Two genuinely different "main comparison" families exist under the CTC umbrella. Do not merge
their numbers.

## 1. The 22-task suite protocol (Qwen3.5 hybrid, one model per task × arm)

The canonical grid: per task, arms `full` / `chunked-mix` (ships as "chunked" in results tables) /
optionally pure `chunked`. Old trainer: `ctc_suite/train_ctc_suite.py` via `run_ctc_local.sbatch`.

- **Hyperparameters:** lr 5e-5, 3 epochs, seq-len 40960, global batch 8 instances, warmup 0.03,
  betas (0.9, 0.95), wd 0.0, grad clip 1.0, fused-linear CE, no YaRN (native RoPE), AC full at
  2b/4b/9b and none at 0.8b. In this tree: `--max-steps = 3 × instances ÷ 8`.
- **Mask-mix curriculum** (`--arch chunked-mix`): p 0.80 → 0.0 linear over the run's per-rank
  forwards; ported 2026-08-20 with the anneal hard-check (commit `8ba448122`).
- **Bases (marker audit PASS at every scale — Qwen3.5 needs NO fixmark):**
  - 0.8b: `/scratch/users/prasann/cpt_mix_ckpts/q35-08b-base-modelonly/model_and_optim`
  - 2b/4b/9b: cubbins `/data/prasann/ctc_suite/bases/q35-{2b,4b,9b}-base-modelonly`
- **Data:** `query_position=after` for all new builds (standing directive). Shards on the nodes
  under `/data/prasann/ctc_suite/data`; rebuildable from the seed pools
  (`ctc-data build --pool auto` → `convert_to_shards.py --query-position after`).

```bash
CTC_NPROC=8 run/train.sh ctc-contradiction-cmix \
    --data /data/$USER/ctc/shards/contradiction:1 \
    --base /data/prasann/ctc_suite/bases/q35-4b-base-modelonly \
    --arch chunked-mix --model qwen3_5_4B --tokenizer qwen3_5 --lr 5e-5 --max-steps 7500
```

## 2. Model-scale sweep (0.8b / 2b / 4b × full / chunked-mix; 4 tasks)

contradiction, reorder, hotpotqa, qdmatch_nq — O(N²)/O(N²)/O(N)/O(N²), hotpotqa the low-CTC
anchor. Same recipe as §1 but **1 epoch** (`--max-steps 2500` at 20k instances), pinned
`global_batch=8` (the 2-node Beaker path silently doubles it — every ledger row pinned 1 node).
Seq-len per task from the shard's max example: contradiction/reorder 40960, hotpotqa 26112,
qdmatch_nq 33792. `qwen3_5_2B` factory ported 2026-08-20 (`54030d5f3`). Ledger of what ran where
(lambda lanes + jsteinhardt + Beaker backfill): old repo `debug/ctc_modelscale/LAUNCH_LEDGER.tsv`.

**Evaluator lock:** the 4B reference table was vLLM-scored, and native-vs-vLLM drift is ~0.08 f1
on contradiction/full@2k (≈2.8× SE). Score every sweep cell with vLLM, never mixed.

## 3. The 5-task mixed-SFT family (Qwen3-4B plain, packed/docchunk 32k)

Mix contradiction:2 nq:1 oolong:1 rerank:1.5 outlier:1.5 (dense-packed arm upsampled
contra→2.9/oolong→1.3 to offset `LongDocStrategy.exclude` drops). lr 1e-5; 1465 steps @ 2 nodes
(dense packed 32768) / 1100 steps @ 4 nodes (docchunk family, seq 40960); YaRN factor 2 past
native. Landmark arm: `--arch landmark --mem-freq 63`, base `q4b-fast-landmark-dolma3longmino`
step2385; compressive arm's base was never staged locally (weka only —
`q4b-base-fast-compressive-landmark-8node`).

- **Bases (weka):** `.../amandab/q4b-{dense,fast-landmark,base-fast-compressive-landmark-8node}-dolma3longmino/step2385/model_and_optim`;
  local copies of dense + fast-landmark in `/scratch/users/prasann/stable_bases/`.
- ⚠ **Open risk from the recovery: no `-fixmark` variant of any Qwen3-4B base appears anywhere in
  the old tree.** Plain Qwen3's box-marker rows are bit-identical out of the box, so every
  Qwen3-4B docchunk/landmark number rests on unaudited marker embeddings. Audit
  (`fix_marker_embeddings.py --check-only`) before reusing those bases, and repair before any new
  run.
- NQ is ALWAYS the p10 hard-neg rebuild (`single_task_ladders_p10/nq`); the 98%-hard original is
  retired.

## 4. Known holes the recovery surfaced

- `docchunk_local40960` (the local twin's data root) does not exist on /scratch and its generator
  (`docchunk_run/convert_local.sh`) is not in any tree — regenerate via the new converter before
  a local 5-task docchunk re-run.
- CPT long-text (`build_cpt_longtext_olmo3.py`) has a data recipe but **no attested training
  invocation anywhere** — any "we CPT'd on it" claim is currently unsupported.
- Multi-landmark (`num_landmarks > 1`) has model code but no launcher in the old memexpress tree;
  the sweeps live on amandab's branches.
