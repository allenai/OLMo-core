# Unified instruction-tuning (IT) suite — setup & handoff

Everything for the corpus-reasoning unified instruction-tuning pipeline. **All code is on GitHub;
all data is on weka or the public GCS bucket** (nothing required is local-only). Written for picking
up development on a fresh machine.

## TL;DR status (2026-06-20)
The full pipeline is **built and validated end-to-end on a 1k debug set**: data → combine → AI2
tokenize → landmark packing → train → checkpoint → native-gen eval with scores. Landmark
**packing + context-parallelism works at 64k** (no memory blocker). Next step is the **full
per-task-balanced build + scaled dense/landmark/sparse runs** (+ optional CPT mix).

## Repos / branches (clone these)
- **OLMo-core** — `github.com/allenai/OLMo-core`, branch **`prasann/landmark`**. Training + data
  tooling + vendored prompts. `pip install -e '.[all]'`.
- **oe-eval-internal** — `github.com/allenai/oe-eval-internal`, branch **`prasann/longctx-eval`**.
  The `cr_*` suite eval tasks (`corpus_reasoning_suite.py`).
- corpus-reasoning (data generators) — `github.com/PrasannS/corpus-reasoning`, branch `main`. Only
  needed to *generate* new suite data; the generated data is already published (below).

## Data — all on weka or the public bucket
**Public GCS bucket (source, read-only, no auth):** `gs://corpus-reasoning-olmo-data-prasann/suite`
(369 files / 22.7 GB: per-task train+eval unified JSONL + `suite_manifest.tsv` mapping
file→task→cot_mode→split). Re-fetch with `src/scripts/data/download_suite_data.sh` (parallel curl).

**On weka (`/weka/oe-training-default/ai2-llm/checkpoints/...`):**
| Path | What |
|---|---|
| `prasanns/cr_suite_data/` | the full suite (train+eval JSONL, 369 files) + `suite_manifest.tsv` + the built `suite_combined_debug.jsonl`. Landed via `land_suite_eval_to_weka_gantry.sh`. **This is the eval-data dir the `cr_*` tasks read by default.** |
| `prasanns/suite_it_sft_qwen/combined_debug/` | 1k debug **AI2-format SFT shards** (`token_ids_part_*.npy` + `labels_mask_*.npy` + `metadata.json`), Qwen3-0.6B-tokenized, 967 instances. |
| `prasanns/q4b-fl-unified-debug/step8` | fast-landmark smoke checkpoint (FSDP, 8k). |
| `prasanns/q4b-fl-unified-debug-cp/step6` | fast-landmark smoke checkpoint (CP=8, **64k** — the CP validation). |
| `q4b-{fast,sparse}-landmark-dolma3longmino/step2385` | CPT base checkpoints (~10B tokens, long-ctx). Init for landmark SFT. |
| `q4b-dense-dolma3longmino/step2385` | dense CPT base (init for dense SFT). |
| `amandab/Qwen3-4B-Base-olmocore/model_and_optim` | raw Qwen3-4B base (no-CPT init option). |

To land the suite onto weka on a new setup: `src/scripts/data/land_suite_eval_to_weka_gantry.sh`
(CPU gantry job; `FULL=1` for train+eval, default eval-only). Writes to `prasanns/cr_suite_data`.

## Pipeline (scripts, in order) — all in OLMo-core
1. **Land data:** `src/scripts/data/land_suite_eval_to_weka_gantry.sh` (bucket → weka cr_suite_data).
2. **Combine:** `src/scripts/data/build_combined_suite_jsonl.py` — per-task train files → one tagged
   (`_task`/`_cot_mode`) JSONL. Equal-rows-per-task (`--per-task-budget N` / `--total-budget`),
   held-out denylist `{redundancy, beir_scifact, beir_fiqa}` (asserted), rerank HELMET→standard
   schema fix, `--cot plain` (default for now). Writes `suite_combined[_debug].jsonl` + a manifest.
3. **AI2 tokenize:** `src/scripts/data/convert_unified_to_sft.py` (+ `_gantry.sh`, **needs
   `--system-python`**) — per-row `build_prompt` (vendored, below) → Qwen3 chat template →
   char-offset loss mask → EOS-separated `token_ids_part_*.npy` + `labels_mask_*.npy` shards.
4. **SFT:** `src/scripts/train/sft/Qwen3-4B-{dense,fast-landmark,sparse-landmark}-unified-SFT.py`.
   Landmark via `LandmarkPackingInstanceSource` (CP=8, `shard_degree=1`); dense via ConcatAndChunk +
   flash + CP. `DATA_ROOT` → the combined shard dir; `-debug` run names read `combined_debug`.
   `UNIFIED_SEQ_LEN` env tunes seq len. Launch: `python <script> launch <run-name> <cluster>`.
5. **Eval:** `launch_cr_suite_evals.sh <weka_checkpoint>` (repo root) → oe-eval `cr_*` tasks
   (`prasann/longctx-eval`), native olmo_core generation, reads `cr_suite_data` by default. 185
   eval tasks auto-registered from a bundled manifest; modes `direct`/`cot`/`cr`.

**Vendored prompt builder** (shared by train + eval so prompts are byte-identical):
`src/olmo_core/data/corpus_reasoning_prompts/` (OLMo-core) and
`oe_eval/tasks/oe_eval_tasks/corpus_reasoning/` (oe-eval) — both copied verbatim from
corpus-reasoning `main@2b20f8d`.

## Validated (1k debug)
combine (1000 ex, balanced, rerank reconciled, 0 held-out leak) → tokenize (967 instances, all 18
tasks, 0 build errors) → fast-landmark train (8 steps, loss↓, checkpoint) → cr_ eval (oolong 0.45,
contradiction 0.003; native gen + scoring works). Plus **CP @64k**: trained 6 steps cleanly. RULER
regression (pre-CP vs CP code) on the fast-landmark CPT ckpt: **identical** (niah_single 1.0@8k,
0.70@32k).

## Landmark long-context (CP + decode)
- **Packing + CP**: implemented (commit `d9b1289d`) and 64k-validated. Landmark SFT trains at full
  64k via Ulysses CP (shards activations). Enabled in the unified + packed fast/sparse scripts.
- **Eval top-k**: landmark decode now defaults to **hard top-k retrieval, top 10%**
  (`GenerationConfig.landmark_top_k_fraction=0.1`).
- **GPU-agent briefs** (repo root, hand to a CUDA box): `landmark-packing-cp-task.md` (✅ done),
  `landmark-sparse-decode-task.md` (TODO — make top-k decode O(k·block) not O(context); the real
  long-ctx eval speedup).

## Env / gotchas (save time on the new machine)
- gantry + olmo-cookbook-eval live in the **`olmoenv` mamba env** (`.../envs/olmoenv/bin`), not default PATH.
- **wandb entity** for these scripts is `prasanns-allen-institute-for-ai` (NOT `ai2-llm`).
- Qwen3 has `bos==eos` → `NumpyDocumentSource` needs `replace(tokenizer, bos_token_id=None)`.
- Internal `launch` tails the job by default; add `--launch.follow=false --launch.step_soft_timeout=null`.
- Write only under `checkpoints/prasanns/` on weka; never touch others' data.

## Open items / next steps
1. **Full build + scaled runs:** `build_combined_suite_jsonl.py --per-task-budget N` → convert
   gantry → full `combined/` shards → launch dense + fast-landmark + sparse SFT (landmark @64k via CP).
2. **CPT mix (v2):** `MixingDocumentSource`, `CPT_FRAC=0.15`, dolma3longmino source; after the
   no-CPT mix is validated. See `src/scripts/train/sft/unified-suite-SFT-plan.md` §9.
3. **Eval coverage:** full held-out buckets (beir_fiqa/scifact, redundancy, RULER-64k).
4. **Decisions still open:** per-task budget; training context (64k now feasible for landmark via CP);
   CoT (plain-only for now). See `src/scripts/data/build-unified-dataset-plan.md` §7.

## Plan docs (more detail)
- `src/scripts/train/sft/unified-suite-SFT-plan.md` — overall SFT-on-Beaker plan.
- `src/scripts/data/build-unified-dataset-plan.md` — dataset build (roster, held-out, CoT, budgets, CPT §9).
