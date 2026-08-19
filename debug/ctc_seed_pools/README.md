# ctc_seed_pools — exporting the published seed pools

Provenance and drivers for the seed-pool artifacts served by `ctc-data build --pool auto`
(HF dataset repo `PrasannSinghal/ctc-seed-pools`; format and codecs in `ctc/src/ctc/data/seeds.py`,
user docs in `ctc/src/ctc/data/README.md` §Seed pools).

## What ran where (2026-08-19)

| piece | what it does | where it ran |
|---|---|---|
| `harvest_contradiction_pairs.py` | Recovers a `pairs_path` file (`contradiction_pairs.jsonl`, gitignored) from the audited 20k train build on cubbins `/data/prasann/ctc_suite_data/contradiction_pool/` + a full-PubMedQA sentence index — the same recovery `debug/ctc_strmatch_redundancy_port/harvest_redundancy_pairs.py` established. The LLM mining run is NOT repeated; these are the pairs the shipped numbers were measured on. | login node |
| `export_seeds.sbatch gpu` | `pool export` for nq / hotpotqa / fiqa / scifact / rerank (pyserini indexes + GPU cross-encoder), then qdmatch_nq / qdmatch_hpqa projected from the fresh nq/hotpotqa seeds. | mooney (indexes + HF cache are mooney-local) |
| `export_seeds.sbatch mooney_cpu` | outlier (6.2 GB article-pool pickle, mooney-local), oolong, contra_fever, outlier_review. | mooney |
| `export_seeds.sbatch horton_cpu` | absence, reorder (the ~11 GB `sedthh/gutenberg_english` cache is horton-local at `/data/prasann/hf`). | horton |
| login-node exports | grouping_labeled (compact OpenAlex JSONLs: `/scratch/.../ctc_suite_data_shared/openalex_compact.jsonl` + the 2024 eval fetch from the pre-migration tree), xabsence, and — once the harvest finishes — contradiction + redundancy (pairs + a PubMedQA filler sample). | login node |

Seeds land in `seeds/` here (login-node builds) and `/data/prasann/ctc_seed_pools` on
mooney/horton (collected over `/net` for upload). Everything heavier than the scripts is
gitignored; the artifacts' home is the Hub repo, not git.

## Caveats carried into the artifacts

- **xabsence's pool is the 659-pair `xabsence_pool_pubmed.jsonl`** — the undersized pool
  BUILD_MATRIX flagged as blocker B3. It seeds small builds fine but cannot supply the large
  rungs; a 50k-pair mining run (needs an LLM endpoint) replaces it when someone pays for one.
- **contradiction/redundancy pairs are harvested, not re-mined.** Pairs whose real half could not
  be located in PubMedQA (both-real / neither-real) are dropped rather than guessed; the harvest
  logs print the loss.
