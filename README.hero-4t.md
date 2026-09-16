# Small heroes: 4.027T continuation campaign

Two independent 64-B300 decay branches restore verified HF optimizer/model/data/RNG
checkpoints at step216000 (3,623,878,656,000 tokens). The live pretraining jobs
remain unchanged. Decays end at step240000 (4,026,531,840,000 tokens), using the
original 16,777,216-token batch, LR1.1e-3 linearly decayed to zero over24000steps.
EMO stays on during the EMO decay only.

| Stage, per lineage | Steps | LR | Schedule / warmup | GPUs | EMO |
|---|---:|---:|---|---:|---|
| MT,100.009Btokens |5961|2.2e-4|Cosine /2000|64|Off|
| LC,100.009Btokens |5961|1.1e-4|Linear /2000|64|Off|
| SFT,2epochs |1810|1e-5,5e-5,1e-4|Linear /3%|8perLR|Off|

MT uses MB4x8192 without recomputation; LC uses MB1x65536 with compatible block
recomputation. Both retain the16Mi batch. SFT uses the existing packed
gptoss120b-deduped OLMo-Think data, assistant masks, 524288-token batch, MB1x65536,
WD0, eager execution, block recomputation and packed FLA0.5.2 kernels.
All posttraining stages reset optimizer/data while preserving transferred weights.

## Automation

- Native controller: src/examples/olmo_ddp/olmoe3_hero_4t_pipeline.py.
  Runs CPU-only on Phobos with no resource requests. Checks all configurations in
  the real pinned runtime on an independent1GPU worker. Each lineage progresses
  independently. MT has a2-step save/reload gate; LC and SFT retain4-step gates.
- Eval controller: src/examples/olmo_ddp/olmoe3_hero_4t_evals.py.
  Separate CPU-only Phobos job; stage subprocesses isolate legacy adapter globals.
  Finished decay/MT/LC exports launch the frozen OLMoBase gen/MCQA, math and code
  recipes. Decay and LC also launch ordinary RULER, never RULER+.
  Both SFT epochs convert; only the six final epochs launch Math500, IFBench,
  HumanEval and AlpacaEval. Eval jobs are allocated urgent,6h minimum runtime.
- The user's inference numerical-parity waiver is explicit in export receipts:
  numerical_parity_skipped_by_user_20260916; numerically_qualified=false.
  No fake passing parity receipts are generated. Source-checkpoint hashes,
  serialized tensor keys/shapes/finiteness, tokenizer/vocabulary/per-head-QK
  metadata and downstream output hashes remain checked. Fast inference remains
  the previously evaluated BF16 grouped-MoE/FLA profile.
- The shared uploader watches separate stage/run namespaces in the existing
  private allenai/olmo-3p5-small bucket. Apply-mode retention keeps two local
  checkpoints with1h grace. Parent/hero registrations are not changed.
  The controllers pause new submissions below12TB free or if the uploader stops.
  CPU controllers never copy checkpoints;1GPU prep/export workers perform I/O.
- Durable submission intents prevent duplicate launches. Failures are logged
  without automatic retry storms. Logs and status receipts expose blocked stages.
- No data is written into Beaker result datasets. Gantry's required /noop-results
  remains empty. No checkpoint cleanup is performed by these launchers/converters.

After both decays have demonstrably started training, the separately scoped
stable-checkpoint evaluation campaign will evaluate hero3T,3.624T and4.027T
endpoints and nearest available OLMo3-7B/hybrid7B revisions. Existing complete
3T evaluations must be reused, not relaunched.
