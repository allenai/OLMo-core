# 3:1/shared-gain hero: 2T endpoint and downstream comparisons

Approved September 21, 2026. The failed 128-GPU trunk resumes its complete
step22000 state (369.098752B tokens), with unchanged architecture, optimizer,
batch16Mi, microbatch4 sequences, LR1.1e-3 and 2,000-step warmup.
The original failure was a distributed collective timeout; no numeric/OOM
failure was identified in the examined logs. No arbitrary node exclusion was added.

## Pretraining

- Warmup/stable schedule through step108000 = 1.811939328T tokens.
- Linear decay to zero over12,000 steps, ending step120000 = 2.013265920T.
- Save cadence remains100 steps through200B,250 through500B,500 afterward.
- A separate immutable hardlink copy preserves the **full pre-decay** step108000
  model, optimizer, trainer, loader and rank RNG state. Its receipt is
  `uploader/automation/olmo35-dolci-hero-20260920/continuation-source.json`.
  Ordinary uploader retention never visits that source directory.
- Future stable continuation must explicitly resume that pre-decay source with
  an extended stable schedule, **not** resume the decayed step120000 endpoint.
- Existing step6000 pin and the independent111.854B decay/MT/LC/SFT chain are untouched.

## New endpoint descendants

All posttraining has EMO disabled. MT and LC match the original full7:1 hero
recipes, not the shorter35.6B comparison stages. Fresh stage starts reset optimizer
and data position; in-stage restarts restore full state and pass the existing audit.

| Stage | GPUs | Tokens/batch | Length | LR | Schedule |
|---|---:|---:|---:|---:|---|
| MT |64|16,777,216|5,961 steps /100.009B|2.2e-4|cosine,2,000-step warmup|
| LC |64|16,777,216|5,961 steps /100.009B|1.1e-4|linear,2,000-step warmup|
| GPT-OSS high SFT |8|2,097,152|2 epochs /840 steps|5e-5|linear,3% warmup|
| Full Dolci-Think SFT |64|8,388,608|2 epochs /5,900 steps|5e-5|linear,3% warmup|

LC and SFT use65,536-token contexts with the established block recomputation;
MT uses8,192. SFT uses assistant masks, packed document boundaries, no weight
decay, and the previously qualified inference chat template and both stop IDs.
The GPT-OSS-high branch uses the recent2Mi comparison recipe; Dolci matches the
latest full-Dolci4T controls. These two datasets are separate runs, not concatenated.

Dataset identities:

- `jacobmorrison/length-investigation-gptoss-120b-high`
- `allenai/Dolci-Think-SFT`, existing audited tokenization/packing reused.

Each new stage has its own run ID, uploader registration and HF export directory.
SFT checkpoints: step0, qualification steps2/4, end of each epoch. Final epoch
automatically converts and evaluates on MATH-500,IFBench,HumanEval,AlpacaEval at
the existingT0.6 recipe. PT/MT/LC export to OlmoBase plus ordinary RULER, not RULER+.

## Existing 2T LC controls

After the128-GPU hero has resumed and logged training, automatically restore
and verify these two exact archived LC sources, then queue full Dolci-Think SFT
with the same64-GPU,8Mi,5e-5,two-epoch recipe and automatic conversion/evaluation:

- EMO PT → **no-EMO MT/LC**: `posttrain-noemo-20260914/lc100b/emo`, step5961.
- No-EMO throughout: `lc100b-after-mt20-decay2t/non-emo`, step5961.

Do not substitute the older EMO-on LC checkpoint or a same-numbered MT source.
The restore checks the bucket's verified receipt, exact inventory and lineage.
It does not modify or delete archival checkpoints.

Training: allocated urgent in `ai2/olmo3p5-training`. Evals: allocated urgent in
`ai2/OLMo-3-moe-experiments`. Controller: Phobos CPU-only with **no requested
resources**. Results datasets contain no model/data payloads. The10TB storage
stop remains enabled; no extra cleanup is authorized by this change.

Real-image CPU validation covers each new model/data/schedule configuration.
Production stage workers additionally do their existing save→resume qualification
before releasing the full run or any descendants. All submission receipts are
durable/idempotent; existing running/completed jobs retain their original specs.
