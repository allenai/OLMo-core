# September 19 matched hybridization/QK-gain experiment

Entrypoints: `olmoe3_qkgain_control.py`, `olmoe3_qkgain_plan.py`,
`olmoe3_qkgain_train.py`, `olmoe3_qkgain_node.py`, `olmoe3_qkgain_eval.py`.

Three lineages: new 3:1 EMO with split gains, new 3:1 EMO with shared gains,
and the original 7:1 EMO/split hero restored at step6000. QK normalization
is head-wise in both modes; only the trainable gain sharing changes.
No attention softmax, kernels, precision, expert/router configuration, or
unrelated running jobs are changed.

| Stage | Token batch | GPUs | Steps | LR/schedule | Checkpoints |
|---|---:|---:|---:|---|---|
| PT |16,777,216|64|0→6667; reference6000→6667|1.1e-3, warmup2000, stable to6000, linear667-step decay|step0 then every100, final6667; reference only6100…6600 +6667|
| MT |786,432|16|45,321|8e-5, linear with2000-step warmup|0; every5000;45321:11|
| LC |786,432|12|45,321|4e-5, linear with2000-step warmup|0; every5000;45321:11|
| SFT |524,288|8|3360 /2epochs|5e-5, linear with3%warmup, WD0|0,1680,3360:3|

PT ends at111.853699072B tokens. MT and LC each process35.641884672B;
the ladder formula requests35,641,421,562 tokens, rounded up to whole batches.
MT/LC deliberately use the old ladder's absolute LRs and batches, not fractions
of the new hero's LR. EMO is ON only during PT/decay and OFF for MT/LC/SFT.
LC uses64K sequences and block recomputation. SFT uses the existing high-reasoning
GPT-OSS120B tokenization with assistant masks and isolated packed documents,
eager FLA, and block recomputation.

Controller runs urgent/unallocated on resource-free Phobos. Training is urgent,
allocated6h, Holmes B300, in ai2/olmo3p5-training. Each stage first performs a
two-step save and two-step full-state resume on its real GPU layout. Native
stage completion gates the next stage; evaluation may overlap downstream training.
The controller validates source paths and final checkpoints, keeps durable
one-submit records, and never creates blind copies of failed/ambiguous jobs.

All scopes live under `/weka/olmo-3p5-checkpoints`:

- Native: `production-qkgain/olmo35-qkgain-20260919/<run-id>`.
- Controller: `uploader/automation/olmo35-qkgain-20260919`.
- Restored reference: controller `sources/emo/step6000/olmo-core`.
- HF exports/evals: `scratch/olmo35-qkgain-20260919/<run-id>/emo/stepN`.
  The inner `emo` is a frozen export-CLI slot; the outer run ID is authoritative.
- Private bucket: `allenai/olmo-3p5-small`, isolated
  `olmo35-qkgain-20260919/<arm>/<stage>` prefixes.

Synchronous full-state saves notify the existing uploader. Explicit registrations
enable verified cleanup with two newest local checkpoints protected and a1h grace.
The controller admits work only with the uploader running and12TB free; the
trainer's existing guard warns at10TB and checkpoints/stops below5TB. No code here
deletes unrelated storage or original reference checkpoints.

Each final PT/MT/LC export gets OLMoBase gen_mc/math/code and ordinary RULER4K–128K.
Each final SFT export gets MATH500, IFBench, HumanEval, AlpacaEval at T=.6/P=.95,
max32768, seed1234 with thinking template and both EOS IDs. Failed reasoning is
audited in saved responses. Export structure, finite tensors, tokenizer and file
hashes are checked. Numerical parity is explicitly NOT claimed: the user's
September16 waiver remains in force. Conversion preserves shared versus split
QK gains from the native source configuration.
