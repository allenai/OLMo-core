# 4T LC SFT LR sweep

Approved September 18. Both 4.027T decay -> EMO-disabled MT -> EMO-disabled LC
parents, step5961. EMO remains disabled in SFT, including the EMO-PT ancestry.
No 2T checkpoints and no five-epoch trials.

Peak LRs: 5e-6, 1e-5, 2e-5, 3.5e-5, 5e-5, 7e-5, 1e-4, 2e-4 (40x span).
Reuse both completed 5e-5 runs and their existing eval identities: 14 new trainings.
The baseline runs() function still selects only the two original runs; the new
controller explicitly uses new_lr_runs(). Existing temperature automation is unchanged.

Each trial uses the pinned GPT-OSS-120B-high dataset/tokenizer/masks/packing,
8 B300 GPUs, 524288 tokens/batch, 65536-token sequences, BF16, packed FLA,
block recomputation, WD0, 3% warmup, linear decay, 3360 steps/two epochs.
Budget roughly 1.5-2 hours per trial once allocated. Save/upload both epochs;
convert/evaluate final epoch only. Math500, IFBench, HumanEval, AlpacaEval use
canonical T=.6, top_p=.95, max_new_tokens32768, seed1234, one sample. Do not
multiply this LR sweep by the six-temperature sweep.

Source-load and full-state-restart GPU gates from e18e332f are reused, with their
audits intact even though the backed-up smoke checkpoint payloads were cleaned.
A real-runtime CPU config gate checks all14 models against saved 5e-5 configs:
model/dataset identical; train module identical after normalizing only scalar LR.
First-batch hashes and fresh optimizer/data reset assertions remain active.
This does not claim the high LRs are stable; training logs check finite losses/norms.

New Phobos CPU gate and controller request no resources. Training is urgent,
allocated1h minimum in ai2/olmo3p5-training; evals urgent allocated8h in the MoE
workspace. Durable exact-name receipts prevent duplicate launches; failures are
reported, not relaunched repeatedly. Native admission12TB, conversion10TB.
Uploader registration: apply, min_local_checkpoints2, grace1h; no manual cleanup
or global policy changes. Estimate ~6.65TB incremental native/export storage.

Numerical inference qualification remains explicitly waived per prior approval;
weight structure, finite values, tokenizer/chat-template and export checks remain.
