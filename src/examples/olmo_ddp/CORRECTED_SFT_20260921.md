# Corrected-tokenizer SFT controls

This campaign leaves PT and all existing eval campaigns untouched. Each run starts
independently from the canonical native 2T/4T LC model, with EMO disabled in SFT
(and disabled during these parents' MT/LC stages). The lineage label describes PT.

| Wave | Parents | Corrected dataset | LR |
|---|---|---|---:|
| 1 | 4T EMO + non-EMO | GPT-OSS medium | 2e-4 |
| 2 | 4T EMO + non-EMO | GPT-OSS high | 2e-4 |
| 3 | 2T EMO + non-EMO | GPT-OSS medium | 2e-4 |
| 4 | 2T EMO + non-EMO | GPT-OSS high | 2e-4 |
| 5 | 4T EMO + non-EMO | Dolci-Think-SFT-32B | 5e-5 |
| 6 | 2T EMO + non-EMO | Dolci-Think-SFT-32B | 5e-5 |

At most two training jobs are in flight. Each uses 64 GPUs on Holmes in
`ai2/olmo3p5-training`, allocated urgent, 8Mi tokens/batch, 64Ki sequence length,
one sequence/rank and two accumulation rounds. Two epochs; 3% warmup then linear
decay to zero; fresh optimizer; no weight decay. The qualified packed eager SFT
path uses FLA with per-block recomputation. No architecture/kernel changes.

All runs save at epoch 1 and epoch 2. The uploader registers independent run
prefixes in the existing private small-model bucket; keeps both checkpoints
locally, and verifies upload. Final checkpoints only get HF conversions and
MATH-500, IFBench, HumanEval and AlpacaEval at temperatures 0.6, 0.8, 1.0. All other
generation/scoring settings match the old tests (top-p .95, 32768-token cap, seed
1234, one sample, both terminal tokens, Think chat formatting). Predictions,
response lengths and overrun counts are saved. 12 models, 144 eval jobs.

Training is gated on a completed corrected-tokenizer manifest, raw/backend/HF
encoding probes, integer-ID and assistant-mask checks, fresh packing, actual
loader probes, source checkpoints, and config validation in the production image.
Dolci gets a deterministic 256-record holdout excluded from training. The GPT
datasets preserve their existing 1024-record holdouts. None use old packing caches.

The controller is CPU-only on Phobos with no resource requests. Data preparation
uses unallocated Rhea capacity. Eval workers run allocated in the MoE workspace.
All payloads go to Weka, not Beaker result datasets. Exact-name durable submission
receipts prevent duplicate launches. Failed jobs are reported for diagnosis;
the controller does not create an unbounded retry storm or change hyperparameters.
