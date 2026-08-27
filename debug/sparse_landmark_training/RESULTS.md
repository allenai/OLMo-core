# Sparse-landmark vs full attention: measured train-step wall-clock (2026-08-26)

`bench_train_step.py` on one horton H200 (job 3476812), exactly the CPT/SFT per-rank setup:
fwd+bwd, full activation checkpointing, bf16, fused-linear loss, 1 sequence per rank, no CP.
Sparse = `AttentionType.sparse_landmark`, block 64 (mem_freq 63), 1 landmark. 8 timed steps after
3 warmups. Raw numbers in `bench_results.json`.

| model | T | dense tok/s/GPU | sparse tok/s/GPU | **speedup** |
| --- | --- | --- | --- | --- |
| Qwen3-4B (all-attn) | 65536 | 3089 | 11721 | **3.79x** |
| Qwen3-4B (all-attn) | 32768 | 4920 | 11850 | **2.41x** |
| Qwen3-4B (all-attn) | 16384 | 7047 | 11707 | **1.66x** |
| Qwen3.5-4B (GDN 3:1 hybrid) | 65536 | 6918 | 12267 | **1.77x** |
| Qwen3.5-4B (GDN 3:1 hybrid) | 32768 | 8725 | 11946 | **1.37x** |
| Qwen3.5-4B (GDN 3:1 hybrid) | 16384 | 9457 | 10974 | **1.16x** |

Reading of the table:

* **Sparse throughput is essentially flat in context length** (~11.7-12.3k tok/s/GPU from 16k to
  64k on both models) — attention cost per token is ~constant by construction, so the whole step is
  MLP-bound. Dense degrades 2.3x from 16k→64k. The speedup therefore *widens* with context; at a
  future 128k/256k rung it will be larger than anything in this table.
* On the hybrid Qwen3.5 only ~1/4 of layers are full attention, which dilutes the win exactly as
  expected (1.77x vs 3.79x at 64k).
* Peak memory is a wash (sparse ~+1GB from the landmark gather buffers).
* Caveats: single-GPU fwd+bwd; real multi-node steps add optimizer + collectives, which are
  attention-agnostic and mostly overlapped, so expect the realized end-to-end ratio to be a bit
  smaller. Sparse sequences carry 1 landmark per 64 tokens, so content-token throughput is x63/64
  (-1.6%), already negligible.

## Wall-clock-matched data scaling (the actionable numbers)

Same hardware, same wall-clock as a dense run → multiply the dense token budget by the speedup
(x63/64 for content tokens):

| run | dense budget | sparse matched budget |
| --- | --- | --- |
| CPT 64k, Qwen3-4B (10B tokens) | 10B | **~37B** |
| CPT 64k, Qwen3.5-4B (10B tokens) | 10B | **~17.4B** |
| SFT 64k longctx, Qwen3-4B, 3 epochs | 3 ep | **~11 ep — or 3.8x unique examples** |
| SFT 32k, Qwen3-4B | 1x | 2.4x |

Constraint: the dolma3_longmino CPT sample holds only **15B tokens**, so the Qwen3-4B matched
budget (37B) means ~2.5 epochs of it — a truly data-scaled run needs a larger sample. Qwen3.5's
17.4B is ~1.2 epochs, nearly feasible as-is.

## No activation checkpointing (2026-08-26, job 3476844)

Same sweep with `--no-ac`. **32k and 64k OOM on the 141GB H200 for every config** — without AC a 4B
model stores every layer's activations (~100GB already at 16k), which is why the real training
configs use full AC at 64k. Only 16k fits:

| model | T | dense tok/s | sparse tok/s | speedup | (with-AC speedup) |
| --- | --- | --- | --- | --- | --- |
| Qwen3-4B | 16384 | 8,891 | 14,770 | **1.66x** | 1.66x |
| Qwen3.5-4B | 16384 | 11,533 | 13,212 | **1.15x** | 1.16x |

Takeaways: full AC is a uniform ~20-26% throughput tax (dense and sparse alike), so the
**sparse/dense ratio is unchanged** by AC — the with-AC speedup table is the right one for
planning, and AC-off is not a viable 64k config anyway. Sparse peak memory is again slightly
*higher* than dense (127 vs 115GB at 16k), confirming no training-memory advantage.
