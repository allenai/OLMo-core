# Provisional final-family throughput qualification

Date: 2026-08-20 (status refreshed 2026-09-04)

All runs use sequence length 8,192, an 8 Mi-token global batch, one 8-GPU
Holmes B300 node, BF16 training, the optimized CuTe KDA kernel, and no
checkpoint or evaluation writes. Throughput is summarized over the last 20
logged optimizer steps once a run completes.

| Active rung | GPUs | EP | Rank microbatch (sequences) | Accumulation | Result | TPS / GPU | TFLOPs / GPU | Active / reserved memory |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| 0.5B | 8 | 1 | 8 | 16 | pass, 50 steps | 201.7k | 503.4 | 159.6 / 164.4 GiB |
| 0.9B | 8 | 1 | 4 | 32 | pass, 50 steps | 77.5k | 415.1 | 240.6 / 243.1 GiB |
| 2.0B | 8 | 4 | 2 | 64 | pass, 50 steps | 37.2k | 443.6 | 221.1 / 225.3 GiB |
| 3.8B | 8 | 8 | 1 | 128 | pass, 50 steps with per-block recomputation | 9.2k | 209.7 | 197.9 / 215.4 GiB |

The next legal rank microbatch at the fixed 8 Mi-token batch is twice the
listed value. The 0.5B MB16 case OOMed in backward, so MB8 is its largest legal
setting. The 0.9B and 2.0B settings are close enough to the B300 memory ceiling
that their next legal settings cannot fit. The initial 3.8B MB1 case OOMed in
backward at approximately 267.3 / 267.7 GiB; its retry completed all 50 steps
with per-block recomputation and shared EP scratch buffers.

## Beaker jobs

- 0.5B: https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01M0G8KR4EB9QB42A0KTE0EKXV
- 0.9B: https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01M0G8ME0QGR89DTNQ7PJT4R7G
- 2.0B: https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01M0G8N2RGE3RD1E3HQHMVQHFJ
- 3.8B recomputation retry: https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01M0G9DVNR4P18MH8X9T7WHZBM
- 3.8B initial OOM: https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01M0G8NR1FQDQP872SNPFPMGND

## Expert-granularity comparison: 256/top-8 with 2x wider experts

This controlled variant changes only the routed-expert granularity from
`512 experts / top-16 / h=1` to `256 experts / top-8 / h=2`. The selected
expert width doubles, so active and total parameter counts remain close while
the number of routed expert calls per token halves. Numbers are last-20-step
means under the same 8 Mi-token, 8-GPU setup as the baseline table.

| Active rung | Layout | Active / total params | TPS / GPU | TPS delta | TFLOPs / GPU | TFLOPs delta | Active / reserved memory |
|---|---|---:|---:|---:|---:|---:|---:|
| 0.5B | 512 / top-16 / h1 | 494.1M / 5.955B | 201.7k | — | 503.4 | — | 159.6 / 164.4 GiB |
| 0.5B | 256 / top-8 / h2 | 492.2M / 5.953B | 212.0k | **+5.08%** | 526.7 | **+4.62%** | 151.7 / 156.2 GiB |
| 0.9B | 512 / top-16 / h1 | 934.0M / 15.757B | 77.5k | — | 415.1 | — | 240.6 / 243.1 GiB |
| 0.9B | 256 / top-8 / h2 | 929.0M / 15.752B | 82.4k | **+6.33%** | 438.9 | **+5.73%** | 229.8 / 232.1 GiB |

The 256/top-8 variant also reduced active memory by 7.85 GiB at 0.5B and
10.84 GiB at 0.9B. Both jobs completed all 50 steps without skipped updates.

- 0.5B 256/top-8: https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01M0GGBKPH9DX017E68GZ2EHY8
- 0.9B 256/top-8: https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01M0GGBNTWMB9642VK6H1496GH

## Deeper production-family candidates

These rows track the replacement 16/24/40-layer family separately from the
shallower configurations above. Throughput is the arithmetic mean over the
last 20 of 100 logged optimizer steps. The run uses Holmes B300s, sequence
length 8,192, BF16, the CuTe KDA PR 837 kernel, and EMO document pools from 16
to 512 experts (512 for evaluation). It disables MXFP8, activation
recomputation, shared EP output buffers, and normal-gradient reduce-scatter.

| Active rung | Active / total params | GPUs | Global batch | PP | EP | Rank microbatch | Accumulation | Result | TPS / GPU | Aggregate TPS | TFLOPs / GPU | MFU | Active / reserved memory | Mean step time |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| Medium | 2.388B / 42.760B | 128 | 8 Mi tokens | 1 | 8 | 2 sequences / 16,384 tokens | 4 | pass, 100 steps | 14.81k | 1.896M | 207.8 | 9.24% | 184.1 / 210.0 GiB | 4.425 s |

The run had zero skipped optimizer updates. Peak active memory during startup
was 205.7 GiB; the table reports its steady-state active allocation and peak
reserved allocation. The two canceled replica attempts shown by Beaker were
rescheduled successfully, and all 16 eight-GPU tasks ultimately succeeded.

- Medium 8 Mi Beaker: https://beaker.org/orgs/ai2/workspaces/olmo3p5-training/work/01M1N6WE5A2B1YQ4MV77N574XY
- Medium 8 Mi W&B: https://wandb.ai/ai2-llm/olmoe3-deep-family-microbatch/runs/2auonolg
