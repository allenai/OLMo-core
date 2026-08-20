# Provisional final-family throughput qualification

Date: 2026-08-20

All runs use sequence length 8,192, an 8 Mi-token global batch, one 8-GPU
Holmes B300 node, BF16 training, the optimized CuTe KDA kernel, and no
checkpoint or evaluation writes. Throughput is summarized over the last 20
logged optimizer steps once a run completes.

| Active rung | GPUs | EP | Rank microbatch (sequences) | Accumulation | Result | TPS / GPU | TFLOPs / GPU | Active / reserved memory |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| 0.5B | 8 | 1 | 8 | 16 | pass, 50 steps | 201.7k | 503.4 | 159.6 / 164.4 GiB |
| 0.9B | 8 | 1 | 4 | 32 | pass, 50 steps | 77.5k | 415.1 | 240.6 / 243.1 GiB |
| 2.0B | 8 | 4 | 2 | 64 | pass, 50 steps | 37.2k | 443.6 | 221.1 / 225.3 GiB |
| 3.8B | 8 | 8 | 1 | 128 | running with per-block recomputation | — | — | — |

The next legal rank microbatch at the fixed 8 Mi-token batch is twice the
listed value. The 0.5B MB16 case OOMed in backward, so MB8 is its largest legal
setting. The 0.9B and 2.0B settings are close enough to the B300 memory ceiling
that their next legal settings cannot fit. The initial 3.8B MB1 case OOMed in
backward at approximately 267.3 / 267.7 GiB; its retry uses per-block
recomputation and shared EP scratch buffers.

## Beaker jobs

- 0.5B: https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01M0G8KR4EB9QB42A0KTE0EKXV
- 0.9B: https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01M0G8ME0QGR89DTNQ7PJT4R7G
- 2.0B: https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01M0G8N2RGE3RD1E3HQHMVQHFJ
- 3.8B recomputation retry: https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01M0G9DVNR4P18MH8X9T7WHZBM
- 3.8B initial OOM: https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01M0G8NR1FQDQP872SNPFPMGND
