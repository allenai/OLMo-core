# JacobM OLMoE Ladder Runs

This file tracks tiny MoE ladder experiments launched from this branch so we can
disambiguate run names, batch sizes, LR sweeps, data roots, and Beaker jobs later.

Unless noted otherwise, runs use:

- Script: `src/scripts/train/jacobm_olmoe_ladder/tiny_275m.py`
- Model: tiny MoE, about 278M active / 1.13B total params
- Data mix: `DataMix.OLMo_mix_0925`
- Data root: `s3://ai2-llm`
- Cluster: `ai2/titan`, 1 node, 8 GPUs
- Image: `tianhuat/olmo-core-torch211-2404-cu128`
- Workspace: `ai2/OLMo-3-moe-experiments`
- Budget: `ai2/oe-other`
- Priority: `urgent`
- WandB: `ai2-llm/jacobm-olmoe-ladder`
- Scheduler: linear warmup then cosine decay to 0.1x final LR
- Warmup fraction: 10%
- Sequence length: 8192

## Run Table

Run names must be unique because the checkpoint path is derived from the run /
experiment name.

| Date | Run | Script | Chinchilla | Batch tokens | Batch seqs | LR | Beaker ID | Beaker link | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 2026-06-02 | `olmoe3-tiny-275m-4xchinchilla-smoketest` | `src/scripts/train/OLMoE3-tiny-275m-active-smoketest.py` | 4x | 1,048,576 | 128 | 2e-4 | `01KT54GVVNM8JRJ94A9ASVVJKX` | https://beaker.org/ex/01KT54GVVNM8JRJ94A9ASVVJKX | Initial successful tiny MoE run. Used the predecessor smoketest script and original WandB project before the ladder project rename. |
| 2026-06-02 | `olmoe3-tiny-275m-cx1-lr1e-4` | `tiny_275m.py` | 1x | 2,097,152 | 256 | 1e-4 | `01KT5JFNT1DEYX814KN5XD3NYZ` | https://beaker.org/ex/01KT5JFNT1DEYX814KN5XD3NYZ | First 2M-batch Cx1 LR sweep run; reached training. |
| 2026-06-02 | `olmoe3-tiny-275m-cx1-lr3e-4` | `tiny_275m.py` | 1x | 2,097,152 | 256 | 3e-4 | `01KT5K59VSV3G6BX7WDE5V156B` | https://beaker.org/ex/01KT5K59VSV3G6BX7WDE5V156B | 2M-batch Cx1 LR sweep. |
| 2026-06-02 | `olmoe3-tiny-275m-cx1-lr8e-4` | `tiny_275m.py` | 1x | 2,097,152 | 256 | 8e-4 | `01KT5K5EM002XR2K3818Y1XV0T` | https://beaker.org/ex/01KT5K5EM002XR2K3818Y1XV0T | 2M-batch Cx1 LR sweep; best visible training CE among `1e-4`, `3e-4`, and `8e-4` in the attached W&B plot. |
| 2026-06-02 | `olmoe3-tiny-275m-cx1-lr1.2e-3` | `tiny_275m.py` | 1x | 2,097,152 | 256 | 1.2e-3 | `01KT5K5MJ19KGCQK5CV96JJ3CS` | https://beaker.org/ex/01KT5K5MJ19KGCQK5CV96JJ3CS | 2M-batch Cx1 LR sweep; queued when the 256k-batch sweep was planned. |

## Planned Sweeps

### Cx1, 256k tokens/step

Motivation: the 2M-batch Cx1 runs only have about 1920 optimizer steps and 192
warmup steps. Dropping to 256k tokens/step gives about 15,360 optimizer steps and
1536 warmup steps, which should make the Cx1 rung a better LR/stability probe.

Recommended LRs:

- `3e-4`
- `5e-4`
- `8e-4`
- `1.2e-3`

Planned run names:

- `olmoe3-tiny-275m-cx1-b256k-lr3e-4`
- `olmoe3-tiny-275m-cx1-b256k-lr5e-4`
- `olmoe3-tiny-275m-cx1-b256k-lr8e-4`
- `olmoe3-tiny-275m-cx1-b256k-lr1.2e-3`

Rationale: `1e-4` is clean but clearly too slow in the 2M-batch plot, while
`8e-4` is currently best among the visible runs. The 256k sweep keeps `3e-4` as
a conservative anchor, adds `5e-4` between the known clean and best-visible
settings, repeats `8e-4`, and keeps `1.2e-3` to test the high side.

Launcher:

```bash
src/scripts/train/jacobm_olmoe_ladder/launch_tiny_275m_lr_sweep_cx1_b256k.sh
```

Reproducibility record / dry-run command printer:

```bash
src/scripts/train/jacobm_olmoe_ladder/reproduce_tiny_275m_lr_sweep_cx1_b256k.sh
```
