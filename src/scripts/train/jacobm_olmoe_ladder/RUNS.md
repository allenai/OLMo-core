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
- Future checkpoint root: `/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3`

Earlier runs before this checkpoint-root cleanup were launched with the local
machine username in the path, so they saved under
`/weka/oe-training-default/ai2-llm/checkpoints/jacob/<run-name>`. Future ladder
launchers default to the `jacobm/olmoe3` root above. Set `CHECKPOINT_ROOT` to
override this in the launcher scripts.

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
| 2026-06-02 | `olmoe3-tiny-275m-cx1-b256k-lr3e-4` | `tiny_275m.py` | 1x | 262,144 | 32 | 3e-4 | `01KT5QEQ3CKEEG2D2XD938RR3S` | https://beaker.org/ex/01KT5QEQ3CKEEG2D2XD938RR3S | 256k-batch Cx1 LR sweep. |
| 2026-06-02 | `olmoe3-tiny-275m-cx1-b256k-lr5e-4` | `tiny_275m.py` | 1x | 262,144 | 32 | 5e-4 | `01KT5QEVY6QNE4VNHF21DZVG4H` | https://beaker.org/ex/01KT5QEVY6QNE4VNHF21DZVG4H | 256k-batch Cx1 LR sweep. |
| 2026-06-02 | `olmoe3-tiny-275m-cx1-b256k-lr8e-4` | `tiny_275m.py` | 1x | 262,144 | 32 | 8e-4 | `01KT5QF1R9GKB17899TN04Z1VD` | https://beaker.org/ex/01KT5QF1R9GKB17899TN04Z1VD | 256k-batch Cx1 LR sweep. |
| 2026-06-02 | `olmoe3-tiny-275m-cx1-b256k-lr1.2e-3` | `tiny_275m.py` | 1x | 262,144 | 32 | 1.2e-3 | `01KT5QF87053QWKAYK53NBKGS2` | https://beaker.org/ex/01KT5QF87053QWKAYK53NBKGS2 | 256k-batch Cx1 LR sweep. |
| 2026-06-02 | `olmoe3-tiny-275m-cx2-b256k-lr5e-4` | `tiny_275m.py` | 2x | 262,144 | 32 | 5e-4 | `01KT61W21PBSX5F2SS0RHNSAPS` | https://beaker.org/ex/01KT61W21PBSX5F2SS0RHNSAPS | Cx2 transfer check; queued before Cx1 follow-ups. |
| 2026-06-02 | `olmoe3-tiny-275m-cx2-b256k-lr7e-4` | `tiny_275m.py` | 2x | 262,144 | 32 | 7e-4 | `01KT61W6Z3S0TQ58CJ9RHRP15G` | https://beaker.org/ex/01KT61W6Z3S0TQ58CJ9RHRP15G | Cx2 transfer check; queued before Cx1 follow-ups. |
| 2026-06-02 | `olmoe3-tiny-275m-cx1-b256k-lr4e-4` | `tiny_275m.py` | 1x | 262,144 | 32 | 4e-4 | `01KT61WCH6S6N0MV4VVB7751RR` | https://beaker.org/ex/01KT61WCH6S6N0MV4VVB7751RR | 256k-batch Cx1 LR refinement. |
| 2026-06-02 | `olmoe3-tiny-275m-cx1-b256k-lr6e-4` | `tiny_275m.py` | 1x | 262,144 | 32 | 6e-4 | `01KT61WJF7MWYVEV8CG6SGG9NM` | https://beaker.org/ex/01KT61WJF7MWYVEV8CG6SGG9NM | 256k-batch Cx1 LR refinement. |
| 2026-06-02 | `olmoe3-tiny-275m-cx1-b256k-lr7e-4` | `tiny_275m.py` | 1x | 262,144 | 32 | 7e-4 | `01KT61WR3EC0A5JHRY35EE1R7A` | https://beaker.org/ex/01KT61WR3EC0A5JHRY35EE1R7A | 256k-batch Cx1 LR refinement. |
| 2026-06-02 | `olmoe3-tiny-275m-cx1-b256k-lr1e-3` | `tiny_275m.py` | 1x | 262,144 | 32 | 1e-3 | `01KT61WY9CK4MHK3CX39KKYYG0` | https://beaker.org/ex/01KT61WY9CK4MHK3CX39KKYYG0 | 256k-batch Cx1 LR refinement. |
| 2026-06-02 | `olmoe3-tiny-275m-cx1-b128k-lr5e-4` | `tiny_275m.py` | 1x | 131,072 | 16 | 5e-4 | `01KT61X45SVNDB6YYNKKEJWJD1` | https://beaker.org/ex/01KT61X45SVNDB6YYNKKEJWJD1 | 128k-batch Cx1 batch-size probe. |
| 2026-06-02 | `olmoe3-tiny-275m-cx1-b128k-lr8e-4` | `tiny_275m.py` | 1x | 131,072 | 16 | 8e-4 | `01KT61XAEYQ7R2A64QYP6CW3Y1` | https://beaker.org/ex/01KT61XAEYQ7R2A64QYP6CW3Y1 | 128k-batch Cx1 batch-size probe. |
| 2026-06-02 | `olmoe3-tiny-275m-cx1-b512k-lr5e-4` | `tiny_275m.py` | 1x | 524,288 | 64 | 5e-4 | `01KT61XFXM7N00EVCYJHF183G5` | https://beaker.org/ex/01KT61XFXM7N00EVCYJHF183G5 | 512k-batch Cx1 batch-size probe. |
| 2026-06-02 | `olmoe3-tiny-275m-cx1-b512k-lr8e-4` | `tiny_275m.py` | 1x | 524,288 | 64 | 8e-4 | `01KT61XPH6PGMNYMGD95ST09PS` | https://beaker.org/ex/01KT61XPH6PGMNYMGD95ST09PS | 512k-batch Cx1 batch-size probe. |
| 2026-06-03 | `olmoe3-tiny-275m-cx1-b256k-n2-lr1.5e-3` | `tiny_275m.py` | 1x | 262,144 | 32 | 1.5e-3 | `01KT73WQH8SDVQ0DS3XGAYD6KT` | https://beaker.org/ex/01KT73WQH8SDVQ0DS3XGAYD6KT | Two-node high-side LR probe. Checkpoints under `jacobm/olmoe3`. |
| 2026-06-03 | `olmoe3-tiny-275m-cx1-b256k-n2-lr2e-3` | `tiny_275m.py` | 1x | 262,144 | 32 | 2e-3 | `01KT73WX76DE1RT678F7R3Y0C1` | https://beaker.org/ex/01KT73WX76DE1RT678F7R3Y0C1 | Two-node high-side LR probe. Checkpoints under `jacobm/olmoe3`. |
| 2026-06-03 | `olmoe3-tiny-275m-cx2-b256k-lr1e-3` | `tiny_275m.py` | 2x | 262,144 | 32 | 1e-3 | `01KT79MKV709DG6F6WQ1NSQT8J` | https://beaker.org/ex/01KT79MKV709DG6F6WQ1NSQT8J | Cx2 high-side check after `7e-4` beat `5e-4`. |
| 2026-06-03 | `olmoe3-tiny-275m-cx4-b512k-lr1e-3` | `tiny_275m.py` | 4x | 524,288 | 64 | 1e-3 | `01KT79MT2NJBZ9QXQS969P9RQ2` | https://beaker.org/ex/01KT79MT2NJBZ9QXQS969P9RQ2 | Cx4 LR sweep at dense-ladder Cx4 batch rule. |
| 2026-06-03 | `olmoe3-tiny-275m-cx4-b512k-lr1.5e-3` | `tiny_275m.py` | 4x | 524,288 | 64 | 1.5e-3 | `01KT79N045S7RHSF80FKE2H77B` | https://beaker.org/ex/01KT79N045S7RHSF80FKE2H77B | Cx4 LR sweep at dense-ladder Cx4 batch rule. |
| 2026-06-03 | `olmoe3-tiny-275m-cx4-b512k-lr2.5e-3` | `tiny_275m.py` | 4x | 524,288 | 64 | 2.5e-3 | `01KT79N5S8NM71V6RXMVNB85AV` | https://beaker.org/ex/01KT79N5S8NM71V6RXMVNB85AV | Cx4 LR sweep at dense-ladder Cx4 batch rule. |
| 2026-06-03 | `olmoe3-tiny-275m-cx4-b512k-lr3.5e-3` | `tiny_275m.py` | 4x | 524,288 | 64 | 3.5e-3 | `01KT79NBSBGCSAJ225ZNHR63ZY` | https://beaker.org/ex/01KT79NBSBGCSAJ225ZNHR63ZY | Cx4 LR sweep at dense-ladder Cx4 batch rule. |
| 2026-06-03 | `olmoe3-tiny-275m-cx1-b256k-lr3e-3` | `tiny_275m.py` | 1x | 262,144 | 32 | 3e-3 | `01KT7A59M00K94XJ59WMK2H517` | https://beaker.org/ex/01KT7A59M00K94XJ59WMK2H517 | Extra high-side Cx1 LR probe for cleaner U-plot. |
| 2026-06-03 | `olmoe3-tiny-275m-cx1-b256k-lr5e-3` | `tiny_275m.py` | 1x | 262,144 | 32 | 5e-3 | `01KT7A5EBAX04FYRRVK8BRXX1B` | https://beaker.org/ex/01KT7A5EBAX04FYRRVK8BRXX1B | Extra high-side Cx1 LR probe for cleaner U-plot. |
| 2026-06-03 | `olmoe3-tiny-275m-cx2-b256k-ep1mb4-lr1e-3` | `tiny_275m.py` | 2x | 262,144 | 32 | 1e-3 | `01KT7JTTXH7ND3CVCM76Y7BDH9` | https://beaker.org/ex/01KT7JTTXH7ND3CVCM76Y7BDH9 | Stopped accidental 8-GPU optimized requeue before the smaller-GPU smoke. EP=1, microbatch=4. |
| 2026-06-03 | `olmoe3-tiny-275m-cx4-b512k-ep1mb8-lr1e-3` | `tiny_275m.py` | 4x | 524,288 | 64 | 1e-3 | `01KT7JV2FD3T57PM5BG28DTJH4` | https://beaker.org/ex/01KT7JV2FD3T57PM5BG28DTJH4 | Stopped accidental 8-GPU optimized requeue before the smaller-GPU smoke. EP=1, microbatch=8. |
| 2026-06-03 | `olmoe3-tiny-275m-cx4-b512k-ep1mb8-lr1.5e-3` | `tiny_275m.py` | 4x | 524,288 | 64 | 1.5e-3 | `01KT7JV8DDQ4ZF7SB3BX7XXPZX` | https://beaker.org/ex/01KT7JV8DDQ4ZF7SB3BX7XXPZX | Stopped accidental 8-GPU optimized requeue before the smaller-GPU smoke. EP=1, microbatch=8. |
| 2026-06-03 | `olmoe3-tiny-275m-cx4-b512k-ep1mb8-lr2.5e-3` | `tiny_275m.py` | 4x | 524,288 | 64 | 2.5e-3 | `01KT7JVE2BVB70KHD37Y39F36V` | https://beaker.org/ex/01KT7JVE2BVB70KHD37Y39F36V | Stopped accidental 8-GPU optimized requeue before the smaller-GPU smoke. EP=1, microbatch=8. |
| 2026-06-03 | `olmoe3-tiny-275m-cx4-b512k-ep1mb8-lr3.5e-3` | `tiny_275m.py` | 4x | 524,288 | 64 | 3.5e-3 | `01KT7JVMPHDH3EN3T3N20F3MJN` | https://beaker.org/ex/01KT7JVMPHDH3EN3T3N20F3MJN | Stopped accidental 8-GPU optimized requeue before the smaller-GPU smoke. EP=1, microbatch=8. |
| 2026-06-03 | `olmoe3-tiny-275m-cx1-b256k-gpu2-ep1mb16-lr3e-3` | `tiny_275m.py` | 1x | 262,144 | 32 | 3e-3 | `01KT7KBFWN0SYRHB9K10S2KXWK` | https://beaker.org/ex/01KT7KBFWN0SYRHB9K10S2KXWK | Two-GPU smoke for preserving the 256k global batch with microbatch=16 and EP=1. |
| 2026-06-03 | `olmoe3-tiny-275m-cx2-b256k-gpu2-ep1mb16-lr1e-3` | `tiny_275m.py` | 2x | 262,144 | 32 | 1e-3 | `01KT7KXRZMJVWYJDBDRFEW7PFX` | https://beaker.org/ex/01KT7KXRZMJVWYJDBDRFEW7PFX | Relaunched with partial-node settings after throughput smoke. 2 GPUs, EP=1, microbatch=16. |
| 2026-06-04 | `olmoe3-tiny-275m-cx1-b256k-gpu2-ep1mb16-lr8e-4-r2` | `tiny_275m.py` | 1x | 262,144 | 32 | 8e-4 | `01KTA3V92BK4FBXXEQF80KSAAA` | https://beaker.org/ex/01KTA3V92BK4FBXXEQF80KSAAA | Current-family Cx1 cold/mid probe. Finished step 15365, avg250M 2.7888, avg500M 2.7915. 2 GPUs, EP=1, microbatch=16. |
| 2026-06-04 | `olmoe3-tiny-275m-cx1-b256k-gpu2-ep1mb16-lr1e-3-r2` | `tiny_275m.py` | 1x | 262,144 | 32 | 1e-3 | `01KTA3W6HMJHQ6SM72NXW55AW3` | https://beaker.org/ex/01KTA3W6HMJHQ6SM72NXW55AW3 | Current-family Cx1 cold/mid probe. Finished step 15365, avg250M 2.7852, avg500M 2.7881. 2 GPUs, EP=1, microbatch=16. |
| 2026-06-04 | `olmoe3-tiny-275m-cx1-b256k-gpu2-ep1mb16-lr1.2e-3-r2` | `tiny_275m.py` | 1x | 262,144 | 32 | 1.2e-3 | `01KTA3X9R2PWD408NG6617M1NS` | https://beaker.org/ex/01KTA3X9R2PWD408NG6617M1NS | Current-family Cx1 cold/mid probe. Finished step 15365, avg250M 2.7843, avg500M 2.7876. 2 GPUs, EP=1, microbatch=16. |
| 2026-06-04 | `olmoe3-tiny-275m-cx1-b256k-gpu2-ep1mb16-lr1.5e-3-r2` | `tiny_275m.py` | 1x | 262,144 | 32 | 1.5e-3 | `01KT9T7WAH2A5D2W14P9P3VF81` | https://beaker.org/ex/01KT9T7WAH2A5D2W14P9P3VF81 | Current-family Cx1 basin rerun to clean up the LR-rule plot. Finished step 15365, avg250M 2.7794, avg500M 2.7831. 2 GPUs, EP=1, microbatch=16. |
| 2026-06-04 | `olmoe3-tiny-275m-cx1-b256k-gpu2-ep1mb16-lr2e-3-r2` | `tiny_275m.py` | 1x | 262,144 | 32 | 2e-3 | `01KT9T9F9A3VNG4YYF8B7TNS84` | https://beaker.org/ex/01KT9T9F9A3VNG4YYF8B7TNS84 | Current-family Cx1 basin rerun to clean up the LR-rule plot. Finished step 15365, avg250M 2.7765, avg500M 2.7809. 2 GPUs, EP=1, microbatch=16. |
| 2026-06-04 | `olmoe3-tiny-275m-cx1-b256k-gpu8-ep8mb4-lr1e-3-sanity` | `tiny_275m.py` | 1x | 262,144 | 32 | 1e-3 | `01KTA98DW402AB3DJH739WBPW5` | https://beaker.org/ex/01KTA98DW402AB3DJH739WBPW5 | Diagnostic-only sanity check for the Cx1 settings-family discrepancy. Finished step 15365, avg250M 2.7671, avg500M 2.7702. Uses EP=8 with the highest legal microbatch for 8 GPUs at 256k global batch. Exclude from ladder LR fits and canonical plots. |
| 2026-06-05 | `olmoe3-tiny-275m-cx1-b256k-gpu8-ep8mb4-lr1e-3-dropless-sanity` | `tiny_275m.py` | 1x | 262,144 | 32 | 1e-3 | `01KTAPHEJMG8DV9ZB4RHTRHMCV` | https://beaker.org/ex/01KTAPHEJMG8DV9ZB4RHTRHMCV | Failed before training because the W&B group name exceeded the 128-character limit. Diagnostic-only follow-up to the EP=8 sanity check; replacement below uses the same EP=8, 8-GPU, microbatch=4 setting with `--no-use-rowwise-a2a`. Exclude from ladder LR fits and canonical plots. |
| 2026-06-05 | `olmoe3-tiny-cx1-ep8drop-lr1e-3` | `tiny_275m.py` | 1x | 262,144 | 32 | 1e-3 | `01KTB86J84BNDZJYXWMPVC9FVG` | https://beaker.org/ex/01KTB86J84BNDZJYXWMPVC9FVG | Replacement dropless EP sanity check with a shorter W&B group name. 8 GPUs, EP=8, microbatch=4, `--no-use-rowwise-a2a`. Finished step 15365, avg100M 2.7646, avg250M 2.7668, avg500M 2.7699, skipped steps 0, token drop rate 0.0. Exclude from ladder LR fits and canonical plots. |
| 2026-06-03 | `olmoe3-tiny-275m-cx4-b512k-gpu4-ep1mb16-lr1e-3` | `tiny_275m.py` | 4x | 524,288 | 64 | 1e-3 | `01KT7KXYNS2CZDHVQ3THE6DGHH` | https://beaker.org/ex/01KT7KXYNS2CZDHVQ3THE6DGHH | Relaunched with partial-node settings after throughput smoke. 4 GPUs, EP=1, microbatch=16. |
| 2026-06-03 | `olmoe3-tiny-275m-cx4-b512k-gpu4-ep1mb16-lr1.5e-3` | `tiny_275m.py` | 4x | 524,288 | 64 | 1.5e-3 | `01KT7KY3WARYXENM35PNFM9M4C` | https://beaker.org/ex/01KT7KY3WARYXENM35PNFM9M4C | Relaunched with partial-node settings after throughput smoke. 4 GPUs, EP=1, microbatch=16. |
| 2026-06-03 | `olmoe3-tiny-275m-cx4-b512k-gpu4-ep1mb16-lr2.5e-3` | `tiny_275m.py` | 4x | 524,288 | 64 | 2.5e-3 | `01KT7KYASSEMQJY7P6P040G7ND` | https://beaker.org/ex/01KT7KYASSEMQJY7P6P040G7ND | Relaunched with partial-node settings after throughput smoke. 4 GPUs, EP=1, microbatch=16. |
| 2026-06-03 | `olmoe3-tiny-275m-cx4-b512k-gpu4-ep1mb16-lr3.5e-3` | `tiny_275m.py` | 4x | 524,288 | 64 | 3.5e-3 | `01KT7KYG086VY082EN6HPSSX9G` | https://beaker.org/ex/01KT7KYG086VY082EN6HPSSX9G | Relaunched with partial-node settings after throughput smoke. 4 GPUs, EP=1, microbatch=16. |
| 2026-06-03 | `olmoe3-tiny-275m-cx1-b256k-gpu2-ep1mb16-lr5e-3` | `tiny_275m.py` | 1x | 262,144 | 32 | 5e-3 | `01KT7KYTYWGV7CAFNNBBWAT91H` | https://beaker.org/ex/01KT7KYTYWGV7CAFNNBBWAT91H | Relaunched high-side Cx1 LR point with partial-node settings. 2 GPUs, EP=1, microbatch=16. |
| 2026-06-03 | `olmoe3-tiny-275m-cx1-b256k-gpu2-ep1mb16-lr8e-3` | `tiny_275m.py` | 1x | 262,144 | 32 | 8e-3 | `01KT7MQ5F42DBCD30P8JDKJ9JH` | https://beaker.org/ex/01KT7MQ5F42DBCD30P8JDKJ9JH | High-side Cx1 follow-up to clearly bracket the right side of the U-plot. 2 GPUs, EP=1, microbatch=16. |
| 2026-06-03 | `olmoe3-tiny-275m-cx2-b256k-gpu2-ep1mb16-lr1.5e-3` | `tiny_275m.py` | 2x | 262,144 | 32 | 1.5e-3 | `01KT7MQAVHRSPW4VG6BVQSY8AW` | https://beaker.org/ex/01KT7MQAVHRSPW4VG6BVQSY8AW | Cx2 high-side follow-up. 2 GPUs, EP=1, microbatch=16. |
| 2026-06-03 | `olmoe3-tiny-275m-cx2-b256k-gpu2-ep1mb16-lr2.5e-3` | `tiny_275m.py` | 2x | 262,144 | 32 | 2.5e-3 | `01KT7MQGH9KF529Q8NC6N63F8K` | https://beaker.org/ex/01KT7MQGH9KF529Q8NC6N63F8K | Cx2 high-side follow-up. 2 GPUs, EP=1, microbatch=16. |
| 2026-06-03 | `olmoe3-tiny-275m-cx2-b256k-gpu2-ep1mb16-lr3.5e-3` | `tiny_275m.py` | 2x | 262,144 | 32 | 3.5e-3 | `01KT7MQPK57YJ9KM7NKN06BDFB` | https://beaker.org/ex/01KT7MQPK57YJ9KM7NKN06BDFB | Cx2 high-side follow-up. 2 GPUs, EP=1, microbatch=16. |
| 2026-06-04 | `olmoe3-tiny-275m-cx2-b256k-gpu2-ep1mb16-lr6e-4-r2` | `tiny_275m.py` | 2x | 262,144 | 32 | 6e-4 | `01KT9RWMECT8AZ63RQH748STYB` | https://beaker.org/ex/01KT9RWMECT8AZ63RQH748STYB | Current-family Cx2 low/mid probe to resolve the old `5e-4`/`7e-4` vs current-family trend mismatch. 2 GPUs, EP=1, microbatch=16. |
| 2026-06-04 | `olmoe3-tiny-275m-cx2-b256k-gpu2-ep1mb16-lr8e-4-r2` | `tiny_275m.py` | 2x | 262,144 | 32 | 8e-4 | `01KT9S05X2WW2BPJVVJXGRYQSV` | https://beaker.org/ex/01KT9S05X2WW2BPJVVJXGRYQSV | Current-family Cx2 low/mid probe to resolve the old `5e-4`/`7e-4` vs current-family trend mismatch. 2 GPUs, EP=1, microbatch=16. |
| 2026-06-03 | `olmoe3-tiny-275m-cx4-b512k-gpu4-ep1mb16-lr5e-3` | `tiny_275m.py` | 4x | 524,288 | 64 | 5e-3 | `01KT7MQWG562KJVZYX2G8A19CZ` | https://beaker.org/ex/01KT7MQWG562KJVZYX2G8A19CZ | Cx4 high-side follow-up after `3.5e-3` looked healthy early. 4 GPUs, EP=1, microbatch=16. |
| 2026-06-03 | `olmoe3-tiny-275m-cx4-b512k-gpu4-ep1mb16-lr5e-4` | `tiny_275m.py` | 4x | 524,288 | 64 | 5e-4 | `01KT7RXNBYGEQSJ50QFNPFSMM6` | https://beaker.org/ex/01KT7RXNBYGEQSJ50QFNPFSMM6 | Cx4 low-side bracket after current Cx4 relaunches favored `1e-3` over higher LRs at matched tokens. 4 GPUs, EP=1, microbatch=16. |
| 2026-06-03 | `olmoe3-tiny-275m-cx4-b512k-gpu4-ep1mb16-lr7e-4` | `tiny_275m.py` | 4x | 524,288 | 64 | 7e-4 | `01KT7RY46X44N47G58WS1G3REQ` | https://beaker.org/ex/01KT7RY46X44N47G58WS1G3REQ | Cx4 low-side bracket after current Cx4 relaunches favored `1e-3` over higher LRs at matched tokens. 4 GPUs, EP=1, microbatch=16. |
| 2026-06-03 | `olmoe3-tiny-275m-cx8-smoke-b768k-gpu2-ep1mb24-lr5e-4` | `tiny_275m.py` | 0.1x | 786,432 | 96 | 5e-4 | `01KT7V50NF2AEH7TM1S3KB3Y0R` | https://beaker.org/ex/01KT7V50NF2AEH7TM1S3KB3Y0R | Failed dry-run OOM. 2 GPUs, EP=1, microbatch=24 was too large. |
| 2026-06-03 | `olmoe3-tiny-275m-cx16-smoke-b1m-gpu2-ep1mb32-lr5e-4` | `tiny_275m.py` | 0.1x | 1,048,576 | 128 | 5e-4 | `01KT7V5CDARY8XGYYJ7DEKHY49` | https://beaker.org/ex/01KT7V5CDARY8XGYYJ7DEKHY49 | Failed dry-run OOM. 2 GPUs, EP=1, microbatch=32 was too large. |
| 2026-06-03 | `olmoe3-moe-a0-810m-smoke-b256k-gpu2-ep1mb16-lr5e-4` | `tiny_275m.py` | 0.02x | 262,144 | 32 | 5e-4 | `01KT7V71MBZ9730W3FBFB1AVEH` | https://beaker.org/ex/01KT7V71MBZ9730W3FBFB1AVEH | Failed before training because model-size support was not present in the launched runtime. Relaunch from a commit with `--model-size`. |
| 2026-06-03 | `olmoe3-moe-a0-810m-smoke-b256k-gpu2-ep1mb8-lr5e-4` | `tiny_275m.py` | 0.02x | 262,144 | 32 | 5e-4 | `01KT7V7CT1MN2XGZ90SF3WT104` | https://beaker.org/ex/01KT7V7CT1MN2XGZ90SF3WT104 | Failed before training because model-size support was not present in the launched runtime. Relaunch from a commit with `--model-size`. |
| 2026-06-03 | `olmoe3-moe-a0-1p2b-smoke-b256k-gpu2-ep1mb8-lr3e-4` | `tiny_275m.py` | 0.02x | 262,144 | 32 | 3e-4 | `01KT7V7RG65QGYZ6KPNJM9ZEW2` | https://beaker.org/ex/01KT7V7RG65QGYZ6KPNJM9ZEW2 | Failed before training because model-size support was not present in the launched runtime. Relaunch from a commit with `--model-size`. |
| 2026-06-03 | `olmoe3-moe-a0-1p2b-smoke-b256k-gpu2-ep1mb4-lr3e-4` | `tiny_275m.py` | 0.02x | 262,144 | 32 | 3e-4 | `01KT7V84489P8W01WEGX4JP8S7` | https://beaker.org/ex/01KT7V84489P8W01WEGX4JP8S7 | Failed before training because model-size support was not present in the launched runtime. Relaunch from a commit with `--model-size`. |
| 2026-06-04 | `olmoe3-tiny-275m-cx8-smoke-b768k-gpu2-ep1mb16-lr5e-4` | `tiny_275m.py` | 0.1x | 786,432 | 96 | 5e-4 | `01KT829008RED7EA12EP2J2KSV` | https://beaker.org/ex/01KT829008RED7EA12EP2J2KSV | Retry after `mb24` OOM. Reached final smoke step 513/513 with skipped steps 0, finite loss, and final logged step about 693 TFLOPs/GPU. |
| 2026-06-04 | `olmoe3-tiny-275m-cx16-smoke-b1m-gpu2-ep1mb16-lr5e-4` | `tiny_275m.py` | 0.1x | 1,048,576 | 128 | 5e-4 | `01KT829CG1KNWB9GRVVZ8HAY0T` | https://beaker.org/ex/01KT829CG1KNWB9GRVVZ8HAY0T | Canonical retry after `mb32` OOM. Reached final smoke step 385/385 and exited 0 with skipped steps 0 and healthy throughput. |
| 2026-06-04 | `olmoe3-tiny-275m-cx16-smoke-b1m-gpu2-ep1mb16-lr5e-4` | `tiny_275m.py` | 0.1x | 1,048,576 | 128 | 5e-4 | `01KT82H1A9XT62VC8XRVZHJPWV` | https://beaker.org/ex/01KT82H1A9XT62VC8XRVZHJPWV | Accidental duplicate of the canonical Cx16 smoke; stopped manually because it used the same checkpoint root. Ignore for analysis. |
| 2026-06-04 | `olmoe3-moe-a0-810m-smoke-b256k-gpu2-ep1mb16-lr5e-4-r2` | `tiny_275m.py` | 0.02x | 262,144 | 32 | 5e-4 | `01KT82MKYTJDPBRN01P0XHXS38` | https://beaker.org/ex/01KT82MKYTJDPBRN01P0XHXS38 | Relaunch from commit with `--model-size` support. Failed dry-run OOM on 2 GPUs, EP=1, microbatch=16. |
| 2026-06-04 | `olmoe3-moe-a0-810m-smoke-b256k-gpu2-ep1mb8-lr5e-4-r2` | `tiny_275m.py` | 0.02x | 262,144 | 32 | 5e-4 | `01KT82MYJXY98QSK4CMES19PH9` | https://beaker.org/ex/01KT82MYJXY98QSK4CMES19PH9 | Relaunch from commit with `--model-size` support. Failed dry-run OOM on 2 GPUs, EP=1, microbatch=8. |
| 2026-06-04 | `olmoe3-moe-a0-1p2b-smoke-b256k-gpu2-ep1mb8-lr3e-4-r2` | `tiny_275m.py` | 0.02x | 262,144 | 32 | 3e-4 | `01KT82NA11D9E3DEH1NPXDR6WJ` | https://beaker.org/ex/01KT82NA11D9E3DEH1NPXDR6WJ | Relaunch from commit with `--model-size` support. Failed dry-run OOM on 2 GPUs, EP=1, microbatch=8. |
| 2026-06-04 | `olmoe3-moe-a0-1p2b-smoke-b256k-gpu2-ep1mb4-lr3e-4-r2` | `tiny_275m.py` | 0.02x | 262,144 | 32 | 3e-4 | `01KT82NN0Q5MPJQ167W5XY8BQA` | https://beaker.org/ex/01KT82NN0Q5MPJQ167W5XY8BQA | Relaunch from commit with `--model-size` support. Failed dry-run OOM on 2 GPUs, EP=1, microbatch=4. |
| 2026-06-04 | `olmoe3-tiny-275m-cx8-b768k-gpu2-ep1mb16-lr3e-4` | `tiny_275m.py` | 8x | 786,432 | 96 | 3e-4 | `01KT836RV6PHBQ3M5SCVYECWC4` | https://beaker.org/ex/01KT836RV6PHBQ3M5SCVYECWC4 | Stopped manually before completion; replaced by coarser factor-of-two grid. Ignore for analysis. |
| 2026-06-04 | `olmoe3-tiny-275m-cx8-b768k-gpu2-ep1mb16-lr5e-4` | `tiny_275m.py` | 8x | 786,432 | 96 | 5e-4 | `01KT8373YTV4C1G08D20XXMHZG` | https://beaker.org/ex/01KT8373YTV4C1G08D20XXMHZG | Stopped manually before completion; replaced by coarser factor-of-two grid. Ignore for analysis. |
| 2026-06-04 | `olmoe3-tiny-275m-cx8-b768k-gpu2-ep1mb16-lr7e-4` | `tiny_275m.py` | 8x | 786,432 | 96 | 7e-4 | `01KT837GAWN3H1XBYASH520M2A` | https://beaker.org/ex/01KT837GAWN3H1XBYASH520M2A | Stopped manually before completion; replaced by coarser factor-of-two grid. Ignore for analysis. |
| 2026-06-04 | `olmoe3-tiny-275m-cx8-b768k-gpu2-ep1mb16-lr1e-3` | `tiny_275m.py` | 8x | 786,432 | 96 | 1e-3 | `01KT837VTNEK7F783YZYHYRJ1Y` | https://beaker.org/ex/01KT837VTNEK7F783YZYHYRJ1Y | Stopped manually before completion; replaced by coarser factor-of-two grid. Ignore for analysis. |
| 2026-06-04 | `olmoe3-tiny-275m-cx16-b1m-gpu2-ep1mb16-lr2e-4` | `tiny_275m.py` | 16x | 1,048,576 | 128 | 2e-4 | `01KT8387116V5NF6QNDCSBN9CA` | https://beaker.org/ex/01KT8387116V5NF6QNDCSBN9CA | Stopped manually before completion; replaced by coarser factor-of-two grid. Ignore for analysis. |
| 2026-06-04 | `olmoe3-tiny-275m-cx16-b1m-gpu2-ep1mb16-lr3e-4` | `tiny_275m.py` | 16x | 1,048,576 | 128 | 3e-4 | `01KT838HX2QMM7RJCYT2P6MW92` | https://beaker.org/ex/01KT838HX2QMM7RJCYT2P6MW92 | Stopped manually before completion; replaced by coarser factor-of-two grid. Ignore for analysis. |
| 2026-06-04 | `olmoe3-tiny-275m-cx16-b1m-gpu2-ep1mb16-lr5e-4` | `tiny_275m.py` | 16x | 1,048,576 | 128 | 5e-4 | `01KT838WN9G1WDPG5RWBS0CGTM` | https://beaker.org/ex/01KT838WN9G1WDPG5RWBS0CGTM | Stopped manually before completion; replaced by coarser factor-of-two grid. Ignore for analysis. |
| 2026-06-04 | `olmoe3-tiny-275m-cx16-b1m-gpu2-ep1mb16-lr7e-4` | `tiny_275m.py` | 16x | 1,048,576 | 128 | 7e-4 | `01KT83989R9R79D6FA2QXX15J9` | https://beaker.org/ex/01KT83989R9R79D6FA2QXX15J9 | Stopped manually before completion; replaced by coarser factor-of-two grid. Ignore for analysis. |
| 2026-06-04 | `olmoe3-moe-a0-810m-smoke-b256k-gpu4-ep1mb8-lr5e-4-r3` | `tiny_275m.py` | 0.02x | 262,144 | 32 | 5e-4 | `01KT840XF9T975KJM3SHXFCH7D` | https://beaker.org/ex/01KT840XF9T975KJM3SHXFCH7D | Failed OOM. 4 GPUs, EP=1, microbatch=8 still too large. |
| 2026-06-04 | `olmoe3-moe-a0-810m-smoke-b256k-gpu4-ep1mb4-lr5e-4-r3` | `tiny_275m.py` | 0.02x | 262,144 | 32 | 5e-4 | `01KT8418KTXB8Z26DVJF8VRSGD` | https://beaker.org/ex/01KT8418KTXB8Z26DVJF8VRSGD | Succeeded. Validated 810M setting: 4 GPUs, EP=1, microbatch=4; skipped steps 0, about 610 actual avg TFLOPs/GPU. |
| 2026-06-05 | `olmoe3-moe-a0-810m-cx1-b256k-gpu4-ep1mb4-lr1.6e-3-pilot` | `tiny_275m.py` | 1x | 262,144 | 32 | 1.6e-3 | `01KTAPFMM1YG2TE3RJX9B94CH1` | https://beaker.org/ex/01KTAPFMM1YG2TE3RJX9B94CH1 | Exploratory 810M Cx1 pilot at the transferred-LR estimate. Stopped intentionally on 2026-06-05 around step 7669 / 2.01B tokens to free 4 GPUs for the `6e-3` hot-side sentinel. Ignore for final LR selection and canonical plots. |
| 2026-06-05 | `olmoe3-moe-a0-810m-cx1-b256k-gpu4-ep1mb4-lr6e-4-r1` | `tiny_275m.py` | 1x | 262,144 | 32 | 6e-4 | `01KTB9D4971X2ZEXK61W1WJ23H` | https://beaker.org/ex/01KTB9D4971X2ZEXK61W1WJ23H | First 810M Cx1 LR sweep, cold-side anchor. Finished step 52648, avg100M 2.4096, avg250M 2.4103, avg500M 2.4131. Best observed point so far, but it is on the low-LR edge. 4 GPUs, EP=1, microbatch=4. |
| 2026-06-05 | `olmoe3-moe-a0-810m-cx1-b256k-gpu4-ep1mb4-lr1.2e-3-r1` | `tiny_275m.py` | 1x | 262,144 | 32 | 1.2e-3 | `01KTB9DGWWHW2C5HJAYC40AZPX` | https://beaker.org/ex/01KTB9DGWWHW2C5HJAYC40AZPX | First 810M Cx1 LR sweep, below transferred estimate. Finished step 52648, avg100M 2.4147, avg250M 2.4156, avg500M 2.4188. 4 GPUs, EP=1, microbatch=4. |
| 2026-06-05 | `olmoe3-moe-a0-810m-cx1-b256k-gpu4-ep1mb4-lr2.4e-3-r1` | `tiny_275m.py` | 1x | 262,144 | 32 | 2.4e-3 | `01KTB9DWX644NVSVP6Y6B6N7J7` | https://beaker.org/ex/01KTB9DWX644NVSVP6Y6B6N7J7 | First 810M Cx1 LR sweep, above transferred estimate. Finished step 52648, avg100M 2.4456, avg250M 2.4465, avg500M 2.4499. 4 GPUs, EP=1, microbatch=4. |
| 2026-06-05 | `olmoe3-moe-a0-810m-cx1-b256k-gpu4-ep1mb4-lr6e-3-r1` | `tiny_275m.py` | 1x | 262,144 | 32 | 6e-3 | `01KTB9E8MXS8TWGFS1F65HJJMD` | https://beaker.org/ex/01KTB9E8MXS8TWGFS1F65HJJMD | First 810M Cx1 LR sweep, hot-side sentinel. Finished step 52648, avg100M 2.5418, avg250M 2.5424, avg500M 2.5457. Confirms the hot side is bracketed. 4 GPUs, EP=1, microbatch=4. |
| 2026-06-05 | `olmoe3-moe-a0-810m-cx1-b256k-gpu4-ep1mb4-lr3e-4-cold-r1` | `tiny_275m.py` | 1x | 262,144 | 32 | 3e-4 | `01KTCBDVG2FJCPTFVR16G6X0TE` | https://beaker.org/ex/01KTCBDVG2FJCPTFVR16G6X0TE | Cold-side extension because the first 810M Cx1 sweep was low-edge best at `6e-4`. Finished step 52648, avg100M 2.4201, avg250M 2.4207, avg500M 2.4234. 4 GPUs, EP=1, microbatch=4. Created from launcher `launch_moe_a0_810m_cx1_cold_ext.sh`. |
| 2026-06-05 | `olmoe3-moe-a0-810m-cx1-b256k-gpu4-ep1mb4-lr1.5e-4-cold-r1` | `tiny_275m.py` | 1x | 262,144 | 32 | 1.5e-4 | `01KTCBE8QQ4S23W3DCBDNGWNQS` | https://beaker.org/ex/01KTCBE8QQ4S23W3DCBDNGWNQS | Cold-side extension because the first 810M Cx1 sweep was low-edge best at `6e-4`. Finished step 52648, avg100M 2.4482, avg250M 2.4486, avg500M 2.4512. 4 GPUs, EP=1, microbatch=4. Created from launcher `launch_moe_a0_810m_cx1_cold_ext.sh`. |
| 2026-06-05 | `olmoe3-moe-a0-810m-cx1-b256k-gpu4-ep1mb4-lr5e-5-cold-sentinel-r1` | `tiny_275m.py` | 1x | 262,144 | 32 | 5e-5 | `01KTCBQ1RGPG86GR8KKBYN6DT1` | https://beaker.org/ex/01KTCBQ1RGPG86GR8KKBYN6DT1 | Failed before training because the generated W&B group exceeded the 128-character limit. Ignore for analysis. |
| 2026-06-05 | `olmoe3-810m-cx1-b256k-gpu8-ep1mb4-lr5e-5-cs-r2` | `tiny_275m.py` | 1x | 262,144 | 32 | 5e-5 | `01KTCD57W068CSD9F72HFAQ21N` | https://beaker.org/ex/01KTCD57W068CSD9F72HFAQ21N | Short-name relaunch of the far cold-side sentinel. Finished step 52648, avg100M 2.5678, avg250M 2.5680, avg500M 2.5705. 8 GPUs, EP=1, microbatch=4, same global batch as the 4-GPU Cx1 runs. Confirms `5e-5` is far too cold. |
| 2026-06-06 | `olmoe3-moe-a0-810m-cx4-b512k-gpu8-ep1mb4-lr2e-4-r1` | `tiny_275m.py` | 4x | 524,288 | 64 | 2e-4 | `01KTDDANJ6V1T8ZPF4VNPEQQHZ` | https://beaker.org/ex/01KTDDANJ6V1T8ZPF4VNPEQQHZ | 810M Cx4 four-point sweep centered around the transfer-calibrated Cx4 estimate. Finished step 105295, 55.205B tokens, avg100M 2.2516, avg250M 2.2578, avg500M 2.2568. 8 GPUs, EP=1, microbatch=4, final-only permanent checkpoint plus ephemeral resume checkpoints. |
| 2026-06-06 | `olmoe3-moe-a0-810m-cx4-b512k-gpu8-ep1mb4-lr4e-4-r1` | `tiny_275m.py` | 4x | 524,288 | 64 | 4e-4 | `01KTDDB3P13BMCY965FKKFC8VV` | https://beaker.org/ex/01KTDDB3P13BMCY965FKKFC8VV | 810M Cx4 four-point sweep centered around the transfer-calibrated Cx4 estimate. Finished step 105295, 55.205B tokens, avg100M 2.2364, avg250M 2.2427, avg500M 2.2417. Best completed Cx4 point. 8 GPUs, EP=1, microbatch=4, final-only permanent checkpoint plus ephemeral resume checkpoints. |
| 2026-06-06 | `olmoe3-moe-a0-810m-cx4-b512k-gpu8-ep1mb4-lr8e-4-r1` | `tiny_275m.py` | 4x | 524,288 | 64 | 8e-4 | `01KTDDBGWNHKZ3F4Q1VJD2RPTS` | https://beaker.org/ex/01KTDDBGWNHKZ3F4Q1VJD2RPTS | 810M Cx4 four-point sweep centered around the transfer-calibrated Cx4 estimate. Finished step 105295, 55.205B tokens, avg100M 2.2387, avg250M 2.2451, avg500M 2.2442. Very close to but slightly worse than `4e-4`. 8 GPUs, EP=1, microbatch=4, final-only permanent checkpoint plus ephemeral resume checkpoints. |
| 2026-06-06 | `olmoe3-moe-a0-810m-cx4-b512k-gpu8-ep1mb4-lr1.6e-3-r1` | `tiny_275m.py` | 4x | 524,288 | 64 | 1.6e-3 | `01KTDDBWP8EC88V8SDW5Y0VTSV` | https://beaker.org/ex/01KTDDBWP8EC88V8SDW5Y0VTSV | 810M Cx4 four-point sweep centered around the transfer-calibrated Cx4 estimate. Finished step 105295, 55.205B tokens, avg100M 2.2622, avg250M 2.2687, avg500M 2.2678. Hot-side point; substantially worse than the `4e-4`/`8e-4` basin. 8 GPUs, EP=1, microbatch=4, final-only permanent checkpoint plus ephemeral resume checkpoints. |
| 2026-06-07 | `olmoe3-moe-a0-1p2b-cx1-b256k-gpu8-ep1mb2-lr1e-4-r1` | `tiny_275m.py` | 1x | 262,144 | 32 | 1e-4 | `01KTG4J00SXZPREAA3A1E463P9` | https://beaker.org/ex/01KTG4J00SXZPREAA3A1E463P9 | First 1.2B Cx1 sweep centered from updated 810M transfer rule. Finished step 81190, 21.283B tokens, avg100M 2.3549, avg250M 2.3550, avg500M 2.3580. W&B `tvx71brh`. 8 GPUs, EP=1, microbatch=2, in-loop fast evals every 2000 steps. |
| 2026-06-07 | `olmoe3-moe-a0-1p2b-cx1-b256k-gpu8-ep1mb2-lr2e-4-r1` | `tiny_275m.py` | 1x | 262,144 | 32 | 2e-4 | `01KTG4JAQ2Z82YSGPAWRBW353H` | https://beaker.org/ex/01KTG4JAQ2Z82YSGPAWRBW353H | First 1.2B Cx1 sweep centered from updated 810M transfer rule. Finished step 81190, 21.283B tokens, avg100M 2.3244, avg250M 2.3246, avg500M 2.3276. W&B `ehcm9znb`. 8 GPUs, EP=1, microbatch=2, in-loop fast evals every 2000 steps. |
| 2026-06-07 | `olmoe3-moe-a0-1p2b-cx1-b256k-gpu8-ep1mb2-lr4e-4-r1` | `tiny_275m.py` | 1x | 262,144 | 32 | 4e-4 | `01KTG4JQ19A27ZC6H0FDD9661S` | https://beaker.org/ex/01KTG4JQ19A27ZC6H0FDD9661S | First 1.2B Cx1 sweep centered from updated 810M transfer rule. Finished step 81190, 21.283B tokens, avg100M 2.3106, avg250M 2.3108, avg500M 2.3139. Best completed Cx1 point. W&B `r9esbx26`. 8 GPUs, EP=1, microbatch=2, in-loop fast evals every 2000 steps. |
| 2026-06-07 | `olmoe3-moe-a0-1p2b-cx1-b256k-gpu8-ep1mb2-lr8e-4-r1` | `tiny_275m.py` | 1x | 262,144 | 32 | 8e-4 | `01KTG4K2MZHZNCYVG5K9RPV4SW` | https://beaker.org/ex/01KTG4K2MZHZNCYVG5K9RPV4SW | First 1.2B Cx1 sweep centered from updated 810M transfer rule. Finished step 81190, 21.283B tokens, avg100M 2.3145, avg250M 2.3148, avg500M 2.3181. W&B `eiuofxc6`. 8 GPUs, EP=1, microbatch=2, in-loop fast evals every 2000 steps. |
| 2026-06-07 | `olmoe3-moe-a0-810m-cx8-b768k-gpu8-ep1mb4-lr1e-4-r1` | `tiny_275m.py` | 8x | 786,432 | 96 | 1e-4 | `01KTHQWMSQ0A4P6RCNKPS7YPYD` | https://beaker.org/ex/01KTHQWMSQ0A4P6RCNKPS7YPYD | Cancelled intentionally on 2026-06-08 after switching to the new 3-point-centered LR policy. This was the extra cold-side insurance point for 810M Cx8; do not include in canonical full-run analysis unless later resumed and completed. 8 GPUs, EP=1, microbatch=4, in-loop fast evals every 2000 steps. |
| 2026-06-07 | `olmoe3-moe-a0-810m-cx8-b768k-gpu8-ep1mb4-lr2e-4-r1` | `tiny_275m.py` | 8x | 786,432 | 96 | 2e-4 | `01KTHQX04RMEK7C7V6DZRZVXM6` | https://beaker.org/ex/01KTHQX04RMEK7C7V6DZRZVXM6 | 810M Cx8 sweep queued while 1.2B Cx1 was near finish. Finished step 140394, 110.410B tokens, avg100M 2.1827, avg250M 2.1844, avg500M 2.1879. W&B `a0k0519k`. 8 GPUs, EP=1, microbatch=4, in-loop fast evals every 2000 steps. |
| 2026-06-07 | `olmoe3-moe-a0-810m-cx8-b768k-gpu8-ep1mb4-lr4e-4-r1` | `tiny_275m.py` | 8x | 786,432 | 96 | 4e-4 | `01KTHQXB575GS84FBP4SNZ1GAA` | https://beaker.org/ex/01KTHQXB575GS84FBP4SNZ1GAA | 810M Cx8 sweep queued while 1.2B Cx1 was near finish. Finished step 140394, 110.410B tokens, avg100M 2.1705, avg250M 2.1721, avg500M 2.1756. Best completed Cx8 point. W&B `dkpaicdc`. 8 GPUs, EP=1, microbatch=4, in-loop fast evals every 2000 steps. |
| 2026-06-07 | `olmoe3-moe-a0-810m-cx8-b768k-gpu8-ep1mb4-lr8e-4-r1` | `tiny_275m.py` | 8x | 786,432 | 96 | 8e-4 | `01KTHQXNN4MFDBAP490ACJTJ07` | https://beaker.org/ex/01KTHQXNN4MFDBAP490ACJTJ07 | 810M Cx8 sweep queued while 1.2B Cx1 was near finish. Finished step 140394, 110.410B tokens, avg100M 2.1752, avg250M 2.1768, avg500M 2.1803. Hot side is worse than `4e-4`, so Cx8 is bracketed. W&B `rhtrhhet`. 8 GPUs, EP=1, microbatch=4, in-loop fast evals every 2000 steps. |
| 2026-06-09 | `olmoe3-moe-a0-810m-cx16-b1m-gpu8-ep1mb4-lr2e-4-r1` | `tiny_275m.py` | 16x | 1,048,576 | 128 | 2e-4 | `01KTNWAFMP6A4PHYWZDD7B00AV` | https://beaker.org/ex/01KTNWAFMP6A4PHYWZDD7B00AV | 810M Cx16 3-point sweep centered from completed Cx1/Cx4/Cx8 LR-rule fit; Cx16 predicted optimum about 4.25e-4. Stopped intentionally on 2026-06-09 after the baseline plan shifted to finish Cx1/Cx2/Cx4/Cx8 first and reserve GPUs for 1.2B/midpoint follow-ups. Exclude from canonical LR fits unless explicitly resumed and completed. 8 GPUs, EP=1, microbatch=4, in-loop fast evals every 2000 steps. |
| 2026-06-09 | `olmoe3-moe-a0-810m-cx16-b1m-gpu8-ep1mb4-lr4e-4-r1` | `tiny_275m.py` | 16x | 1,048,576 | 128 | 4e-4 | `01KTNWATWQYX39RJVQE3141N8P` | https://beaker.org/ex/01KTNWATWQYX39RJVQE3141N8P | 810M Cx16 3-point sweep centered from completed Cx1/Cx4/Cx8 LR-rule fit; Cx16 predicted optimum about 4.25e-4. Stopped intentionally on 2026-06-09 after the baseline plan shifted to finish Cx1/Cx2/Cx4/Cx8 first and reserve GPUs for 1.2B/midpoint follow-ups. Exclude from canonical LR fits unless explicitly resumed and completed. 8 GPUs, EP=1, microbatch=4, in-loop fast evals every 2000 steps. |
| 2026-06-09 | `olmoe3-moe-a0-810m-cx16-b1m-gpu8-ep1mb4-lr8e-4-r1` | `tiny_275m.py` | 16x | 1,048,576 | 128 | 8e-4 | `01KTNWB6SW5H07ED0YC63S897X` | https://beaker.org/ex/01KTNWB6SW5H07ED0YC63S897X | 810M Cx16 3-point sweep centered from completed Cx1/Cx4/Cx8 LR-rule fit; Cx16 predicted optimum about 4.25e-4. Stopped while queued on 2026-06-09 after the baseline plan shifted to finish Cx1/Cx2/Cx4/Cx8 first and reserve GPUs for 1.2B/midpoint follow-ups. Exclude from canonical LR fits unless explicitly resumed and completed. 8 GPUs, EP=1, microbatch=4, in-loop fast evals every 2000 steps. |
| 2026-06-07 | `olmoe3-moe-a0-1p2b-cx4-b512k-gpu8-ep1mb2-lr1.5e-4-r1` | `tiny_275m.py` | 4x | 524,288 | 64 | 1.5e-4 | `01KTHW5XZXCNW9VV7FAMCS1C8F` | https://beaker.org/ex/01KTHW5XZXCNW9VV7FAMCS1C8F | 1.2B Cx4 sweep centered from completed 1.2B Cx1 plus updated 810M transfer rule. Finished step 162379, 85.133B tokens; repaired W&B history scan on 2026-06-10 saw full 85.133B tokens, avg100M 2.1655, avg250M 2.1654, avg500M 2.1679. W&B `5u5iumvr`. 8 GPUs, EP=1, microbatch=2, in-loop fast evals every 2000 steps. |
| 2026-06-07 | `olmoe3-moe-a0-1p2b-cx4-b512k-gpu8-ep1mb2-lr3e-4-r1` | `tiny_275m.py` | 4x | 524,288 | 64 | 3e-4 | `01KTHW68C59T1XE9WNFW3EP3G1` | https://beaker.org/ex/01KTHW68C59T1XE9WNFW3EP3G1 | 1.2B Cx4 sweep centered from completed 1.2B Cx1 plus updated 810M transfer rule. Finished step 162379, 85.133B tokens; repaired W&B history scan on 2026-06-10 saw avg100M 2.1500, avg250M 2.1508, avg500M 2.1531. W&B `rkjs2sze`. 8 GPUs, EP=1, microbatch=2, in-loop fast evals every 2000 steps. |
| 2026-06-07 | `olmoe3-moe-a0-1p2b-cx4-b512k-gpu8-ep1mb2-lr6e-4-r1` | `tiny_275m.py` | 4x | 524,288 | 64 | 6e-4 | `01KTHW6KH3XFR790J6J4G8ZAJ6` | https://beaker.org/ex/01KTHW6KH3XFR790J6J4G8ZAJ6 | 1.2B Cx4 sweep centered from completed 1.2B Cx1 plus updated 810M transfer rule. Finished step 162379, 85.133B tokens; repaired W&B history scan on 2026-06-10 saw avg100M 2.1549, avg250M 2.1548, avg500M 2.1573. W&B `1tzma107`. Close to, but worse than, `3e-4`; this is not a strict hot-side bracket, so the `1.2e-3` run was resumed. 8 GPUs, EP=1, microbatch=2, in-loop fast evals every 2000 steps. |
| 2026-06-07 | `olmoe3-moe-a0-1p2b-cx4-b512k-gpu8-ep1mb2-lr1.2e-3-r1` | `tiny_275m.py` | 4x | 524,288 | 64 | 1.2e-3 | `01KTHW6ZSXGD1P8NEA7S3KM198` | https://beaker.org/ex/01KTHW6ZSXGD1P8NEA7S3KM198 | Initially stopped on 2026-06-08 after switching to the 3-point-centered LR policy, but resumed on 2026-06-10 because `3e-4`/`6e-4` are too close to count as a strict hot-side bracket. New Beaker job attempt `01KTSB2H1TMF7Z1T2MY40J2QM0`; existing checkpoint folder contains `step10500`, so this should resume rather than restart. Include only after the resumed full run completes. 8 GPUs, EP=1, microbatch=2, in-loop fast evals every 2000 steps. |
| 2026-06-10 | `olmoe3-moe-a0-1p2b-cx8-b768k-gpu32-ep1mb1-lr2e-4-r1` | `tiny_275m.py` | 8x | 786,432 | 96 | 2e-4 | `01KTQZ0CTP2MBNHS1KT4BRGXEB` | https://beaker.org/ex/01KTQZ0CTP2MBNHS1KT4BRGXEB | 1.2B Cx8 3-point sweep centered from the validated 1.2B Cx4 transfer result. 4 nodes x 8 GPUs, EP=1, microbatch=1. Stopped intentionally on 2026-06-11 after the 4-node jobs showed only about 300 TFLOPs/GPU; replaced by one-node `gpu8-ep1mb4` `r2` below. Exclude this 4-node attempt from canonical LR analysis unless explicitly resumed and completed. |
| 2026-06-10 | `olmoe3-moe-a0-1p2b-cx8-b768k-gpu32-ep1mb1-lr4e-4-r1` | `tiny_275m.py` | 8x | 786,432 | 96 | 4e-4 | `01KTQZ0Q834TV6CCW5FZGKG12G` | https://beaker.org/ex/01KTQZ0Q834TV6CCW5FZGKG12G | 1.2B Cx8 3-point sweep centered from the validated 1.2B Cx4 transfer result. Kept running as the 4-node systems comparison point after the 4-node jobs showed only about 300 TFLOPs/GPU. Finished step 216505, 170.266B tokens, avg100M 2.0871, avg250M 2.0835, avg500M 2.0852. W&B `gbt7khqj`, skipped steps 0 at ~6.29B tokens. 4 nodes x 8 GPUs, EP=1, microbatch=1, in-loop fast evals every 2000 steps. Treat as systems-comparison data; prefer one-node `gpu8-ep1mb4` for canonical 1.2B Cx8. |
| 2026-06-10 | `olmoe3-moe-a0-1p2b-cx8-b768k-gpu32-ep1mb1-lr8e-4-r1` | `tiny_275m.py` | 8x | 786,432 | 96 | 8e-4 | `01KTQZ13TY5D8ZZFZ6AZ9CDYKT` | https://beaker.org/ex/01KTQZ13TY5D8ZZFZ6AZ9CDYKT | 1.2B Cx8 3-point sweep centered from the validated 1.2B Cx4 transfer result. 4 nodes x 8 GPUs, EP=1, microbatch=1. Stopped intentionally on 2026-06-11 after the 4-node jobs showed only about 300 TFLOPs/GPU; replaced by one-node `gpu8-ep1mb4` `r2` below. Exclude this 4-node attempt from canonical LR analysis unless explicitly resumed and completed. |
| 2026-06-11 | `olmoe3-moe-a0-1p2b-cx8-b768k-gpu8-ep1mb4-lr2e-4-r2` | `tiny_275m.py` | 8x | 786,432 | 96 | 2e-4 | `01KTWB5V3CBHWS868FKGBX342D` | https://beaker.org/ex/01KTWB5V3CBHWS868FKGBX342D | One-node replacement for the stopped 4-node `2e-4` attempt. 1 node x 8 GPUs, EP=1, microbatch=4, same 96-sequence global batch, in-loop fast evals every 2000 steps. Queued/created at launch. |
| 2026-06-11 | `olmoe3-moe-a0-1p2b-cx8-b768k-gpu8-ep1mb4-lr8e-4-r2` | `tiny_275m.py` | 8x | 786,432 | 96 | 8e-4 | `01KTWB65YRYR8K44RYXBZ7T5WJ` | https://beaker.org/ex/01KTWB65YRYR8K44RYXBZ7T5WJ | One-node replacement for the stopped 4-node `8e-4` attempt. 1 node x 8 GPUs, EP=1, microbatch=4, same 96-sequence global batch, in-loop fast evals every 2000 steps. Queued/created at launch. |
| 2026-06-07 | `olmoe3-moe-a0-810m-cx2-b512k-gpu8-ep1mb4-lr1.5e-4-r1` | `tiny_275m.py` | 2x | 524,288 | 64 | 1.5e-4 | `01KTHW7HB59AMPSZBP8FJHS5QG` | https://beaker.org/ex/01KTHW7HB59AMPSZBP8FJHS5QG | Paused intentionally on 2026-06-08 to free 8 GPUs for another user's urgent job. Resumed/requeued on 2026-06-08 with fresh job `01KTMJHQVXKD9FXCT3YTNQPVYJ`, started again at 2026-06-08 21:43 UTC, then paused again at 2026-06-08 22:00 UTC so the midpoint smoke could run first. Resumed/requeued again after midpoint smoke validation with fresh job `01KTMMFXYQ39FSYQ4JGHM91N53`. Finished step 52648, 27.603B tokens, avg100M 2.3544, avg250M 2.3608, avg500M 2.3589. W&B `fcqkb55w`. 8 GPUs, EP=1, microbatch=4, in-loop fast evals every 2000 steps. |
| 2026-06-07 | `olmoe3-moe-a0-810m-cx2-b512k-gpu8-ep1mb4-lr3e-4-r1` | `tiny_275m.py` | 2x | 524,288 | 64 | 3e-4 | `01KTHW7WY6Z2NFAP8FNT1HP3XN` | https://beaker.org/ex/01KTHW7WY6Z2NFAP8FNT1HP3XN | 810M Cx2 sweep queued after 1.2B Cx4 launch. Finished step 52648, 27.603B tokens, avg100M 2.3245, avg250M 2.3308, avg500M 2.3291. W&B `ogp6mrt6`. 8 GPUs, EP=1, microbatch=4, in-loop fast evals every 2000 steps. |
| 2026-06-07 | `olmoe3-moe-a0-810m-cx2-b512k-gpu8-ep1mb4-lr6e-4-r1` | `tiny_275m.py` | 2x | 524,288 | 64 | 6e-4 | `01KTHW88Q43J8M8CRCDN9VZDHV` | https://beaker.org/ex/01KTHW88Q43J8M8CRCDN9VZDHV | Stopped while queued on 2026-06-08 so another user's urgent 8-GPU job could go first. Resumed/requeued on 2026-06-08 with fresh job `01KTMJHR97JYCGJG05F5QQ555N`, then stopped again at 2026-06-08 22:00 UTC so the midpoint smoke could run first. Resumed/requeued again after midpoint smoke validation with fresh queued job `01KTMMFXNT05K156KYEGXFG7H6`. Finished step 52648, 27.603B tokens, avg100M 2.3096, avg250M 2.3160, avg500M 2.3144. Best completed Cx2 point so far. W&B `okb4e1u0`. 8 GPUs, EP=1, microbatch=4, in-loop fast evals every 2000 steps. |
| 2026-06-07 | `olmoe3-moe-a0-810m-cx2-b512k-gpu8-ep1mb4-lr1.2e-3-r1` | `tiny_275m.py` | 2x | 524,288 | 64 | 1.2e-3 | `01KTHW8MCKRJH3PW0W58KRVXA4` | https://beaker.org/ex/01KTHW8MCKRJH3PW0W58KRVXA4 | Stopped while queued on 2026-06-08 so another user's urgent 8-GPU job could go first. Resumed/requeued on 2026-06-08 with fresh job `01KTMJHRPEG8QHDWE2N16Z2QZR`, then stopped again at 2026-06-08 22:00 UTC so the midpoint smoke could run first. Resumed/requeued again after midpoint smoke validation with fresh queued job `01KTMMFXSJH84D3Q74PQ815VVS`. Finished step 52648, 27.603B tokens, avg100M 2.3112, avg250M 2.3176, avg500M 2.3161. Hot side is slightly worse than `6e-4`, so Cx2 is bracketed. W&B `d13uavyt`. 8 GPUs, EP=1, microbatch=4, in-loop fast evals every 2000 steps. |
| 2026-06-08 | `olmoe3-moe-a0-mid-480m-smoke-b256k-gpu4-ep1mb8-lr1.2e-3-r1` | `tiny_275m.py` | 0.02x | 262,144 | 32 | 1.2e-3 | `01KTMJY87YW09KHB4H6ERGZQ4K` | https://beaker.org/ex/01KTMJY87YW09KHB4H6ERGZQ4K | Midpoint baseline smoke for planned `mid_480m` rung. Failed before training because W&B group name exceeded the 128-character limit; no useful smoke signal. Relaunched with shorter name in `01KTMM5YQTGA9TXKFYMF5NPB46`. |
| 2026-06-08 | `m480-smoke-gpu4-ep1mb8-lr12-r2` | `tiny_275m.py` | 0.02x | 262,144 | 32 | 1.2e-3 | `01KTMM5YQTGA9TXKFYMF5NPB46` | https://beaker.org/ex/01KTMM5YQTGA9TXKFYMF5NPB46 | Short-name retry of midpoint baseline smoke. Passed startup: reached step 436, skipped steps 0, CE loss fell to about 4.5, and throughput stabilized around 632-644 TFLOPs/GPU with about 621 actual avg TFLOPs/GPU. Stopped intentionally after validation so queued ladder jobs could proceed. 4 GPUs, EP=1, microbatch=8, no in-loop evals. |
| 2026-06-08 | `m480-cx1-b256k-gpu4-ep1mb8-lr6e-4-r1` | `tiny_275m.py` | 1x | 262,144 | 32 | 6e-4 | `01KTMMJCV3818NDPK51R89MH08` | https://beaker.org/ex/01KTMMJCV3818NDPK51R89MH08 | Midpoint `mid_480m` Cx1 3-point sweep centered from transferred LR estimate after smoke validation. Finished step 29022, 7.608B tokens, avg100M 2.5696, avg250M 2.5676, avg500M 2.5725. W&B `56vuwauw`. 4 GPUs, EP=1, microbatch=8, in-loop fast evals every 2000 steps. |
| 2026-06-08 | `m480-cx1-b256k-gpu4-ep1mb8-lr1.2e-3-r1` | `tiny_275m.py` | 1x | 262,144 | 32 | 1.2e-3 | `01KTMMJSTMY3TSR7MHH5G7M22H` | https://beaker.org/ex/01KTMMJSTMY3TSR7MHH5G7M22H | Midpoint `mid_480m` Cx1 3-point sweep centered from transferred LR estimate after smoke validation. Finished step 29022, 7.608B tokens, avg100M 2.5653, avg250M 2.5636, avg500M 2.5690. Best completed Cx1 point. W&B `49mybsr0`. 4 GPUs, EP=1, microbatch=8, in-loop fast evals every 2000 steps. |
| 2026-06-08 | `m480-cx1-b256k-gpu4-ep1mb8-lr2.4e-3-r1` | `tiny_275m.py` | 1x | 262,144 | 32 | 2.4e-3 | `01KTMMK7VN9BXSCBYX2HQKQQWH` | https://beaker.org/ex/01KTMMK7VN9BXSCBYX2HQKQQWH | Midpoint `mid_480m` Cx1 3-point sweep centered from transferred LR estimate after smoke validation. Finished step 29022, 7.608B tokens, avg100M 2.5839, avg250M 2.5826, avg500M 2.5889. Hot side is worse than `1.2e-3`, so Cx1 is bracketed. W&B `7zz7c1zu`. 4 GPUs, EP=1, microbatch=8, in-loop fast evals every 2000 steps. |
| 2026-06-08 | `m480-cx2-b512k-gpu4-ep1mb8-lr3e-4-r1` | `tiny_275m.py` | 2x | 524,288 | 64 | 3e-4 | `01KTMMKN716ZSRZN473CV4BC23` | https://beaker.org/ex/01KTMMKN716ZSRZN473CV4BC23 | Midpoint `mid_480m` Cx2 3-point sweep centered from transferred LR estimate after smoke validation. Finished step 29022, 15.216B tokens, avg100M 2.4828, avg250M 2.4889, avg500M 2.4874. W&B `ridb7me5`. 4 GPUs, EP=1, microbatch=8, in-loop fast evals every 2000 steps. |
| 2026-06-08 | `m480-cx2-b512k-gpu4-ep1mb8-lr6e-4-r1` | `tiny_275m.py` | 2x | 524,288 | 64 | 6e-4 | `01KTMMM35QKDE15XCSKG76Z6ST` | https://beaker.org/ex/01KTMMM35QKDE15XCSKG76Z6ST | Midpoint `mid_480m` Cx2 3-point sweep centered from transferred LR estimate after smoke validation. Finished step 29022, 15.216B tokens, avg100M 2.4597, avg250M 2.4658, avg500M 2.4643. W&B `9bf5s9lf`. 4 GPUs, EP=1, microbatch=8, in-loop fast evals every 2000 steps. |
| 2026-06-08 | `m480-cx2-b512k-gpu4-ep1mb8-lr1.2e-3-r1` | `tiny_275m.py` | 2x | 524,288 | 64 | 1.2e-3 | `01KTMMMHBEV4JW3N0X4X3MFHK8` | https://beaker.org/ex/01KTMMMHBEV4JW3N0X4X3MFHK8 | Midpoint `mid_480m` Cx2 3-point sweep centered from transferred LR estimate after smoke validation. Finished step 29022, 15.216B tokens, avg100M 2.4519, avg250M 2.4580, avg500M 2.4567. Best completed Cx2 point so far, but on the high edge. W&B `roj7jv11`. 4 GPUs, EP=1, microbatch=8, in-loop fast evals every 2000 steps. |
| 2026-06-09 | `m480-cx2-b512k-gpu4-ep1mb8-lr2.4e-3-r1` | `tiny_275m.py` | 2x | 524,288 | 64 | 2.4e-3 | `01KTPWRQD0Z7SN3KEA6EBMTCB2` | https://beaker.org/ex/01KTPWRQD0Z7SN3KEA6EBMTCB2 | Midpoint `mid_480m` Cx2 high-side extension after the initial Cx2 triplet was monotonic and high-edge-best at `1.2e-3`. Finished step 29022, 15.216B tokens by W&B summary. W&B history is temporarily short: analyzer still sees only 8.577B tokens, avg100M 2.6475, avg250M 2.6518, avg500M 2.6612. Re-run analyzer before plotting/fitting this point. W&B `t8c3jnru`. Launched from `launch_moe_a0_mid_480m_cx2_followups.sh`. 4 GPUs, EP=1, microbatch=8, in-loop fast evals every 2000 steps. |
| 2026-06-09 | `m480-cx2-b512k-gpu4-ep1mb8-lr9.6e-3-r1` | `tiny_275m.py` | 2x | 524,288 | 64 | 9.6e-3 | `01KTQ3V5C3BJNDXHD38BV76KTH` | https://beaker.org/ex/01KTQ3V5C3BJNDXHD38BV76KTH | Midpoint `mid_480m` Cx2 far hot-side sentinel to quickly find the right-side turnover after the initial triplet was monotonic. Finished step 29022, 15.216B tokens by W&B summary. W&B history is temporarily short: analyzer still sees only 6.070B tokens, avg100M 2.9432, avg250M 2.9405, avg500M 2.9432. This is already clearly hot, but re-run analyzer before final plotting/fitting. W&B `q1n36aql`. Launched from `launch_moe_a0_mid_480m_cx2_followups.sh`. 4 GPUs, EP=1, microbatch=8, in-loop fast evals every 2000 steps. |
| 2026-06-08 | `m480-cx4-b512k-gpu4-ep1mb8-lr4e-4-r1` | `tiny_275m.py` | 4x | 524,288 | 64 | 4e-4 | `01KTMMMZ1539AV33SHB12S17Q4` | https://beaker.org/ex/01KTMMMZ1539AV33SHB12S17Q4 | Midpoint `mid_480m` Cx4 3-point sweep centered from transferred LR estimate after smoke validation. Finished step 58044, 30.432B tokens; refreshed history avg100M 2.3854, avg250M 2.3903, avg500M 2.3887. W&B `qtomqoti`. 4 GPUs, EP=1, microbatch=8, in-loop fast evals every 2000 steps. |
| 2026-06-08 | `m480-cx4-b512k-gpu4-ep1mb8-lr8e-4-r1` | `tiny_275m.py` | 4x | 524,288 | 64 | 8e-4 | `01KTMMNC9R56MX1MSGZQ865SXA` | https://beaker.org/ex/01KTMMNC9R56MX1MSGZQ865SXA | Midpoint `mid_480m` Cx4 3-point sweep centered from transferred LR estimate after smoke validation. Finished step 58044, 30.432B tokens; refreshed history avg100M 2.3739, avg250M 2.3788, avg500M 2.3772. Best completed Cx4 point; Cx4 is now bracketed by `4e-4` and `1.6e-3`. W&B `r5jfqoq8`. 4 GPUs, EP=1, microbatch=8, in-loop fast evals every 2000 steps. |
| 2026-06-08 | `m480-cx4-b512k-gpu4-ep1mb8-lr1.6e-3-r1` | `tiny_275m.py` | 4x | 524,288 | 64 | 1.6e-3 | `01KTMMNTA3NN9K4THQXCKGP717` | https://beaker.org/ex/01KTMMNTA3NN9K4THQXCKGP717 | Midpoint `mid_480m` Cx4 3-point sweep centered from transferred LR estimate after smoke validation. Finished step 58044, 30.432B tokens; refreshed history avg100M 2.3820, avg250M 2.3869, avg500M 2.3855. W&B `0sv8ymgk`. 4 GPUs, EP=1, microbatch=8, in-loop fast evals every 2000 steps. |
| 2026-06-10 | `m480-cx4-b512k-gpu4-ep1mb8-lr1e-4-r2` | `tiny_275m.py` | 4x | 524,288 | 64 | 1e-4 | `01KTSC4J4KGTZXY0XP5P0AXQXM` | https://beaker.org/ex/01KTSC4J4KGTZXY0XP5P0AXQXM | Midpoint `mid_480m` Cx4 cold-side sentinel launched before the stale W&B history refresh showed Cx4 was already bracketed. Finished step 58044, 30.432B tokens, avg100M 2.4642, avg250M 2.4689, avg500M 2.4673. Extra cold-side insurance point; W&B `0mvi3nov`. 4 GPUs, EP=1, microbatch=8, in-loop fast evals every 2000 steps. |
| 2026-06-10 | `m480-cx8-b768k-gpu8-ep1mb4-lr2e-4-r1` | `tiny_275m.py` | 8x | 786,432 | 96 | 2e-4 | `01KTQYY60VX9NNSHTS81CK4DZX` | https://beaker.org/ex/01KTQYY60VX9NNSHTS81CK4DZX | Midpoint `mid_480m` Cx8 3-point sweep centered from Cx4/transfer evidence. Finished step 77392, 60.864B tokens; avg100M 2.3437, avg250M 2.3428, avg500M 2.3449. W&B `o88mknat`. 8 GPUs total, EP=1, microbatch=4, in-loop fast evals every 2000 steps. |
| 2026-06-10 | `m480-cx8-b768k-gpu8-ep1mb4-lr4e-4-r1` | `tiny_275m.py` | 8x | 786,432 | 96 | 4e-4 | `01KTQYYHA1W5SE0Y955JBH17X9` | https://beaker.org/ex/01KTQYYHA1W5SE0Y955JBH17X9 | Midpoint `mid_480m` Cx8 3-point sweep centered from Cx4/transfer evidence. Finished step 77392, 60.864B tokens; avg100M 2.3172, avg250M 2.3163, avg500M 2.3185. W&B `pochsiqs`. 8 GPUs total, EP=1, microbatch=4, in-loop fast evals every 2000 steps. |
| 2026-06-10 | `m480-cx8-b768k-gpu8-ep1mb4-lr8e-4-r1` | `tiny_275m.py` | 8x | 786,432 | 96 | 8e-4 | `01KTQYYW8GRSTHMBNH5WA69Q7H` | https://beaker.org/ex/01KTQYYW8GRSTHMBNH5WA69Q7H | Midpoint `mid_480m` Cx8 3-point sweep centered from Cx4/transfer evidence. Finished step 77392, 60.864B tokens; avg100M 2.3085, avg250M 2.3076, avg500M 2.3099. Best completed Cx8 point so far, on the hot edge. W&B `ehjbiul4`. 8 GPUs total, EP=1, microbatch=4, in-loop fast evals every 2000 steps. |
| 2026-06-10 | `m480-cx8-b768k-gpu8-ep1mb4-lr3.2e-3-r2` | `tiny_275m.py` | 8x | 786,432 | 96 | 3.2e-3 | `01KTSC51ZXE3YQAZMANDP15QT7` | https://beaker.org/ex/01KTSC51ZXE3YQAZMANDP15QT7 | Midpoint `mid_480m` Cx8 hot-side sentinel after `8e-4` beat `4e-4` among finished points. Finished step 77392, 60.864B tokens, avg100M 2.3494, avg250M 2.3486, avg500M 2.3510. This brackets Cx8 on the hot side; W&B `fvbz0h7v`. 8 GPUs total, EP=1, microbatch=4, in-loop fast evals every 2000 steps. |
| 2026-06-05 | `olmoe3-eval-275m-cx1-lr1e-3-r2` | `tiny_275m.py` | eval-only | n/a | n/a | n/a | `01KTD0JFDAE3CXYKZV85CYVRYC` | https://beaker.org/ex/01KTD0JFDAE3CXYKZV85CYVRYC | Eval backfill smoke over checkpoint `/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx1-b256k-gpu2-ep1mb16-lr1e-3-r2/step15365`. Finished successfully with 178 W&B eval keys, including downstream BPB/BPB v2 metrics and LM validation components. Created from launcher `launch_moe_a0_eval_backfill_smoke.sh`; ignore for LR U-plots. |
| 2026-06-05 | `olmoe3-275m-cx1-evaltest-lr1e-3` | `tiny_275m.py` | 1x | 262,144 | 32 | 1e-3 | `01KTD0K416CRVYAN5CN5SQPK5Y` | https://beaker.org/ex/01KTD0K416CRVYAN5CN5SQPK5Y | 275M Cx1 ladder-eval training test. 2 GPUs, EP=1, microbatch=16, `--ladder-evals --eval-task-set=fast --eval-interval=100`. Passed startup and logged 178 W&B eval keys by step 197 with skipped steps 0, but the 100-step eval cadence made the run too slow. Stopped intentionally on 2026-06-06 after proving the eval hook. Ignore for canonical LR-selection unless we explicitly promote it. Future in-loop eval training should use interval 2000 plus final eval/checkpoint. |
| 2026-06-04 | `olmoe3-moe-a0-1p2b-smoke-b256k-gpu4-ep1mb4-lr3e-4-r3` | `tiny_275m.py` | 0.02x | 262,144 | 32 | 3e-4 | `01KT841MWJKFK5KWCAXCGA9WC1` | https://beaker.org/ex/01KT841MWJKFK5KWCAXCGA9WC1 | Succeeded. Preferred 1.2B setting: 4 GPUs, EP=1, microbatch=4; skipped steps 0, about 662 actual avg TFLOPs/GPU. |
| 2026-06-04 | `olmoe3-moe-a0-1p2b-smoke-b256k-gpu4-ep2mb4-lr3e-4-r3` | `tiny_275m.py` | 0.02x | 262,144 | 32 | 3e-4 | `01KT8420RGXWJ8C3JFCNG67W2T` | https://beaker.org/ex/01KT8420RGXWJ8C3JFCNG67W2T | Succeeded but slower than EP=1. Keep as memory fallback only; skipped steps 0, about 607 actual avg TFLOPs/GPU. |
| 2026-06-04 | `olmoe3-tiny-275m-cx8-b768k-gpu4-ep1mb8-lr2e-4` | `tiny_275m.py` | 8x | 786,432 | 96 | 2e-4 | `01KT8445FT6GZFKPE7JKS3F8RY` | https://beaker.org/ex/01KT8445FT6GZFKPE7JKS3F8RY | Replacement coarse factor-of-two Cx8 LR grid. 4 GPUs, EP=1, microbatch=8. |
| 2026-06-04 | `olmoe3-tiny-275m-cx8-b768k-gpu4-ep1mb8-lr4e-4` | `tiny_275m.py` | 8x | 786,432 | 96 | 4e-4 | `01KT844H6AQWZNNSXJZRV258VZ` | https://beaker.org/ex/01KT844H6AQWZNNSXJZRV258VZ | Replacement coarse factor-of-two Cx8 LR grid. 4 GPUs, EP=1, microbatch=8. |
| 2026-06-04 | `olmoe3-tiny-275m-cx8-b768k-gpu4-ep1mb8-lr8e-4` | `tiny_275m.py` | 8x | 786,432 | 96 | 8e-4 | `01KT844XB38AM31SCZDSTA1EAA` | https://beaker.org/ex/01KT844XB38AM31SCZDSTA1EAA | Replacement coarse factor-of-two Cx8 LR grid. 4 GPUs, EP=1, microbatch=8. |
| 2026-06-04 | `olmoe3-tiny-275m-cx8-b768k-gpu4-ep1mb8-lr1.6e-3` | `tiny_275m.py` | 8x | 786,432 | 96 | 1.6e-3 | `01KT84589PJ0VDVSS7CDQPBSCV` | https://beaker.org/ex/01KT84589PJ0VDVSS7CDQPBSCV | Replacement coarse factor-of-two Cx8 LR grid. 4 GPUs, EP=1, microbatch=8. |
| 2026-06-04 | `olmoe3-tiny-275m-cx16-b1m-gpu4-ep1mb16-lr1e-4` | `tiny_275m.py` | 16x | 1,048,576 | 128 | 1e-4 | `01KT845KM5CF6JZHJB6KW6WARW` | https://beaker.org/ex/01KT845KM5CF6JZHJB6KW6WARW | Replacement coarse factor-of-two Cx16 LR grid. 4 GPUs, EP=1, microbatch=16. |
| 2026-06-04 | `olmoe3-tiny-275m-cx16-b1m-gpu4-ep1mb16-lr2e-4` | `tiny_275m.py` | 16x | 1,048,576 | 128 | 2e-4 | `01KT845WN987DZTN03Q7NSXAK4` | https://beaker.org/ex/01KT845WN987DZTN03Q7NSXAK4 | Replacement coarse factor-of-two Cx16 LR grid. 4 GPUs, EP=1, microbatch=16. |
| 2026-06-04 | `olmoe3-tiny-275m-cx16-b1m-gpu4-ep1mb16-lr4e-4` | `tiny_275m.py` | 16x | 1,048,576 | 128 | 4e-4 | `01KT8466QCKVK2WDKW7F75TK9H` | https://beaker.org/ex/01KT8466QCKVK2WDKW7F75TK9H | Replacement coarse factor-of-two Cx16 LR grid. 4 GPUs, EP=1, microbatch=16. |
| 2026-06-04 | `olmoe3-tiny-275m-cx16-b1m-gpu4-ep1mb16-lr8e-4` | `tiny_275m.py` | 16x | 1,048,576 | 128 | 8e-4 | `01KT846JGMA8TDYZGGH4E34K3P` | https://beaker.org/ex/01KT846JGMA8TDYZGGH4E34K3P | Replacement coarse factor-of-two Cx16 LR grid. 4 GPUs, EP=1, microbatch=16. |

Status update for the replacement Cx8/Cx16 grids above: all eight jobs failed
before completion when `/weka/oe-training-default` filled up during checkpoint
writes. Their intermediate checkpoints were deleted during cleanup, so they
cannot be resumed. Treat them as partial-only W&B curves and exclude them from
completed-run U-plots and final LR rules.

Accidental relaunches from commit `bdd30f9` were stopped quickly after we
realized the deleted checkpoint dirs prevented resume. Ignore these jobs and
their W&B runs for analysis:

- Cx8 `2e-4`: `01KT8HHKES4H7D59RY9WB80MNJ`
- Cx8 `4e-4`: `01KT8HHZPVWKAHB2MCTJFFZA4H`
- Cx8 `8e-4`: `01KT8HJB8JPBEKHFCD9EZCXJQZ`
- Cx8 `1.6e-3`: `01KT8HJPVC15B839NFH1W17G5B`
- Cx16 `1e-4`: `01KT8HK34D5T5JPEN454R2NZR2`
- Cx16 `2e-4`: `01KT8HKFBYQC1MXAGJXDBP96FQ`
- Cx16 `4e-4`: `01KT8HKT4P0ZB8QXDPP57Y6MDP`
- Cx16 `8e-4`: `01KT8HM745W5A1J1Y3TWRZ3QT0`

Fresh `r2` reruns from commit `2cfd4c56` use final-only permanent checkpoints
plus latest ephemeral checkpoints. These are the canonical Cx8/Cx16 completion
runs unless a job fails and is explicitly replaced:

- Cx8 `2e-4`, `gpu4-ep1mb8`: `01KT8JPNQTTSQFDCGNV7HT8VV1`
- Cx8 `4e-4`, `gpu4-ep1mb8`: `01KT8JQ0V85RVSY309P3BXQ85Y`
- Cx8 `6e-4`, `gpu4-ep1mb8`: `01KT8JQCM49JRFVNMT7WRV701V`
- Cx8 `8e-4`, `gpu4-ep1mb8`: `01KT8JQR750TFJKE13ZXY7JYTT`
- Cx16 `2e-4`, `gpu8-ep1mb16`: `01KT8JR3WKXCR6TN8897A57DHS`
- Cx16 `4e-4`, `gpu8-ep1mb16`: `01KT8JRFSG3J7AJ5PV7E32Z46K`
- Cx16 `6e-4`, `gpu8-ep1mb16`: `01KT8JRVAG6RVGT231477NGQD9`

Status update: the Cx16 `2e-4` `r2` experiment
(`01KT8JR3WKXCR6TN8897A57DHS`) had its first job fail at step ~4507 with a
CUDA/NCCL watchdog abort, not a storage-full checkpoint failure. The experiment
was resumed in Beaker so it can pick up from the latest checkpoint:

- Cx16 `2e-4`, `gpu8-ep1mb16`, `r2` resumed job:
  `01KT8P9WZJ20XGTY44BH38M9W2`

An accidental fresh-from-scratch replacement was briefly launched and stopped
early; ignore it for analysis:

- Cx16 `2e-4`, `gpu8-ep1mb16`, `r3`: `01KT8NJ55CHAKYCG1E1J7Q9QBJ`

Status update: the four Cx8 `r2` runs finished successfully on 2026-06-04:

- Cx8 `2e-4`: finished, step 40971, 32.221B tokens, avg250M 2.5429,
  avg500M 2.5422.
- Cx8 `4e-4`: finished, step 40971, 32.221B tokens, avg250M 2.5092,
  avg500M 2.5085.
- Cx8 `6e-4`: finished, step 40971, 32.221B tokens, avg250M 2.4978,
  avg500M 2.4972.
- Cx8 `8e-4`: finished, step 40971, 32.221B tokens, avg250M 2.4909,
  avg500M 2.4903.
- Cx8 `1.6e-3`: finished, step 40971, 32.221B tokens, avg250M 2.4864,
  avg500M 2.4859.

The best completed Cx8 point was initially the high-edge `1.6e-3` run, so
launched high-side extensions:

- Cx8 `1.6e-3`, `gpu4-ep1mb8`, `r2`: `01KT9D6W9F4RGA5RSA8XSSMEP3`
- Cx8 `3.2e-3`, `gpu4-ep1mb8`, `r2`: `01KT9Q661N0YHYHC9A9T9AGV1J`;
  stopped intentionally after Cx8 `1.6e-3` was already clearly worse than the
  completed `8e-4` best. Ignore for full-run analysis.
- Cx8 `3.2e-3`, `gpu4-ep1mb8`, `r3`: `01KTAA55V6QXN45QZFBHTY6B65`;
  launched after the full `1.6e-3` run finished better than `8e-4`, leaving the
  rung high-edge-best. Finished successfully on 2026-06-05 at step 40971,
  32.221B tokens, avg250M 2.4987, avg500M 2.4982. This brackets Cx8 on the
  right side; best observed remains `1.6e-3`.
- Cx8 `6.4e-3`, `gpu4-ep1mb8`, `r2`: `01KT9Q6HX5X6KFW5RD1VSC9BV4`;
  stopped intentionally after lower high-side probes were already clearly worse.
  Ignore for full-run analysis.
- Cx8 `6.4e-3`, `gpu4-ep1mb8`, `r3`: `01KTAC584AMG9645NEKF479R15`;
  far right-side sentinel launched after adopting the policy of occasionally
  jumping farther when a completed curve is still monotonically improving at the
  high edge. Finished successfully on 2026-06-05 at step 40971, 32.221B
  tokens, avg250M 2.5347, avg500M 2.5341.
- Cx8 `1.6e-2`, `gpu4-ep1mb8`, `sentinel`: `01KTACFJ4D4FQG33ZPT4R306WT`;
  true order-of-magnitude sentinel to quickly find a right-side upturn. Finished
  successfully on 2026-06-05 at step 40971, 32.221B tokens, avg250M 2.6285,
  avg500M 2.6278.

Status update: the canonical Cx16 `r2` runs finished successfully on 2026-06-04.

- Cx16 `2e-4`: finished after Beaker resume, step 61457, 64.442B tokens,
  avg250M 2.4759, avg500M 2.4744.
- Cx16 `4e-4`: finished, step 61457, 64.442B tokens, avg250M 2.4474,
  avg500M 2.4461.
- Cx16 `6e-4`: finished, step 61457, 64.442B tokens, avg250M 2.4367,
  avg500M 2.4354.
- Cx16 `1.2e-3`: finished, step 61457, 64.442B tokens, avg250M 2.4301,
  avg500M 2.4288.

The best completed Cx16 point is the high-edge `1.2e-3` run, so Cx16 is not yet
bracketed. Launched high-side extensions:

- Cx16 `1.2e-3`, `gpu8-ep1mb16`, `r2`: `01KT9H6XQJ2GEMKPKHKPCED5B1`
- Cx16 `2.4e-3`, `gpu8-ep1mb16`, `r2`: `01KT9Q6X0B6PG3G6ZSBZGTPSVQ`;
  stopped intentionally after Cx16 `1.2e-3` was already clearly worse than the
  completed `6e-4` best. Ignore for full-run analysis.
- Cx16 `2.4e-3`, `gpu8-ep1mb16`, `r3`: `01KTAC763FP2W34ZX6N4CT21QD`;
  far right-side sentinel launched after `1.2e-3` improved enough in flight to
  make Cx16 plausibly high-edge-best. Finished successfully on 2026-06-05 at
  step 61457, 64.442B tokens, avg250M 2.4413, avg500M 2.4400.
- Cx16 `4.8e-3`, `gpu8-ep1mb16`, `r2`: `01KT9Q774FWC6NZDSGTD0Y2W7K`;
  stopped intentionally while queued after lower high-side probes were already
  clearly worse. Ignore for full-run analysis.
- Cx16 `6e-3`, `gpu8-ep1mb16`, `sentinel`: `01KTACHG3Z4Y1G9HW9ESYZK58Q`;
  true order-of-magnitude sentinel to quickly find a right-side upturn. Finished
  successfully on 2026-06-05 at step 61457, 64.442B tokens, avg250M 2.4876,
  avg500M 2.4862.

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

Launched run names:

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

## Eval Backfills Launched 2026-06-06

The eval smoke `olmoe3-eval-275m-cx1-lr1e-3-r2`
(`01KTD0JFDAE3CXYKZV85CYVRYC`) proved that final-checkpoint eval backfills work.
After that, the in-loop eval-test training run
`olmoe3-275m-cx1-evaltest-lr1e-3`
(`01KTD0K416CRVYAN5CN5SQPK5Y`) was stopped intentionally because the temporary
`--eval-interval=100` setting made it too slow. Future in-loop eval training
should use `--eval-interval=2000` plus the final checkpoint eval.

The jobs below are eval-only, tagged `eval-backfill`, and should be excluded
from LR U-plots and training-loss analysis. All 32 finished on 2026-06-06, and
their 180 eval-related W&B summary metrics were copied onto the corresponding
source training W&B run summaries with `eval_backfill/...` metadata.

| Name | Beaker experiment | Source checkpoint family |
| --- | --- | --- |
| `eval-275m-275-01` | https://beaker.org/ex/01KTDSRH22CQEQK9N3MPSHK5C6 | 275M Cx1 `1.2e-3-r2`, final checkpoint |
| `eval-275m-275-02` | https://beaker.org/ex/01KTDSRY3TWWNV7GSK4N04X0V5 | 275M Cx1 `1.5e-3-r2`, final checkpoint |
| `eval-275m-275-03` | https://beaker.org/ex/01KTDSSAC83NWFMTEJQR0QZHMN | 275M Cx1 `1e-3-r2`, final checkpoint |
| `eval-275m-275-04` | https://beaker.org/ex/01KTDSSPBC9T7Z5NWBYP8KJH4H | 275M Cx1 `2e-3-r2`, final checkpoint |
| `eval-275m-275-05` | https://beaker.org/ex/01KTDST1YNTWWX8W33J9EXZTA1 | 275M Cx1 `8e-4-r2`, final checkpoint |
| `eval-275m-275-06` | https://beaker.org/ex/01KTDSTDSPTPARPCJEQH1N6F5T | 275M Cx16 `1.2e-3-r2`, final checkpoint |
| `eval-275m-275-07` | https://beaker.org/ex/01KTDSTSPH5P839P0D2T1MDMPK | 275M Cx16 `2.4e-3-r3`, final checkpoint |
| `eval-275m-275-08` | https://beaker.org/ex/01KTDSV5REYVM86SMEGFJQW56X | 275M Cx16 `2e-4-r2`, final checkpoint |
| `eval-275m-275-09` | https://beaker.org/ex/01KTDSVH6SM4YSQ8RTK8A7GRWE | 275M Cx16 `4e-4-r2`, final checkpoint |
| `eval-275m-275-10` | https://beaker.org/ex/01KTDSVX1NZFSSR847CDBTEB76 | 275M Cx16 `6e-3-sentinel`, final checkpoint |
| `eval-275m-275-11` | https://beaker.org/ex/01KTDSWA6D2VCFG5A0F989V08H | 275M Cx16 `6e-4-r2`, final checkpoint |
| `eval-275m-275-12` | https://beaker.org/ex/01KTDSWPR0GYAWWVRQAX0ZJPX7 | 275M Cx2 `1e-3`, final checkpoint |
| `eval-275m-275-13` | https://beaker.org/ex/01KTDSX27M9FJ6B6PXW7D65JK4 | 275M Cx2 `6e-4-r2`, final checkpoint |
| `eval-275m-275-14` | https://beaker.org/ex/01KTDSXE8J4YHX0K1RX7K9NZQ0 | 275M Cx2 `8e-4-r2`, final checkpoint |
| `eval-275m-275-15` | https://beaker.org/ex/01KTDSXSX5PRXM01J6M918KDBF | 275M Cx4 `1.5e-3`, final checkpoint |
| `eval-275m-275-16` | https://beaker.org/ex/01KTDSY6DTECEGRQ3ZGRRQE2RR | 275M Cx4 `1e-3`, final checkpoint |
| `eval-275m-275-17` | https://beaker.org/ex/01KTDSYJCDCBZ6MKDVJ3PM12A9 | 275M Cx4 `2.5e-3`, final checkpoint |
| `eval-275m-275-18` | https://beaker.org/ex/01KTDSYXN8MNT2517TFM9K2SYA | 275M Cx8 `1.6e-2-sentinel`, final checkpoint |
| `eval-275m-275-19` | https://beaker.org/ex/01KTDSZ9NEW1SKAZCMY2W6PWJA | 275M Cx8 `1.6e-3-r2`, final checkpoint |
| `eval-275m-275-20` | https://beaker.org/ex/01KTDSZNERKPY0N0WVBC8XMJWT | 275M Cx8 `2e-4-r2`, final checkpoint |
| `eval-275m-275-21` | https://beaker.org/ex/01KTDT015E5PXQG3HCRD27EYA4 | 275M Cx8 `3.2e-3-r3`, final checkpoint |
| `eval-275m-275-22` | https://beaker.org/ex/01KTDT0E3TX5HJ55E9PR9CZ4QW | 275M Cx8 `4e-4-r2`, final checkpoint |
| `eval-275m-275-23` | https://beaker.org/ex/01KTDT0TMXNWV7K2HSGRZ1TF8F | 275M Cx8 `6.4e-3-r3`, final checkpoint |
| `eval-275m-275-24` | https://beaker.org/ex/01KTDT17B1M54RPSKRS23K916Q | 275M Cx8 `6e-4-r2`, final checkpoint |
| `eval-275m-275-25` | https://beaker.org/ex/01KTDT1JPCZTJ7Y7282MAP9PJN | 275M Cx8 `8e-4-r2`, final checkpoint |
| `eval-810m-810-01` | https://beaker.org/ex/01KTDT1XT861M6E2W34A1F5Z7A | 810M Cx1 `5e-5-cs-r2`, final checkpoint |
| `eval-810m-810-02` | https://beaker.org/ex/01KTDT2AKPF46Y2T3M8H5G8VRS | 810M Cx1 `1.2e-3-r1`, final checkpoint |
| `eval-810m-810-03` | https://beaker.org/ex/01KTDT2NYWRZBCW8XAYB80YYC3 | 810M Cx1 `1.5e-4-cold-r1`, final checkpoint |
| `eval-810m-810-04` | https://beaker.org/ex/01KTDT31TADK1X5SQFWCNAA41J | 810M Cx1 `2.4e-3-r1`, final checkpoint |
| `eval-810m-810-05` | https://beaker.org/ex/01KTDT3DMNDCWEXPTCM3H84D1G | 810M Cx1 `3e-4-cold-r1`, final checkpoint |
| `eval-810m-810-06` | https://beaker.org/ex/01KTDT3RA3DWXEZF68B6234A4B | 810M Cx1 `6e-3-r1`, final checkpoint |
| `eval-810m-810-07` | https://beaker.org/ex/01KTDT41V1R22KWH57NTHCBKVF | 810M Cx1 `6e-4-r1`, final checkpoint |

## 810M Cx4 Eval Backfills Launched 2026-06-06

The Cx4 training runs did not have in-loop evals enabled. Eval-only final
checkpoint backfills are copied back to the source training W&B summaries with
`copy_eval_backfills_to_wandb.py` as they finish. The `2e-4`, `4e-4`, and
`8e-4` backfills have finished and were copied back on 2026-06-06. The
`1.6e-3` backfill finished and was copied back on 2026-06-07.

| Name | Beaker experiment | Source checkpoint |
| --- | --- | --- |
| `eval-810m-cx4-lr2e-4-r1` | https://beaker.org/ex/01KTFFM30FQYJ6RR76826V58QM | `olmoe3-moe-a0-810m-cx4-b512k-gpu8-ep1mb4-lr2e-4-r1/step105295` |
| `eval-810m-cx4-lr4e-4-r1` | https://beaker.org/ex/01KTFFMFKM2D7FR4EMRKEWXGFP | `olmoe3-moe-a0-810m-cx4-b512k-gpu8-ep1mb4-lr4e-4-r1/step105295` |
| `eval-810m-cx4-lr8e-4-r1` | https://beaker.org/ex/01KTFK9M9SP2EZ7DRY4VHZCHSA | `olmoe3-moe-a0-810m-cx4-b512k-gpu8-ep1mb4-lr8e-4-r1/step105295` |
| `eval-810m-cx4-lr1.6e-3-r1` | https://beaker.org/ex/01KTG4AXWFBSNYPTAK4JE9FFTS | `olmoe3-moe-a0-810m-cx4-b512k-gpu8-ep1mb4-lr1.6e-3-r1/step105295` |

## Expert Granularity Experiment Launched 2026-06-11

Experiment plan:
`src/scripts/train/jacobm_olmoe_ladder/experiment_plans/expert_granularity.md`.

Code support:

- `--expert-geometry={baseline_48e_top4,coarse_24e_top2,fine_96e_top8}`
- launchers under
  `src/scripts/train/jacobm_olmoe_ladder/experiments/expert_granularity/`

Dry-run counts at 275M:

- `coarse_24e_top2`: ~0.28B active / ~0.20B active non-embedding / ~1.13B total
- `fine_96e_top8`: ~0.28B active / ~0.20B active non-embedding / ~1.14B total

Smoke tests:

| Name | Beaker experiment | W&B | Status | Notes |
| --- | --- | --- | --- | --- |
| `eg-smoke-eg24e2k-lr2e-3-r2` | https://beaker.org/ex/01KTT5PNBZ3THN3CF1R729PB18 | `deu60y28` | finished cleanly | 275M Cx1 smoke, batch 262,144 / 32 seqs, 1 GPU, EP=1, microbatch=16, skipped steps 0. |
| `eg-smoke-eg96e8k-lr2e-3-r2` | https://beaker.org/ex/01KTT5Q0E969CCFESR0SNJ4QE6 | `8cheit54` | failed | OOM during trainer dry-run at 1 GPU, EP=1, microbatch=16. Exclude from analysis. |
| `eg-smoke-eg96e8k-lr2e-3-r4` | https://beaker.org/ex/01KTT5ZNZ3Z748YXGSNZHSEXC2 | `hh2qh4i9` | finished cleanly | Retry with 1 GPU, EP=1, microbatch=8. Finished step 308, 80.740M tokens, skipped steps 0. |

Accidental/ambiguous smoke jobs stopped before use:

- `01KTT5KMH3JD0QV1R1ETB6F1PZ`: launched before expert-geometry code was committed.
- `01KTT5M01CG1DF523DVJZFKZ0H`: launched before expert-geometry code was committed.
- `01KTT5VBMDWBCR3085PZHQ71ZB`: accidental duplicate coarse smoke; stopped.
- `01KTT5VQ2V955SZ5033FMHGH4W`: accidental fine smoke still using mb16; stopped.

275M Cx1 transfer probes:

| Name | Variant | LR | Batch tokens | Batch seqs | GPUs | EP | Microbatch | Beaker experiment | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `eg-275m-cx1-eg24e2k-lr1e-3-r1` | `coarse_24e_top2` | 1e-3 | 262,144 | 32 | 1 | 1 | 16 | https://beaker.org/ex/01KTT68X1E45RNN2M5N250NNE1 | Finished step 15349, 4.024B tokens, avg100M 2.7834, avg250M 2.7873, avg500M 2.7894. W&B `6o5xk53j`. |
| `eg-275m-cx1-eg24e2k-lr2e-3-r1` | `coarse_24e_top2` | 2e-3 | 262,144 | 32 | 1 | 1 | 16 | https://beaker.org/ex/01KTT697TSG6X0JTD316YN5BD6 | Finished step 15349, 4.024B tokens, avg100M 2.7769, avg250M 2.7814, avg500M 2.7849. W&B `ndkmprhm`. Best coarse Cx1 point. |
| `eg-275m-cx1-eg24e2k-lr4e-3-r1` | `coarse_24e_top2` | 4e-3 | 262,144 | 32 | 1 | 1 | 16 | https://beaker.org/ex/01KTT69KTCH2FXME9E63T11DM3 | Finished step 15349, 4.024B tokens, avg100M 2.7850, avg250M 2.7904, avg500M 2.7965. W&B `8eoup1kb`. Coarse Cx1 is bracketed. |
| `eg-275m-cx1-eg96e8k-lr1e-3-r1` | `fine_96e_top8` | 1e-3 | 262,144 | 32 | 1 | 1 | 8 | https://beaker.org/ex/01KTT69YPF7F1GAQH4ZPM6JT1A | Finished step 15396, 4.036B tokens, avg100M 2.7675, avg250M 2.7683, avg500M 2.7722. W&B `bpdo8b6b`. Uses mb8 fallback after mb16 OOM. |
| `eg-275m-cx1-eg96e8k-lr2e-3-r1` | `fine_96e_top8` | 2e-3 | 262,144 | 32 | 1 | 1 | 8 | https://beaker.org/ex/01KTT6A9WYW6Y3RB31X433F87X | Finished step 15396, 4.036B tokens, avg100M 2.7628, avg250M 2.7641, avg500M 2.7695. W&B `lyky04e5`. Best fine Cx1 point. |
| `eg-275m-cx1-eg96e8k-lr4e-3-r1` | `fine_96e_top8` | 4e-3 | 262,144 | 32 | 1 | 1 | 8 | https://beaker.org/ex/01KTT6AN3CMWPT5E9NGWZJH3DX | Finished step 15396, 4.036B tokens, avg100M 2.7652, avg250M 2.7673, avg500M 2.7750. W&B `4l7axfwy`. Fine Cx1 is bracketed. |

275M Cx4 baseline-centered probes:

These were queued before the Cx1 probes finished so the overnight queue can keep
working. They are centered around the 275M baseline Cx4 optimum region. If the
variant-specific LR multiplier differs from 1.0, extend each curve with at most
one targeted follow-up after the first three Cx4 jobs finish.

| Name | Variant | LR | Batch tokens | Batch seqs | GPUs | EP | Microbatch | Beaker experiment | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `eg-275m-cx4-eg24e2k-lr8e-4-r1` | `coarse_24e_top2` | 8e-4 | 524,288 | 64 | 4 | 1 | 16 | https://beaker.org/ex/01KTT7A7ZS7EV70TR3XBJMSAVG | Finished step 30698, 16.095B tokens, avg100M 2.5758, avg250M 2.5796, avg500M 2.5822. W&B `5talvqd1`. |
| `eg-275m-cx4-eg24e2k-lr1.6e-3-r1` | `coarse_24e_top2` | 1.6e-3 | 524,288 | 64 | 4 | 1 | 16 | https://beaker.org/ex/01KTT7AJT93MT6DW89FE1A04RE | Finished step 30698, 16.095B tokens, avg100M 2.5676, avg250M 2.5713, avg500M 2.5740. W&B `eq0vqyj9`. Best coarse Cx4 point. |
| `eg-275m-cx4-eg24e2k-lr3.2e-3-r1` | `coarse_24e_top2` | 3.2e-3 | 524,288 | 64 | 4 | 1 | 16 | https://beaker.org/ex/01KTT7AYPX74Y4QG4C6GWGX3A0 | Finished step 30698, 16.095B tokens, avg100M 2.5768, avg250M 2.5805, avg500M 2.5833. W&B `mrpoyk8n`. Coarse Cx4 is bracketed. |
| `eg-275m-cx4-eg96e8k-lr8e-4-r1` | `fine_96e_top8` | 8e-4 | 524,288 | 64 | 4 | 1 | 8 | https://beaker.org/ex/01KTT7B9XNDT2D578HYR8VBVYA | Finished step 30791, 16.143B tokens, avg100M 2.5577, avg250M 2.5582, avg500M 2.5599. W&B `0vr98te9`. |
| `eg-275m-cx4-eg96e8k-lr1.6e-3-r1` | `fine_96e_top8` | 1.6e-3 | 524,288 | 64 | 4 | 1 | 8 | https://beaker.org/ex/01KTT7BMTTSW6VAY20FM8XKVWB | Finished step 30791, 16.143B tokens, avg100M 2.5516, avg250M 2.5523, avg500M 2.5541. W&B `gsqree2x`. Best fine Cx4 point. |
| `eg-275m-cx4-eg96e8k-lr3.2e-3-r1` | `fine_96e_top8` | 3.2e-3 | 524,288 | 64 | 4 | 1 | 8 | https://beaker.org/ex/01KTT7C0KBTY8TVNC361Z2G3KV | Finished step 30791, 16.143B tokens, avg100M 2.5623, avg250M 2.5629, avg500M 2.5650. W&B `589cgpj0`. Fine Cx4 is bracketed; uses mb8 fallback after mb16 OOM at Cx1 smoke. |

275M Cx2/Cx8 transferred-LR probes:

These were approved after the Cx1/Cx4 curves showed clean LR transfer from the
baseline 275M optimum region. Cx2 was launched first, then Cx8. The Cx8
microbatch is `4` so the 96-sequence global batch divides evenly over 8 GPUs.

| Name | Variant | LR | Batch tokens | Batch seqs | GPUs | EP | Microbatch | Beaker experiment | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `eg-275m-cx2-eg24e2k-lr5e-4-r1` | `coarse_24e_top2` | 5e-4 | 524,288 | 64 | 2 | 1 | 16 | https://beaker.org/ex/01KTVTB301X3KC47AFF6PMDGWH | Transferred Cx2 sweep; started at launch check. |
| `eg-275m-cx2-eg24e2k-lr1e-3-r1` | `coarse_24e_top2` | 1e-3 | 524,288 | 64 | 2 | 1 | 16 | https://beaker.org/ex/01KTVTBEG3E629TDE325DPFQK6 | Transferred Cx2 sweep; started at launch check. |
| `eg-275m-cx2-eg24e2k-lr2e-3-r1` | `coarse_24e_top2` | 2e-3 | 524,288 | 64 | 2 | 1 | 16 | https://beaker.org/ex/01KTVTBRF083397K9V6HM278YC | Transferred Cx2 sweep; started at launch check. |
| `eg-275m-cx2-eg96e8k-lr5e-4-r1` | `fine_96e_top8` | 5e-4 | 524,288 | 64 | 2 | 1 | 8 | https://beaker.org/ex/01KTVTC4VB90FP20CSF459VSTX | Transferred Cx2 sweep; started at launch check. |
| `eg-275m-cx2-eg96e8k-lr1e-3-r1` | `fine_96e_top8` | 1e-3 | 524,288 | 64 | 2 | 1 | 8 | https://beaker.org/ex/01KTVTCF42VT40VADJNRQ8DZ3S | Transferred Cx2 sweep; started at launch check. |
| `eg-275m-cx2-eg96e8k-lr2e-3-r1` | `fine_96e_top8` | 2e-3 | 524,288 | 64 | 2 | 1 | 8 | https://beaker.org/ex/01KTVTCTVEAQ480W38BMGK21C1 | Transferred Cx2 sweep; started at launch check. |
| `eg-275m-cx8-eg24e2k-lr8e-4-r1` | `coarse_24e_top2` | 8e-4 | 786,432 | 96 | 8 | 1 | 4 | https://beaker.org/ex/01KTVTDHG4W7D77KTMK287N36Y | Finished step 40930, 32.189B tokens, avg100M 2.4990, avg250M 2.5035, avg500M 2.5017. W&B `zu643dvj`. |
| `eg-275m-cx8-eg24e2k-lr1.6e-3-r1` | `coarse_24e_top2` | 1.6e-3 | 786,432 | 96 | 8 | 1 | 4 | https://beaker.org/ex/01KTVTDXBT5YKA0Z2Q45P2TCNM | Finished step 40930, 32.189B tokens, avg100M 2.4947, avg250M 2.4990, avg500M 2.4973. W&B `ff9vq2dh`. Current best coarse Cx8 point, pending final `3.2e-3`. |
| `eg-275m-cx8-eg24e2k-lr3.2e-3-r1` | `coarse_24e_top2` | 3.2e-3 | 786,432 | 96 | 8 | 1 | 4 | https://beaker.org/ex/01KTVTE88FV3A9W5SQRYQ3Q4JP | Finished step 40930, 32.189B tokens, avg100M 2.5046, avg250M 2.5092, avg500M 2.5074. W&B `1d8wo3d7`. Coarse Cx8 is bracketed; `1.6e-3` remains best. |
| `eg-275m-cx8-eg96e8k-lr8e-4-r1` | `fine_96e_top8` | 8e-4 | 786,432 | 96 | 8 | 1 | 4 | https://beaker.org/ex/01KTVTEMFWXR4N9AW58HFDZ31Y | Running at 2026-06-12 status check; W&B `djnuz8yq` had just started, so no useful signal yet. |
| `eg-275m-cx8-eg96e8k-lr1.6e-3-r1` | `fine_96e_top8` | 1.6e-3 | 786,432 | 96 | 8 | 1 | 4 | https://beaker.org/ex/01KTVTF04AHCR2VP8DCPERSY7D | Transferred Cx8 sweep; queued/created at launch check. |
| `eg-275m-cx8-eg96e8k-lr3.2e-3-r1` | `fine_96e_top8` | 3.2e-3 | 786,432 | 96 | 8 | 1 | 4 | https://beaker.org/ex/01KTVTFBWY8CPQQNMXBCHFK029 | Transferred Cx8 sweep; queued/created at launch check. |

810M Cx1/Cx4 best-observed baseline LR transfer checks:

These runs test whether the expert-granularity LR transfer observed at 275M
continues to hold at 810M. They use the best observed trained baseline LRs,
not a fresh LR sweep: Cx1 `6e-4` and Cx4 `4e-4`. Launches use the canonical
810M settings from `SETTINGS_AUDIT.md`: EP=1, microbatch=4, 8 GPUs, with
Cx1 batch 262,144 / 32 seqs and Cx4 batch 524,288 / 64 seqs.

| Name | Variant | Cx | LR | Batch tokens | Batch seqs | GPUs | EP | Microbatch | Beaker experiment | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `eg-810m-cx1-eg24e2k-lr6e-4-r1` | `coarse_24e_top2` | 1 | 6e-4 | 262,144 | 32 | 8 | 1 | 4 | https://beaker.org/ex/01KTX8DY64MRW5DCWAJZPQY1YR | Finished successfully by 2026-06-13 status check. W&B `1nqxk9iw`, 13.790B tokens, avg250M `2.4191`; worse than baseline Cx1 `2.4104`. |
| `eg-810m-cx4-eg24e2k-lr4e-4-r1` | `coarse_24e_top2` | 4 | 4e-4 | 524,288 | 64 | 8 | 1 | 4 | https://beaker.org/ex/01KTXR4J7FN4ERB9BYDKC261F5 | Finished successfully by 2026-06-13 status check. W&B `q50qk891`, 55.158B tokens, avg250M `2.2585`; worse than baseline Cx4 `2.2424`. |
| `eg-810m-cx1-eg96e8k-lr6e-4-r1` | `fine_96e_top8` | 1 | 6e-4 | 262,144 | 32 | 8 | 1 | 4 | https://beaker.org/ex/01KTXR7563GGMW6FE57TTVACSY | Finished successfully by 2026-06-13 status check. W&B `wjto6qtp`, 13.825B tokens, avg250M `2.3985`; better than baseline Cx1 `2.4104`. |
| `eg-810m-cx4-eg96e8k-lr4e-4-r1` | `fine_96e_top8` | 4 | 4e-4 | 524,288 | 64 | 8 | 1 | 4 | https://beaker.org/ex/01KTXR9YA2QAR7HB1XS1R0FTBW | Finished successfully by 2026-06-13 status check. W&B `7cbm4c9b`, 55.299B tokens, avg250M `2.2353`; better than baseline Cx4 `2.2424`. |

275M Cx2 `b384k` batch-repair reruns:

These rerun the three comparable 275M Cx2 curves after discovering that the
original baseline Cx2 used 262,144 tokens / 32 seqs while the first
expert-granularity Cx2 jobs used 524,288 tokens / 64 seqs. The new canonical
Cx2 setting is 393,216 tokens / 48 seqs, 2 GPUs, EP=1, microbatch=8.
Treat the older Cx2 curves as diagnostic until these finish. The first two
`b384k` attempts were stopped before meaningful training:

- `r1`, LRs `5e-4`, `1e-3`, `2e-3`, Beaker IDs:
  `01KTWRZERD6B0AMVFFVM9S8NQB`, `01KTWRZTRYS6ZNY7JA0BQ104KF`,
  `01KTWS072YT07TW1GBWK7GT8FN`, `01KTWS0KBHE36Z442JHESC5ME8`,
  `01KTWS0YRMJ6YS7A68R5Z9N4KW`, `01KTWS1BRBPCMNYFMMG45J4EN2`,
  `01KTWS1R4QG212PVE1SDG5ZDKX`, `01KTWS2448JWCTHSQTSEJHEFQM`,
  `01KTWS2GJFBH1NJZ59JGA8DYZ9`.
- `r2`, LRs `8e-4`, `1.6e-3`, `3.2e-3`, Beaker IDs:
  `01KTWSNHBWR13VH3MNMEP3GSQ6`, `01KTWSNW9M0J8PM844036DWT9Z`,
  `01KTWSP8D5X4922FGAFNT3HFAC`, `01KTWSPM8W068K9R879H1K5H8J`,
  `01KTWSQ08PYQJ42HF8DPSPKJ1X`, `01KTWSQC03WSA1VZJQK0M1SQF1`,
  `01KTWSQQG44CKZNMG09VPPZEDD`, `01KTWSR3GJF9TNM12TBX6XG918`,
  `01KTWSRF3XZ8M981H3JRHAWGBQ`.

The canonical `r3` grid is centered on the fitted 275M Cx2 prediction
(`~1.75e-3` to `1.8e-3`) using `9e-4`, `1.8e-3`, and `3.6e-3`.

| Name | Variant | LR | Batch tokens | Batch seqs | GPUs | EP | Microbatch | Beaker experiment | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `olmoe3-tiny-275m-cx2-b384k-gpu2-ep1mb8-lr9e-4-r3` | baseline A0 | 9e-4 | 393,216 | 48 | 2 | 1 | 8 | https://beaker.org/ex/01KTWSSSBRVT08888H1QA5ZAMA | Finished step 20486, 8.055B tokens, avg100M 2.6613, avg250M 2.6605, avg500M 2.6619. W&B `atxrokcu`. |
| `olmoe3-tiny-275m-cx2-b384k-gpu2-ep1mb8-lr1.8e-3-r3` | baseline A0 | 1.8e-3 | 393,216 | 48 | 2 | 1 | 8 | https://beaker.org/ex/01KTWST9K76CKHW8PJEJNATKBE | Finished step 20486, 8.055B tokens, avg100M 2.6546, avg250M 2.6541, avg500M 2.6560. Best observed repaired 275M Cx2 point. W&B `lq4zvsx4`. |
| `olmoe3-tiny-275m-cx2-b384k-gpu2-ep1mb8-lr3.6e-3-r3` | baseline A0 | 3.6e-3 | 393,216 | 48 | 2 | 1 | 8 | https://beaker.org/ex/01KTWSTTHAPYKGXHACBYQ4AH9T | Finished step 20486, 8.055B tokens, avg100M 2.6612, avg250M 2.6610, avg500M 2.6637. W&B `pv6y1aqx`. |
| `eg-275m-cx2-b384k-eg24e2k-lr9e-4-r3` | `coarse_24e_top2` | 9e-4 | 393,216 | 48 | 2 | 1 | 8 | https://beaker.org/ex/01KTWSV732H19AQCQVNTW3MF86 | Canonical Cx2 batch-repair rerun; queued/created at launch check. |
| `eg-275m-cx2-b384k-eg24e2k-lr1.8e-3-r3` | `coarse_24e_top2` | 1.8e-3 | 393,216 | 48 | 2 | 1 | 8 | https://beaker.org/ex/01KTWSVGQD9TPX9ERFPXXYESF5 | Canonical Cx2 batch-repair rerun; queued/created at launch check. |
| `eg-275m-cx2-b384k-eg24e2k-lr3.6e-3-r3` | `coarse_24e_top2` | 3.6e-3 | 393,216 | 48 | 2 | 1 | 8 | https://beaker.org/ex/01KTWSVW0JAJSVFQ1Q9FCS4FQ0 | Canonical Cx2 batch-repair rerun; queued/created at launch check. |
| `eg-275m-cx2-b384k-eg96e8k-lr9e-4-r3` | `fine_96e_top8` | 9e-4 | 393,216 | 48 | 2 | 1 | 8 | https://beaker.org/ex/01KTWSW8G04HREZK5W7W8FF43Y | Canonical Cx2 batch-repair rerun; queued/created at launch check. |
| `eg-275m-cx2-b384k-eg96e8k-lr1.8e-3-r3` | `fine_96e_top8` | 1.8e-3 | 393,216 | 48 | 2 | 1 | 8 | https://beaker.org/ex/01KTWSWN0F4QTW6DMYHG0GY1VV | Canonical Cx2 batch-repair rerun; queued/created at launch check. |
| `eg-275m-cx2-b384k-eg96e8k-lr3.6e-3-r3` | `fine_96e_top8` | 3.6e-3 | 393,216 | 48 | 2 | 1 | 8 | https://beaker.org/ex/01KTWSX298YRHX43R7SA55T9PY | Canonical Cx2 batch-repair rerun; queued/created at launch check. |

mid_480m and 810M Cx2 `b384k` batch-repair reruns:

These extend the smoother Cx2 batch repair to larger baseline sizes. 1.2B Cx2
is intentionally not launched yet. Treat the older `b512k` Cx2 curves for
mid_480m and 810M as diagnostic once these finish.

| Name | Model | LR | Batch tokens | Batch seqs | GPUs | EP | Microbatch | Beaker experiment | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `m480-cx2-b384k-gpu4-ep1mb4-lr4.5e-4-r1` | `mid_480m` | 4.5e-4 | 393,216 | 48 | 4 | 1 | 4 | https://beaker.org/ex/01KTWV11QM0BKFWXZ8QGPQXHTP | Cx2 repair sweep centered at predicted LR `9e-4`; queued/created at launch check. |
| `m480-cx2-b384k-gpu4-ep1mb4-lr9e-4-r1` | `mid_480m` | 9e-4 | 393,216 | 48 | 4 | 1 | 4 | https://beaker.org/ex/01KTWV1E704BPEXJK95HPJ22T4 | Cx2 repair sweep centered at predicted LR `9e-4`; queued/created at launch check. |
| `m480-cx2-b384k-gpu4-ep1mb4-lr1.8e-3-r1` | `mid_480m` | 1.8e-3 | 393,216 | 48 | 4 | 1 | 4 | https://beaker.org/ex/01KTWV1TRD9BQG0JFB5BXYZ2C6 | Cx2 repair sweep centered at predicted LR `9e-4`; queued/created at launch check. |
| `olmoe3-moe-a0-810m-cx2-b384k-gpu8-ep1mb2-lr2.8e-4-r1` | `810m` | 2.8e-4 | 393,216 | 48 | 8 | 1 | 2 | https://beaker.org/ex/01KTWV28D8KFYH2KJ9835S3C48 | Failed before training on 2026-06-12 with distributed startup/checkpointer error: `RuntimeError: ... gloo/transport/tcp/pair.cc:547] Connection closed by peer` during `Checkpointer.pre_train()`. Treat as infrastructure/startup failure, not a model/LR result. Retried as `r2`. |
| `olmoe3-moe-a0-810m-cx2-b384k-gpu8-ep1mb2-lr5.6e-4-r1` | `810m` | 5.6e-4 | 393,216 | 48 | 8 | 1 | 2 | https://beaker.org/ex/01KTWV2PB16G6R4JP34CHY9NC3 | Failed before training on 2026-06-12 with the same distributed startup/checkpointer error as the other 810M Cx2 `r1` repair jobs. Treat as infrastructure/startup failure. Retried as `r2`. |
| `olmoe3-moe-a0-810m-cx2-b384k-gpu8-ep1mb2-lr1.12e-3-r1` | `810m` | 1.12e-3 | 393,216 | 48 | 8 | 1 | 2 | https://beaker.org/ex/01KTWV33CYVNDPZ3FGWSXWYXPA | Failed before training on 2026-06-12 with the same distributed startup/checkpointer error as the other 810M Cx2 `r1` repair jobs. Treat as infrastructure/startup failure. Retried as `r2`. |
| `m480-cx2-b384k-gpu4-ep1mb4-lr4.5e-4-r2` | `mid_480m` | 4.5e-4 | 393,216 | 48 | 4 | 1 | 4 | https://beaker.org/ex/01KTXRKKTJMXPCR64DMQT2CR41 | Accidental duplicate retry created while relaunching only 810M Cx2 repairs; stopped immediately on 2026-06-12. Ignore. |
| `m480-cx2-b384k-gpu4-ep1mb4-lr9e-4-r2` | `mid_480m` | 9e-4 | 393,216 | 48 | 4 | 1 | 4 | https://beaker.org/ex/01KTXRKYEBSZY146G0EYDJ4FXD | Accidental duplicate retry created while relaunching only 810M Cx2 repairs; stopped immediately on 2026-06-12. Ignore. |
| `m480-cx2-b384k-gpu4-ep1mb4-lr1.8e-3-r2` | `mid_480m` | 1.8e-3 | 393,216 | 48 | 4 | 1 | 4 | https://beaker.org/ex/01KTXRMAHNAHEV0SV04BTPJ60F | Accidental duplicate retry created while relaunching only 810M Cx2 repairs; stopped immediately on 2026-06-12. Ignore. |
| `olmoe3-moe-a0-810m-cx2-b384k-gpu8-ep1mb2-lr2.8e-4-r2` | `810m` | 2.8e-4 | 393,216 | 48 | 8 | 1 | 2 | https://beaker.org/ex/01KTXRMPNEYF37JVNREN6WZV5K | Failed before training on 2026-06-12 because W&B rejected the generated group name: `invalid parameters: 128 limit exceeded for GroupName`. Code fix `07df5ad` shortens W&B groups; retried as `r3`. Ignore for analysis. |
| `olmoe3-moe-a0-810m-cx2-b384k-gpu8-ep1mb2-lr5.6e-4-r2` | `810m` | 5.6e-4 | 393,216 | 48 | 8 | 1 | 2 | https://beaker.org/ex/01KTXRN76Y3J0TXX9HZN7YNMQX | Failed before training on 2026-06-12 with the same W&B `GroupName` length error as the other `r2` jobs. Code fix `07df5ad` shortens W&B groups; retried as `r3`. Ignore for analysis. |
| `olmoe3-moe-a0-810m-cx2-b384k-gpu8-ep1mb2-lr1.12e-3-r2` | `810m` | 1.12e-3 | 393,216 | 48 | 8 | 1 | 2 | https://beaker.org/ex/01KTXRNNPV1134T95XAKC2HJWE | Failed before training on 2026-06-12 with the same W&B `GroupName` length error as the other `r2` jobs. Code fix `07df5ad` shortens W&B groups; retried as `r3`. Ignore for analysis. |
| `olmoe3-moe-a0-810m-cx2-b384k-gpu8-ep1mb2-lr2.8e-4-r3` | `810m` | 2.8e-4 | 393,216 | 48 | 8 | 1 | 2 | https://beaker.org/ex/01KTYEJH9S58TDA837YZANCJ9C | Finished successfully by 2026-06-13 status check. W&B `uh4el1df`, 27.603B tokens, avg250M `2.3333`. |
| `olmoe3-moe-a0-810m-cx2-b384k-gpu8-ep1mb2-lr5.6e-4-r3` | `810m` | 5.6e-4 | 393,216 | 48 | 8 | 1 | 2 | https://beaker.org/ex/01KTYEJWXR7X0KFBFQ57G9YC9V | Finished successfully by 2026-06-13 status check. W&B `v5puakhq`, 27.603B tokens, avg250M `2.3204`; best observed repaired 810M Cx2 point. |
| `olmoe3-moe-a0-810m-cx2-b384k-gpu8-ep1mb2-lr1.12e-3-r3` | `810m` | 1.12e-3 | 393,216 | 48 | 8 | 1 | 2 | https://beaker.org/ex/01KTYEK7S5MXJYSJMGB3BNR1HE | Finished successfully by 2026-06-13 status check. W&B `sxivrph5`, 27.603B tokens, avg250M `2.3268`; hot side is worse than `5.6e-4`, so repaired 810M Cx2 is bracketed. |

275M extreme-granularity Cx1 probes:

These diagnostic probes test whether the `fine_96e_top8` improvement continues
to even higher top-k and smaller per-expert hidden sizes. They use the same Cx1
LR triplet as the earlier expert-granularity transfer probes. Treat these as
small probes, not automatic full-ladder variants.

| Name | Variant | LR | Batch tokens | Batch seqs | GPUs | EP | Microbatch | Beaker experiment | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `eg-275m-cx1-eg192e16k-lr1e-3-r1` | `extreme_192e_top16` | 1e-3 | 262,144 | 32 | 1 | 1 | 4 | https://beaker.org/ex/01KTVV5B6ZWBR64FACMZH96TMQ | Extreme granularity probe; finalized by 2026-06-12 status check. |
| `eg-275m-cx1-eg192e16k-lr2e-3-r1` | `extreme_192e_top16` | 2e-3 | 262,144 | 32 | 1 | 1 | 4 | https://beaker.org/ex/01KTVV5QVXKG7D0MJC304SVSDM | Extreme granularity probe; finalized by 2026-06-12 status check. |
| `eg-275m-cx1-eg192e16k-lr4e-3-r1` | `extreme_192e_top16` | 4e-3 | 262,144 | 32 | 1 | 1 | 4 | https://beaker.org/ex/01KTVV63GS5WFC8VVFYDY24183 | Extreme granularity probe; finalized by 2026-06-12 status check. |
| `eg-275m-cx1-eg384e32k-lr1e-3-r1` | `ultra_384e_top32` | 1e-3 | 262,144 | 32 | 1 | 1 | 2 | https://beaker.org/ex/01KTVV6EZ981F8YJ8B2M6X1S57 | Ultra granularity probe; finalized by 2026-06-12 status check. |
| `eg-275m-cx1-eg384e32k-lr2e-3-r1` | `ultra_384e_top32` | 2e-3 | 262,144 | 32 | 1 | 1 | 2 | https://beaker.org/ex/01KTVV6TWVNSJGTMHPWY9JS4Y6 | Ultra granularity probe; running at 2026-06-12 status check. |
| `eg-275m-cx1-eg384e32k-lr4e-3-r1` | `ultra_384e_top32` | 4e-3 | 262,144 | 32 | 1 | 1 | 2 | https://beaker.org/ex/01KTVV76SNY7XF5AER7ST06F04 | Ultra granularity probe; running at 2026-06-12 status check. |

## 2026-06-11 Total Sparsity Experiment

First-wave variants:

- `sp96e4k`: 96 experts, top-4, `moe_hidden_size=d_model`; exact 275M dry-run
  count is 278,856,192 active params / 2,069,561,856 total params = 13.47%
  active/total.
- `sp192e4k`: 192 experts, top-4, `moe_hidden_size=d_model`; exact 275M dry-run
  count is 279,667,200 active params / 3,938,935,296 total params = 7.10%
  active/total.

The first wave intentionally does not include `sp24e4k`, because that is less
sparse than the baseline and is reserved for a later diagnostic only.

Smoke tests:

| Name | Variant | LR | Batch tokens | Batch seqs | GPUs | EP | Microbatch | Beaker experiment | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `sp-smoke-sp96e4k-lr2e-3-r1` | `high_total_96e_top4` | 2e-3 | 262,144 | 32 | 4 | 1 | 4 | https://beaker.org/ex/01KTWFC73099P7Y0TVCRGJSZHZ | Total-sparsity smoke from commit `22aeaab`; finished cleanly at step 308, skipped steps 0, peak reserved memory 55.9 GiB. |
| `sp-smoke-sp192e4k-lr2e-3-r1` | `huge_total_192e_top4` | 2e-3 | 262,144 | 32 | 4 | 1 | 4 | https://beaker.org/ex/01KTWFCK6QBQ4QH5X3TBKF98MA | Total-sparsity smoke from commit `22aeaab`; finished cleanly at step 310, skipped steps 0, peak reserved memory 79.3 GiB. |

275M Cx1/Cx4 transferred LR grids:

| Name | Variant | Cx | LR | Batch tokens | Batch seqs | GPUs | EP | Microbatch | Beaker experiment | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `sp-275m-cx1-sp96e4k-lr1e-3-r1` | `high_total_96e_top4` | 1 | 1e-3 | 262,144 | 32 | 4 | 1 | 4 | https://beaker.org/ex/01KTWGMXVT9N3GSXEBYC2H5KV3 | Finished normally on 2026-06-12 before the sparsity pause. Treat as diagnostic-only for now; do not launch further sparsity work until explicitly resumed. |
| `sp-275m-cx1-sp96e4k-lr2e-3-r1` | `high_total_96e_top4` | 1 | 2e-3 | 262,144 | 32 | 4 | 1 | 4 | https://beaker.org/ex/01KTWGNAEEZZ71FPN3GPGFRW89 | Stopped intentionally on 2026-06-12 after starting, to pause sparsity work and focus on expert granularity. Ignore unless explicitly resumed. |
| `sp-275m-cx1-sp96e4k-lr4e-3-r1` | `high_total_96e_top4` | 1 | 4e-3 | 262,144 | 32 | 4 | 1 | 4 | https://beaker.org/ex/01KTWGNPBFPAJWYPQ0Z2VFN4YY | Stopped intentionally on 2026-06-12 before start. Ignore unless explicitly resumed. |
| `sp-275m-cx4-sp96e4k-lr8e-4-r1` | `high_total_96e_top4` | 4 | 8e-4 | 524,288 | 64 | 4 | 1 | 4 | https://beaker.org/ex/01KTWGP25M12KH4YYA9W0FXMFJ | Stopped intentionally on 2026-06-12 before start. Ignore unless explicitly resumed. |
| `sp-275m-cx4-sp96e4k-lr1.6e-3-r1` | `high_total_96e_top4` | 4 | 1.6e-3 | 524,288 | 64 | 4 | 1 | 4 | https://beaker.org/ex/01KTWGPDKXH3953BDF61N0G57Q | Stopped intentionally on 2026-06-12 before start. Ignore unless explicitly resumed. |
| `sp-275m-cx4-sp96e4k-lr3.2e-3-r1` | `high_total_96e_top4` | 4 | 3.2e-3 | 524,288 | 64 | 4 | 1 | 4 | https://beaker.org/ex/01KTWGPSSMJZ5Z002E57BK64RE | Stopped intentionally on 2026-06-12 before start. Ignore unless explicitly resumed. |
| `sp-275m-cx1-sp192e4k-lr1e-3-r1` | `huge_total_192e_top4` | 1 | 1e-3 | 262,144 | 32 | 4 | 1 | 4 | https://beaker.org/ex/01KTWGQ57J8FW4F95CM27ASC99 | Stopped intentionally on 2026-06-12 before start. Ignore unless explicitly resumed. |
| `sp-275m-cx1-sp192e4k-lr2e-3-r1` | `huge_total_192e_top4` | 1 | 2e-3 | 262,144 | 32 | 4 | 1 | 4 | https://beaker.org/ex/01KTWGQH2M6JDTXZ2VKNYF7Z0E | Stopped intentionally on 2026-06-12 before start. Ignore unless explicitly resumed. |
| `sp-275m-cx1-sp192e4k-lr4e-3-r1` | `huge_total_192e_top4` | 1 | 4e-3 | 262,144 | 32 | 4 | 1 | 4 | https://beaker.org/ex/01KTWGQWXG9TZAT8SSZ231TM0M | Stopped intentionally on 2026-06-12 before start. Ignore unless explicitly resumed. |
| `sp-275m-cx4-sp192e4k-lr8e-4-r1` | `huge_total_192e_top4` | 4 | 8e-4 | 524,288 | 64 | 4 | 1 | 4 | https://beaker.org/ex/01KTWGR9QHSDAS5RCV7NZV5GJ7 | Stopped intentionally on 2026-06-12 before start. Ignore unless explicitly resumed. |
| `sp-275m-cx4-sp192e4k-lr1.6e-3-r1` | `huge_total_192e_top4` | 4 | 1.6e-3 | 524,288 | 64 | 4 | 1 | 4 | https://beaker.org/ex/01KTWGRN09127CB35E5SGS0B03 | Stopped intentionally on 2026-06-12 before start. Ignore unless explicitly resumed. |
| `sp-275m-cx4-sp192e4k-lr3.2e-3-r1` | `huge_total_192e_top4` | 4 | 3.2e-3 | 524,288 | 64 | 4 | 1 | 4 | https://beaker.org/ex/01KTWGS0RHZ743VAGKQ9R85JG5 | Stopped intentionally on 2026-06-12 before start. Ignore unless explicitly resumed. |

## 2026-06-12 Stable-Name Launch Bundle

This bundle uses semantic, resume-stable names: model/variant, Cx, batch policy
when optimizer-relevant, LR, and attempt id. Node count, GPU count, EP, and
microbatch are recorded in W&B/Beaker tags/config and in this ledger, not in the
new run names. The Beaker runtime still uses the committed compatibility script
`src/scripts/train/jacobm_olmoe_ladder/tiny_275m.py`; local docs/scripts may
refer to the preferred `moe_a0_ladder.py` name after that file is committed.

An initial launch attempt used the local-only `moe_a0_ladder.py` path before it
was present in the remote commit used by Beaker. These experiments were stopped
immediately and should be ignored: `01KTZ0WKC2KM7YMBFJRG3SEJ3H`,
`01KTZ0WZD0EGG8JBAQTW8RWTX6`, `01KTZ0XBA98Z2HSVMWP4355PNA`,
`01KTZ0Y0GDH21QDKNK61Q7DZDT`, `01KTZ0YC8NEE1GEFMDX9Y4DAZ9`,
`01KTZ0YR3VBQ8PN4SBQ34DDEMV`, `01KTZ0Z26Q4QK3HC0RXM9ZFYJ0`,
`01KTZ0ZE3018K63YY9XDXZJG16`, `01KTZ0ZRJGZFF8YG0N8B2T9GDS`,
`01KTZ104BRE6X8HKNQR42BZFXE`, `01KTZ10FXNJ555R9S2EDENDRGG`.

1.2B Cx2 `b384k` baseline:

| Name | Model | LR | Batch tokens | Batch seqs | GPUs | EP | Microbatch | Beaker experiment | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `olmoe3-moe-a0-1p2b-cx2-b384k-lr1.5e-4-r1` | `1p2b` | 1.5e-4 | 393,216 | 48 | 8 | 1 | 2 | https://beaker.org/ex/01KTZ15R9Q87WDSPN6740YQX06 | Running at 2026-06-13 status check. W&B `dtd8qeiv`, 15.639B tokens, live CE summary `2.5166`. Stable run name; systems settings in tags. |
| `olmoe3-moe-a0-1p2b-cx2-b384k-lr3e-4-r1` | `1p2b` | 3e-4 | 393,216 | 48 | 8 | 1 | 2 | https://beaker.org/ex/01KTZ163MP8B3VFGGY7YWHSFB6 | First job failed before training on 2026-06-13 with `ModuleNotFoundError: olmo_core` on a B200 node after Gantry setup; experiment now has a fresh pending retry. Do not treat as model/LR signal. |
| `olmoe3-moe-a0-1p2b-cx2-b384k-lr6e-4-r1` | `1p2b` | 6e-4 | 393,216 | 48 | 8 | 1 | 2 | https://beaker.org/ex/01KTZ16ETE4R5XZYHKSB8SXDXM | Running at 2026-06-13 status check. W&B `54pt8zj7`, 14.950B tokens, live CE summary `2.3018`. Stable run name; systems settings in tags. |

480M expert-granularity full ladder at predicted LRs:

| Name | Variant | Cx | LR | Batch tokens | Batch seqs | GPUs | EP | Microbatch | Beaker experiment | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `eg-480m-cx1-eg24e2k-lr9e-4-r1` | `coarse_24e_top2` | 1 | 9e-4 | 262,144 | 32 | 4 | 1 | 8 | https://beaker.org/ex/01KTZ17462R909ZKQB3X6SXDH8 | Finished by 2026-06-13 status check. W&B `rcgxm5qv`, 7.601B tokens, avg250M `2.5817`. Runtime argument uses `--model-size=mid_480m` for compatibility; name/docs use canonical `480m`. |
| `eg-480m-cx2-eg24e2k-lr1e-3-r1` | `coarse_24e_top2` | 2 | 1e-3 | 393,216 | 48 | 4 | 1 | 4 | https://beaker.org/ex/01KTZ17FZN0J2F45HRYPP8R0R2 | Finished by 2026-06-13 status check. W&B `ksfrmhct`, 15.201B tokens, avg250M `2.4767`. Predicted-LR full-ladder run with repaired `b384k` Cx2. Runtime argument uses `--model-size=mid_480m` for compatibility. |
| `eg-480m-cx4-eg24e2k-lr8e-4-r1` | `coarse_24e_top2` | 4 | 8e-4 | 524,288 | 64 | 4 | 1 | 8 | https://beaker.org/ex/01KTZ17W6A94BW60A8DS1C9CXA | Running at 2026-06-13 status check. W&B `wq8gib5l`, 18.315B tokens, live CE summary `2.5611`. Runtime argument uses `--model-size=mid_480m` for compatibility. |
| `eg-480m-cx8-eg24e2k-lr8e-4-r1` | `coarse_24e_top2` | 8 | 8e-4 | 786,432 | 96 | 8 | 1 | 4 | https://beaker.org/ex/01KTZ187DS1DCVR3F67HTMPGWF | Running at 2026-06-13 status check. W&B `epx7o7ty`, 29.148B tokens, live CE summary `2.4715`. Runtime argument uses `--model-size=mid_480m` for compatibility. |
| `eg-480m-cx1-eg96e8k-lr1e-3-r1` | `fine_96e_top8` | 1 | 1e-3 | 262,144 | 32 | 4 | 1 | 8 | https://beaker.org/ex/01KTZ18KQDJ02P9B7C074Q5S86 | Finished by 2026-06-13 status check. W&B `nvndg2tr`, 7.623B tokens, avg250M `2.5546`. Runtime argument uses `--model-size=mid_480m` for compatibility. |
| `eg-480m-cx2-eg96e8k-lr1e-3-r1` | `fine_96e_top8` | 2 | 1e-3 | 393,216 | 48 | 4 | 1 | 4 | https://beaker.org/ex/01KTZ18Z8EK00WCE2G3AVCFCNM | Running at 2026-06-13 status check. W&B `fzk2affn`, 7.058B tokens, live CE summary `2.7996`. Predicted-LR full-ladder run with repaired `b384k` Cx2. Runtime argument uses `--model-size=mid_480m` for compatibility. |
| `eg-480m-cx4-eg96e8k-lr8e-4-r1` | `fine_96e_top8` | 4 | 8e-4 | 524,288 | 64 | 4 | 1 | 8 | https://beaker.org/ex/01KTZ19B5WKD9EEMGWYRH3HQ7B | Running at 2026-06-13 status check. W&B `ezokso90`, 7.711B tokens, live CE summary `2.6682`. Runtime argument uses `--model-size=mid_480m` for compatibility. |
| `eg-480m-cx8-eg96e8k-lr8e-4-r1` | `fine_96e_top8` | 8 | 8e-4 | 786,432 | 96 | 8 | 1 | 4 | https://beaker.org/ex/01KTZ19QVD4G1MJN612YENYA7T | Running at 2026-06-13 status check. W&B `8676ezla`, 10.789B tokens, live CE summary `2.7340`. Runtime argument uses `--model-size=mid_480m` for compatibility. |
