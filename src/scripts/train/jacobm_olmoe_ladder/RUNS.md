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
| 2026-06-06 | `olmoe3-moe-a0-810m-cx4-b512k-gpu8-ep1mb4-lr4e-4-r1` | `tiny_275m.py` | 4x | 524,288 | 64 | 4e-4 | `01KTDDB3P13BMCY965FKKFC8VV` | https://beaker.org/ex/01KTDDB3P13BMCY965FKKFC8VV | 810M Cx4 four-point sweep centered around the transfer-calibrated Cx4 estimate. Finished step 105295, 55.205B tokens, avg100M 2.2364, avg250M 2.2427, avg500M 2.2417. Best finished Cx4 point so far while `8e-4` and `1.6e-3` are still running. 8 GPUs, EP=1, microbatch=4, final-only permanent checkpoint plus ephemeral resume checkpoints. |
| 2026-06-06 | `olmoe3-moe-a0-810m-cx4-b512k-gpu8-ep1mb4-lr8e-4-r1` | `tiny_275m.py` | 4x | 524,288 | 64 | 8e-4 | `01KTDDBGWNHKZ3F4Q1VJD2RPTS` | https://beaker.org/ex/01KTDDBGWNHKZ3F4Q1VJD2RPTS | 810M Cx4 four-point sweep centered around the transfer-calibrated Cx4 estimate. Finished step 105295, 55.205B tokens, avg100M 2.2387, avg250M 2.2451, avg500M 2.2442. Very close to but slightly worse than `4e-4`; `1.6e-3` still running. 8 GPUs, EP=1, microbatch=4, final-only permanent checkpoint plus ephemeral resume checkpoints. |
| 2026-06-06 | `olmoe3-moe-a0-810m-cx4-b512k-gpu8-ep1mb4-lr1.6e-3-r1` | `tiny_275m.py` | 4x | 524,288 | 64 | 1.6e-3 | `01KTDDBWP8EC88V8SDW5Y0VTSV` | https://beaker.org/ex/01KTDDBWP8EC88V8SDW5Y0VTSV | 810M Cx4 four-point sweep centered around the transfer-calibrated Cx4 estimate. 8 GPUs, EP=1, microbatch=4, final-only permanent checkpoint plus ephemeral resume checkpoints. |
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
checkpoint backfills should be copied back to the source training W&B summaries
with `copy_eval_backfills_to_wandb.py` as they finish.

| Name | Beaker experiment | Source checkpoint |
| --- | --- | --- |
| `eval-810m-cx4-lr2e-4-r1` | https://beaker.org/ex/01KTFFM30FQYJ6RR76826V58QM | `olmoe3-moe-a0-810m-cx4-b512k-gpu8-ep1mb4-lr2e-4-r1/step105295` |
| `eval-810m-cx4-lr4e-4-r1` | https://beaker.org/ex/01KTFFMFKM2D7FR4EMRKEWXGFP | `olmoe3-moe-a0-810m-cx4-b512k-gpu8-ep1mb4-lr4e-4-r1/step105295` |
| `eval-810m-cx4-lr8e-4-r1` | https://beaker.org/ex/01KTFK9M9SP2EZ7DRY4VHZCHSA | `olmoe3-moe-a0-810m-cx4-b512k-gpu8-ep1mb4-lr8e-4-r1/step105295` |
